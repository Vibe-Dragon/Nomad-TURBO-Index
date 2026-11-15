"""Handoff rules for stock NOMAD.

These rules follow the stock scanner, not any in-container speed patch.
Setting the ingest policy to Manual does not stop jobs already in
file-embeddings. BullMQ pause does not stop handlers that are already
active. Offsets are safe to read only after that queue's active count is 0.
"""

from __future__ import annotations

import re


def offsets_are_safe(active_count: int) -> bool:
    """True only when file-embeddings has no active job.

    Setting ingest to Manual does not stop a job stock NOMAD already started.
    Turbo waits for this before it reads resume offsets.
    """
    return int(active_count) == 0


# After a download disappears, NOMAD may still be deleting that file's points.
DOWNLOAD_SETTLE_SECONDS = 120

# /proc/pressure/io percentages. Full means every task was stalled on disk.
IO_FULL_BUSY = 40.0
IO_SOME_BUSY = 70.0


def remembered_policy(value: str | None) -> str:
    """Only an exact Always is turned back on.

    A blank or unknown value stays Manual. A missed read must not switch
    stock indexing on.
    """
    if (value or "").strip() == "Always":
        return "Always"
    return "Manual"


def policy_after_stop(previous: str | None, partial_remains: bool) -> str:
    """Put the ingest policy back to the value turbo found at Start.

    Always plus a half-finished file makes NOMAD queue that file again from
    the first article. That one case stays Manual. Anything else returns the
    found value. An unknown value stays Manual and is not turned into Always.
    """
    found = remembered_policy(previous)
    if partial_remains and found != "Manual":
        return "Manual"
    return found


def stock_should_continue(previous: str | None) -> bool:
    """Hand unfinished files to stock NOMAD only when the run started on Always.

    Manual means stock indexing stays off. Turbo keeps the checkpoint.
    """
    return remembered_policy(previous) == "Always"


def stop_status_note(previous: str | None) -> str:
    """Status after Stop. Manual runs do not hand unfinished files back.

    Always is the only saved policy that gives those files to stock NOMAD.
    Anything else stays Manual.
    """
    if stock_should_continue(previous):
        return "Stopped. Unfinished files were handed back to NOMAD."
    return (
        "Stopped. Unfinished files were not handed back to NOMAD. "
        "The ingest policy stayed Manual."
    )


def finished_status_note(previous: str | None) -> str:
    """Status when the queue finishes. Manual does not start stock indexing.

    The policy found at Start is restored. Always turns stock indexing back on.
    Manual leaves it off.
    """
    if stock_should_continue(previous):
        return "Finished. Indexing was handed back to NOMAD."
    return (
        "Finished. Indexing was not handed back to NOMAD. "
        "The ingest policy stayed Manual."
    )


def idle_handoff_note(previous: str | None) -> str:
    """Note when Stay-on is off and the queue is empty.

    Off restores the policy found at Start. A Manual start does not resume
    stock embedding.
    """
    if stock_should_continue(previous):
        return "Nothing left to index. NOMAD is embedding again."
    return (
        "Nothing left to index. The ingest policy stayed Manual. "
        "Stock indexing was not started."
    )


def parse_io_pressure(text: str) -> tuple[float, float]:
    """avg10 some and full stalls from /proc/pressure/io. Missing lines are 0."""
    some = 0.0
    full = 0.0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        avg10 = 0.0
        for item in parts[1:]:
            if not item.startswith("avg10="):
                continue
            try:
                avg10 = float(item.split("=", 1)[1])
            except ValueError:
                avg10 = 0.0
        if parts[0] == "some":
            some = avg10
        elif parts[0] == "full":
            full = avg10
    return some, full


def write_hold(
    *,
    downloads_pending: bool,
    seconds_since_download: float | None,
    io_some_avg10: float,
    io_full_avg10: float,
    settle_seconds: float = DOWNLOAD_SETTLE_SECONDS,
) -> str:
    """Why a knowledge-base write should wait. Empty means write now.

    download: a file is still arriving. settle: one just finished and NOMAD
    may still be updating Qdrant. disk: the disks are stalled, so the next
    write waits once and then goes ahead.
    """
    if downloads_pending:
        return "download"
    if seconds_since_download is not None and seconds_since_download < settle_seconds:
        return "settle"
    if io_full_avg10 >= IO_FULL_BUSY or io_some_avg10 >= IO_SOME_BUSY:
        return "disk"
    return ""


def drop_should_wait(reason: str, waited_s: float, *, limit_s: float = 600) -> bool:
    """A replaced file's point delete waits out a download. A disk stall does not wait forever."""
    if reason in ("download", "settle"):
        return True
    if reason == "disk" and waited_s < limit_s:
        return True
    return False


def hand_back_when_idle(*, keep_control: bool, downloads_pending: bool) -> bool:
    """An empty queue ends the run, unless the user asked turbo to stay.

    Ending the run restores the ingest policy found at Start. A run that
    found Manual does not start stock indexing. A download still being
    written is not an empty queue. Turbo waits for that file either way.
    """
    if keep_control or downloads_pending:
        return False
    return True


def hold_queue_on_exit(touched: bool, failed_before_work: bool) -> bool:
    """A startup failure must not release the next NOMAD batches.

    Pause already let the in-flight batches finish and held their continuations.
    Resuming after a crash before turbo writes anything is what starts those
    new batches. A clean stop, or a failure after turbo has indexed something,
    hands the queue back.
    """
    return failed_before_work and not touched


def partial_row_state(existing_state: str | None) -> str | None:
    """Row to store for a file turbo did not finish.

    No row plus any Qdrant points becomes backfill_indexed on the next scan,
    which abandons the rest of the file. A new partial file therefore gets
    pending_decision. An existing indexed row is left alone: stock treats
    indexed plus points as skip, and the continuation job still finishes it.
    """
    if existing_state is None:
        return "pending_decision"
    return None


def ingest_write(*, done: bool, chunks: int, existing_state: str | None) -> tuple[str, int] | None:
    """Ingest row to write when a file opens, restarts, or finishes.

    A finished file is indexed with the chunk count from that finish.
    A partial file stays pending_decision, including when a restart finds
    the same unfinished file. An indexed row is not demoted.
    """
    count = max(0, int(chunks))
    if done:
        return "indexed", count
    if existing_state == "indexed":
        return None
    return "pending_decision", count


def record_file_finish(store, path: str, *, done: bool, chunks: int, existing_state: str | None) -> str | None:
    """Write the finish decision through a store that records rows."""
    choice = ingest_write(done=done, chunks=chunks, existing_state=existing_state)
    if choice is None:
        return None
    state, count = choice
    if state == "indexed":
        store.mark_indexed(path, count)
    else:
        store.mark_pending(path, count)
    return state


_EDITION_NAME = re.compile(r"^(?P<family>.+)_(?P<version>\d{4}-\d{2})\.zim$", re.IGNORECASE)


def zim_edition(path: str) -> tuple[str, str] | None:
    """Family name and YYYY-MM version from a Kiwix ZIM path. None for other files."""
    name = (path or "").replace("\\", "/").rsplit("/", 1)[-1]
    match = _EDITION_NAME.match(name)
    if match is None:
        return None
    return match.group("family"), match.group("version")


def previous_edition_path(new_path: str, candidates: list[str]) -> str | None:
    """The newest other edition of this ZIM. The new path itself is not returned."""
    parsed = zim_edition(new_path)
    if parsed is None:
        return None
    family, version = parsed
    parent = (new_path or "").replace("\\", "/").rsplit("/", 1)[0]
    best_path = None
    best_version = ""
    for candidate in candidates:
        other = zim_edition(candidate)
        if other is None or other[0] != family or other[1] == version:
            continue
        text = candidate.replace("\\", "/")
        if text.rsplit("/", 1)[0] != parent:
            continue
        if text == (new_path or "").replace("\\", "/"):
            continue
        if other[1] > best_version:
            best_version = other[1]
            best_path = candidate
    return best_path


def edition_offered_only(path: str, offered_version: str | None, offered_on_disk: bool) -> bool:
    """True when a newer edition is only offered and this file is still installed."""
    parsed = zim_edition(path)
    if parsed is None or not (offered_version or "").strip():
        return False
    if offered_on_disk:
        return False
    return parsed[1] != str(offered_version).strip()


def offered_edition_path(installed_path: str, offered_version: str | None) -> str | None:
    """Path the offered edition would have, beside the file that is installed."""
    parsed = zim_edition(installed_path)
    if parsed is None or not (offered_version or "").strip():
        return None
    family, _version = parsed
    text = installed_path.replace("\\", "/")
    parent, _name = text.rsplit("/", 1)
    return f"{parent}/{family}_{str(offered_version).strip()}.zim"


def replacement_delete(
    *,
    new_path: str,
    previous_path: str | None,
    previous_on_disk: bool,
    previous_state: str | None,
    upsert_in_flight: bool,
    offered_only: bool,
) -> str | None:
    """Old path to delete before the new edition is embedded.

    A pending_decision row and a missing row both still name the old path.
    Nothing is deleted while an upsert is in flight, and a file that is still
    installed is kept when a newer edition is only offered. The new path is
    never the delete target.
    """
    del previous_state
    if upsert_in_flight or offered_only or previous_on_disk:
        return None
    if not previous_path or previous_path == new_path:
        return None
    return previous_path


def delete_replaced_edition(
    store,
    *,
    new_path: str,
    previous_path: str | None,
    previous_on_disk: bool,
    previous_state: str | None,
    upsert_in_flight: bool,
    offered_only: bool,
) -> str | None:
    """Delete only the previous path, through a store that records the call."""
    target = replacement_delete(
        new_path=new_path,
        previous_path=previous_path,
        previous_on_disk=previous_on_disk,
        previous_state=previous_state,
        upsert_in_flight=upsert_in_flight,
        offered_only=offered_only,
    )
    if not target:
        return None
    store.delete_source_points(target)
    return target


def choose_resume(redis_offsets: list[int], checkpoint_articles: int | None) -> int:
    """Article offset to continue from. Never an in-flight batch-start offset."""
    redis_at = max(redis_offsets) if redis_offsets else 0
    checkpoint = int(checkpoint_articles or 0)
    return max(redis_at, checkpoint)


def continuation_keep(job_offsets: list[tuple[str, int]]) -> tuple[str | None, list[str]]:
    """One continuation per file. The highest offset wins. The rest are removed."""
    if not job_offsets:
        return None, []
    keep_id, keep_off = max(job_offsets, key=lambda item: (item[1], item[0]))
    remove = [job_id for job_id, _off in job_offsets if job_id != keep_id]
    if keep_off < 0:
        return None, [job_id for job_id, _off in job_offsets]
    return keep_id, remove


def article_outcome(skip: str | None, *, already_stored: bool, stored_chunks: int) -> str:
    """indexed, skipped, or bad.

    A read or parse failure is bad. So is an article the embedder rejected
    down to a piece too small to split. A page with no text is skipped. A page
    that is not English is skipped. A page already stored, or one that
    produced chunks, is indexed. A long page is indexed in full, so it counts
    as indexed.
    """
    if already_stored:
        return "indexed"
    if (skip or "").strip() in ("read", "parse", "embed"):
        return "bad"
    if int(stored_chunks) <= 0:
        return "skipped"
    return "indexed"


def repeat_failure(seen: set[tuple[str, int]], source: str, articles: int) -> bool:
    """True when this file already failed once at this same article."""
    key = (source, int(articles))
    if key in seen:
        return True
    seen.add(key)
    return False


def qdrant_write_held(message: str) -> bool:
    """True when Qdrant stopped finishing a write.

    Turbo stays on that file and tries the same article again. An embedder
    timeout uses the same words and still fails the file after one retry.
    A write Qdrant rejected, such as HTTP 400, is not a hold.
    """
    text = (message or "").lower()
    if "qdrant" not in text:
        return False
    return "unreachable" in text or "timed out" in text or "timeout" in text


def nomad_embed_guard(policy: str | None, queue_paused: bool) -> tuple[str, bool]:
    """While turbo is indexing, keep NOMAD from embedding the same files.

    Returns the policy to write (empty when it is already Manual) and whether
    the embed queue has to be paused again.
    """
    corrected = "" if policy == "Manual" else "Manual"
    return corrected, not queue_paused


def file_is_embeddable(name: str) -> str | None:
    """Stock determineFileType. Unknown files, including .tmp downloads, are skipped."""
    lower = name.lower()
    dot = lower.rfind(".")
    ext = lower[dot:] if dot >= 0 else ""
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}:
        return "image"
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "docx"
    if ext in {".txt", ".md", ".rtf"}:
        return "text"
    if ext == ".epub":
        return "epub"
    if ext == ".zim":
        return "zim"
    return None
