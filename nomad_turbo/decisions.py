"""Handoff rules for stock NOMAD.

These rules follow the stock scanner, not any in-container speed patch.
Setting the ingest policy to Manual does not stop jobs already in
file-embeddings. BullMQ pause does not stop handlers that are already
active. Offsets are safe to read only after that queue's active count is 0.
"""

from __future__ import annotations


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
    already stored, or one that produced chunks, is indexed. A long page is
    indexed in full, so it counts as indexed.
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
