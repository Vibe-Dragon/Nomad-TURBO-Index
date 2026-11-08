"""Run turbo outside the admin container, then hand embedding back.

Turbo does not patch that container. It pauses only the file-embeddings
queue, indexes from the saved offsets, and restores the ingest policy it
found at Start. A run that found Manual stays Manual and does not hand
unfinished files back to stock NOMAD.
"""

from __future__ import annotations

import json
import logging
import queue
import stat
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from nomad_turbo.chunking import DOCUMENT_PREFIX, keywords, mean_tokens
from nomad_turbo.decisions import (
    article_outcome,
    choose_resume,
    continuation_keep,
    file_is_embeddable,
    hand_back_when_idle,
    hold_queue_on_exit,
    nomad_embed_guard,
    offsets_are_safe,
    partial_row_state,
    repeat_failure,
)
from nomad_turbo.catalog import origin_for, wipe_allowed
from nomad_turbo.desktop import write_shortcut
from nomad_turbo.discover import Install, load_install, state_dir
from nomad_turbo.packing import ArticleFeed, PreparedArticle, prepare_article
from nomad_turbo.passes import (
    ArticlePrints,
    article_plain,
    library_ledger,
    pass_label,
    reset_library_ledger,
    sections_from_article,
)
from nomad_turbo.progress import ocr_note, ocr_page, ocr_status_level, remember_ocr_workers
from nomad_turbo.plain import read_text
from nomad_turbo.pool import EmbedPool
from nomad_turbo.runtime import (
    active_embeds,
    ChunkRejected,
    container_ocr_env,
    embed,
    host_to_container,
    list_jobs,
    queue_is_paused,
    queuectl,
    release_extract,
    read_visual,
    read_zim_date,
    start_extract,
    stop_ours,
)
from nomad_turbo.schedule import (
    SWITCH_REASONS,
    OrderFile,
    apply_sample_batch,
    engine_switch_reason,
    file_needs_measure,
    file_type_label,
    interrupt_outcome,
    measure_ending,
    measure_leave_reason,
    open_index_now,
    release_measure,
    pick_next,
)
from nomad_turbo.session import Session
from nomad_turbo.slots import install_embed_chunks, safe_call_chunks
from nomad_turbo.tune import content_kind
from nomad_turbo.store import (
    clear_ingest_rows,
    collection_ready,
    count_source,
    delete_collection,
    delete_source_points,
    existing_paths,
    forget_ingest,
    get_policy,
    ingest_states,
    mark_indexed,
    mark_pending,
    point_id,
    set_policy,
    source_active,
    upsert,
)

log = logging.getLogger("nomad_turbo")

# A small file can spare a path list, used to skip articles already stored.
# A larger file resumes from its checkpoint. Listing every path would walk
# the index, and an empty skip set is not the same as an empty file.
_PATH_LIST_LIMIT = 2000

# How often the library is compared with the queue while a run is going.
SCAN_SECONDS = 5

# After a Qdrant write times out, stay on this file instead of walking the library.
QDRANT_HOLD_SECONDS = 20
QDRANT_HOLD_NOTE = (
    "Qdrant is still writing this file. Do not restart Qdrant or delete the index. "
    "Turbo is waiting here."
)

_rules_mod = None
_rules_lock = threading.Lock()
_io_lock = threading.Lock()
_drops_lock = threading.Lock()
_last_download_at: float | None = None
_hold_announced = ""


def _rules():
    """Decisions as of this Start. The open window reloads engine, not decisions."""
    global _rules_mod
    with _rules_lock:
        if _rules_mod is None:
            import importlib

            from nomad_turbo import decisions as decisions_mod

            importlib.reload(decisions_mod)
            _rules_mod = decisions_mod
        return _rules_mod


def _drops_path() -> Path:
    return state_dir() / "pending_drops.json"


def _read_drops() -> dict[str, float]:
    path = _drops_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    found: dict[str, float] = {}
    for key, value in data.items():
        try:
            found[str(key)] = float(value)
        except (TypeError, ValueError):
            found[str(key)] = time.time()
    return found


def _write_drops(items: dict[str, float]) -> None:
    path = _drops_path()
    if not items:
        if path.exists():
            path.unlink()
        return
    path.write_text(json.dumps(items))


def _note_downloads(install: Install, known: bool | None = None) -> tuple[bool, float | None]:
    """True while a .tmp or .part file exists, then seconds since the last one vanished.

    `known` is the answer from a walk that already happened. Passing it does
    not walk the folders again.
    """
    global _last_download_at, _downloads_seen
    if known is None:
        pending = _downloads_pending(install)
    else:
        pending = bool(known)
        if _library_live is None or not _library_live.current():
            _downloads_seen = pending
    now = time.monotonic()
    with _io_lock:
        if pending:
            _last_download_at = now
            return True, 0.0
        if _last_download_at is None:
            return False, None
        return False, now - _last_download_at


def _io_pressure() -> tuple[float, float]:
    try:
        text = Path("/proc/pressure/io").read_text()
    except OSError:
        return 0.0, 0.0
    return _rules().parse_io_pressure(text)


def _library_hold(install: Install, downloads_pending: bool | None = None) -> str:
    pending, age = _note_downloads(install, downloads_pending)
    some, full = _io_pressure()
    return _rules().write_hold(
        downloads_pending=pending,
        seconds_since_download=age,
        io_some_avg10=some,
        io_full_avg10=full,
    )


def _announce_hold(session: Session | None, reason: str) -> None:
    global _hold_announced
    if reason == _hold_announced:
        return
    if reason == "download":
        message = "A ZIM download is in progress. Knowledge-base writes are waiting."
        logged = message
    elif reason == "settle":
        message = "A download just finished. Waiting for its knowledge-base update before writing."
        logged = message
    elif reason == "disk":
        message = "The disk is busy. Slowing knowledge-base writes."
        logged = message
    elif reason == "segment":
        logged = (
            "Indexing is paused so a segment can be optimized. "
            "A plain segment is past the 4 GiB vector cap. New points are waiting."
        )
        message = (
            f"{logged} Points already stored stay. "
            "The optimization bar on this page starts over for this pause. "
            "Qdrant does not report a percent while it builds the segment."
        )
    elif reason == "space":
        logged = (
            "Indexing is paused so a segment can be optimized. "
            "Free space is under twice the segment that would be copied. New points are waiting."
        )
        message = (
            f"{logged} Points already stored stay. "
            "The optimization bar on this page starts over for this pause."
        )
    else:
        message = "Knowledge-base writes resumed."
        logged = message
    log.info(logged)
    _hold_announced = reason
    hold = reason if reason in ("segment", "space") else ""
    setter = getattr(session, "set_optimize_hold", None) if session is not None else None
    if callable(setter):
        setter(hold)
    if session is not None and reason:
        session.set_note(message)
    elif session is not None:
        session.set_note("Indexing.")


def _note_embed_call(
    session,
    size: int,
    chunks: int,
    seconds: float,
    ok: bool,
    kind: str = "",
    article_size: int = 0,
    tokens: int = 0,
) -> None:
    note = getattr(session, "note_embed_call", None)
    if not callable(note):
        return
    try:
        note(size, chunks, seconds, ok, kind, article_size, tokens)
    except TypeError as exc:
        # A window opened before this argument existed still has the shorter
        # method. Start reloads this file and not that object. Raising here
        # abandons the current file and walks the rest of the library.
        if "positional" not in str(exc):
            raise
        note(size, chunks, seconds, ok, kind, article_size)


def _next_call_chunks(pool, session, install) -> int:
    """Chunks the next embed call sends.

    The session stores a chunk count. A pool that only knows the old percent
    helper is asked for that count anyway, which is what the segment tests stub.
    """
    if session is not None and hasattr(session, "get_chunks"):
        chunks = int(session.get_chunks())
    else:
        use_gpu = bool(getattr(pool, "use_gpu", False))
        total = int(getattr(pool, "total_mib", 0) or 0)
        gpu_total = total if use_gpu and total > 0 else None
        chunks = install_embed_chunks(
            install,
            gpu_total_mib=gpu_total,
            host_total_mib=int(getattr(pool, "host_total_mib", 0) or 0),
            cpu_count=int(getattr(pool, "cpu_count", 1) or 1),
        )
    direct = getattr(pool, "call_width", None)
    if callable(direct):
        return max(1, int(direct(chunks)))
    legacy = getattr(pool, "batch_width", None)
    if callable(legacy):
        return max(1, int(legacy(chunks)))
    return max(1, int(chunks))


def _segment_hold(install: Install) -> str:
    """Wait while a segment rewrite would not fit. Import is local so a new Start sees it."""
    from nomad_turbo import segment_guard

    return segment_guard.segment_upsert_hold(segment_guard.cached_write_facts(install)).reason


def _wait_to_write(
    install: Install,
    stop: threading.Event,
    session: Session | None,
    interval: float = 5.0,
    downloads: bool | None = None,
) -> bool:
    """Hold Qdrant writes during a download or while a segment cannot be rewritten.

    A download holds until it is gone and the settle period has passed. A
    stalled disk waits five seconds, then the write goes ahead so indexing
    does not stop for the whole run. A segment that is past the vector cap,
    or free space under twice that segment, keeps waiting. Points stay.
    `downloads` is the flag from the walk between files. The first check uses
    it. A check after a wait looks again.
    """
    known = downloads
    while not stop.is_set():
        # Segment and space are hard holds. A disk stall must not skip them.
        segment = _segment_hold(install)
        library = _library_hold(install, known)
        known = None
        reason = segment or library
        if not reason:
            if _hold_announced:
                _announce_hold(session, "")
            return True
        _announce_hold(session, reason)
        if stop.wait(interval):
            return False
        if reason == "disk":
            return not stop.is_set()
    return False


def _remember_drop(source: str) -> None:
    with _drops_lock:
        pending = _read_drops()
        pending.setdefault(source, time.time())
        _write_drops(pending)


def _forget_drop(source: str) -> None:
    with _drops_lock:
        pending = _read_drops()
        pending.pop(source, None)
        _write_drops(pending)


def _request_drop(install: Install, session: Session | None, source: str) -> None:
    """Delete a removed file's points, unless a download or a busy disk is in the way."""
    with _drops_lock:
        pending = _read_drops()
        since = pending.get(source, time.time())
        known = source in pending
    reason = _library_hold(install)
    if _rules().drop_should_wait(reason, 0 if not known else time.time() - since):
        _remember_drop(source)
        if not known:
            log.info(
                "%s is gone. Its knowledge-base entries will be cleared after the download settles.",
                source,
            )
        _hide_source(session, source)
        return
    log.info("%s is gone. Deleting its knowledge-base entries.", source)
    if _drop_source(install, source):
        _forget_drop(source)
        _hide_source(session, source)
        return
    _remember_drop(source)


def _flush_drops(
    install: Install,
    session: Session | None,
    stop: threading.Event,
    downloads: bool | None = None,
) -> None:
    with _drops_lock:
        pending = list(_read_drops().items())
    for source, since in pending:
        if stop.is_set():
            return
        reason = _library_hold(install, downloads)
        if _rules().drop_should_wait(reason, time.time() - since):
            return
        log.info("clearing knowledge-base entries for %s", source)
        if not _drop_source(install, source):
            return
        _forget_drop(source)
        _hide_source(session, source)


@dataclass
class FileWork:
    host: Path
    kind: str
    source: str
    articles: int = 0
    path: str | None = None
    chunks: int = 0
    done: bool = False
    deferred: bool = False
    removed: bool = False
    jobs: list[tuple[str, int]] = field(default_factory=list)
    total_articles: int | None = None
    touched: bool = False
    mtime: float | None = None
    skipped: int = 0
    bad: int = 0
    # Copied onto every point. False only when the ingest row turned the file off.
    point_active: bool = True
    # html, text, or ocr. A checkpoint without it is still the HTML walk.
    zim_pass: str = "html"
    pass_done: int = 0
    pass_total: int | None = None
    label: str = ""
    phase: str = "html"
    # PDFs from this ZIM that are still in the OCR queue, and ones already stored.
    ocr_pending: list[str] = field(default_factory=list)
    ocr_done: list[str] = field(default_factory=list)


def _checkpoint_file(source: str) -> Path:
    import hashlib

    folder = state_dir() / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / (hashlib.sha256(source.encode()).hexdigest()[:16] + ".json")


def _read_checkpoint(source: str) -> dict | None:
    path = _checkpoint_file(source)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_checkpoint(work: FileWork) -> None:
    _checkpoint_file(work.source).write_text(
        json.dumps(
            {
                "source": work.source,
                "articles": work.articles,
                "path": work.path,
                "chunks": work.chunks,
                "done": work.done,
                "jobs": work.jobs,
                "total_articles": work.total_articles,
                "pass": getattr(work, "zim_pass", None) or "html",
                "ocr_pending": list(getattr(work, "ocr_pending", []) or []),
                "ocr_done": list(getattr(work, "ocr_done", []) or []),
            }
        )
    )


def _ocr_tally_note(total: int) -> str:
    """Status line while a ZIM walk is still finding files that need OCR."""
    count = max(0, int(total))
    if count == 1:
        return "Indexing. 1 file in this ZIM is waiting for OCR."
    return f"Indexing. {count} files in this ZIM are waiting for OCR."


def _clear_zim_ocr_ahead(spill_id: str) -> None:
    """Drop OCR text read during a ZIM walk once that ZIM is stored."""
    if not spill_id:
        return
    folder = state_dir() / "ocr-ahead" / spill_id
    if not folder.is_dir():
        return
    import shutil

    shutil.rmtree(folder, ignore_errors=True)


def _spill_paths(source: str) -> dict[str, Path]:
    """On-disk lists for the text pass and the OCR pass. The id matches the checkpoint."""
    import hashlib

    folder = state_dir() / "spill"
    folder.mkdir(parents=True, exist_ok=True)
    spill_id = hashlib.sha256(source.encode()).hexdigest()[:16]
    return {
        "id": spill_id,
        "text": folder / f"{spill_id}-text.jsonl",
        "ocr": folder / f"{spill_id}-ocr.jsonl",
        "prints": folder / f"{spill_id}.prints",
        "go": folder / f"{spill_id}.go",
    }


def _extract_resume(work: FileWork) -> tuple[str | None, str | None, str]:
    """HTML after-path, later-pass resume path, and the pass name.

    A text or OCR resume does not walk the HTML again. The saved path is the
    last entry committed in that pass. A path from the HTML walk is not in
    the later list, and the reader starts that pass at the beginning.
    """
    name = getattr(work, "zim_pass", None) or "html"
    if name == "queued":
        return None, None, "queued"
    if name not in ("html", "text", "ocr"):
        name = "html"
    if name in ("text", "ocr"):
        return None, work.path, name
    return work.path, None, name


def _ocr_quality(session: Session | None, install: Install) -> str:
    from nomad_turbo.passes import normalize_ocr_quality

    getter = getattr(session, "ocr_quality", None) if session is not None else None
    if callable(getter):
        return normalize_ocr_quality(getter())
    return normalize_ocr_quality(getattr(install, "ocr_quality", "medium"))


def _ocr_assignment(session: Session | None, install: Install):
    from nomad_turbo.devices import assignment_sentence, list_devices, resolve_assignment, saved_choice
    from nomad_turbo.ocr_budget import set_runtime
    from nomad_turbo.ocr_engine import ocr_python, probe_map

    devices = list_devices()
    ocr_choice = saved_choice(session, install, "ocr_device")
    probes = probe_map(devices) if ocr_choice != "cpu" and ocr_python() else {}
    assignment = resolve_assignment(
        devices,
        saved_choice(session, install, "index_device"),
        ocr_choice,
        probes,
    )
    set_runtime(share=assignment.share, index_cpu=assignment.index.kind == "cpu", cpus=os_cpu())
    told = getattr(_ocr_assignment, "told", "")
    sentence = assignment_sentence(assignment)
    if sentence != told:
        _ocr_assignment.told = sentence
        log.info("%s", sentence)
    return assignment


def _ocr_container(session: Session | None, install: Install, *, wait: bool) -> dict[str, str]:
    """Budget is always published. The GPU socket is set only when that reader is up, or will be."""
    from nomad_turbo.devices import reader_label
    from nomad_turbo.ocr_engine import gpu_ocr

    assignment = _ocr_assignment(session, install)
    socket = False
    if assignment.uses_gpu_ocr:
        if wait:
            socket = gpu_ocr.ensure(assignment)
            if not socket:
                log.warning("GPU OCR did not start. This file uses the CPU reader.")
        else:
            gpu_ocr.plan(assignment)
            socket = True
    if socket:
        return container_ocr_env(
            socket=True,
            device=assignment.ocr.name,
            reader=reader_label(assignment.backend.name),
        )
    return container_ocr_env(socket=False, device="CPU", reader="Tesseract")


def _release_ocr(work: FileWork) -> None:
    """Hand the article prints to the OCR pass, then let it draw.

    A GPU reader is loaded here, after HTML is fingerprinted and before the
    first page is drawn. The same device waits inside the CPU budget.
    """
    from nomad_turbo.ocr_engine import gpu_ocr

    paths = getattr(work, "spill", None) or _spill_paths(work.source)
    prints = getattr(work, "prints", None) or ArticlePrints()
    prints.merge_file(paths["prints"])
    prints.write(paths["prints"])
    work.prints = prints
    planned = gpu_ocr.planned()
    if planned is not None and not gpu_ocr.ensure(planned):
        log.warning("GPU OCR did not start. This file uses the CPU reader.")
    paths["go"].write_text("go\n", encoding="utf-8")


def _wants_control(session: Session | None, install: Install) -> bool:
    getter = getattr(session, "get_keep_control", None)
    if callable(getter):
        return bool(getter())
    return bool(getattr(install, "keep_control", False))


def _queue_has_work(session: Session | None, finished: set[str], failed: set[str]) -> bool | None:
    """False when the window already knows there is nothing left to index.

    A failed file is still work when Index now or priority names it. The
    batch step is what puts that file back, and only when it is still on disk.
    """
    book = session.get_book() if session else None
    listed = getattr(book, "listed", None)
    if not callable(listed):
        return None
    retry: set[str] = set()
    if session is not None:
        now = session.index_now() if hasattr(session, "index_now") else None
        if now:
            retry.add(now)
        priorities = session.priorities() if hasattr(session, "priorities") else ()
        retry.update(priorities)
    for source, state in listed():
        if source in finished:
            continue
        if state in ("indexed", "deferred"):
            continue
        if (state == "failed" or source in failed) and source not in retry:
            continue
        return True
    return False


def _forget_source_jobs(install: Install, source: str) -> None:
    """Drop NOMAD embed jobs for one path so a finished file is not embedded twice."""
    try:
        jobs = list_jobs(install)
    except Exception:
        log.exception("could not list embed jobs for %s", source)
        return
    for job in jobs:
        data = job.get("data") or {}
        if data.get("filePath") != source:
            continue
        try:
            queuectl(install, "remove", str(job["id"]))
        except Exception:
            log.exception("could not remove the embed job for %s", source)


def _drop_source(install: Install, source: str) -> bool:
    """Delete the points and ingest row for a file that left the disk."""
    try:
        delete_source_points(install, source)
    except Exception:
        log.exception("could not delete knowledge-base entries for %s", source)
        return False
    try:
        forget_ingest(install, source)
    except Exception:
        log.exception("could not clear the ingest row for %s", source)
    _forget_source_jobs(install, source)
    checkpoint = _checkpoint_file(source)
    if checkpoint.exists():
        checkpoint.unlink()
    return True


def _hide_source(session: Session | None, source: str) -> None:
    book = session.get_book() if session else None
    drop = getattr(book, "drop", None)
    if callable(drop):
        drop(source)


class _LiveFile:
    """The file being indexed, and a signal that it disappeared."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.source: str | None = None
        self.retire = threading.Event()

    def begin(self, source: str) -> None:
        with self._lock:
            self.source = source
            self.retire.clear()

    def end(self) -> None:
        with self._lock:
            self.source = None
            self.retire.clear()

    def current(self) -> str | None:
        with self._lock:
            return self.source


# Set while a run is inside a file. A write hold then reuses the last
# download check instead of walking zim/ and kb_uploads again.
_library_live: _LiveFile | None = None
_downloads_seen = False


def _sync_library(
    install: Install,
    session: Session | None,
    live: _LiveFile,
    snap: LibrarySnap | None = None,
) -> None:
    """Show files that just arrived, and clear entries for files NOMAD removed.

    The folders are walked when nothing is indexing. During a file, only
    that file's own path is checked, which is not a folder walk. A snapshot
    from the pause between files is reused, so this does not walk again.
    """
    current = live.current()
    if current:
        host = _library_host(install, current)
        if host is not None and not host.exists():
            if not live.retire.is_set():
                log.info("%s disappeared while it was indexing. Its entries will be cleared.", current)
                live.retire.set()
        return
    if snap is None:
        snap = _walk_library(install)
    book = session.get_book() if session else None
    present = getattr(book, "ensure_present", None)
    for item in snap.files:
        if not callable(present):
            continue
        if present(
            item.source,
            name=item.host.name,
            origin=origin_for(item.source),
            kind=item.kind,
            size_bytes=item.size,
        ):
            log.info("new file in the queue: %s", item.host.name)
    listed = getattr(book, "listed", None)
    if not callable(listed):
        return
    for source, _state in listed():
        host = _library_host(install, source)
        if host is None or host.exists():
            continue
        if source == live.current():
            if not live.retire.is_set():
                log.info("%s disappeared while it was indexing. Its entries will be cleared.", source)
                live.retire.set()
            continue
        _request_drop(install, session, source)


def _keep_nomad_quiet(install: Install, stop: threading.Event) -> None:
    """Put Manual and the embed-queue pause back if someone changed them.

    NOMAD's own screens can still switch the policy or resume the queue.
    Either one lets NOMAD embed a file turbo is already writing, which
    stores a second copy of those points. This does not block a question
    to the NOMAD AI, and it does not block Re-embed or Reset.
    """
    if stop.is_set():
        return
    try:
        policy = get_policy(install)
        paused = queue_is_paused(install)
    except Exception:
        log.exception("could not check whether NOMAD is still leaving embeds to turbo")
        return
    if stop.is_set():
        return
    corrected, must_pause = nomad_embed_guard(policy, paused)
    if corrected:
        log.warning(
            "ingest policy was %s while turbo is running. Setting it back to Manual.",
            policy,
        )
        try:
            set_policy(install, corrected)
        except Exception:
            log.exception("could not set ingest policy back to Manual")
    if stop.is_set():
        return
    if must_pause:
        log.warning("the embed queue was turned back on while turbo is running. Pausing it again.")
        try:
            queuectl(install, "pause")
        except Exception:
            log.exception("could not pause the embed queue")


def _watch_library(install: Install, session: Session | None, stop: threading.Event, live: _LiveFile) -> None:
    while not stop.is_set():
        try:
            _flush_drops(install, session, stop)
            _sync_library(install, session, live)
            _keep_nomad_quiet(install, stop)
        except Exception:
            log.exception("library scan failed")
        try:
            _collection_logbook().watch_collection_logs(state_dir(), install)
        except Exception:
            log.exception("could not check log files against the knowledge index")
        if stop.wait(SCAN_SECONDS):
            return


def _open_state(work: FileWork) -> str:
    """Partial means some of the file is stored. Nothing stored is still waiting."""
    if int(work.articles or 0) > 0 or int(work.chunks or 0) > 0:
        return "partial"
    return "pending"


def _drop_priority(session: Session | None, source: str) -> None:
    """A finished file does not keep its priority mark."""
    if session is None or not source:
        return
    setter = getattr(session, "set_priority", None)
    if callable(setter):
        setter(source, False)
    stats = session.file_stats() if hasattr(session, "file_stats") else None
    if stats is None or not hasattr(stats, "set_priority"):
        return
    stats.set_priority(source, False)
    saver = getattr(stats, "save", None)
    if not callable(saver):
        return
    try:
        saver()
    except Exception:
        log.exception("could not save the priority for %s", source)


def _clear_saved_priorities(session: Session | None) -> None:
    """Drop every priority mark. The index they pointed at is gone."""
    from nomad_turbo.file_stats import STATS_NAME, FileStats

    stats = None
    if session is not None and hasattr(session, "file_stats"):
        getter = session.file_stats
        stats = getter() if callable(getter) else None
    if stats is None:
        stats = FileStats(state_dir() / STATS_NAME)
        loader = getattr(stats, "load", None)
        if callable(loader):
            loader()
    sources = list(stats.priorities()) if hasattr(stats, "priorities") else []
    clearer = getattr(stats, "clear_priorities", None)
    if callable(clearer):
        clearer()
    else:
        for source in sources:
            stats.set_priority(source, False)
    saver = getattr(stats, "save", None)
    if callable(saver):
        saver()
    if session is None:
        return
    session_clear = getattr(session, "clear_priorities", None)
    if callable(session_clear):
        session_clear()
        return
    setter = getattr(session, "set_priority", None)
    if callable(setter):
        for source in sources:
            setter(source, False)


def _report(session: Session | None, work: FileWork, state: str) -> None:
    if state == "indexed":
        _drop_priority(session, work.source)
    if session is None:
        return
    book = session.get_book()
    if book is None:
        return
    book.update(
        work.source,
        articles=work.articles,
        total=work.total_articles,
        chunks=work.chunks,
        state=state,
        label=getattr(work, "label", "") or "",
        pass_done=int(getattr(work, "pass_done", 0) or 0),
        pass_total=getattr(work, "pass_total", None),
    )


def _collection_logbook():
    """The logbook module. Reload once when this process still has the old copy.

    A window opened before clear_collection_logs existed keeps that old module
    in memory. Importing the name from it fails after the collection is
    already gone.
    """
    import importlib

    import nomad_turbo.logbook as logbook

    if not hasattr(logbook, "clear_collection_logs"):
        importlib.reload(logbook)
    return logbook


def wipe_knowledge_index(install: Install) -> None:
    """Delete vectors and ingest rows. ZIM files and uploads stay on disk.

    The embed queue has to be idle. Clearing it only removes this queue.
    """
    if not offsets_are_safe(active_embeds(install)):
        raise RuntimeError("refusing to delete the index while an embed job is active")
    queuectl(install, "clear")
    delete_collection(install)
    clear_collection_logs = _collection_logbook().clear_collection_logs
    clear_collection_logs(state_dir())
    collection_ready(install)
    clear_ingest_rows(install)
    folder = state_dir() / "checkpoints"
    if folder.is_dir():
        for path in folder.glob("*.json"):
            path.unlink()
    reset_library_ledger()
    log.warning("knowledge index deleted. Library files are kept.")


def delete_index_only(stop: threading.Event, session: Session | None = None) -> None:
    """Delete the knowledge index and leave indexing stopped.

    NOMAD's current embed batches are allowed to finish. Ingest policy stays
    Manual afterwards so nothing starts embedding on its own.
    """
    install = load_install()
    previous = get_policy(install)
    saved = previous if previous in ("Manual", "Always") else "Manual"
    (state_dir() / "policy").write_text(saved)
    set_policy(install, "Manual")
    log.info("ingest policy set to Manual. Waiting until NOMAD's current embed batches finish.")
    if session:
        session.set_phase("deleting")
        session.set_note("Waiting for NOMAD to finish its current embed batches.")
    queuectl(install, "pause")
    _status(phase="deleting", paused_queue=True, slots=0)
    cancelled = False
    try:
        while not stop.is_set():
            active = active_embeds(install)
            if offsets_are_safe(active):
                break
            log.info("NOMAD still has %s embed job(s) in progress", active)
            if session:
                session.set_note(f"Waiting for NOMAD to finish {active} embed job(s).")
            _keep_nomad_quiet(install, stop)
            stop.wait(5)
        if stop.is_set():
            cancelled = True
            log.info("delete cancelled before the index was removed")
            set_policy(install, saved)
            if session:
                session.set_phase("idle")
                session.set_note("Delete cancelled. The index was not removed.")
            return
        if not offsets_are_safe(active_embeds(install)):
            raise RuntimeError("refusing to delete the index while an embed job is active")
        if session:
            session.set_note("Deleting the knowledge index.")
        wipe_knowledge_index(install)
        _clear_saved_priorities(session)
        if session and session.get_book():
            session.get_book().mark_all_unindexed()
        if session:
            session.set_phase("idle")
            session.set_note("Knowledge index deleted. Indexing has not started.")
        log.warning("knowledge index deleted. Indexing was not started.")
    finally:
        try:
            queuectl(install, "resume")
            if cancelled:
                log.info("file-embeddings queue resumed.")
            else:
                log.info("file-embeddings queue resumed. Ingest policy stays Manual.")
        except Exception:
            log.exception("could not resume file-embeddings")
        _status(phase="off", paused_queue=False, slots=0, repair_hold=False)


def _status(**fields: object) -> None:
    path = state_dir() / "status.json"
    current = {}
    if path.exists():
        try:
            current = json.loads(path.read_text())
        except json.JSONDecodeError:
            current = {}
    current.update(fields)
    path.write_text(json.dumps(current))


@dataclass(frozen=True)
class LibraryFile:
    host: Path
    kind: str
    source: str
    mtime: float
    size: int


@dataclass(frozen=True)
class LibrarySnap:
    """Embeddable files and the download flag from one pass over the library."""

    files: tuple[LibraryFile, ...]
    downloads: bool


def _walk_library(install: Install) -> LibrarySnap:
    """One pass over zim and kb_uploads.

    Each embeddable file is measured once. A name ending in .tmp or .part is a
    download still arriving, including when that name is not embeddable.
    """
    root = Path(install.storage_host)
    found: list[LibraryFile] = []
    downloads = False
    for folder in ("zim", "kb_uploads"):
        base = root / folder
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            name = path.name
            if name.endswith(".tmp") or name.endswith(".part"):
                downloads = True
            kind = file_is_embeddable(name)
            if not kind:
                continue
            try:
                info = path.stat()
            except OSError:
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            found.append(
                LibraryFile(
                    host=path,
                    kind=kind,
                    source=host_to_container(install, str(path)),
                    mtime=info.st_mtime,
                    size=info.st_size,
                )
            )
    return LibrarySnap(tuple(found), downloads)


def _indexable(snap: LibrarySnap, *, include_fresh: bool = False) -> tuple[LibraryFile, ...]:
    """Files the queue may open. One touched in the last 10 seconds stays out."""
    if include_fresh:
        return snap.files
    now = time.time()
    return tuple(item for item in snap.files if now - item.mtime >= 10)


def _sources(
    install: Install,
    *,
    include_fresh: bool = False,
    snap: LibrarySnap | None = None,
) -> list[tuple[Path, str, str]]:
    if snap is None:
        snap = _walk_library(install)
    return [(item.host, item.kind, item.source) for item in _indexable(snap, include_fresh=include_fresh)]


def _look_between_files(install: Install, session: Session | None, live: _LiveFile) -> LibrarySnap | None:
    """Walk once while no file is open, and publish that picture.

    The queue, the new-file list, and the download check read the snapshot.
    An open file returns None and does not walk.
    """
    if live.current() is not None:
        return None
    snap = _walk_library(install)
    global _downloads_seen
    _downloads_seen = snap.downloads
    _sync_library(install, session, live, snap)
    return snap


def _library_host(install: Install, source: str) -> Path | None:
    """Host path of a ZIM or upload. Other sources are left alone."""
    root = install.container_storage.rstrip("/")
    if not source.startswith(root + "/"):
        return None
    relative = source[len(root) + 1 :]
    if not (relative.startswith("zim/") or relative.startswith("kb_uploads/")):
        return None
    return Path(install.storage_host) / relative


def _downloads_pending(install: Install) -> bool:
    """True when a download is still being written into the library.

    While a file is indexing this returns the last answer and does not walk
    the folders. The pause between files uses the shared walk instead.
    """
    global _downloads_seen
    live = _library_live
    if live is not None and live.current():
        return _downloads_seen
    root = Path(install.storage_host)
    found = False
    for folder in ("zim", "kb_uploads"):
        base = root / folder
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.name.endswith(".tmp") or path.name.endswith(".part"):
                found = True
                break
        if found:
            break
    _downloads_seen = found
    return found


def _work_list(
    install: Install,
    jobs: list[dict],
    finished: set[str],
    snap: LibrarySnap | None = None,
) -> list[FileWork]:
    grouped: dict[str, list[tuple[str, int, dict]]] = {}
    for job in jobs:
        data = job.get("data") or {}
        source = data.get("filePath")
        if not source:
            continue
        grouped.setdefault(source, []).append((str(job["id"]), int(data.get("batchOffset") or 0), data))
    states = ingest_states(install)
    if snap is None:
        snap = _walk_library(install)
    work: list[FileWork] = []
    for item in _indexable(snap):
        host, kind, source = item.host, item.kind, item.source
        if source in finished:
            continue
        entries = grouped.get(source, [])
        checkpoint = _read_checkpoint(source)
        indexed = states.get(source) == "indexed"
        if indexed and not entries and not (checkpoint and not checkpoint.get("done")):
            continue
        redis_offsets = [offset for _job_id, offset, _data in entries]
        articles = choose_resume(redis_offsets, (checkpoint or {}).get("articles"))
        resume_path = None
        if checkpoint and int(checkpoint.get("articles") or 0) >= articles and checkpoint.get("path"):
            resume_path = checkpoint["path"]
            articles = int(checkpoint["articles"])
        zim_pass = str((checkpoint or {}).get("pass") or "html")
        if zim_pass not in ("html", "text", "ocr", "queued"):
            zim_pass = "ocr" if zim_pass == "done" else "html"
        ocr_pending = [str(path) for path in ((checkpoint or {}).get("ocr_pending") or []) if path]
        ocr_done = [str(path) for path in ((checkpoint or {}).get("ocr_done") or []) if path]
        total = None
        chunks = int((checkpoint or {}).get("chunks") or 0)
        for _job_id, _offset, data in entries:
            if data.get("totalArticles"):
                total = int(data["totalArticles"])
            chunks = max(chunks, int(data.get("chunksSoFar") or 0))
        mtime = item.mtime
        work.append(
            FileWork(
                host=host,
                kind=kind,
                source=source,
                articles=articles,
                path=resume_path,
                chunks=chunks,
                jobs=[(job_id, offset) for job_id, offset, _data in entries],
                total_articles=total,
                mtime=mtime,
                zim_pass=zim_pass,
                ocr_pending=ocr_pending,
                ocr_done=ocr_done,
            )
        )
    return work


def _points_for(
    install: Install,
    vectors: list[list[float]],
    payload_texts: list[str],
    meta: dict,
    section_ordinal: int,
) -> list[dict]:
    stamp = int(time.time() * 1000)
    points = []
    section_key = f"{section_ordinal}:{meta.get('section_title') or ''}"
    for index, (vector, text) in enumerate(zip(vectors, payload_texts)):
        if len(vector) != install.dimension:
            raise RuntimeError(f"{install.embed_model} returned {len(vector)} dimensions, stock NOMAD stores {install.dimension}")
        points.append(
            {
                "id": point_id(
                    str(meta.get("source") or ""),
                    str(meta.get("article_path") or ""),
                    section_key,
                    index,
                    text,
                ),
                "vector": vector,
                "payload": {
                    **meta,
                    "text": text,
                    "chunk_index": index,
                    "total_chunks": len(payload_texts),
                    "keywords": keywords(meta.get("full_title") or meta.get("article_title") or "", text),
                    "char_count": len(text),
                    "created_at": stamp,
                    "source": meta["source"],
                    "active": bool(meta.get("active", True)),
                },
            }
        )
    return points


_EXTRACT_DONE = object()
_EXTRACT_CLOSED = object()


def _points_for_prepared(install: Install, prepared: PreparedArticle, vectors: list[list[float]]) -> list[dict]:
    if len(vectors) != len(prepared.inputs):
        raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(prepared.inputs)} chunks")
    grouped: list[list[list[float]]] = [[] for _ in prepared.sections]
    for vector, owner in zip(vectors, prepared.section_for_input):
        grouped[owner].append(vector)
    points: list[dict] = []
    for ordinal, (section, section_vectors) in enumerate(zip(prepared.sections, grouped)):
        if len(section_vectors) != len(section.payloads):
            raise RuntimeError("section vectors do not match the stored text")
        points.extend(_points_for(install, section_vectors, section.payloads, section.meta, ordinal))
    return points


def _note_bad_chunk(work: FileWork, session: Session | None) -> None:
    """One chunk was dropped. The rest of the article is still stored."""
    work.bad += 1
    work.skipped += 1
    book = session.get_book() if session else None
    note = getattr(book, "note_article", None)
    if callable(note):
        note("bad")


def _commit_article(work: FileWork, seen: set[str], session: Session | None, article: dict, new_chunks: int) -> None:
    work.chunks += int(new_chunks)
    work.articles = int(article["index"])
    work.path = article.get("path")
    work.touched = True
    if getattr(work, "pass_total", None):
        work.pass_done = int(getattr(work, "pass_done", 0) or 0) + 1
    path = article.get("path")
    if path:
        seen.add(path)
    _write_checkpoint(work)
    _status(file=work.source, articles=work.articles, chunks=work.chunks)
    _report(session, work, "running")
    _clear_qdrant_hold(session)
    log.debug("%s article %s, %s chunks", Path(work.source).name, work.articles, work.chunks)


def _clear_qdrant_hold(session: Session | None) -> None:
    """A stored article means Qdrant accepted the write. The waiting line can go."""
    if session is None:
        return
    if session.view().get("note") == QDRANT_HOLD_NOTE:
        session.set_note("Indexing.")


def _store_ready(
    install: Install,
    work: FileWork,
    seen: set[str],
    session: Session | None,
    pairs: list[tuple[PreparedArticle, list[list[float]]]],
    stop: threading.Event | None = None,
) -> None:
    """Upsert finished articles, then checkpoint each one.

    Point ids are random, so a crash after the upsert and before the
    checkpoint can write that call's articles again on resume. The gap is
    one embed call.
    """
    staged: list[tuple[PreparedArticle, list[dict]]] = []
    flat: list[dict] = []
    for prepared, vectors in pairs:
        if prepared.already_stored or not vectors:
            staged.append((prepared, []))
            continue
        points = _points_for_prepared(install, prepared, vectors)
        staged.append((prepared, points))
        flat.extend(points)
    if flat:
        if stop is not None and not _wait_to_write(install, stop, session):
            raise RuntimeError("stopped")
        upsert(install, flat)
        ledger = getattr(work, "ledger", None)
        if ledger is not None:
            for point in flat:
                ledger.remember(str((point.get("payload") or {}).get("text") or ""))
    for prepared, points in staged:
        _count_article(work, session, prepared, len(points))
        _commit_article(work, seen, session, prepared.article, len(points))


def _count_article(work: FileWork, session: Session | None, prepared: PreparedArticle, stored_chunks: int) -> None:
    kind = article_outcome(
        prepared.article.get("skip"),
        already_stored=prepared.already_stored,
        stored_chunks=stored_chunks,
    )
    if kind == "indexed":
        return
    if kind == "bad":
        work.bad += 1
        work.skipped += 1
    else:
        work.skipped += 1
    book = session.get_book() if session else None
    note = getattr(book, "note_article", None)
    if callable(note):
        note(kind)


def _prepare_article(install: Install, work: FileWork, article: dict, seen: set[str]) -> PreparedArticle:
    if article.get("path") in seen:
        return PreparedArticle(
            article=article,
            sections=[],
            inputs=[],
            section_for_input=[],
            already_stored=True,
        )
    article = dict(article)
    drop_short = getattr(work, "kind", "") == "zim"
    sections = sections_from_article(article, drop_short_image=drop_short)
    article["sections"] = sections
    if article.get("kind") == "image" and drop_short and not sections and not article.get("skip"):
        article["skip"] = "photo" if str(article.get("ocr") or "").strip() else "empty"
    if not sections and not article.get("skip"):
        article["skip"] = "empty"
    plain = ""
    prints = getattr(work, "prints", None)
    if not article.get("skip"):
        plain = article_plain(article)
        # HTML is first and is not compared with itself. A later copy is.
        if prints is not None and getattr(work, "phase", "html") != "html" and prints.duplicate(plain):
            article["skip"] = "duplicate"
            article["sections"] = []
            article["text"] = ""
            plain = ""
    if article.get("skip"):
        article["sections"] = []
        article["text"] = ""
    ledger = getattr(work, "ledger", None)

    def _keep(payload: str) -> bool:
        return not ledger.seen(payload)

    prepared = prepare_article(
        article,
        work.source,
        article.get("_archive") or {},
        str(uuid.uuid4()),
        install.document_prefix or DOCUMENT_PREFIX,
        keep_chunk=_keep if ledger is not None else None,
    )
    for section in prepared.sections:
        section.meta["active"] = work.point_active
    if prints is not None and not article.get("skip") and plain:
        prints.keep(plain)
    return prepared


def _load_point_active(install: Install, work: FileWork) -> None:
    """Read the file's search flag once. A lookup failure leaves points searchable."""
    try:
        work.point_active = source_active(install, work.source)
    except Exception as exc:
        log.warning(
            "could not read the active flag for %s. New points stay searchable: %s",
            work.source,
            exc,
        )
        work.point_active = True


class _UpsertPipe:
    """One stored batch may overlap the next embed call. Checkpoints stay in order."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="turbo-upsert")
        self._future = None

    def submit(self, fn) -> None:
        self.retire()
        self._future = self._executor.submit(fn)

    def retire(self) -> None:
        future = self._future
        self._future = None
        if future is not None:
            future.result()

    def close(self) -> None:
        try:
            self.retire()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)


def _after_batch(pool: EmbedPool, session: Session | None, install: Install) -> None:
    """Start the one embed server. Changing the chunk count does not start another."""
    chunks = _next_call_chunks(pool, session, install)
    pool.apply_target(chunks, batch_open=False)
    pool.publish(session)


def resume_log(source: str, articles: int, points: int | None, paths: set[str] | None) -> str:
    """One line when a file starts or continues.

    `paths` is the article-path skip set when it was loaded. None means that
    set was not loaded. A large file still continues at `articles` from its
    checkpoint. The point count is what is already stored.
    """
    if points is None:
        detail = "point count unavailable"
    elif paths is None:
        detail = f"{points} points already stored, article paths not re-listed"
    else:
        detail = f"{len(paths)} paths already stored"
    return f"indexing {source} from article {articles} ({detail})"


def _index_zim(
    install: Install,
    pool: EmbedPool,
    work: FileWork,
    stop: threading.Event,
    session: Session | None,
    retire: threading.Event | None = None,
    switch_check=None,
    on_idle=None,
    on_enqueue=None,
    on_body_done=None,
) -> None:
    _load_point_active(install, work)
    already = count_source(install, work.source)
    paths: set[str] | None = None
    if already is not None and already <= _PATH_LIST_LIMIT:
        paths = existing_paths(install, work.source)
    stored = paths or set()
    after_path, resume_path, zim_pass = _extract_resume(work)
    work.zim_pass = zim_pass
    spill = _spill_paths(work.source)
    work.spill = spill
    if zim_pass == "html" and not after_path:
        spill["go"].unlink(missing_ok=True)
        spill["prints"].unlink(missing_ok=True)
    prints = ArticlePrints()
    prints.merge_file(spill["prints"])
    prints.bind(spill["prints"])
    work.prints = prints
    work.ledger = library_ledger()
    if zim_pass in ("text", "ocr"):
        work.phase = zim_pass
        work.label = pass_label(zim_pass)
    else:
        work.phase = "html"
        work.label = "HTML"
    log.info("%s", resume_log(work.source, work.articles, already, paths))
    _report(session, work, "running")
    ocr_ahead = False
    if zim_pass == "html":
        try:
            from nomad_turbo.overlap import cpu_ocr_overlaps_gpu

            ocr_ahead = cpu_ocr_overlaps_gpu(_ocr_assignment(session, install))
        except Exception:
            log.exception("could not see where OCR will run")
            ocr_ahead = False
    if ocr_ahead:
        log.info("PDFs found in this ZIM go to the front of the OCR queue while its pages are indexed.")
    proc = start_extract(
        install,
        str(work.host),
        0 if after_path else work.articles,
        after_path,
        work.articles,
        quality=_ocr_quality(session, install),
        spill_id=spill["id"],
        zim_pass=zim_pass,
        resume_path=resume_path,
        ocr_env=_ocr_container(session, install, wait=False),
        ocr_ahead=ocr_ahead and on_enqueue is None,
        ocr_queue=on_enqueue is not None,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    def _drain() -> None:
        name = work.host.name
        for line in proc.stderr:
            text = line.strip()
            if not text:
                continue
            if text.startswith("ocr ") or text.startswith("pdf text layer"):
                log.log(ocr_status_level(text), "extract: %s", text[:300])
                remember_ocr_workers(session, text)
                found = ocr_page(text)
                if found and getattr(work, "phase", "") == "ocr" and session is not None:
                    session.set_note(ocr_note(name, found[0], found[1]))
                continue
            log.info("extract: %s", text[:300])

    threading.Thread(target=_drain, daemon=True).start()

    def _read_stdout() -> None:
        try:
            for line in proc.stdout:
                if stop.is_set():
                    return
                while not stop.is_set():
                    try:
                        lines.put(line, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                else:
                    return
        finally:
            if stop.is_set():
                return
            while not stop.is_set():
                try:
                    lines.put(None, timeout=0.5)
                    break
                except queue.Full:
                    continue

    lines: queue.Queue = queue.Queue(maxsize=64)
    threading.Thread(target=_read_stdout, daemon=True).start()
    archive: dict = {}

    def _pull(timeout: float):
        nonlocal archive
        deadline = time.monotonic() + timeout
        while not stop.is_set() and not (retire is not None and retire.is_set()) and not (switch_check and switch_check()):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                line = lines.get(timeout=remaining)
            except queue.Empty:
                return None
            if line is None:
                return _EXTRACT_CLOSED
            if not str(line).strip():
                continue
            event = json.loads(line)
            if event.get("type") == "meta":
                archive = event
                if event.get("totalArticles"):
                    work.total_articles = int(event["totalArticles"])
                    _report(session, work, "running")
                note_date = getattr(session, "note_zim_date", None)
                if callable(note_date) and "date" in event:
                    note_date(work.source, str(event.get("date") or ""))
                continue
            if event.get("type") == "ocr-enqueue":
                if on_enqueue is not None:
                    try:
                        on_enqueue(event)
                    except Exception:
                        log.exception("could not queue a PDF from %s", work.source)
                continue
            if event.get("type") == "ocr-tally":
                try:
                    total = max(0, int(event.get("total") or 0))
                except (TypeError, ValueError):
                    total = 0
                if total and session is not None:
                    session.set_note(_ocr_tally_note(total))
                if event.get("complete") and total:
                    log.info("%s", _ocr_tally_note(total).removeprefix("Indexing. "))
                continue
            if event.get("type") == "pass":
                name = str(event.get("pass") or "")
                if name in ("text", "ocr"):
                    work.zim_pass = name
                    work.phase = name
                    work.label = pass_label(name)
                    total = int(event.get("total") or 0)
                    work.pass_total = total or None
                    work.pass_done = int(event.get("done") or 0)
                    if session is not None:
                        session.set_note("Indexing.")
                    _write_checkpoint(work)
                    _report(session, work, "running")
                continue
            if event.get("type") == "hold":
                try:
                    _release_ocr(work)
                except Exception:
                    log.exception("could not release the OCR pass for %s", work.source)
                    held = getattr(work, "spill", None)
                    if held is not None:
                        held["go"].write_text("go\n", encoding="utf-8")
                continue
            if event.get("type") == "phase":
                phase = str(event.get("phase") or "")
                if phase in ("html", "text", "ocr"):
                    work.phase = phase
                    work.label = pass_label(phase)
                    if phase != "ocr" and session is not None:
                        session.set_note("Indexing.")
                    _report(session, work, "running")
                continue
            if event.get("type") == "article":
                if getattr(work, "phase", "") == "ocr":
                    work.phase = "ocr-text"
                    work.label = pass_label("ocr-text")
                    if session is not None:
                        session.set_note("Indexing.")
                    _report(session, work, "running")
                event["_archive"] = archive
                return event
            if event.get("type") == "done":
                return _EXTRACT_DONE
        return None

    try:
        outcome = _consume_articles(
            install, pool, work, stop, session, stored, _pull, retire, switch_check, on_idle
        )
        if outcome == "done" and on_body_done is not None:
            on_body_done()
        elif outcome == "done":
            work.done = True
            work.zim_pass = "done"
            work.touched = True
            _write_checkpoint(work)
            _clear_zim_ocr_ahead(spill["id"])
            _report(session, work, "indexed")
            log.info(
                "finished %s, %s chunks, %s skipped, %s bad",
                work.source,
                work.chunks,
                work.skipped,
                work.bad,
            )
        if proc.wait(timeout=10) not in (0, -15, None) and not work.done and not stop.is_set():
            raise RuntimeError(f"extractor exited {proc.returncode} for {work.source}")
    finally:
        close_prints = getattr(getattr(work, "prints", None), "close", None)
        if callable(close_prints):
            close_prints()
        release_extract(proc)


def _halt_reason(stop: threading.Event, retire: threading.Event | None = None, switch_check=None) -> str | None:
    """Why to leave the file. `replaced` drops points. A switch keeps them.

    The check runs between embed calls, so the call already running finishes first.
    """
    if retire is not None and retire.is_set():
        return "replaced"
    if switch_check is not None:
        asked = switch_check()
        if asked:
            return str(asked)
    if stop.is_set():
        return "stopped"
    return None


def _consume_articles(install, pool, work, stop, session, seen, pull, retire=None, switch_check=None, on_idle=None) -> str:
    """Embed across articles. Checkpoint an article only after its points are stored."""
    feed = ArticleFeed()
    pipe = _UpsertPipe()
    ended = False
    saw_done = False
    last_width = None
    error = None
    try:
        while _halt_reason(stop, retire, switch_check) is None:
            width = _next_call_chunks(pool, session, install)
            if width != last_width:
                log.debug("%s chunks per call", width)
                last_width = width
            kind = content_kind(getattr(work, "source", "") or "")
            while feed.pending() < width and not ended and _halt_reason(stop, retire, switch_check) is None:
                item = pull(0.5)
                if item is None:
                    break
                if item is _EXTRACT_DONE:
                    ended = True
                    saw_done = True
                    break
                if item is _EXTRACT_CLOSED:
                    ended = True
                    break
                feed.add(_prepare_article(install, work, item, seen))
                _publish_flight(session, work, feed)
            if _halt_reason(stop, retire, switch_check):
                break
            if feed.pending() == 0:
                ready = feed.drain_ready()
                if ready:
                    pipe.submit(lambda batch=ready: _store_ready(install, work, seen, session, batch, stop))
                    continue
                if ended:
                    break
                # The ZIM has nothing to embed on this turn. A finished OCR
                # read can use the GPU, then this file is checked again.
                if on_idle is not None:
                    pipe.retire()
                    if _halt_reason(stop, retire, switch_check):
                        break
                    if on_idle(seen):
                        continue
                continue
            if not pool.urls:
                raise RuntimeError("the embed server is not running")
            texts = feed.next_inputs(width)
            article_size = feed.call_article_size()
            token_mean = mean_tokens(texts)
            started = time.monotonic()
            try:
                from nomad_turbo.ocr_budget import note_embed

                note_embed(True)
                try:
                    vectors = embed(pool.urls[0], install.embed_model, texts)
                finally:
                    note_embed(False)
            except ChunkRejected as exc:
                action, article_path = feed.split_rejected(exc.index)
                _publish_flight(session, work, feed)
                name = Path(work.source).name
                if action == "split":
                    log.info(
                        "splitting one chunk of %s (%s) that did not fit the embedder: %s",
                        name,
                        article_path,
                        exc.detail,
                    )
                elif action == "dropped":
                    _note_bad_chunk(work, session)
                    log.warning(
                        "skipping one chunk of %s (%s) the embedder rejected: %s",
                        name,
                        article_path,
                        exc.detail,
                    )
                else:
                    log.warning(
                        "skipping a chunk of %s (%s) the embedder rejected: %s",
                        name,
                        article_path,
                        exc.detail,
                    )
                continue
            except Exception:
                elapsed = time.monotonic() - started
                _note_embed_call(
                    session, width, len(texts), elapsed, False, kind, article_size, token_mean
                )
                raise
            else:
                elapsed = time.monotonic() - started
                # A short leftover batch is not the size that was asked for.
                if len(texts) >= max(4, width // 2):
                    _note_embed_call(
                        session, width, len(texts), elapsed, True, kind, article_size, token_mean
                    )
                note_sequences = getattr(pool, "note_sequences", None)
                if callable(note_sequences):
                    note_sequences(token_mean, len(texts), elapsed, stop)
            ready = feed.push_vectors(vectors)
            _publish_flight(session, work, feed)
            if ready:
                pipe.submit(lambda batch=ready: _store_ready(install, work, seen, session, batch, stop))
    except Exception as exc:
        error = exc
    finally:
        try:
            pipe.close()
        except Exception as exc:
            if error is None:
                error = exc
            else:
                log.error("upsert failed after another error: %s", exc)
    if error is not None:
        raise error
    reason = _halt_reason(stop, retire, switch_check)
    # A file that finished during the call already running is indexed, not abandoned.
    if reason and not saw_done:
        raise RuntimeError(reason)
    return "done" if saw_done else "closed"


def _publish_flight(session, work, feed) -> None:
    """Show chunk progress for a file that is not a ZIM.

    A ZIM reports each stored article. A PDF or other upload is one article,
    so the row would stay empty until that article was upserted.
    """
    if session is None or getattr(work, "kind", "") == "zim":
        return
    book = session.get_book() if hasattr(session, "get_book") else None
    note = getattr(book, "note_flight", None) if book is not None else None
    if not callable(note):
        return
    done, total = feed.vector_progress()
    if total <= 0:
        return
    note(work.source, done, total)


def _skip_stored_plain(install: Install, session: Session | None, work: FileWork) -> bool:
    """A file that already has points is indexed. OCR does not read it again."""
    existing = count_source(install, work.source)
    if not existing:
        return False
    log.info("skipping %s, points already stored", work.source)
    work.done = True
    work.touched = True
    work.chunks = existing
    work.total_articles = 1
    work.articles = 1
    _write_checkpoint(work)
    _report(session, work, "indexed")
    return True


def _show_ocr(session, work: FileWork, name: str, page: int | None, total: int | None, *, announce: bool) -> None:
    """The OCR row. `announce` also writes the status line.

    A second file can be embedding. Its status line stays put when this read
    was started beside it.
    """
    note = ocr_note(name, page, total) if page and total else ocr_note(name)
    work.phase = "ocr"
    work.label = "OCR"
    if session is not None:
        setter = getattr(session, "set_ocr_activity", None)
        if callable(setter):
            setter(note)
        if announce:
            session.set_note(note)
    if "\n" not in str(getattr(work, "source", "")):
        _report(session, work, "running")


def _clear_ocr_note(session, name: str) -> None:
    if session is None:
        return
    clear = getattr(session, "clear_ocr_activity", None)
    if callable(clear):
        clear(name)


def _rearm_measure(session) -> None:
    """The five-minute sample is embedding time. The OCR read does not use it up."""
    if session is None or not callable(getattr(session, "measuring", None)) or not session.measuring():
        return
    arm = getattr(session, "arm_measure_clock", None)
    if callable(arm):
        arm()


def _read_plain_visual(install, work: FileWork, stop, session, *, announce: bool) -> dict | None:
    """Read one uploaded PDF or picture. None means the read failed."""
    name = getattr(work, "title", "") or work.host.name
    quality = _ocr_quality(session, install)
    drawing = work.kind == "image" or quality != "low"
    if drawing:
        _show_ocr(session, work, name, None, None, announce=announce)
    else:
        work.phase = "text"
        work.label = "text"
        if "\n" not in str(work.source):
            _report(session, work, "running")
        if session is not None and announce:
            session.set_note("Indexing.")

    def _ocr_status(line: str) -> None:
        log.log(ocr_status_level(line), "ocr: %s", line[:300])
        remember_ocr_workers(session, line)
        found = ocr_page(line)
        if found is None or str(line).rstrip().endswith(" text"):
            return
        _show_ocr(session, work, name, found[0], found[1], announce=announce)

    visual = read_visual(
        install,
        str(work.host),
        work.kind,
        stop,
        _ocr_status,
        quality,
        ocr_env=_ocr_container(session, install, wait=True),
    )
    if visual is None:
        return None
    drew = bool(visual.get("drew"))
    pages = visual.get("pages") or []
    total = len(pages) if pages else None
    work.phase = "ocr-text" if drew else "text"
    work.label = "OCR text" if drew else "text"
    if session is not None and session.get_book() is not None and "\n" not in str(work.source):
        session.get_book().update(
            work.source,
            articles=int(work.articles or 0),
            total=work.total_articles,
            chunks=int(work.chunks or 0),
            state="running",
            label=work.label,
            pass_done=int(total or 0),
            pass_total=total,
        )
    return visual


def _article_from_visual(work: FileWork, visual: dict) -> dict:
    if visual.get("pages"):
        return {
            "index": 1,
            "path": work.source,
            "title": work.host.name,
            "kind": "pdf",
            "pages": visual.get("pages") or [],
            "sections": [],
        }
    if "ocr" in visual:
        return {
            "index": 1,
            "path": work.source,
            "title": work.host.name,
            "kind": "image",
            "ocr": visual.get("ocr") or "",
            "sections": [],
        }
    body = str(visual.get("text") or "")
    return {
        "index": 1,
        "path": work.source,
        "title": work.host.name,
        "text": body,
        "sections": [{"heading": "", "text": body, "level": 2}],
    }


def _embed_plain_article(
    install: Install,
    pool: EmbedPool,
    work: FileWork,
    stop: threading.Event,
    session: Session | None,
    retire: threading.Event | None,
    switch_check,
    article,
) -> None:
    if article is None:
        log.warning("%s could not be read. Marked needs NOMAD.", work.source)
        work.done = True
        work.deferred = True
        _report(session, work, "deferred")
        if session is not None:
            session.set_note("Indexing.")
        _clear_ocr_note(session, work.host.name)
        return
    # One document is one article. The total is known before its chunks are.
    if not work.total_articles:
        work.total_articles = 1
    _report(session, work, "running")
    sent = False

    def _pull(_timeout: float):
        nonlocal sent
        if sent:
            return _EXTRACT_DONE
        sent = True
        return article

    outcome = _consume_articles(install, pool, work, stop, session, set(), _pull, retire, switch_check)
    if outcome != "done":
        return
    work.done = True
    work.touched = True
    work.total_articles = 1
    _write_checkpoint(work)
    _report(session, work, "indexed")
    log.info(
        "finished %s, %s chunks, %s skipped, %s bad",
        work.source,
        work.chunks,
        work.skipped,
        work.bad,
    )


def _begin_plain_embed(install: Install, work: FileWork) -> None:
    _load_point_active(install, work)
    work.prints = ArticlePrints()
    work.ledger = library_ledger()


def _index_prepared_plain(
    install: Install,
    pool: EmbedPool,
    work: FileWork,
    stop: threading.Event,
    session: Session | None,
    retire: threading.Event | None,
    switch_check,
    visual: dict | None,
) -> None:
    """Embed a PDF or picture whose text is already read."""
    reason = _halt_reason(stop, retire, switch_check)
    if reason == "replaced" or (retire is not None and retire.is_set()):
        raise RuntimeError("replaced")
    if reason in SWITCH_REASONS:
        raise RuntimeError(reason)
    if reason:
        return
    if _skip_stored_plain(install, session, work):
        return
    _begin_plain_embed(install, work)
    _rearm_measure(session)
    article = None
    if visual is not None:
        drew = bool(visual.get("drew"))
        work.phase = "ocr-text" if drew else "text"
        work.label = "OCR text" if drew else "text"
        if session is not None:
            session.set_note("Indexing.")
        _clear_ocr_note(session, work.host.name)
        article = _article_from_visual(work, visual)
    _embed_plain_article(install, pool, work, stop, session, retire, switch_check, article)


def _index_plain(
    install: Install,
    pool: EmbedPool,
    work: FileWork,
    stop: threading.Event,
    session: Session | None,
    retire: threading.Event | None = None,
    switch_check=None,
) -> None:
    reason = _halt_reason(stop, retire, switch_check)
    if reason == "replaced" or (retire is not None and retire.is_set()):
        raise RuntimeError("replaced")
    if reason in SWITCH_REASONS:
        raise RuntimeError(reason)
    if reason:
        return
    if _skip_stored_plain(install, session, work):
        return
    _begin_plain_embed(install, work)
    article = None
    if work.kind in ("pdf", "image"):
        visual = _read_plain_visual(install, work, stop, session, announce=True)
        _rearm_measure(session)
        if visual is not None:
            drew = bool(visual.get("drew"))
            work.phase = "ocr-text" if drew else "text"
            work.label = "OCR text" if drew else "text"
            if session is not None:
                session.set_note("Indexing.")
            _clear_ocr_note(session, work.host.name)
            article = _article_from_visual(work, visual)
    else:
        text = read_text(work.host, work.kind)
        if text is not None:
            article = {
                "index": 1,
                "path": work.source,
                "title": work.host.name,
                "text": text,
                "sections": [{"heading": "", "text": text, "level": 2}],
            }
    _embed_plain_article(install, pool, work, stop, session, retire, switch_check, article)


def _finalize(install: Install, previous: str, files: list[FileWork]) -> bool:
    partial = False
    for work in files:
        if work.removed or not work.touched or work.deferred:
            continue
        if work.done:
            mark_indexed(install, work.source, work.chunks)
            for job_id, _offset in work.jobs:
                queuectl(install, "remove", job_id)
            continue
        partial = True
        if not _rules().stock_should_continue(previous):
            for job_id, _offset in work.jobs:
                queuectl(install, "remove", job_id)
            state = partial_row_state(ingest_states(install).get(work.source))
            if state == "pending_decision":
                mark_pending(install, work.source, work.chunks)
            continue
        keep, remove = continuation_keep(work.jobs)
        for job_id in remove:
            queuectl(install, "remove", job_id)
        payload = {
            "filePath": work.source,
            "fileName": Path(work.source).name,
            "batchOffset": work.articles,
            "totalArticles": work.total_articles or work.articles,
            "isFinalBatch": False,
            "chunksSoFar": work.chunks,
        }
        if keep:
            queuectl(install, "update", keep, stdin=json.dumps(payload))
        elif work.articles > 0:
            queuectl(install, "add", stdin=json.dumps(payload))
        state = partial_row_state(ingest_states(install).get(work.source))
        if state == "pending_decision":
            mark_pending(install, work.source, work.chunks)
    chosen = _rules().policy_after_stop(previous, partial)
    set_policy(install, chosen)
    if not _rules().stock_should_continue(previous):
        log.info(
            "ingest policy left on Manual, which is how turbo found it. Stock indexing was not started.",
        )
    elif partial and previous == "Always":
        log.info(
            "turbo stopped with unfinished files. Ingest policy stays Manual so NOMAD will not start those files over. It was Always when turbo started.",
        )
    elif previous in ("Manual", "Always") and chosen == previous:
        log.info("ingest policy restored to %s, which is how turbo found it.", chosen)
    else:
        log.info("ingest policy left on %s.", chosen)
    (state_dir() / "result").write_text("partial" if partial else "done")
    return partial


def settle_file_stop(install: Install, session: Session | None, work: FileWork, reason: str) -> str:
    """Checkpoint a switch, or drop a file that left the disk.

    A switch keeps stored points and does not hand the file to NOMAD.
    The disappeared-file path is the only one that deletes points.
    """
    outcome = interrupt_outcome(reason="replaced" if reason == "replaced" else reason, finished=bool(work.done))
    if outcome == "drop":
        _request_drop(install, session, work.source)
        work.removed = True
        work.touched = False
        work.done = False
        return outcome
    _write_checkpoint(work)
    if outcome == "indexed":
        work.done = True
        _report(session, work, "indexed")
    else:
        _report(session, work, _open_state(work))
    return outcome


def finish_measure_pass(session, stop: threading.Event) -> str:
    """End Measure. Idle starts stop the run. Then index, or a run already going, keeps indexing."""
    then_index = bool(session.measure_then_index()) if hasattr(session, "measure_then_index") else False
    action = measure_ending(
        from_idle=bool(session.measure_from_idle()),
        pass_complete=True,
        then_index=then_index,
    )
    session.set_measuring(False)
    if action == "stop":
        session.set_note("Measure finished.")
        stop.set()
    else:
        session.set_note("Indexing.")
    return action


def _recommended_chunks(pool) -> int:
    """The same safe call the Recommended control uses."""
    use_gpu = bool(getattr(pool, "use_gpu", False))
    total = int(getattr(pool, "total_mib", 0) or 0)
    gpu_total = total if use_gpu and total > 0 else None
    return safe_call_chunks(
        gpu_total_mib=gpu_total,
        host_total_mib=int(getattr(pool, "host_total_mib", 0) or 0),
        cpu_count=int(getattr(pool, "cpu_count", 1) or 1),
    )


def _learn_zim_dates(install: Install, session, works: list[FileWork], stop: threading.Event | None = None) -> None:
    """Read every missing ZIM Date before the next file is chosen.

    One archive at a time, through nomad-turbo-zimdate. This does not start
    indexing and does not take the extractor. An unreadable date is remembered
    so the file's modification time is used and docker is not asked again.
    """
    if session is None or not str(session.get_order()).startswith("date"):
        return
    known = getattr(session, "zim_date_known", None)
    if not callable(known):
        return
    try:
        for work in works:
            if stop is not None and stop.is_set():
                return
            if work.kind != "zim" or known(work.source):
                continue
            try:
                raw = read_zim_date(install, str(work.host))
            except Exception:
                log.exception("could not read the ZIM date for %s", work.host.name)
                raw = None
            session.note_zim_date(work.source, raw)
    finally:
        stats = session.file_stats() if hasattr(session, "file_stats") else None
        if stats is not None and hasattr(stats, "save"):
            try:
                stats.save()
            except Exception:
                log.exception("could not save file stats")


def _order_rows(works: list[FileWork], session) -> list[OrderFile]:
    book = session.get_book() if session is not None and hasattr(session, "get_book") else None
    snap = None
    if book is not None and hasattr(book, "snapshot"):
        try:
            snap = book.snapshot()
        except Exception:
            snap = None
    by_source = {item.source: item for item in getattr(snap, "files", ())} if snap is not None else {}
    priorities = session.priorities() if session is not None and hasattr(session, "priorities") else set()
    dates = session.zim_dates() if session is not None and hasattr(session, "zim_dates") else {}
    rows: list[OrderFile] = []
    for index, work in enumerate(works):
        item = by_source.get(work.source)
        total = item.total_articles if item is not None else work.total_articles
        articles = item.articles if item is not None else work.articles
        chunks = item.chunks if item is not None else work.chunks
        size = int(item.size_bytes) if item is not None else 0
        if size <= 0:
            try:
                size = int(work.host.stat().st_size)
            except OSError:
                size = 0
        state = item.state if item is not None else "pending"
        if state == "indexed":
            completed = 1.0
        elif total:
            completed = min(1.0, max(0.0, int(articles) / int(total)))
        else:
            completed = 0.0
        eta = None
        if item is not None and item.eta_seconds is not None and state != "indexed":
            eta = float(item.eta_seconds)
        raw_date = dates.get(work.source) if dates else None
        rows.append(
            OrderFile(
                source=work.source,
                name=work.host.name,
                file_type=file_type_label(work.host.name),
                size_bytes=int(size),
                mtime=work.mtime,
                zim_date=raw_date or None,
                articles_total=int(total) if total else None,
                completed=completed,
                chunks=int(chunks),
                measured_eta=eta,
                state=state,
                walk_index=index,
                priority=work.source in priorities,
            )
        )
    return rows


def _note_sample(session, work: FileWork) -> None:
    if session is None or not hasattr(session, "measuring") or not session.measuring():
        return
    session.note_sampled(work.source)
    book = session.get_book()
    keep = getattr(book, "keep_measured_rate", None)
    if callable(keep):
        keep(work.source)


def _measure_defer(works: list[FileWork], session) -> set[str]:
    """Files this Measure pass should not sample.

    Already sampled this pass, plus anything that already has progress.
    The book and the checkpoint are both read. The larger counts win, so a
    stale row cannot send Measure back over a file that has already started.
    """
    defer = set(session.sampled()) if hasattr(session, "sampled") else set()
    book = session.get_book() if hasattr(session, "get_book") else None
    by_source: dict = {}
    if book is not None and hasattr(book, "snapshot"):
        try:
            snap = book.snapshot()
        except Exception:
            snap = None
        if snap is not None:
            by_source = {item.source: item for item in getattr(snap, "files", ())}
    for work in works:
        item = by_source.get(work.source)
        articles = int(work.articles or 0)
        chunks = int(work.chunks or 0)
        state = ""
        eta = None
        if item is not None:
            articles = max(articles, int(getattr(item, "articles", 0) or 0))
            chunks = max(chunks, int(getattr(item, "chunks", 0) or 0))
            state = getattr(item, "state", "") or ""
            saved = getattr(item, "eta_seconds", None)
            if saved is not None and state != "running":
                eta = saved
        if not file_needs_measure(articles, chunks, state=state, eta_seconds=eta):
            defer.add(work.source)
    return defer


def _queue_batch(
    install: Install,
    session,
    jobs: list[dict],
    finished: set[str],
    failed: set[str],
    snap: LibrarySnap | None = None,
) -> list[FileWork]:
    """Unfinished files in library-walk order. Index now and priority can retry a failure.

    A snapshot from the pause between files is the file list. Omitting it
    walks once.
    """
    if session is not None:
        now = session.index_now() if hasattr(session, "index_now") else None
        priorities = session.priorities() if hasattr(session, "priorities") else set()
        for source in priorities:
            failed.discard(source)
        if now:
            failed.discard(now)
    return [item for item in _work_list(install, jobs, finished, snap) if item.source not in failed]


def article_stay_reason(session, source: str, sample_started, now: float, armed_epoch: int = 0):
    """Whether to leave the file in progress after the embed call already running.

    Index now, a priority mark, and Measure come from the session. An ordinary
    order change stays. This does not snapshot the library.
    """
    return engine_switch_reason(session, source, sample_started, now, armed_epoch)


def _release_finished_index_now(install: Install, session, batch: list[FileWork]) -> None:
    """Drop Index now when this pass cannot open that file.

    The window can still call a file partial after the knowledge base has
    finished it. The work list then skips it. Leaving the request set aborts
    the file in progress and starts that same file again, because the next
    pick never sees the request. A file written in the last 10 seconds is
    kept: it is waiting to settle and the next walk can open it.
    """
    if session is None or not hasattr(session, "index_now"):
        return
    target = session.index_now()
    if not target:
        return
    if any(item.source == target for item in batch):
        return
    host = _library_host(install, target)
    if host is not None and host.exists():
        try:
            if time.time() - host.stat().st_mtime < 10:
                return
        except OSError:
            pass
    log.info(
        "Index now left %s alone. It is already finished, so the file in progress keeps going.",
        Path(target).name,
    )
    session.clear_index_now()


def _switch_check(install: Install, session, work: FileWork, sample_started, armed: int, batch: list[FileWork]):
    """The between-article check for the file now in progress.

    Measure on a file that already has articles or chunks leaves at the next
    boundary and does not replace its ETA. A file with nothing stored yet is
    sampled for the usual five minutes, and the articles that sample stores
    do not end it early. Index now on a file this pass cannot open is cleared
    here, so the file in progress is not stopped and started again.
    """
    held = {"progress": None}

    def switch_check(started=sample_started, source=work.source, epoch=armed, file=work):
        _release_finished_index_now(install, session, batch)
        # A PDF re-arms this clock after OCR, so the sample starts then.
        measuring_now = False
        asked_now = None
        if session is not None:
            ask = getattr(session, "measuring", None)
            measuring_now = bool(callable(ask) and ask())
            if measuring_now:
                asked_now = open_index_now(session)
        reason, held["progress"] = measure_leave_reason(
            sample_started=started,
            measuring=measuring_now,
            articles=int(file.articles or 0),
            chunks=int(file.chunks or 0),
            index_now=asked_now,
            had_progress=held["progress"],
        )
        if reason:
            return reason
        moment = started
        if measuring_now:
            clock = getattr(session, "measure_clock", None)
            if callable(clock):
                live = clock()
                if live is not None:
                    moment = live
        return article_stay_reason(session, source, moment, time.monotonic(), epoch)

    return switch_check


def _next_work(install: Install, pool, session, batch: list[FileWork], stop: threading.Event):
    """The one file this pass should index, plus the measure clock and priority epoch.

    An empty work means Measure finished or nothing matched. The action is
    `stop`, `resume`, or empty.
    """
    if session is not None:
        _learn_zim_dates(install, session, batch, stop)
    if not batch:
        return None, None, 0, ""
    _release_finished_index_now(install, session, batch)
    choice = session.get_order() if session is not None and hasattr(session, "get_order") else "library"
    rows = _order_rows(batch, session)
    index_now = open_index_now(session, rows) if session is not None else None
    if session is not None and hasattr(session, "index_now") and session.index_now() and index_now is None:
        session.clear_index_now()
    measuring = bool(session is not None and hasattr(session, "measuring") and session.measuring())
    defer = _measure_defer(batch, session) if measuring else set()
    picked = pick_next(
        rows,
        choice=choice,
        current=None,
        index_now=index_now,
        defer=defer,
    )
    if picked.target is None:
        if measuring:
            return None, None, 0, finish_measure_pass(session, stop)
        return None, None, 0, ""
    work = next(item for item in batch if item.source == picked.target)
    armed = session.arm_priority() if session is not None and hasattr(session, "arm_priority") else 0
    sample_started = None
    if measuring and index_now != work.source and pool is not None:
        auto = bool(session.tune_auto()) if hasattr(session, "tune_auto") else False
        apply_sample_batch(session, auto=auto, recommended=_recommended_chunks(pool))
        arm = getattr(session, "arm_measure_clock", None)
        sample_started = arm() if callable(arm) else time.monotonic()
    return work, sample_started, armed, ""


def _ocr_job_source(owner: str, article_path: str) -> str:
    if article_path and article_path != owner:
        return owner + "\n" + article_path
    return owner


def _export_pdf_name(article_path: str) -> str:
    import hashlib

    return hashlib.sha256(article_path.encode()).hexdigest()[:16] + ".pdf"


def _export_host(container_file: str) -> Path:
    text = str(container_file or "").replace("\\", "/")
    marker = "/turbo-state/"
    if marker in text:
        return state_dir() / text.split(marker, 1)[1]
    return Path(text)


def _read_spill_list(path: Path) -> list[str]:
    if not path.is_file():
        return []
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            rows.append(line)
            continue
        if isinstance(value, str) and value:
            rows.append(value)
    return rows


def _clear_ocr_queue_dir(spill_id: str) -> None:
    if not spill_id:
        return
    folder = state_dir() / "ocr-queue" / spill_id
    if not folder.is_dir():
        return
    import shutil

    shutil.rmtree(folder, ignore_errors=True)


def _upload_job(work: FileWork):
    from nomad_turbo.overlap import OcrJob

    return OcrJob(
        source=work.source,
        host=work.host,
        kind=work.kind,
        title=work.host.name,
        owner=work.source,
        article_path=work.source,
    )


def _zim_job(owner: str, article_path: str, host: Path, title: str):
    from nomad_turbo.overlap import OcrJob

    return OcrJob(
        source=_ocr_job_source(owner, article_path),
        host=host,
        kind="pdf",
        title=title or Path(article_path).name,
        owner=owner,
        article_path=article_path,
    )


def _remember_touched(touched: list[FileWork], work: FileWork) -> None:
    if all(item.source != work.source for item in touched):
        touched.append(work)


def _fill_upload_queue(lane, install: Install, session, batch: list[FileWork], finished: set[str], touched: list[FileWork]) -> None:
    """Put every uploaded PDF and picture at the back of the OCR queue."""
    reserved = lane.reserved()
    jobs = []
    for work in batch:
        if work.kind not in ("pdf", "image") or work.done or work.source in finished:
            continue
        if work.source in reserved:
            continue
        if _skip_stored_plain(install, session, work):
            finished.add(work.source)
            _remember_touched(touched, work)
            continue
        jobs.append(_upload_job(work))
    if jobs:
        lane.offer(jobs)


def _accept_zim_enqueue(lane, work: FileWork, event: dict) -> None:
    path = str(event.get("path") or "")
    raw = str(event.get("file") or "")
    if not path or not raw:
        return
    done = set(getattr(work, "ocr_done", []) or [])
    if path in done:
        return
    pending = list(getattr(work, "ocr_pending", []) or [])
    if path not in pending:
        pending.append(path)
        work.ocr_pending = pending
    lane.offer([_zim_job(work.source, path, _export_host(raw), str(event.get("title") or path))], front=True)


def _catch_up_exports(lane, work: FileWork) -> None:
    """Queue ZIM PDFs whose bytes are already on disk."""
    if work.kind != "zim":
        return
    spill = getattr(work, "spill", None) or _spill_paths(work.source)
    work.spill = spill
    done = set(getattr(work, "ocr_done", []) or [])
    pending = [path for path in (getattr(work, "ocr_pending", []) or []) if path not in done]
    for path in _read_spill_list(spill["ocr"]):
        if path in done or path in pending:
            continue
        pending.append(path)
    spill_id = str(spill["id"])
    kept: list[str] = []
    for path in pending:
        host = state_dir() / "ocr-queue" / spill_id / _export_pdf_name(path)
        source = _ocr_job_source(work.source, path)
        if lane.pending(source) or lane.has_result(source):
            kept.append(path)
            continue
        if host.is_file():
            kept.append(path)
            lane.offer([_zim_job(work.source, path, host, Path(path).name)], front=True)
            continue
        log.warning("the OCR copy of %s in %s is missing, so that PDF is left unread", path, work.source)
    work.ocr_pending = kept


def _finish_queued_zim(work: FileWork, lane, session, finished: set[str]) -> None:
    """A ZIM is finished once its pages and its queued PDFs are stored."""
    if work.kind != "zim" or work.done or getattr(work, "zim_pass", "") != "queued":
        return
    done = set(getattr(work, "ocr_done", []) or [])
    pending = [path for path in (getattr(work, "ocr_pending", []) or []) if path not in done]
    work.ocr_pending = pending
    if pending:
        return
    if lane is not None and lane.owner_busy(work.source):
        return
    work.done = True
    work.zim_pass = "done"
    work.touched = True
    _write_checkpoint(work)
    spill = getattr(work, "spill", None) or _spill_paths(work.source)
    _clear_zim_ocr_ahead(str(spill["id"]))
    _clear_ocr_queue_dir(str(spill["id"]))
    finished.add(work.source)
    _report(session, work, "indexed")
    log.info(
        "finished %s, %s chunks, %s skipped, %s bad",
        work.source,
        work.chunks,
        work.skipped,
        work.bad,
    )


def _embed_one_article(install, pool, work, stop, session, retire, switch_check, article, seen) -> bool:
    if article is None:
        return False
    sent = False

    def _pull(_timeout: float):
        nonlocal sent
        if sent:
            return _EXTRACT_DONE
        sent = True
        return article

    outcome = _consume_articles(install, pool, work, stop, session, seen, _pull, retire, switch_check)
    return outcome == "done"


def _embed_ready_source(
    lane,
    source: str,
    install: Install,
    pool,
    stop: threading.Event,
    session,
    batch: list[FileWork],
    finished: set[str],
    touched: list[FileWork],
    retire,
    switch_check,
    seen,
) -> bool:
    """Embed one finished OCR read. The GPU is between ZIM articles or between ZIMs."""
    job = lane.job(source)
    visual, err = lane.take(source)
    if job is None:
        return False
    if str(err) == "stopped":
        raise RuntimeError("stopped")
    owner = str(getattr(job, "owner", "") or getattr(job, "source", ""))
    article_path = str(getattr(job, "article_path", "") or owner)
    by_source = {item.source: item for item in batch}
    work = by_source.get(owner)
    if work is None:
        return False
    _remember_touched(touched, work)
    if "\n" in str(getattr(job, "source", "")):
        work.zim_pass = "queued"
        title = str(getattr(job, "title", "") or article_path)
        if err is not None or visual is None:
            log.warning("ocr failed for %s in %s: %s", article_path, owner, err)
            article = {
                "index": int(work.articles or 0) + 1,
                "path": article_path,
                "title": title,
                "sections": [],
                "skip": "read",
            }
        else:
            drew = bool(visual.get("drew"))
            work.phase = "ocr-text" if drew else "text"
            work.label = "OCR text" if drew else "text"
            if session is not None:
                session.set_note("Indexing.")
            article = _article_from_visual(work, visual)
            article["index"] = int(work.articles or 0) + 1
            article["path"] = article_path
            article["title"] = title
        use_seen = seen if seen is not None else set()
        if not _embed_one_article(install, pool, work, stop, session, retire, switch_check, article, use_seen):
            return False
        stored = list(getattr(work, "ocr_done", []) or [])
        if article_path not in stored:
            stored.append(article_path)
        work.ocr_done = stored
        work.ocr_pending = [path for path in (work.ocr_pending or []) if path != article_path]
        work.zim_pass = "queued"
        _write_checkpoint(work)
        _finish_queued_zim(work, lane, session, finished)
        return True
    if err is not None:
        log.warning("ocr failed for %s: %s", owner, err)
        work.done = True
        work.deferred = True
        work.touched = True
        _report(session, work, "deferred")
        finished.add(work.source)
        return True
    try:
        _index_prepared_plain(install, pool, work, stop, session, retire, switch_check, visual)
    except RuntimeError as exc:
        text = str(exc)
        if text in ("stopped", "replaced") or text in SWITCH_REASONS:
            raise
        log.exception("could not embed %s", owner)
        work.done = True
        work.deferred = True
        _report(session, work, "deferred")
        finished.add(work.source)
        return True
    if work.done:
        finished.add(work.source)
        if session is not None and hasattr(session, "clear_index_now"):
            session.clear_index_now(work.source)
    return True


def _drain_ready_ocr(
    lane,
    install: Install,
    pool,
    stop: threading.Event,
    session,
    batch: list[FileWork],
    finished: set[str],
    touched: list[FileWork],
    retire=None,
    switch_check=None,
    seen=None,
    *,
    one: bool = False,
    hold: str | None = None,
) -> int:
    """Embed OCR text that is already ready. Leave a read that is still going.

    `hold` is the ZIM whose pages are still being embedded. Its own PDFs wait
    until that embed finishes, so a crash can resume the walk.
    """
    if lane is None:
        return 0
    count = 0
    while not stop.is_set():
        ready = []
        for source in lane.ready_sources():
            job = lane.job(source)
            if hold and job is not None and getattr(job, "owner", None) == hold:
                continue
            ready.append(source)
        if not ready:
            return count
        if _embed_ready_source(
            lane,
            ready[0],
            install,
            pool,
            stop,
            session,
            batch,
            finished,
            touched,
            retire,
            switch_check,
            seen,
        ):
            count += 1
        if one:
            return count
    return count


def _offer_overlap(lane, install: Install, session, batch: list[FileWork], hold: str | None) -> list[FileWork]:
    """Queue the PDFs and pictures that can be read before the next ordinary file.

    A file that already has points is marked indexed and returned. The caller
    records it finished so it is not read.
    """
    from nomad_turbo.overlap import preview_order, visual_reads

    rows = _order_rows(batch, session)
    choice = session.get_order() if session is not None and hasattr(session, "get_order") else "library"
    index_now = open_index_now(session, rows) if session is not None else None
    measuring = bool(session is not None and hasattr(session, "measuring") and session.measuring())
    defer = _measure_defer(batch, session) if measuring else set()
    kinds = {item.source: item.kind for item in batch}
    # A ZIM can run for hours with nothing for the CPU to read. Uploaded
    # PDFs and pictures later in the queue are read during that time.
    past_ordinary = kinds.get(hold or "", "") == "zim"
    order = preview_order(
        rows,
        choice=choice,
        index_now=index_now,
        defer=defer,
        kind_of=lambda source: kinds.get(source, ""),
        limit=None,
        past_ordinary=past_ordinary,
    )
    reserved = lane.reserved()
    wanted = visual_reads(order, reserved=reserved, hold=hold, limit=None, past_ordinary=past_ordinary)
    by_source = {item.source: item for item in batch}
    offer = []
    done: list[FileWork] = []
    for source in wanted:
        work = by_source.get(source)
        if work is None or source in reserved:
            continue
        if _skip_stored_plain(install, session, work):
            done.append(work)
            continue
        offer.append(work)
    if past_ordinary and offer:
        log.info(
            "CPU OCR will read %s uploaded file%s while this ZIM is indexed.",
            len(offer),
            "" if len(offer) == 1 else "s",
        )
    lane.offer(offer)
    return done


def _await_visual(lane, work: FileWork, stop: threading.Event, session):
    """Wait until this file's read finishes.

    Returns the reader dict, an error, and whether a result arrived. A failed
    read is a dict of None with no error, and the file is marked needs NOMAD.
    Stopping before a result returns finished False.
    """
    while not stop.is_set():
        if lane.has_result(work.source):
            visual, err = lane.take(work.source)
            return visual, err, True
        if not lane.pending(work.source):
            return None, RuntimeError("ocr did not start"), True
        if session is not None:
            note = session.ocr_activity() if hasattr(session, "ocr_activity") else ""
            if note:
                session.set_note(note)
        stop.wait(0.3)
    if lane.has_result(work.source):
        visual, err = lane.take(work.source)
        return visual, err, True
    return None, None, False


def execute(
    stop: threading.Event,
    session: Session | None = None,
    wipe: bool = False,
    manage_icon: bool = True,
) -> int:
    do_wipe = wipe_allowed(wipe, bool(session and session.wipe_confirmed))
    if wipe and not do_wipe:
        raise RuntimeError("refusing to delete the index without confirmation")
    install = load_install()
    if manage_icon:
        write_shortcut(True)
    pool: EmbedPool | None = None
    lane = None
    paused = False
    budget_stop = threading.Event()
    budget_on = False
    previous = None
    touched: list[FileWork] = []
    finished: set[str] = set()
    failed_before_work = False
    try:
        try:
            _collection_logbook().align_running_logs(state_dir(), install)
        except Exception:
            log.exception("could not align log files with the knowledge index")
        collection_ready(install)
        previous = get_policy(install)
        (state_dir() / "policy").write_text(previous)
        set_policy(install, "Manual")
        log.info(
            "NOMAD ingest policy was %s. Turbo keeps Manual until this run stops, then puts that back.",
            previous,
        )
        if session:
            session.set_phase("waiting")
            session.set_note("Waiting for NOMAD to finish its current embed batches.")
        queuectl(install, "pause")
        paused = True
        _status(phase="waiting", paused_queue=True, slots=0)
        while not stop.is_set():
            active = active_embeds(install)
            if offsets_are_safe(active):
                break
            log.info("NOMAD still has %s embed job(s) in progress", active)
            if session:
                session.set_note(f"Waiting for NOMAD to finish {active} embed job(s).")
            _keep_nomad_quiet(install, stop)
            stop.wait(5)
        if stop.is_set():
            log.info("stopped before turbo wrote anything")
            return 0
        if not offsets_are_safe(active_embeds(install)):
            raise RuntimeError("refusing to read offsets while file-embeddings still has active jobs")
        if do_wipe:
            wipe_knowledge_index(install)
            if session and session.get_book():
                session.get_book().mark_all_unindexed()
            jobs: list[dict] = []
        else:
            jobs = list_jobs(install)
        from nomad_turbo.devices import list_devices, pick_index, saved_choice
        from nomad_turbo.ocr_budget import arm, serve_budget

        index_device = pick_index(list_devices(), saved_choice(session, install, "index_device"))
        if index_device.kind == "nvidia" and index_device.index is not None:
            gpu_index, total, used = index_device.index, index_device.total_mib, index_device.used_mib
            has_gpu = True
        else:
            gpu_index, total, used, has_gpu = None, 0, 0, False
        arm(state_dir() / "ocr-budget", os_cpu())
        threading.Thread(
            target=serve_budget,
            args=(budget_stop,),
            name="nomad-turbo-ocr-budget",
            daemon=True,
        ).start()
        budget_on = True
        pool = EmbedPool(
            install,
            use_gpu=has_gpu,
            gpu_index=gpu_index,
            total_mib=total,
            used_before=used,
            per_slot_mib=install.per_slot_mib,
            cpu_count=os_cpu(),
        )
        _after_batch(pool, session, install)
        if session:
            session.set_phase("running")
            session.set_note("Indexing.")
        _status(phase="running", paused_queue=True, slots=len(pool.urls))
        try:
            import importlib

            import nomad_turbo.overlap as overlap_mod

            # This window reloads engine on Start. Reload the reader here so
            # that Start also picks up a change in the overlap module.
            importlib.reload(overlap_mod)
            from nomad_turbo.overlap import VisualLane, cpu_ocr_overlaps_gpu

            if cpu_ocr_overlaps_gpu(_ocr_assignment(session, install)):
                holder: dict = {"lane": None}

                def _ahead(item, _holder=holder):
                    flag = _holder["lane"].stop_flag() if _holder["lane"] is not None else stop
                    return _read_plain_visual(install, item, flag, session, announce=False)

                spill = state_dir() / "ocr-ahead"
                spill.mkdir(parents=True, exist_ok=True)
                for old in spill.glob("*.json"):
                    try:
                        old.unlink()
                    except OSError:
                        log.exception("could not clear a leftover OCR read")
                lane = VisualLane(_ahead, stop, spill_dir=spill)
                holder["lane"] = lane
                log.info(
                    "CPU OCR reads uploaded PDFs and pictures while the GPU indexes. "
                    "A PDF found in a ZIM goes to the front of that queue. "
                    "Finished OCR text is embedded after the ZIM the GPU is on, before the next ZIM. "
                    "A plain upload, such as markdown, stays in the embed list and is indexed when the GPU reaches it. "
                    "Finished reads stay in memory until they pass a quarter of system memory, then they wait on disk."
                )
        except Exception:
            log.exception("could not see where OCR will run")
            if lane is not None:
                try:
                    lane.close()
                except Exception:
                    log.exception("could not stop the OCR read")
            lane = None
        failed: set[str] = set()
        retried: set[tuple[str, int]] = set()
        live = _LiveFile()
        watch_stop = threading.Event()
        watch_thread = threading.Thread(
            target=_watch_library,
            args=(install, session, watch_stop, live),
            name="nomad-turbo-library",
            daemon=True,
        )
        global _library_live
        _library_live = live
        watch_thread.start()
        idle_note = ""
        queue_told = False
        try:
            while not stop.is_set():
                # One walk while no file is open. New files, the queue, and
                # the download check all read it. An open file does not walk.
                snap = None
                try:
                    snap = _look_between_files(install, session, live)
                except Exception:
                    log.exception("library scan failed")
                pending = _queue_has_work(session, finished, failed)
                if pending is False:
                    batch = []
                else:
                    try:
                        jobs = list_jobs(install)
                    except Exception:
                        log.exception("could not refresh the embed queue")
                    batch = _queue_batch(install, session, jobs, finished, failed, snap)
                if not batch:
                    if session is not None and hasattr(session, "measuring") and session.measuring() and pending is False:
                        action = finish_measure_pass(session, stop)
                        if action == "stop" or stop.is_set():
                            break
                    keep = _wants_control(session, install)
                    if snap is not None:
                        downloads = snap.downloads
                    else:
                        downloads = _downloads_pending(install)
                    settling = pending is True
                    if hand_back_when_idle(keep_control=keep, downloads_pending=downloads or settling):
                        note = _rules().idle_handoff_note(previous)
                        log.info(note)
                        if session:
                            session.set_note(note)
                        break
                    if downloads:
                        note = "Waiting for a download to finish."
                    elif settling:
                        note = "New file found. It will start once it has finished writing."
                    else:
                        note = "Queue is empty. Turbo is staying on and will index new files."
                    if note != idle_note:
                        log.info(note)
                        idle_note = note
                    if session:
                        session.set_note(note)
                    stop.wait(SCAN_SECONDS)
                    continue
                idle_note = ""
                if lane is not None:
                    _fill_upload_queue(lane, install, session, batch, finished, touched)
                    for queued in batch:
                        if queued.kind == "zim" and queued.zim_pass == "queued":
                            queued.spill = getattr(queued, "spill", None) or _spill_paths(queued.source)
                            _catch_up_exports(lane, queued)
                            _finish_queued_zim(queued, lane, session, finished)
                    _drain_ready_ocr(lane, install, pool, stop, session, batch, finished, touched)
                    if not queue_told:
                        uploads = sum(
                            1 for item in batch if item.kind in ("pdf", "image") and item.source not in finished
                        )
                        log.info(
                            "CPU OCR queue has %s uploaded PDFs and pictures. "
                            "A PDF found in a ZIM goes to the front. "
                            "Finished OCR text is embedded after the current ZIM, before the next one. "
                            "A plain upload, such as markdown, stays in the embed list.",
                            uploads,
                        )
                        queue_told = True
                    now = open_index_now(session) if session is not None else None
                    from nomad_turbo.overlap import gpu_candidates

                    gpu_batch = gpu_candidates(batch, overlap=True, index_now=now, finished=finished)
                    if not gpu_batch:
                        if stop.wait(0.3):
                            break
                        continue
                else:
                    gpu_batch = batch
                work, sample_started, armed, action = _next_work(install, pool, session, gpu_batch, stop)
                if work is None:
                    if action == "stop" or stop.is_set():
                        break
                    continue
                if stop.is_set() or not _wait_to_write(
                    install,
                    stop,
                    session,
                    downloads=None if snap is None else snap.downloads,
                ):
                    break
                _flush_drops(
                    install,
                    session,
                    stop,
                    downloads=None if snap is None else snap.downloads,
                )
                touched.append(work)
                if not work.host.exists():
                    _request_drop(install, session, work.source)
                    work.removed = True
                    work.touched = False
                    continue

                switch_check = _switch_check(install, session, work, sample_started, armed, batch)
                if lane is not None and work.kind in ("pdf", "image"):
                    lane.offer([_upload_job(work)], front=True)
                    lane.prefer(work.source)

                live.begin(work.source)
                try:
                    if work.done:
                        pass
                    elif lane is not None and work.kind in ("pdf", "image"):
                        prepared_visual, read_error, read_done = _await_visual(lane, work, stop, session)
                        if not read_done or (read_error is not None and str(read_error) == "stopped"):
                            _report(session, work, _open_state(work))
                            break
                        if read_error is not None:
                            raise read_error
                        _index_prepared_plain(
                            install, pool, work, stop, session, live.retire, switch_check, prepared_visual
                        )
                    elif work.kind == "zim":
                        def _idle(seen, _work=work, _switch=switch_check):
                            if lane is None:
                                return False
                            did = _drain_ready_ocr(
                                lane,
                                install,
                                pool,
                                stop,
                                session,
                                batch,
                                finished,
                                touched,
                                live.retire,
                                _switch,
                                seen,
                                one=True,
                                hold=_work.source,
                            )
                            if did and session is not None:
                                _report(session, _work, "running")
                            return bool(did)

                        def _enq(event, _work=work):
                            if lane is not None:
                                _accept_zim_enqueue(lane, _work, event)

                        def _body(_work=work):
                            if lane is None:
                                return
                            _catch_up_exports(lane, _work)
                            if _work.ocr_pending:
                                _work.done = False
                                _work.zim_pass = "queued"
                                _work.touched = True
                                _write_checkpoint(_work)
                                _report(session, _work, "partial")
                                count = len(_work.ocr_pending)
                                log.info(
                                    "%s pages are stored. %s PDF%s from it %s in the OCR queue.",
                                    Path(_work.source).name,
                                    count,
                                    "" if count == 1 else "s",
                                    "is" if count == 1 else "are",
                                )
                                return
                            _work.done = True
                            _work.zim_pass = "done"
                            _work.touched = True
                            _write_checkpoint(_work)
                            spill = getattr(_work, "spill", None) or _spill_paths(_work.source)
                            _clear_zim_ocr_ahead(str(spill["id"]))
                            _clear_ocr_queue_dir(str(spill["id"]))
                            _report(session, _work, "indexed")
                            log.info(
                                "finished %s, %s chunks, %s skipped, %s bad",
                                _work.source,
                                _work.chunks,
                                _work.skipped,
                                _work.bad,
                            )

                        _index_zim(
                            install,
                            pool,
                            work,
                            stop,
                            session,
                            live.retire,
                            switch_check,
                            on_idle=_idle if lane is not None else None,
                            on_enqueue=_enq if lane is not None else None,
                            on_body_done=_body if lane is not None else None,
                        )
                    else:
                        _index_plain(install, pool, work, stop, session, live.retire, switch_check)
                except Exception as exc:
                    text = str(exc)
                    if text == "replaced" or live.retire.is_set():
                        settle_file_stop(install, session, work, "replaced")
                        continue
                    if text in SWITCH_REASONS:
                        outcome = settle_file_stop(install, session, work, text)
                        if text == "measure":
                            _note_sample(session, work)
                        elif text == "measure-skip":
                            log.info(
                                "leaving %s at article %s. It already has progress, so Measure will not sample it.",
                                work.source,
                                work.articles,
                            )
                        if outcome == "indexed":
                            finished.add(work.source)
                            _forget_source_jobs(install, work.source)
                            if session is not None and hasattr(session, "clear_index_now"):
                                session.clear_index_now(work.source)
                        continue
                    if stop.is_set() or text == "stopped":
                        log.info("stopping at %s article %s", work.source, work.articles)
                        _report(session, work, _open_state(work))
                        break
                    if _rules().qdrant_write_held(text):
                        log.warning(
                            "qdrant is not taking writes. Staying on %s at article %s: %s",
                            work.source,
                            work.articles,
                            exc,
                        )
                        _report(session, work, _open_state(work))
                        if session:
                            session.set_note(QDRANT_HOLD_NOTE)
                        if stop.wait(QDRANT_HOLD_SECONDS):
                            log.info("stopping at %s article %s", work.source, work.articles)
                        break
                    if repeat_failure(retried, work.source, work.articles):
                        log.exception("file failed: %s", work.source)
                        _report(session, work, "failed")
                        failed.add(work.source)
                    else:
                        log.warning(
                            "will try %s again from article %s after an error: %s",
                            work.source,
                            work.articles,
                            exc,
                        )
                        _report(session, work, _open_state(work))
                else:
                    if work.done:
                        finished.add(work.source)
                        _forget_source_jobs(install, work.source)
                        if session is not None and hasattr(session, "clear_index_now"):
                            session.clear_index_now(work.source)
                        if session is not None and hasattr(session, "measuring") and session.measuring():
                            _note_sample(session, work)
                    elif stop.is_set():
                        _report(session, work, _open_state(work))
                        break
                finally:
                    live.end()
        finally:
            _library_live = None
            watch_stop.set()
            watch_thread.join(timeout=30)
            if watch_thread.is_alive():
                log.warning("library watch is still stopping. Handoff will continue.")
        return 0
    except Exception:
        failed_before_work = not any(item.touched for item in touched)
        raise
    finally:
        if lane is not None:
            try:
                lane.close()
            except Exception:
                log.exception("could not stop the OCR read")
        if budget_on:
            budget_stop.set()
            from nomad_turbo.ocr_budget import disarm
            from nomad_turbo.ocr_engine import gpu_ocr

            disarm()
            gpu_ocr.stop()
        if pool is not None:
            pool.stop_all()
            pool.publish(session)
        stop_ours()
        hold = hold_queue_on_exit(any(item.touched for item in touched), failed_before_work)
        if previous is not None and not hold:
            try:
                _finalize(install, previous, touched)
            except Exception:
                log.exception("handoff back to NOMAD failed")
                try:
                    set_policy(install, "Manual")
                except Exception:
                    log.exception("could not keep ingest policy on Manual")
        elif hold:
            log.info("turbo failed before it indexed anything. Ingest stays Manual and the embed queue stays paused.")
        if paused and not hold:
            try:
                queuectl(install, "resume")
                log.info("file-embeddings queue resumed")
            except Exception:
                log.exception("could not resume file-embeddings")
        status_fields = {"phase": "off", "paused_queue": bool(paused and hold), "slots": 0}
        if not hold:
            status_fields["repair_hold"] = False
        _status(**status_fields)
        if manage_icon:
            write_shortcut(False)
        # After the thread's work is done, including a sample Stop cut short.
        # Clearing earlier would make Measure look idle while handoff is still running.
        release_measure(session)


def os_cpu() -> int:
    import os

    return os.cpu_count() or 1


def recover_paused() -> bool:
    """A dead turbo left the embed queue paused. Restore continuations and resume it.

    Returns False when a repair delete may still be running. The queue stays paused.
    """
    status_path = state_dir() / "status.json"
    if status_path.exists():
        try:
            if json.loads(status_path.read_text()).get("repair_hold"):
                log.warning("a repair delete may still be running in Qdrant. The embed queue stays paused.")
                return False
        except json.JSONDecodeError:
            pass
    install = load_install()
    folder = state_dir() / "checkpoints"
    files: list[FileWork] = []
    if folder.is_dir():
        for path in folder.glob("*.json"):
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            source = data.get("source")
            if not source:
                continue
            jobs = []
            for item in data.get("jobs") or []:
                if isinstance(item, list) and len(item) == 2:
                    jobs.append((str(item[0]), int(item[1])))
            files.append(
                FileWork(
                    host=Path(source),
                    kind="zim",
                    source=source,
                    articles=int(data.get("articles") or 0),
                    path=data.get("path"),
                    chunks=int(data.get("chunks") or 0),
                    done=bool(data.get("done")),
                    jobs=jobs,
                    touched=True,
                )
            )
    saved = state_dir() / "policy"
    previous = saved.read_text().strip() if saved.exists() else "Manual"
    if files:
        _finalize(install, previous, files)
    else:
        from nomad_turbo.store import set_policy

        set_policy(install, previous if previous in ("Manual", "Always") else "Manual")
    stop_ours()
    queuectl(install, "resume")
    _status(phase="off", paused_queue=False, slots=0, repair_hold=False)
    log.info("recovered a paused embed queue")
    return True
