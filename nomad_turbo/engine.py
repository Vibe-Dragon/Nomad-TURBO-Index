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
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from nomad_turbo.chunking import DOCUMENT_PREFIX, keywords
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
from nomad_turbo.plain import read_text
from nomad_turbo.pool import EmbedPool
from nomad_turbo.runtime import (
    active_embeds,
    ChunkRejected,
    embed,
    gpu_rows,
    host_to_container,
    list_jobs,
    queue_is_paused,
    queuectl,
    release_extract,
    read_visual,
    start_extract,
    stop_ours,
)
from nomad_turbo.session import Session
from nomad_turbo.slots import install_boost_percent, pick_gpu
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


def _note_downloads(install: Install) -> tuple[bool, float | None]:
    """True while a .tmp or .part file exists, then seconds since the last one vanished."""
    global _last_download_at
    pending = _downloads_pending(install)
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


def _library_hold(install: Install) -> str:
    pending, age = _note_downloads(install)
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
    elif reason == "settle":
        message = "A download just finished. Waiting for its knowledge-base update before writing."
    elif reason == "disk":
        message = "The disk is busy. Slowing knowledge-base writes."
    elif reason == "segment":
        message = (
            "Indexing is paused so a segment can be optimized. "
            "A plain segment is past the 4 GiB vector cap. "
            "New points are waiting. Points already stored stay. "
            "The optimization bar on this page starts over for this pause. "
            "Qdrant does not report a percent while it builds the segment."
        )
    elif reason == "space":
        message = (
            "Indexing is paused so a segment can be optimized. "
            "Free space is under twice the segment that would be copied. "
            "New points are waiting. Points already stored stay. "
            "The optimization bar on this page starts over for this pause."
        )
    else:
        message = "Knowledge-base writes resumed."
    log.info(message)
    _hold_announced = reason
    hold = reason if reason in ("segment", "space") else ""
    setter = getattr(session, "set_optimize_hold", None) if session is not None else None
    if callable(setter):
        setter(hold)
    if session is not None and reason:
        session.set_note(message)
    elif session is not None:
        session.set_note("Indexing.")


def _note_embed_call(session, percent: int, chunks: int, seconds: float, ok: bool) -> None:
    note = getattr(session, "note_embed_call", None)
    if callable(note):
        note(percent, chunks, seconds, ok)


def _segment_hold(install: Install) -> str:
    """Wait while a segment rewrite would not fit. Import is local so a new Start sees it."""
    from nomad_turbo import segment_guard

    return segment_guard.segment_upsert_hold(segment_guard.cached_write_facts(install)).reason


def _wait_to_write(
    install: Install,
    stop: threading.Event,
    session: Session | None,
    interval: float = 5.0,
) -> bool:
    """Hold Qdrant writes during a download or while a segment cannot be rewritten.

    A download holds until it is gone and the settle period has passed. A
    stalled disk waits five seconds, then the write goes ahead so indexing
    does not stop for the whole run. A segment that is past the vector cap,
    or free space under twice that segment, keeps waiting. Points stay.
    """
    while not stop.is_set():
        # Segment and space are hard holds. A disk stall must not skip them.
        segment = _segment_hold(install)
        library = _library_hold(install)
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


def _flush_drops(install: Install, session: Session | None, stop: threading.Event) -> None:
    with _drops_lock:
        pending = list(_read_drops().items())
    for source, since in pending:
        if stop.is_set():
            return
        reason = _library_hold(install)
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
    skipped: int = 0
    bad: int = 0
    # Copied onto every point. False only when the ingest row turned the file off.
    point_active: bool = True


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
            }
        )
    )


def _wants_control(session: Session | None, install: Install) -> bool:
    getter = getattr(session, "get_keep_control", None)
    if callable(getter):
        return bool(getter())
    return bool(getattr(install, "keep_control", False))


def _queue_has_work(session: Session | None, finished: set[str], failed: set[str]) -> bool | None:
    """False when the window already knows there is nothing left to index."""
    book = session.get_book() if session else None
    listed = getattr(book, "listed", None)
    if not callable(listed):
        return None
    for source, state in listed():
        if state in ("indexed", "deferred", "failed"):
            continue
        if source in finished or source in failed:
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


def _sync_library(install: Install, session: Session | None, live: _LiveFile) -> None:
    """Show files that just arrived, and clear entries for files NOMAD removed."""
    book = session.get_book() if session else None
    present = getattr(book, "ensure_present", None)
    for host, kind, source in _sources(install, include_fresh=True):
        if not callable(present):
            continue
        try:
            size = host.stat().st_size
        except OSError:
            size = 0
        if present(
            source,
            name=host.name,
            origin=origin_for(source),
            kind=kind,
            size_bytes=size,
        ):
            log.info("new file in the queue: %s", host.name)
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
        if stop.wait(SCAN_SECONDS):
            return


def _report(session: Session | None, work: FileWork, state: str) -> None:
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
    )


def wipe_knowledge_index(install: Install) -> None:
    """Delete vectors and ingest rows. ZIM files and uploads stay on disk.

    The embed queue has to be idle. Clearing it only removes this queue.
    """
    if not offsets_are_safe(active_embeds(install)):
        raise RuntimeError("refusing to delete the index while an embed job is active")
    log.warning("deleting the knowledge index. Library files are kept.")
    queuectl(install, "clear")
    delete_collection(install)
    collection_ready(install)
    clear_ingest_rows(install)
    folder = state_dir() / "checkpoints"
    if folder.is_dir():
        for path in folder.glob("*.json"):
            path.unlink()
    log.warning("knowledge index deleted.")


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


def _sources(install: Install, *, include_fresh: bool = False) -> list[tuple[Path, str, str]]:
    root = Path(install.storage_host)
    found: list[tuple[Path, str, str]] = []
    for folder in ("zim", "kb_uploads"):
        base = root / folder
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            kind = file_is_embeddable(path.name)
            if not kind:
                continue
            try:
                if not include_fresh and time.time() - path.stat().st_mtime < 10:
                    continue
            except OSError:
                continue
            found.append((path, kind, host_to_container(install, str(path))))
    return found


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
    root = Path(install.storage_host)
    for folder in ("zim", "kb_uploads"):
        base = root / folder
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.name.endswith(".tmp") or path.name.endswith(".part"):
                return True
    return False


def _work_list(install: Install, jobs: list[dict], finished: set[str]) -> list[FileWork]:
    grouped: dict[str, list[tuple[str, int, dict]]] = {}
    for job in jobs:
        data = job.get("data") or {}
        source = data.get("filePath")
        if not source:
            continue
        grouped.setdefault(source, []).append((str(job["id"]), int(data.get("batchOffset") or 0), data))
    states = ingest_states(install)
    work: list[FileWork] = []
    for host, kind, source in _sources(install):
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
        total = None
        chunks = int((checkpoint or {}).get("chunks") or 0)
        for _job_id, _offset, data in entries:
            if data.get("totalArticles"):
                total = int(data["totalArticles"])
            chunks = max(chunks, int(data.get("chunksSoFar") or 0))
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
    prepared = prepare_article(
        article,
        work.source,
        article.get("_archive") or {},
        str(uuid.uuid4()),
        install.document_prefix or DOCUMENT_PREFIX,
    )
    for section in prepared.sections:
        section.meta["active"] = work.point_active
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
    """Start the one embed server. Moving the slider does not start another."""
    percent = session.get_percent() if session else install_boost_percent(install)
    pool.apply_target(percent, batch_open=False)
    pool.publish(session)


def _index_zim(
    install: Install,
    pool: EmbedPool,
    work: FileWork,
    stop: threading.Event,
    session: Session | None,
    retire: threading.Event | None = None,
) -> None:
    _load_point_active(install, work)
    already = count_source(install, work.source)
    stored = existing_paths(install, work.source) or set() if already is not None and already <= 2000 else set()
    log.info("indexing %s from article %s (%s paths already stored)", work.source, work.articles, len(stored))
    _report(session, work, "running")
    proc = start_extract(
        install,
        str(work.host),
        0 if work.path else work.articles,
        work.path,
        work.articles,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    def _drain() -> None:
        for line in proc.stderr:
            text = line.strip()
            if text:
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
        while not stop.is_set() and not (retire is not None and retire.is_set()):
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
                continue
            if event.get("type") == "article":
                event["_archive"] = archive
                return event
            if event.get("type") == "done":
                return _EXTRACT_DONE
        return None

    try:
        outcome = _consume_articles(install, pool, work, stop, session, stored, _pull, retire)
        if outcome == "done":
            work.done = True
            work.touched = True
            _write_checkpoint(work)
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
        release_extract(proc)


def _halt_reason(stop: threading.Event, retire: threading.Event | None) -> str | None:
    if retire is not None and retire.is_set():
        return "replaced"
    if stop.is_set():
        return "stopped"
    return None


def _consume_articles(install, pool, work, stop, session, seen, pull, retire=None) -> str:
    """Embed across articles. Checkpoint an article only after its points are stored."""
    feed = ArticleFeed()
    pipe = _UpsertPipe()
    ended = False
    saw_done = False
    last_width = None
    error = None
    try:
        while _halt_reason(stop, retire) is None:
            percent = session.get_percent() if session else install_boost_percent(install)
            width = pool.batch_width(percent)
            if width != last_width:
                log.info("%s chunks per call at %s%%", width, percent)
                last_width = width
            while feed.pending() < width and not ended and _halt_reason(stop, retire) is None:
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
            if _halt_reason(stop, retire):
                break
            if feed.pending() == 0:
                ready = feed.drain_ready()
                if ready:
                    pipe.submit(lambda batch=ready: _store_ready(install, work, seen, session, batch, stop))
                    continue
                if ended:
                    break
                continue
            if not pool.urls:
                raise RuntimeError("the embed server is not running")
            texts = feed.next_inputs(width)
            started = time.monotonic()
            try:
                vectors = embed(pool.urls[0], install.embed_model, texts)
            except ChunkRejected as exc:
                action, article_path = feed.split_rejected(exc.index)
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
                _note_embed_call(session, percent, len(texts), time.monotonic() - started, False)
                raise
            else:
                # A short leftover batch is not the size the slider asked for.
                if len(texts) >= max(4, width // 2):
                    _note_embed_call(session, percent, len(texts), time.monotonic() - started, True)
            ready = feed.push_vectors(vectors)
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
    reason = _halt_reason(stop, retire)
    if reason:
        raise RuntimeError(reason)
    return "done" if saw_done else "closed"


def _index_plain(
    install: Install,
    pool: EmbedPool,
    work: FileWork,
    stop: threading.Event,
    session: Session | None,
    retire: threading.Event | None = None,
) -> None:
    if _halt_reason(stop, retire):
        if retire is not None and retire.is_set():
            raise RuntimeError("replaced")
        return
    _load_point_active(install, work)
    existing = count_source(install, work.source)
    if existing:
        log.info("skipping %s, points already stored", work.source)
        work.done = True
        work.touched = True
        work.chunks = existing
        work.total_articles = 1
        work.articles = 1
        _write_checkpoint(work)
        _report(session, work, "indexed")
        return
    if work.kind in ("pdf", "image"):
        text = read_visual(install, str(work.host), work.kind, stop)
    else:
        text = read_text(work.host, work.kind)
    if text is None:
        log.warning("%s could not be read. Marked needs NOMAD.", work.source)
        work.done = True
        work.deferred = True
        _report(session, work, "deferred")
        return
    _report(session, work, "running")
    article = {
        "index": 1,
        "path": work.source,
        "title": work.host.name,
        "text": text,
        "sections": [{"heading": "", "text": text, "level": 2}],
    }
    sent = False

    def _pull(_timeout: float):
        nonlocal sent
        if sent:
            return _EXTRACT_DONE
        sent = True
        return article

    outcome = _consume_articles(install, pool, work, stop, session, set(), _pull, retire)
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
    paused = False
    previous = None
    touched: list[FileWork] = []
    finished: set[str] = set()
    failed_before_work = False
    try:
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
        rows = gpu_rows()
        chosen = pick_gpu(rows)
        if chosen:
            gpu_index, total, used = chosen
            has_gpu = True
        else:
            gpu_index, total, used, has_gpu = None, 0, 0, False
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
        watch_thread.start()
        idle_note = ""
        try:
            while not stop.is_set():
                pending = _queue_has_work(session, finished, failed)
                if pending is False:
                    batch = []
                else:
                    try:
                        jobs = list_jobs(install)
                    except Exception:
                        log.exception("could not refresh the embed queue")
                    batch = [item for item in _work_list(install, jobs, finished) if item.source not in failed]
                if not batch:
                    keep = _wants_control(session, install)
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
                stopped_early = False
                for work in batch:
                    if stop.is_set():
                        stopped_early = True
                        break
                    if not _wait_to_write(install, stop, session):
                        stopped_early = True
                        break
                    _flush_drops(install, session, stop)
                    touched.append(work)
                    if not work.host.exists():
                        _request_drop(install, session, work.source)
                        work.removed = True
                        work.touched = False
                        continue
                    live.begin(work.source)
                    try:
                        if work.kind == "zim":
                            _index_zim(install, pool, work, stop, session, live.retire)
                        else:
                            _index_plain(install, pool, work, stop, session, live.retire)
                    except Exception as exc:
                        if str(exc) == "replaced" or live.retire.is_set():
                            _request_drop(install, session, work.source)
                            work.removed = True
                            work.touched = False
                            work.done = False
                            continue
                        if stop.is_set() or str(exc) == "stopped":
                            log.info("stopping at %s article %s", work.source, work.articles)
                            _report(session, work, "partial")
                            stopped_early = True
                            break
                        if _rules().qdrant_write_held(str(exc)):
                            log.warning(
                                "qdrant is not taking writes. Staying on %s at article %s: %s",
                                work.source,
                                work.articles,
                                exc,
                            )
                            _report(session, work, "partial")
                            if session:
                                session.set_note(QDRANT_HOLD_NOTE)
                            if stop.wait(QDRANT_HOLD_SECONDS):
                                log.info("stopping at %s article %s", work.source, work.articles)
                                stopped_early = True
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
                            _report(session, work, "partial")
                    else:
                        if work.done:
                            finished.add(work.source)
                            _forget_source_jobs(install, work.source)
                        elif stop.is_set():
                            _report(session, work, "partial")
                            stopped_early = True
                    finally:
                        live.end()
                    if stopped_early:
                        break
                if stopped_early:
                    break
        finally:
            watch_stop.set()
            watch_thread.join(timeout=30)
            if watch_thread.is_alive():
                log.warning("library watch is still stopping. Handoff will continue.")
        return 0
    except Exception:
        failed_before_work = not any(item.touched for item in touched)
        raise
    finally:
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
