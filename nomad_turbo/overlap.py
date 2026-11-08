"""CPU OCR while a GPU indexes.

Uploaded PDFs and pictures start in the OCR queue. A PDF found in a ZIM
goes to the front. The file already being read finishes first. Finished
text waits, and the GPU embeds it after the ZIM it is on. GPU OCR stays
one file: that reader uses the indexing card. Finished text stays in
memory until it reaches about a quarter of system memory. Further finished
reads wait in a temporary file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from nomad_turbo.schedule import pick_next

log = logging.getLogger(__name__)

# Share of system RAM that may hold finished OCR text. Past this, reads wait on disk.
RAM_FRACTION = 0.25

VISUAL_KINDS = frozenset({"pdf", "image"})


def ocr_ram_budget(total_bytes: int | None = None) -> int:
    """Bytes of finished OCR text kept in memory.

    `total_bytes` is system RAM. When it is omitted, the budget is a quarter
    of the installed RAM. A machine that does not report its RAM spills every
    finished read to disk.
    """
    if total_bytes is None:
        total_bytes = _mem_total_bytes()
    total = max(0, int(total_bytes or 0))
    return int(total * RAM_FRACTION)


def _mem_total_bytes() -> int:
    try:
        from nomad_turbo.slots import host_memory_mib

        memory = host_memory_mib()
    except Exception:
        return 0
    if not memory or not memory[0]:
        return 0
    return int(memory[0]) * 1024 * 1024


def text_bytes(visual) -> int:
    """UTF-8 size of the text a finished read is holding."""
    if not isinstance(visual, dict):
        return 0
    total = 0
    pages = visual.get("pages")
    if isinstance(pages, list) and pages:
        for page in pages:
            if isinstance(page, dict):
                for key in ("text", "layer", "ocr"):
                    total += len(str(page.get(key) or "").encode("utf-8"))
            elif isinstance(page, str):
                total += len(page.encode("utf-8"))
        return total
    for key in ("text", "ocr"):
        total += len(str(visual.get(key) or "").encode("utf-8"))
    return total


def cpu_ocr_overlaps_gpu(assignment) -> bool:
    """True when the reader is Tesseract and indexing is on a GPU.

    A GPU reader stays on one file. CPU indexing shares its cores with
    Tesseract, so a second file would only contend with the first.
    """
    if assignment is None:
        return False
    index = getattr(assignment, "index", None)
    backend = getattr(assignment, "backend", None)
    if index is None or backend is None:
        return False
    if getattr(index, "kind", "") == "cpu":
        return False
    return getattr(backend, "name", "") == "tesseract"


def visual_reads(
    order: list[tuple[str, str]],
    *,
    reserved: set[str],
    hold: str | None,
    limit: int | None = None,
    past_ordinary: bool = False,
) -> list[str]:
    """PDF and picture sources to read now, in index order.

    `order` is `(source, kind)` in the order files will be indexed.
    `reserved` sources are already reading or waiting. `hold` is the file
    the GPU is on or about to take. Reading stops at a file that is not a
    PDF or picture, except `hold` itself when that file is already on the
    GPU. `past_ordinary` keeps going past those files, which is how uploaded
    PDFs are read while a ZIM is on the GPU. `limit` caps how many new reads
    are taken. None keeps every PDF and picture up to that boundary. `hold`
    is still read when a cap is full.
    """
    room = None if limit is None else max(0, int(limit) - len(reserved))
    found: list[str] = []
    seen_hold = hold is None
    for source, kind in order:
        visual = kind in VISUAL_KINDS
        if source == hold:
            seen_hold = True
        if not seen_hold and not visual and not past_ordinary:
            break
        if not visual:
            if source != hold and not past_ordinary:
                break
            continue
        if source in reserved:
            continue
        if room is not None and source != hold and len(found) >= room:
            break
        found.append(source)
    return found


def preview_order(
    rows,
    *,
    choice: str,
    index_now: str | None,
    defer: set[str],
    kind_of,
    limit: int | None = None,
    past_ordinary: bool = False,
) -> list[tuple[str, str]]:
    """The index order up to the next file that is not a PDF or picture.

    `limit` stops after that many PDFs and pictures. None follows the run
    until a file that is not a PDF or picture. `past_ordinary` follows the
    whole queue so PDFs after a ZIM are included. Files before that boundary
    stay in front of later pictures.
    """
    skip: set[str] = set()
    now = index_now
    out: list[tuple[str, str]] = []
    visuals = 0
    cap = None if limit is None else max(1, int(limit))
    while True:
        picked = pick_next(rows, choice=choice, current=None, index_now=now, skip=skip, defer=defer)
        if not picked.target:
            break
        now = None
        skip.add(picked.target)
        kind = str(kind_of(picked.target) or "")
        out.append((picked.target, kind))
        if kind in VISUAL_KINDS:
            visuals += 1
            if cap is not None and visuals >= cap:
                break
        elif len(out) > 1 and not past_ordinary:
            break
    return out


@dataclass
class OcrJob:
    """One PDF or picture for the OCR queue.

    `source` is unique in the queue. An upload uses its library path. A PDF
    inside a ZIM uses the ZIM path plus the entry path. `owner` is the library
    file the stored points belong to. `host` is the file the reader opens.
    """

    source: str
    host: Path
    kind: str
    title: str = ""
    owner: str = ""
    article_path: str = ""
    # The reader reports progress through the same fields as a library file.
    articles: int = 0
    total_articles: int | None = None
    chunks: int = 0

    def __post_init__(self) -> None:
        self.host = Path(self.host)
        if not self.owner:
            self.owner = self.source
        if not self.article_path:
            self.article_path = self.source


def gpu_candidates(files, *, overlap: bool, index_now: str | None = None, finished: set[str] | None = None):
    """Files the GPU can embed on this pass.

    With the OCR queue on, a PDF or picture waits until its text is ready.
    A ZIM whose pages are stored and whose PDFs are still in that queue waits
    too. A markdown file or other plain upload stays in the list. Index now
    still names its file.
    """
    done = finished or set()
    chosen = []
    for item in files:
        source = item.source
        if source in done:
            continue
        if index_now and source == index_now:
            chosen.append(item)
            continue
        if not overlap:
            chosen.append(item)
            continue
        kind = getattr(item, "kind", "") or ""
        if kind in VISUAL_KINDS:
            continue
        if kind == "zim" and getattr(item, "zim_pass", "") == "queued":
            continue
        chosen.append(item)
    return chosen


class _StopPair:
    """One flag read_visual can poll. Either side stopping ends the read."""

    def __init__(self, first: threading.Event, second: threading.Event) -> None:
        self._first = first
        self._second = second

    def is_set(self) -> bool:
        return self._first.is_set() or self._second.is_set()


class VisualLane:
    """One OCR read at a time, for files the GPU has not reached yet.

    `reader(work)` returns the reader dict, or None when the file could not
    be read. It raises RuntimeError('stopped') when the stop flag is set.
    Finished text stays in memory until `budget` bytes. The rest is written
    under `spill_dir` and removed when that file is indexed or the lane stops.
    """

    def __init__(self, reader, stop: threading.Event, spill_dir: Path | None = None, budget: int | None = None) -> None:
        self._reader = reader
        self._run_stop = stop
        self._spill_dir = Path(spill_dir) if spill_dir is not None else None
        self._budget = ocr_ram_budget() if budget is None else max(0, int(budget))
        self._abort = threading.Event()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._order: deque[str] = deque()
        self._meta: dict[str, object] = {}
        self._finished: list[str] = []
        self._inflight: str | None = None
        self._ready: dict[str, dict | None] = {}
        self._spilled: dict[str, Path] = {}
        self._sizes: dict[str, int] = {}
        self._ram = 0
        self._errors: dict[str, BaseException] = {}
        self._closed = False
        self._told_spill = False
        self._thread = threading.Thread(target=self._loop, name="nomad-turbo-ocr-ahead", daemon=True)
        self._thread.start()

    def stop_flag(self) -> _StopPair:
        """The flag the reader polls. Closing the lane stops the open read."""
        return _StopPair(self._run_stop, self._abort)

    def _known(self) -> set[str]:
        found = set(self._order) | set(self._ready) | set(self._spilled) | set(self._errors)
        if self._inflight:
            found.add(self._inflight)
        return found

    def reserved(self) -> set[str]:
        with self._lock:
            return self._known()

    def pending(self, source: str) -> bool:
        with self._lock:
            return source in self._known()

    def has_result(self, source: str) -> bool:
        with self._lock:
            return source in self._ready or source in self._spilled or source in self._errors

    def has_work(self) -> bool:
        """True while a read is queued, running, or waiting to be embedded."""
        with self._lock:
            return bool(self._known())

    def ready_sources(self) -> list[str]:
        """Finished reads, in the order they finished."""
        with self._lock:
            live = set(self._ready) | set(self._spilled) | set(self._errors)
            return [source for source in self._finished if source in live]

    def job(self, source: str):
        """The queued item. Call this before `take`, which drops it."""
        with self._lock:
            return self._meta.get(source)

    def owner_busy(self, owner: str) -> bool:
        """True while this library file still has a read queued or finished."""
        with self._lock:
            sources = list(self._order)
            if self._inflight:
                sources.append(self._inflight)
            sources.extend(self._ready)
            sources.extend(self._spilled)
            sources.extend(self._errors)
            for source in sources:
                job = self._meta.get(source)
                if getattr(job, "owner", None) == owner:
                    return True
            return False

    def prefer(self, source: str) -> None:
        """Move a queued file to the front. The file already being read stays."""
        with self._cv:
            if source not in self._order:
                return
            self._order.remove(source)
            self._order.appendleft(source)
            self._cv.notify()

    def offer(self, works, *, front: bool = False) -> None:
        """Queue these reads. A source already known is left where it is.

        `front` places them ahead of reads that have not started. The first
        item in `works` stays ahead of the others in that group.
        """
        with self._cv:
            if self._closed:
                return
            known = self._known()
            items = list(works)
            if front:
                items = list(reversed(items))
            for work in items:
                source = work.source
                if source in known:
                    continue
                if front:
                    self._order.appendleft(source)
                else:
                    self._order.append(source)
                self._meta[source] = work
                known.add(source)
            self._cv.notify()

    def take(self, source: str) -> tuple[dict | None, BaseException | None]:
        """The finished read. The source must already have a result."""
        with self._cv:
            self._meta.pop(source, None)
            self._finished = [item for item in self._finished if item != source]
            if source in self._errors:
                return None, self._errors.pop(source)
            if source in self._ready:
                self._ram -= self._sizes.pop(source, 0)
                if self._ram < 0:
                    self._ram = 0
                return self._ready.pop(source), None
            path = self._spilled.pop(source, None)
        if path is None:
            return None, RuntimeError("ocr did not start")
        try:
            visual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, exc
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        if not isinstance(visual, dict):
            return None, RuntimeError("ocr spill was empty")
        return visual, None

    def close(self) -> None:
        """Stop the open read and drop anything still queued, including temp files."""
        with self._cv:
            self._closed = True
            self._order.clear()
            self._ready.clear()
            self._ram = 0
            self._sizes.clear()
            self._cv.notify_all()
        self._abort.set()
        self._thread.join(timeout=30)
        self._discard_spills()

    def _discard_spills(self) -> None:
        with self._lock:
            paths = list(self._spilled.values())
            self._spilled.clear()
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        folder = self._spill_dir
        if folder is None or not folder.is_dir():
            return
        for path in folder.glob("*.json"):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _remember(self, source: str) -> None:
        if source not in self._finished:
            self._finished.append(source)

    def _loop(self) -> None:
        while not self._abort.is_set() and not self._run_stop.is_set():
            with self._cv:
                if self._closed or self._abort.is_set() or self._run_stop.is_set():
                    return
                if not self._order:
                    self._cv.wait(timeout=0.3)
                    continue
                source = self._order.popleft()
                work = self._meta.get(source)
                self._inflight = source
            if work is None:
                continue
            try:
                visual = self._reader(work)
            except Exception as exc:
                with self._cv:
                    self._inflight = None
                    self._errors[source] = exc
                    self._remember(source)
                    self._cv.notify_all()
                if self._abort.is_set() or self._run_stop.is_set() or str(exc) == "stopped":
                    break
                continue
            self._keep(source, visual)

    def _keep(self, source: str, visual) -> None:
        size = text_bytes(visual)
        path = None
        with self._lock:
            spill = self._spill_dir is not None and size > max(0, self._budget - self._ram)
        if spill:
            path = self._write_spill(source, visual)
        with self._cv:
            self._inflight = None
            if self._closed:
                self._cv.notify_all()
                if path is not None:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
                return
            if path is not None:
                self._spilled[source] = path
                if not self._told_spill:
                    self._told_spill = True
                    log.info(
                        "OCR text reached a quarter of system memory. Further finished reads wait on disk."
                    )
            else:
                self._ready[source] = visual
                self._sizes[source] = size
                self._ram += size
            self._remember(source)
            self._cv.notify_all()

    def _write_spill(self, source: str, visual) -> Path | None:
        folder = self._spill_dir
        if folder is None:
            return None
        try:
            folder.mkdir(parents=True, exist_ok=True)
            name = hashlib.sha256(source.encode()).hexdigest()[:16] + ".json"
            path = folder / name
            path.write_text(json.dumps(visual, ensure_ascii=False), encoding="utf-8")
        except (OSError, TypeError, ValueError):
            log.exception("could not set OCR text aside for %s", source)
            return None
        return path
