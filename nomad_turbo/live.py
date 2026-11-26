"""A painted reading, and the process that runs embed and OCR.

The window timer only copies a reading that is already in memory. The
publisher builds that reading off the window thread. Embed, OCR, and a
Python stretch run in a child process so a stall there does not hold the
window's interpreter lock.
"""

from __future__ import annotations

import copy
import importlib
import logging
import os
import pickle
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, replace

from nomad_turbo.qdrant_status import changes_allowed, delete_allowed

log = logging.getLogger("nomad_turbo")

_WORKLOAD_ENV = "NOMAD_TURBO_WORKLOAD"


@dataclass(frozen=True)
class ButtonFacts:
    """What the buttons need. Copied before the window paints."""

    thread_alive: bool = False
    delete_alive: bool = False
    repair_alive: bool = False
    repair_starting: bool = False
    optimize_busy: bool = False
    rows_ready: bool = False
    qdrant_ready: bool = False
    collection_missing: bool = False
    scanning: bool = False
    measuring: bool = False
    measure_then_index: bool = False
    repair_kind: str = ""
    # The Qdrant page is writing quantization or the graph. A storage rebuild uses delete_alive.
    qdrant_busy: bool = False


@dataclass(frozen=True)
class Reading:
    """One frame. The window paints this and does not ask the workers again."""

    seq: int = 0
    view: dict = field(default_factory=dict)
    snap: object = None
    priorities: frozenset = field(default_factory=frozenset)
    activity: str = ""
    ocr_activity: str = ""
    lane: object = None
    embedding: object = None
    embed_order: object = None
    ocr_order: object = None
    order: str = "library"
    zim_dates: dict = field(default_factory=dict)
    open_files: tuple = ()
    queue_open: bool = False
    library_items: tuple = ()
    structure_epoch: int = 0
    doc_sig: tuple = (False,)
    doc_groups: tuple = ()
    hide_finished: bool = False
    row_snaps: dict = field(default_factory=dict)
    repair: object = None
    qdrant_text: str = ""
    qdrant_ready: bool = False
    collection_status: str = ""
    optimizer_error: str = ""
    collection_missing: bool = False
    qdrant_unreadable: bool = False
    gpu: tuple | None = None
    host_points: tuple = ()
    host_on: bool = False
    log_lines: tuple = ()
    log_reset: bool = False
    scanning: bool = False
    elapsed: int | None = None
    embed_calls: tuple = ()
    embed_call_count: int = 0
    ocr_workers: tuple | None = None
    running_source: str = ""
    tune_auto: bool = False
    optimize_live: tuple | None = None
    facts: ButtonFacts = field(default_factory=ButtonFacts)
    painted_at: float = 0.0


class LiveBoard:
    """The latest reading. The lock only swaps the pointer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reading: Reading | None = None
        self._seq = 0

    def publish(self, reading: Reading) -> Reading:
        with self._lock:
            self._seq += 1
            stored = replace(reading, seq=self._seq)
            self._reading = stored
            return stored

    def current(self) -> Reading | None:
        with self._lock:
            return self._reading


def control_senses(phase: str, facts: ButtonFacts) -> dict:
    """Which buttons are enabled. The window and the tests both call this."""
    running = phase in ("waiting", "running") and facts.thread_alive
    deleting = facts.delete_alive
    file_repairing = facts.repair_alive or facts.repair_starting
    optimizing = facts.optimize_busy
    repairing = file_repairing or optimizing
    # A quantization or graph write keeps the points, and it still must not overlap Start.
    settling = deleting or facts.qdrant_busy
    library_idle = facts.rows_ready and not running and not repairing and not settling
    changes = changes_allowed(facts.qdrant_ready)
    can_delete = delete_allowed(facts.qdrant_ready, facts.collection_missing)
    measuring = facts.measuring
    measure = (
        facts.rows_ready
        and changes
        and not measuring
        and not repairing
        and not settling
        and (library_idle or running)
    )
    then_index = measuring and not repairing and not settling
    idle_repair = not running and not repairing and not settling and not facts.scanning and changes
    return {
        "running": running,
        "deleting": deleting,
        "file_repairing": file_repairing,
        "repairing": repairing,
        "library_idle": library_idle,
        "changes": changes,
        "can_delete": can_delete,
        "measuring": measuring,
        "start": library_idle and changes,
        "delete": library_idle and can_delete,
        "rescan": library_idle,
        "measure": measure,
        "then_index": then_index,
        "stop": running or deleting,
        "repair_scan": idle_repair,
        "repair_start": idle_repair,
        "repair_stop": file_repairing,
        "repair_quick": not repairing,
        "repair_complete": not repairing,
        "optimize_segments": idle_repair,
        "then_index_armed": measuring and facts.measure_then_index,
    }


def button_facts(window) -> ButtonFacts:
    """Copy the flags the buttons read. Each session call takes its lock and lets go."""
    session = window.session
    measuring = bool(session.measuring()) if hasattr(session, "measuring") else False
    then_index = False
    if measuring and hasattr(session, "measure_then_index"):
        then_index = bool(session.measure_then_index())
    repair_thread = getattr(window, "repair_thread", None)
    delete_thread = getattr(window, "delete_thread", None)
    qdrant_thread = getattr(window, "qdrant_thread", None)
    thread = getattr(window, "thread", None)
    board = getattr(window, "repair_board", None)
    kind = str(getattr(board, "kind", "") or "") if board is not None else ""
    optimize = bool(window._optimize_busy()) if hasattr(window, "_optimize_busy") else False
    return ButtonFacts(
        thread_alive=thread is not None and thread.is_alive(),
        delete_alive=delete_thread is not None and delete_thread.is_alive(),
        repair_alive=repair_thread is not None and repair_thread.is_alive(),
        repair_starting=bool(getattr(window, "repair_starting", False)),
        optimize_busy=optimize,
        rows_ready=bool(getattr(window, "rows_ready", False)),
        qdrant_ready=bool(getattr(window, "qdrant_ready", False)),
        collection_missing=bool(getattr(window, "collection_missing", False)),
        scanning=bool(getattr(window, "scanning", False)),
        measuring=measuring,
        measure_then_index=then_index,
        repair_kind=kind,
        qdrant_busy=qdrant_thread is not None and qdrant_thread.is_alive(),
    )


def tray_indicator_label(view: dict, *, indexing: bool, repairing: bool, hidden: bool, repair_kind: str) -> tuple[str, str]:
    """The label and guide passed to the tray icon."""
    if hidden and (indexing or repairing):
        if indexing:
            label = str(view.get("chunks", view.get("percent", "")))
        elif repair_kind == "scan":
            label = "scan"
        else:
            label = "repair"
        return label, "100%"
    if indexing:
        return str(view.get("chunks", view.get("percent", ""))), "chunks"
    return "", "chunks"


def tray_status_text(repair_overall: str, snap) -> str:
    """The tray menu line. A repair overall wins. Otherwise the library bar."""
    if repair_overall:
        return str(repair_overall)[:120]
    if snap is None:
        return ""
    return str(getattr(snap, "overall_text", "") or "")[:120]


def tray_stop_label(*, repairing: bool, repair_kind: str) -> str:
    if repairing:
        return "Stop scan" if repair_kind == "scan" else "Stop repair"
    return "Stop indexing"


def hold_interpreter(seconds: float, started=None) -> int:
    """Busy-loop in this process. The clock is checked in stretches, so the loop holds the lock.

    ``started`` may be an event with ``set``, or a file path. A path is written
    when the loop begins. A spawn child cannot be handed a lock after it starts.
    """
    if started is not None:
        mark = getattr(started, "set", None)
        if callable(mark):
            mark()
        else:
            with open(started, "w", encoding="utf-8") as handle:
                handle.write("started")
    deadline = time.monotonic() + float(seconds)
    token = 0
    while time.monotonic() < deadline:
        for _ in range(100000):
            token = (token + 1) & 0xFFFFFFFF
    return token


def _call(session, name: str, default=None):
    getter = getattr(session, name, None)
    if not callable(getter):
        return default
    return getter()


def _running_source(snap, lane, embedding) -> str:
    from nomad_turbo.queue_view import pinned_files, running_column

    files = [] if snap is None else list(getattr(snap, "files", []) or [])
    library_running = [item for item in files if getattr(item, "state", "") == "running"]
    running = pinned_files(library_running, lane, embedding)
    lead = next((item for item in running if running_column(item) == "embed"), None)
    current = lead or (running[0] if running else None)
    return current.source if current is not None else ""


def _copy_library(window, session, book, snap) -> tuple:
    """Book copies. The book lock is not held across a disk walk or nvidia-smi."""
    items = tuple(getattr(window, "_publish_items", ()) or ())
    groups = tuple(getattr(window, "_publish_groups", ()) or ())
    doc_sig = getattr(window, "_publish_doc_sig", (False,))
    row_snaps = dict(getattr(window, "_publish_rows", {}) or {})
    epoch = int(getattr(snap, "structure_epoch", 0) or 0) if snap is not None else 0
    if book is None:
        return items, groups, doc_sig if isinstance(doc_sig, tuple) else (False,), row_snaps, epoch
    zim = False
    getter = getattr(session, "zim_documents", None)
    if callable(getter):
        zim = bool(getter())
    elif getattr(getattr(window, "install", None), "zim_documents", False):
        zim = True
    if zim:
        from nomad_turbo.discover import state_dir
        from nomad_turbo.zimdocs import document_stamp, load_document_groups

        try:
            doc_sig = (True, document_stamp(state_dir()))
        except Exception:
            log.debug("document stamp failed", exc_info=True)
            doc_sig = getattr(window, "_publish_doc_sig", (True, None))
        if doc_sig != getattr(window, "_publish_doc_sig", None):
            try:
                groups = tuple(load_document_groups(state_dir()))
            except Exception:
                log.debug("document groups failed", exc_info=True)
                groups = tuple(getattr(window, "_publish_groups", ()) or ())
    else:
        doc_sig = (False,)
        groups = ()
    full = epoch != getattr(window, "_publish_epoch", None) or not items
    running_now = {
        item.source
        for item in (getattr(snap, "files", []) or [])
        if getattr(item, "state", "") == "running"
    }
    if full and hasattr(book, "library_items"):
        epoch, found = book.library_items()
        items = tuple(found)
        sources = [item.source for item in items]
        row_snaps = book.files_for(sources) if hasattr(book, "files_for") and sources else {}
    elif hasattr(book, "files_for"):
        previous = set(getattr(window, "_publish_running", ()) or ())
        need = previous | running_now
        if need:
            row_snaps = dict(row_snaps)
            row_snaps.update(book.files_for(list(need)))
    window._publish_epoch = epoch
    window._publish_items = items
    window._publish_groups = groups
    window._publish_doc_sig = doc_sig
    window._publish_rows = row_snaps
    window._publish_running = running_now
    return items, groups, doc_sig, row_snaps, int(epoch or 0)


def _gpu_sample(window):
    """nvidia-smi, with no book lock and no session lock held by this function."""
    now = time.monotonic()
    last = float(getattr(window, "_gpu_sample_at", 0) or 0)
    if now - last < 2.0 and hasattr(window, "_gpu_sample"):
        return getattr(window, "_gpu_sample", None)
    try:
        from nomad_turbo.runtime import gpu_rows
        from nomad_turbo.slots import pick_gpu

        rows = gpu_rows()
        window._gpu_sample = pick_gpu(rows) if rows else getattr(window, "_gpu_sample", getattr(window, "live_gpu", None))
    except Exception:
        log.debug("gpu sample failed", exc_info=True)
        window._gpu_sample = getattr(window, "_gpu_sample", None)
    window._gpu_sample_at = now
    return getattr(window, "_gpu_sample", None)


def collect_reading(window) -> Reading:
    """One reading. Locks are taken only to copy, and never across embed, OCR, or nvidia-smi."""
    gpu = _gpu_sample(window)
    panel = getattr(window, "host_panel", None)
    published = ()
    if panel is not None:
        published = tuple(getattr(panel.sampler, "published", ()) or ())
    host_on = bool(getattr(getattr(window, "install", None), "host_open", False))
    session = window.session
    view = _call(session, "view", {}) or {}
    if not isinstance(view, dict):
        view = {}
    priorities = frozenset(_call(session, "priorities", ()) or ())
    activity = str(view.get("note") or "")
    ocr_activity = str(_call(session, "ocr_activity", "") or "")
    lane = _call(session, "lane_view", None)
    if isinstance(lane, dict):
        try:
            lane = copy.deepcopy(lane)
        except Exception:
            lane = dict(lane)
    embedding = _call(session, "embedding_file", None)
    embed_order = _call(session, "embed_queue", None)
    ocr_order = _call(session, "ocr_queue", None)
    order = _call(session, "get_order", "library") or "library"
    dates = _call(session, "zim_dates", {}) or {}
    if not isinstance(dates, dict):
        dates = {}
    calls = tuple(_call(session, "embed_calls", ()) or ())
    call_count = int(_call(session, "embed_call_count", 0) or 0)
    workers = _call(session, "ocr_workers", None)
    tune_auto = bool(_call(session, "tune_auto", False))
    book = _call(session, "get_book", None)
    snap = None
    if book is not None:
        snap = book.pulse() if hasattr(book, "pulse") else book.snapshot()
    items, groups, doc_sig, row_snaps, epoch = _copy_library(window, session, book, snap)
    queue_open = getattr(window, "queue_window", None) is not None
    open_files = ()
    if queue_open and book is not None and hasattr(book, "open_snaps"):
        open_files = tuple(book.open_snaps())
    repair = None
    board = getattr(window, "repair_board", None)
    if board is not None and hasattr(board, "snapshot"):
        try:
            repair = board.snapshot()
        except Exception:
            log.debug("repair snapshot failed", exc_info=True)
    reset = bool(getattr(window, "filter_dirty", False))
    cursor = int(getattr(window, "last_seq", 0) or 0)
    memory = window.memory
    if reset:
        log_lines = tuple(memory.snapshot())
    else:
        log_lines = tuple(memory.since(cursor))
    started = getattr(window, "started_at", None)
    elapsed = None if not started else max(0, int(time.monotonic() - float(started)))
    live = getattr(window, "_optimize_live", None)
    optimize_live = live if isinstance(live, tuple) else None
    hide = bool(getattr(getattr(window, "install", None), "hide_finished", False))
    return Reading(
        view=dict(view),
        snap=snap,
        priorities=priorities,
        activity=activity,
        ocr_activity=ocr_activity,
        lane=lane,
        embedding=embedding,
        embed_order=None if embed_order is None else list(embed_order),
        ocr_order=None if ocr_order is None else list(ocr_order),
        order=str(order),
        zim_dates=dict(dates),
        open_files=open_files,
        queue_open=queue_open,
        library_items=tuple(items),
        structure_epoch=epoch,
        doc_sig=tuple(doc_sig),
        doc_groups=tuple(groups),
        hide_finished=hide,
        row_snaps=dict(row_snaps),
        repair=repair,
        qdrant_text=str(getattr(window, "qdrant_text", "") or ""),
        qdrant_ready=bool(getattr(window, "qdrant_ready", False)),
        collection_status=str(getattr(window, "qdrant_collection_status", "") or ""),
        optimizer_error=str(getattr(window, "qdrant_optimizer_error", "") or ""),
        collection_missing=bool(getattr(window, "collection_missing", False)),
        qdrant_unreadable=bool(getattr(window, "qdrant_unreadable", False)),
        gpu=gpu,
        host_points=published,
        host_on=host_on,
        log_lines=log_lines,
        log_reset=reset,
        scanning=bool(getattr(window, "scanning", False)),
        elapsed=elapsed,
        embed_calls=calls,
        embed_call_count=call_count,
        ocr_workers=None if not workers else tuple(workers),
        running_source=_running_source(snap, lane, embedding),
        tune_auto=tune_auto,
        optimize_live=optimize_live,
        facts=button_facts(window),
        painted_at=time.monotonic(),
    )


class Publisher:
    """Publishes a reading a few times a second. The window timer does not call this."""

    def __init__(self, window) -> None:
        self.window = window
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="nomad-turbo-paint", daemon=True)
        self._loops = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.window.live.publish(collect_reading(self.window))
            except Exception:
                log.exception("paint publish failed")
            self._loops += 1
            if self._loops % 4 == 0:
                try:
                    self.window._flush_stats()
                except Exception:
                    log.exception("could not save file stats")
            self._stop.wait(0.25)


def _resolve(qualname: str):
    module_name, _, func_name = qualname.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _load_exc_class(qualname: str):
    module_name, _, name = qualname.partition(":")
    obj = importlib.import_module(module_name)
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def _error_spec(exc: BaseException) -> tuple:
    """Class, constructor args, and attributes.

    A "Name: message" string drops the class and, for a rejected chunk, the index.
    """
    args = []
    for item in exc.args:
        try:
            pickle.dumps(item)
        except Exception:
            args.append(str(item))
        else:
            args.append(item)
    state = {}
    try:
        raw_state = vars(exc)
    except TypeError:
        raw_state = {}
    for key, value in raw_state.items():
        if str(key).startswith("_"):
            continue
        try:
            pickle.dumps(value)
        except Exception:
            continue
        state[key] = value
    return (f"{type(exc).__module__}:{type(exc).__qualname__}", tuple(args), state)


def _build_exc(cls, args: tuple) -> BaseException:
    try:
        return cls(*args)
    except Exception:
        # ChunkRejected stores the index on the instance. BaseException.args is only the message.
        exc = cls.__new__(cls)
        BaseException.__init__(exc, *args)
        return exc


def _restore_error(payload) -> BaseException:
    """The exception the child raised, with its class and attributes."""
    if isinstance(payload, BaseException):
        return payload
    qualname, args, state = payload
    exc = _build_exc(_load_exc_class(qualname), tuple(args))
    if isinstance(state, dict):
        for key, value in state.items():
            try:
                setattr(exc, key, value)
            except Exception:
                continue
    return exc


def _send_error(results, token, exc: BaseException) -> None:
    spec = _error_spec(exc)
    try:
        pickle.dumps(spec)
    except Exception:
        spec = (f"{type(exc).__module__}:{type(exc).__qualname__}", (str(exc),), {})
    results.put((token, "err", spec))


class _StopBoard:
    """OCR stop flags in the child. The parent sends a token, not an Event.

    A spawn queue cannot carry an Event created after the child has started.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, threading.Event] = {}
        self._pending: set[str] = set()

    def watch(self, stops) -> None:
        while True:
            try:
                token = stops.get(timeout=0.2)
            except queue.Empty:
                continue
            if token is None:
                return
            self.fire(token)

    def fire(self, token: str) -> None:
        with self._lock:
            event = self._events.get(token)
            if event is None:
                self._pending.add(token)
            else:
                event.set()

    def register(self, token: str) -> threading.Event:
        event = threading.Event()
        with self._lock:
            self._events[token] = event
            if token in self._pending:
                self._pending.discard(token)
                event.set()
        return event

    def forget(self, token: str) -> None:
        with self._lock:
            self._events.pop(token, None)
            self._pending.discard(token)


def _run_task(token, qualname, args, kwargs, results) -> None:
    try:
        value = _resolve(qualname)(*args, **kwargs)
        results.put((token, "ok", value))
    except Exception as exc:
        _send_error(results, token, exc)


def _run_ocr(token, install, host_path, kind, quality, ocr_env, ocr_every_pdf, results, board: _StopBoard) -> None:
    local = board.register(token)

    def on_status(text: str) -> None:
        results.put((token, "status", text))

    try:
        from nomad_turbo.runtime import read_visual

        value = read_visual(
            install,
            host_path,
            kind,
            local,
            on_status,
            quality,
            ocr_env,
            ocr_every_pdf,
        )
        results.put((token, "ok", value))
    except Exception as exc:
        _send_error(results, token, exc)
    finally:
        board.forget(token)


def _workload_main(tasks, results, stops, extra_env, sys_path) -> None:
    os.environ[_WORKLOAD_ENV] = "1"
    for entry in sys_path:
        if entry and entry not in sys.path:
            sys.path.insert(0, entry)
    if extra_env:
        os.environ.update({str(key): str(value) for key, value in extra_env.items()})
    board = _StopBoard()
    threading.Thread(target=board.watch, args=(stops,), name="nomad-turbo-ocr-stop", daemon=True).start()
    while True:
        try:
            msg = tasks.get(timeout=0.2)
        except queue.Empty:
            continue
        if not msg or msg[0] == "stop":
            return
        kind = msg[0]
        if kind == "go":
            _token, qualname, args, kwargs = msg[1], msg[2], msg[3], msg[4]
            threading.Thread(
                target=_run_task,
                args=(_token, qualname, args, kwargs, results),
                name="nomad-turbo-task",
                daemon=True,
            ).start()
            continue
        if kind == "ocr":
            threading.Thread(
                target=_run_ocr,
                args=(msg[1], msg[2], msg[3], msg[4], msg[5], msg[6], msg[7], results, board),
                name="nomad-turbo-ocr-task",
                daemon=True,
            ).start()


class Workload:
    """One child process for embed, OCR, and other stalls. Spawn, so the window is not forked."""

    _current: "Workload | None" = None

    def __init__(self, extra_env: dict | None = None) -> None:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self.ctx = ctx
        self.tasks = ctx.Queue()
        self.results = ctx.Queue()
        self.stops = ctx.Queue()
        self.proc = ctx.Process(
            target=_workload_main,
            args=(self.tasks, self.results, self.stops, dict(extra_env or {}), list(sys.path)),
            name="nomad-turbo-workload",
            daemon=True,
        )
        self.proc.start()
        self._waiters: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self._demux = threading.Thread(target=self._pump, name="nomad-turbo-workload-demux", daemon=True)
        self._demux.start()

    @classmethod
    def current(cls) -> "Workload | None":
        return cls._current

    @classmethod
    def inside(cls) -> bool:
        return os.environ.get(_WORKLOAD_ENV) == "1"

    @classmethod
    def install(cls, extra_env: dict | None = None) -> "Workload | None":
        if cls.inside():
            return None
        current = cls._current
        if current is not None and current.proc.is_alive():
            return current
        host = cls(extra_env)
        cls._current = host
        return host

    def submit(self, qualname: str, args: tuple = ()) -> str:
        token = uuid.uuid4().hex
        self.tasks.put(("go", token, qualname, args, {}))
        return token

    def call(self, qualname: str, args: tuple = (), timeout: float | None = None):
        token = uuid.uuid4().hex
        box: queue.Queue = queue.Queue()
        with self._lock:
            self._waiters[token] = box
        self.tasks.put(("go", token, qualname, args, {}))
        return self._wait(box, timeout)

    def read_visual(self, install, host_path, kind, stop, on_status, quality, ocr_env, ocr_every_pdf):
        token = uuid.uuid4().hex
        box: queue.Queue = queue.Queue()
        with self._lock:
            self._waiters[token] = box
        self.tasks.put(("ocr", token, install, host_path, kind, quality, ocr_env, ocr_every_pdf))
        return self._wait(box, None, stop=stop, on_status=on_status, stop_token=token)

    def _wait(self, box: queue.Queue, timeout, stop=None, on_status=None, stop_token=None):
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        signaled = False
        while True:
            if stop is not None and stop.is_set() and stop_token is not None and not signaled:
                try:
                    self.stops.put(stop_token)
                except Exception:
                    log.debug("could not signal ocr stop", exc_info=True)
                else:
                    signaled = True
            if not self.proc.is_alive() and box.empty():
                raise RuntimeError("workload process stopped")
            try:
                kind, payload = box.get(timeout=0.2)
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("workload timed out")
                continue
            if kind == "status":
                if on_status is not None:
                    try:
                        on_status(payload)
                    except Exception:
                        log.debug("ocr status failed", exc_info=True)
                continue
            if kind == "err":
                raise _restore_error(payload)
            return payload

    def _pump(self) -> None:
        while True:
            try:
                token, kind, payload = self.results.get(timeout=0.2)
            except queue.Empty:
                if not self.proc.is_alive():
                    return
                continue
            with self._lock:
                box = self._waiters.get(token)
            if box is not None:
                box.put((kind, payload))

    def close(self) -> None:
        try:
            self.tasks.put(("stop",))
        except Exception:
            pass
        self.proc.join(0.5)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(1)
        if Workload._current is self:
            Workload._current = None
