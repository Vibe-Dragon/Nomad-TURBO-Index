"""How many CPU OCR workers may run while Qdrant is working.

The OCR process reads one number from the state directory and parks workers
down to that number at the next page. The Settings control can lower that
number, and the next page uses it. Qdrant writes, an optimizer pass, and
a 429 or 503 still win. A quiet collection gives the cores back, up to the
number the control allows. Two threads stay free.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class _Pressure:
    embed: int = 0
    upserts: int = 0
    ready: bool | None = None
    status: str = ""
    stalled_until: float = 0.0
    share: bool = False
    index_cpu: bool = False
    long_zim: bool = False
    # Consecutive ZIM embed calls. Three long calls turn the cap on.
    # Three short calls turn it off. A file boundary zeroes both.
    long_run: int = 0
    short_run: int = 0


_lock = threading.Lock()
_state = _Pressure()
_path: Path | None = None
_cpus = 1
# None means every worker the computer allows. A lower number is the user's ceiling.
_worker_max: int | None = None


def reset_pressure() -> None:
    """Test helper. Drops the armed file, the counters, and the user's ceiling."""
    global _state, _path, _worker_max
    with _lock:
        _state = _Pressure()
        _path = None
        _worker_max = None


def set_runtime(*, share: bool, index_cpu: bool, cpus: int | None = None) -> None:
    global _cpus
    with _lock:
        _state.share = bool(share)
        _state.index_cpu = bool(index_cpu)
        if cpus is not None:
            _cpus = max(1, int(cpus))
    publish()


def note_embed(active: bool) -> None:
    with _lock:
        _state.embed = max(0, _state.embed + (1 if active else -1))
    publish()


def note_upsert(active: bool) -> None:
    with _lock:
        _state.upserts = max(0, _state.upserts + (1 if active else -1))
    publish()


def upserts_busy() -> bool:
    """True while a point batch is still being written."""
    with _lock:
        return _state.upserts > 0


def note_long_zim(active: bool) -> None:
    """Force the long-ZIM OCR cap, or clear it at a file boundary.

    False publishes the normal allowance and forgets the call run, so the
    next file starts uncapped. The document being read is not removed.
    Embed calls use note_zim_embed_call, which needs three long calls.
    """
    with _lock:
        _state.long_zim = bool(active)
        if not active:
            _state.long_run = 0
            _state.short_run = 0
    publish()


def long_zim_cap_state(
    active: bool,
    kind: str,
    mean_tokens: int,
    long_run: int,
    short_run: int,
) -> tuple[bool, int, int]:
    """Decide the cap from one embed call.

    A ZIM call at or above the long-call size counts toward the cap. The
    cap turns on only after three of those in a row, and turns off only
    after three shorter ZIM calls in a row. One long call does not pin a
    short dictionary. A PDF or picture never turns the cap on.
    """
    if (kind or "") != "zim":
        return False, 0, 0
    from nomad_turbo.slots import LONG_CALL_SAMPLES, LONG_CALL_TOKENS

    long_run = max(0, int(long_run))
    short_run = max(0, int(short_run))
    if int(mean_tokens) >= LONG_CALL_TOKENS:
        long_run = min(LONG_CALL_SAMPLES, long_run + 1)
        short_run = 0
        if long_run >= LONG_CALL_SAMPLES:
            active = True
    else:
        short_run = min(LONG_CALL_SAMPLES, short_run + 1)
        long_run = 0
        if short_run >= LONG_CALL_SAMPLES:
            active = False
    return bool(active), long_run, short_run


def note_zim_embed_call(kind: str, mean_tokens: int) -> None:
    """Publish the OCR cap from one embed call of the file being indexed."""
    with _lock:
        active, long_run, short_run = long_zim_cap_state(
            _state.long_zim,
            kind,
            mean_tokens,
            _state.long_run,
            _state.short_run,
        )
        _state.long_zim = active
        _state.long_run = long_run
        _state.short_run = short_run
    publish()


def embedding_file_is_long_zim(kind: str, mean_tokens: int) -> bool:
    """A ZIM whose embed calls are long. A PDF or picture keeps its own OCR job."""
    from nomad_turbo.slots import LONG_CALL_TOKENS

    if (kind or "") != "zim":
        return False
    return int(mean_tokens) >= LONG_CALL_TOKENS


def long_zim_worker_cap(allowance: int) -> int:
    """Workers while a long ZIM is embedding. Strictly below the full allowance."""
    allowance = max(0, int(allowance))
    if allowance <= 1:
        return 0
    return min(allowance - 1, max(1, allowance // 5))


def apply_long_zim_cap(cap: int, allowance: int, long_zim: bool) -> int:
    """Leave a quiet cap alone. A long ZIM publishes a lower one."""
    if not long_zim:
        return int(cap)
    return min(int(cap), long_zim_worker_cap(allowance))


def note_qdrant(ready: bool, status: str) -> None:
    with _lock:
        _state.ready = bool(ready)
        _state.status = str(status or "").strip().lower()
    publish()


def note_qdrant_fault(message: str) -> None:
    """A refused or unreachable write. OCR parks for a short stretch."""
    text = str(message or "")
    if not any(token in text for token in ("429", "503", "unreachable")):
        return
    with _lock:
        _state.stalled_until = time.monotonic() + 30
    publish()


def ocr_worker_limit(cpus: int) -> int:
    """The most workers the control may offer. Two threads stay free.

    A 12-thread computer offers 10. A computer with one or two threads keeps
    one worker, matching the reader's quiet pool.
    """
    cpus = max(1, int(cpus or 1))
    reserve = min(2, max(0, cpus - 1))
    return max(1, cpus - reserve)


def clamp_ocr_worker_max(count: int, cpus: int) -> int:
    """A count inside 1 and the thread limit. Two threads stay free."""
    limit = ocr_worker_limit(cpus)
    try:
        value = int(count)
    except (TypeError, ValueError):
        return limit
    if value < 1:
        return 1
    if value > limit:
        return limit
    return value


def shown_worker_max(saved: int | None, cpus: int) -> int:
    """The number the slider shows. A missing choice is the thread limit."""
    if saved is None:
        return ocr_worker_limit(cpus)
    return clamp_ocr_worker_max(saved, cpus)


def stored_worker_max(count: int, cpus: int) -> int | None:
    """None when the choice is the whole allowance. A lower count is kept."""
    chosen = clamp_ocr_worker_max(count, cpus)
    if chosen >= ocr_worker_limit(cpus):
        return None
    return chosen


def set_worker_max(count: int | None, *, cpus: int | None = None) -> int:
    """Remember the user's ceiling and publish it when a run is armed.

    None means every worker the computer allows, still leaving two threads free.
    The reader uses the new number on its next page.
    """
    global _worker_max, _cpus
    with _lock:
        if cpus is not None:
            _cpus = max(1, int(cpus))
        if count is None:
            _worker_max = None
            chosen = ocr_worker_limit(_cpus)
        else:
            chosen = clamp_ocr_worker_max(count, _cpus)
            _worker_max = stored_worker_max(chosen, _cpus)
    publish()
    return chosen


def cpu_ocr_cap(
    cpus: int,
    *,
    share: bool,
    embed: bool,
    upserts: bool,
    ready: bool | None,
    status: str,
    stalled: bool,
    index_cpu: bool,
) -> int:
    """Workers the CPU reader may keep. 0 parks every worker after its page.

    A quiet 12-thread machine keeps 10, matching the reader. While Qdrant is
    taking points that drops to 2. An optimizer pass keeps 1. A refusal, a
    red collection, or the same device as the embedder parks OCR completely.
    """
    cpus = max(1, int(cpus or 1))
    reserve = min(2, max(0, cpus - 1))
    ceiling = max(0, cpus - reserve)
    if share and (embed or upserts):
        return 0
    color = (status or "").lower()
    if stalled or ready is False or color == "red":
        return 0
    if index_cpu and embed:
        return 1 if cpus >= 2 else 0
    if color in ("yellow", "grey"):
        return 1 if cpus >= 4 else 0
    if upserts:
        if cpus < 4:
            return 0
        return max(1, cpus // 6)
    return ceiling


def current_cap(cpus: int | None = None) -> int:
    with _lock:
        stalled = time.monotonic() < _state.stalled_until
        host = _cpus if cpus is None else max(1, int(cpus))
        cap = cpu_ocr_cap(
            host,
            share=_state.share,
            embed=_state.embed > 0,
            upserts=_state.upserts > 0,
            ready=_state.ready,
            status=_state.status,
            stalled=stalled,
            index_cpu=_state.index_cpu,
        )
        allowance = ocr_worker_limit(host)
        if _worker_max is None:
            chosen = allowance
        else:
            chosen = min(allowance, max(1, int(_worker_max)))
        return apply_long_zim_cap(min(cap, chosen), allowance, _state.long_zim)


def publish(path: Path | None = None) -> int:
    """Write the cap. No path and no armed file means the reader is uncapped."""
    target = path if path is not None else _path
    cap = current_cap()
    if target is None:
        return cap
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{cap}\n", encoding="utf-8")
    except OSError:
        return cap
    return cap


def arm(path: Path, cpus: int) -> None:
    global _path, _cpus
    with _lock:
        _path = path
        _cpus = max(1, int(cpus))
    publish(path)


def disarm() -> None:
    """Remove the cap so a later read is not stuck on a number from this run."""
    global _path
    with _lock:
        path = _path
        _path = None
    if path is not None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def serve_budget(stop: threading.Event) -> None:
    """Refresh the file so a 30 second refusal expires without another event."""
    while not stop.is_set():
        publish()
        stop.wait(1)
