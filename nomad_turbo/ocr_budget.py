"""How many CPU OCR workers may run while Qdrant is working.

The OCR process reads one number from the state directory and parks workers
down to that number at the next page. Qdrant writes, an optimizer pass, and
a 429 or 503 win. A quiet collection gives the cores back.
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


_lock = threading.Lock()
_state = _Pressure()
_path: Path | None = None
_cpus = 1


def reset_pressure() -> None:
    """Test helper. Drops the armed file and the counters."""
    global _state, _path
    with _lock:
        _state = _Pressure()
        _path = None


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
        return cpu_ocr_cap(
            _cpus if cpus is None else cpus,
            share=_state.share,
            embed=_state.embed > 0,
            upserts=_state.upserts > 0,
            ready=_state.ready,
            status=_state.status,
            stalled=stalled,
            index_cpu=_state.index_cpu,
        )


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
