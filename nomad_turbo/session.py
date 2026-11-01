"""Shared state between the control window and the indexing thread."""

from __future__ import annotations

import threading
import time

from nomad_turbo.progress import ProgressBook
from nomad_turbo.schedule import normalize_order
from nomad_turbo.slots import clamp_embed_chunks


class Session:
    def __init__(self, percent: int):
        self.stop = threading.Event()
        self.close_requested = False
        self.wipe_confirmed = False
        self._lock = threading.Lock()
        self.chunks = clamp_embed_chunks(percent)
        self.keep_control = False
        self.phase = "idle"
        self.error = ""
        self.note = ""
        self.optimize_hold = ""
        self._embed_calls: list[tuple] = []
        self._embed_call_count = 0
        self._book: ProgressBook | None = None
        self.active_slots = 0
        self.total_mib = 0
        self.used_before = 0
        self.per_slot = 1024
        self.measured: int | None = None
        self.use_gpu = False
        self.gpu_index: int | None = None
        self.forced: int | None = None
        self.max_slots = 64
        self.host_total_mib = 0
        self.host_available_mib = 0
        self._order = "library"
        self._priority: set[str] = set()
        self._priority_epoch = 0
        self._index_now: str | None = None
        self._measuring = False
        self._measure_from_idle = False
        self._measure_then_index = False
        self._measure_clock: float | None = None
        self._sampled: set[str] = set()
        self._zim_dates: dict[str, str] = {}
        self._file_stats = None
        self._tune_auto = False

    def set_chunks(self, chunks: int) -> None:
        with self._lock:
            self.chunks = clamp_embed_chunks(chunks)

    def get_chunks(self) -> int:
        with self._lock:
            return self.chunks

    def set_percent(self, percent: int) -> None:
        """Older name. The stored value is a chunk count, not a percent."""
        self.set_chunks(percent)

    def get_percent(self) -> int:
        """Older name. Returns the chunk count."""
        return self.get_chunks()

    def set_keep_control(self, value: bool) -> None:
        with self._lock:
            self.keep_control = bool(value)

    def get_keep_control(self) -> bool:
        with self._lock:
            return self.keep_control

    def set_phase(self, phase: str, error: str = "") -> None:
        with self._lock:
            self.phase = phase
            if error:
                self.error = error

    def view(self) -> dict:
        with self._lock:
            return {
                "percent": self.chunks,
                "chunks": self.chunks,
                "phase": self.phase,
                "error": self.error,
                "note": self.note,
                "optimize_hold": self.optimize_hold,
                "active_slots": self.active_slots,
                "total_mib": self.total_mib,
                "used_before": self.used_before,
                "per_slot": self.per_slot,
                "measured": self.measured,
                "use_gpu": self.use_gpu,
                "gpu_index": self.gpu_index,
                "forced": self.forced,
                "max_slots": self.max_slots,
                "host_total_mib": self.host_total_mib,
                "host_available_mib": self.host_available_mib,
            }

    def set_note(self, note: str) -> None:
        with self._lock:
            self.note = note

    def note_embed_call(
        self,
        size: int,
        chunks: int,
        seconds: float,
        ok: bool,
        kind: str = "",
        article_size: int = 0,
        tokens: int = 0,
    ) -> None:
        """Remember one embed call so the batch advice can follow the pace, not the card."""
        with self._lock:
            self._embed_call_count += 1
            self._embed_calls.append(
                (
                    int(size),
                    int(chunks),
                    float(seconds),
                    bool(ok),
                    str(kind or ""),
                    int(article_size or 0),
                    int(tokens or 0),
                )
            )
            if len(self._embed_calls) > 48:
                del self._embed_calls[:-48]

    def embed_calls(self) -> list[tuple[int, int, float, bool]]:
        with self._lock:
            return list(self._embed_calls)

    def embed_call_count(self) -> int:
        """How many embed calls have been recorded. The kept list is only the latest."""
        with self._lock:
            return self._embed_call_count

    def set_optimize_hold(self, reason: str) -> None:
        """segment or space while indexing is waiting on a rewrite. Empty otherwise."""
        with self._lock:
            self.optimize_hold = reason if reason in ("segment", "space") else ""

    def bind_book(self, book: ProgressBook) -> None:
        with self._lock:
            self._book = book

    def get_book(self) -> ProgressBook | None:
        with self._lock:
            return self._book

    def configure_gpu(
        self,
        *,
        total: int,
        used_before: int,
        per_slot: int,
        measured: int | None,
        use_gpu: bool,
        gpu_index: int | None,
        active: int,
        forced: int | None,
        max_slots: int,
        host_total: int = 0,
        host_available: int = 0,
    ) -> None:
        with self._lock:
            self.total_mib = int(total)
            self.used_before = int(used_before)
            self.per_slot = int(per_slot)
            self.measured = int(measured) if measured else None
            self.use_gpu = bool(use_gpu)
            self.gpu_index = gpu_index
            self.active_slots = int(active)
            self.forced = int(forced) if forced else None
            self.max_slots = int(max_slots)
            self.host_total_mib = int(host_total)
            self.host_available_mib = int(host_available)

    def set_order(self, choice: str) -> None:
        with self._lock:
            self._order = normalize_order(choice)

    def get_order(self) -> str:
        with self._lock:
            return self._order

    def set_priority(self, source: str, on: bool) -> None:
        """Mark or clear one file. A new mark is what makes a running file switch."""
        with self._lock:
            if on:
                if source not in self._priority:
                    self._priority.add(source)
                    self._priority_epoch += 1
            else:
                self._priority.discard(source)

    def priorities(self) -> set[str]:
        with self._lock:
            return set(self._priority)

    def arm_priority(self) -> int:
        """Epoch at the moment a file starts. Later marks are the ones that switch."""
        with self._lock:
            return self._priority_epoch

    def wants_priority_switch(self, current: str, armed_epoch: int) -> bool:
        """True when a file was marked after this one started, and this one is not marked."""
        with self._lock:
            if current in self._priority:
                return False
            return self._priority_epoch != int(armed_epoch)

    def request_index_now(self, source: str) -> None:
        """The latest request wins until that file finishes."""
        with self._lock:
            self._index_now = source

    def index_now(self) -> str | None:
        with self._lock:
            return self._index_now

    def clear_index_now(self, source: str | None = None) -> None:
        with self._lock:
            if source is None or self._index_now == source:
                self._index_now = None

    def start_measure(self, *, from_idle: bool, then_index: bool = False) -> None:
        """Begin a measure pass and start the 300s clock for the file already running.

        `then_index` keeps indexing after the samples when this pass started idle.
        A pass started during a run already keeps indexing.
        """
        with self._lock:
            self._measuring = True
            self._measure_from_idle = bool(from_idle)
            self._measure_then_index = bool(then_index)
            self._sampled = set()
            self._measure_clock = time.monotonic()

    def arm_measure_clock(self, started: float | None = None) -> float:
        """Start a fresh 300s sample. The next file gets its own clock."""
        with self._lock:
            self._measure_clock = time.monotonic() if started is None else float(started)
            return self._measure_clock

    def measure_clock(self) -> float | None:
        """When the current sample started, or None when Measure is off."""
        with self._lock:
            if not self._measuring:
                return None
            return self._measure_clock

    def set_measuring(self, on: bool) -> None:
        with self._lock:
            self._measuring = bool(on)
            if not on:
                self._sampled = set()
                self._measure_clock = None
                self._measure_then_index = False

    def measuring(self) -> bool:
        with self._lock:
            return self._measuring

    def measure_from_idle(self) -> bool:
        with self._lock:
            return self._measure_from_idle

    def measure_then_index(self) -> bool:
        with self._lock:
            # A session built before this field existed still answers after its class is rebound.
            return bool(getattr(self, "_measure_then_index", False))

    def arm_then_index(self) -> bool:
        """Keep indexing after the measurement that is already running.

        Returns False when nothing is being measured. Does not restart the
        pass, reset the sample clock, or clear files already sampled.
        """
        with self._lock:
            if not self._measuring:
                return False
            self._measure_then_index = True
            return True

    def note_sampled(self, source: str) -> None:
        with self._lock:
            self._sampled.add(source)

    def sampled(self) -> set[str]:
        with self._lock:
            return set(self._sampled)

    def bind_file_stats(self, stats) -> None:
        with self._lock:
            self._file_stats = stats

    def file_stats(self):
        with self._lock:
            return self._file_stats

    def note_zim_date(self, source: str, raw: str | None) -> None:
        """Remember a ZIM Date. An empty value means the archive had none."""
        with self._lock:
            self._zim_dates[source] = "" if raw is None else str(raw)
            stats = self._file_stats
        if stats is not None and hasattr(stats, "note_zim_date"):
            stats.note_zim_date(source, raw)

    def zim_date_known(self, source: str) -> bool:
        with self._lock:
            return source in self._zim_dates

    def zim_date(self, source: str) -> str | None:
        with self._lock:
            if source not in self._zim_dates:
                return None
            value = self._zim_dates[source]
            return value or None

    def zim_dates(self) -> dict[str, str]:
        with self._lock:
            return dict(self._zim_dates)

    def set_tune_auto(self, on: bool) -> None:
        with self._lock:
            self._tune_auto = bool(on)

    def tune_auto(self) -> bool:
        with self._lock:
            return self._tune_auto
