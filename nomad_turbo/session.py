"""Shared state between the control window and the indexing thread."""

from __future__ import annotations

import threading

from nomad_turbo.progress import ProgressBook
from nomad_turbo.slots import clamp_boost_percent


class Session:
    def __init__(self, percent: int):
        self.stop = threading.Event()
        self.close_requested = False
        self.wipe_confirmed = False
        self._lock = threading.Lock()
        self.percent = clamp_boost_percent(percent)
        self.keep_control = False
        self.phase = "idle"
        self.error = ""
        self.note = ""
        self.optimize_hold = ""
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

    def set_percent(self, percent: int) -> None:
        with self._lock:
            self.percent = clamp_boost_percent(percent)

    def get_percent(self) -> int:
        with self._lock:
            return self.percent

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
                "percent": self.percent,
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
