"""Skipped and bad article counts for the current knowledge index.

The numbers stay on disk across a restart. They are cleared by the same
collection clock as the log files: a timeout keeps them, a confirmed missing
collection drops them, and a record older than the collection is dropped.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path


class RunCounts:
    """One small JSON file of the run's skipped and bad totals."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.skipped = 0
        self.bad = 0
        self.saved_at: float | None = None

    def load(self) -> None:
        path = self.path
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            return
        if not isinstance(raw, dict):
            return
        self.skipped = max(0, int(raw.get("skipped") or 0))
        self.bad = max(0, int(raw.get("bad") or 0))
        stamp = raw.get("saved_at")
        self.saved_at = float(stamp) if isinstance(stamp, (int, float)) else None

    def note(self, kind: str) -> None:
        """Count one article that was not stored. A bad article is also skipped."""
        if kind == "bad":
            self.bad += 1
            self.skipped += 1
        elif kind == "skipped":
            self.skipped += 1
        else:
            return
        self.saved_at = time.time()

    def save(self) -> None:
        if self.skipped == 0 and self.bad == 0:
            self.clear()
            return
        payload = {
            "skipped": int(self.skipped),
            "bad": int(self.bad),
            "saved_at": self.saved_at,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(self.path)

    def clear(self) -> None:
        self.skipped = 0
        self.bad = 0
        self.saved_at = None
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass

    def apply_clock(self, kind: str, epoch: int) -> bool:
        """Drop the counts when the collection clock says the record is not this index.

        Returns True when the counts were cleared.
        """
        from nomad_turbo.logbook import record_survives

        stamp = datetime.fromtimestamp(self.saved_at) if self.saved_at else None
        if record_survives(stamp, kind, epoch):
            return False
        self.clear()
        return True

    def load_and_apply(self, kind: str, epoch: int) -> None:
        self.load()
        self.apply_clock(kind, epoch)
