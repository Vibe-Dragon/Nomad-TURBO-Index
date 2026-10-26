"""Progress fractions and ETAs. File size carries the overall bar."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

from nomad_turbo.catalog import CatalogRow, origin_for, row_color, status_word


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "measuring"
    if seconds <= 0:
        return "done"
    whole = int(seconds)
    if whole < 90:
        return f"{whole}s"
    minutes = whole // 60
    if minutes < 90:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h" if hours else f"{days}d"


# The pane shows this many lines. The file keeps the whole session.
LOG_PANE_LINES = 4000

# Pixel height of the log text. The drag bar under the Log row sets this.
# Older builds stored a window split in the same setting. A split taken from
# the top of the window is outside this range and falls back to the default.
LOG_HEIGHT_MIN = 120
LOG_HEIGHT_MAX = 720
LOG_HEIGHT_DEFAULT = 220


def saved_log_height(saved: int) -> int:
    """Log text height. 0 and an old window-split position use the default."""
    value = int(saved or 0)
    if LOG_HEIGHT_MIN <= value <= LOG_HEIGHT_MAX:
        return value
    return LOG_HEIGHT_DEFAULT


def next_log_height(start: int, upward: int, *, room: int) -> int:
    """Height after a drag. `upward` is how far the pointer moved up, in pixels."""
    ceiling = min(LOG_HEIGHT_MAX, max(LOG_HEIGHT_MIN, int(room)))
    grown = int(start) + int(upward)
    if grown < LOG_HEIGHT_MIN:
        return LOG_HEIGHT_MIN
    if grown > ceiling:
        return ceiling
    return grown


def log_lines_to_drop(line_count: int, limit: int = LOG_PANE_LINES) -> int:
    """Leading lines to remove so the on-screen log stays within limit.

    A long run logs one line per article. Leaving those in the text view makes
    the window stop answering the close button.
    """
    extra = int(line_count) - int(limit)
    if extra > 0:
        return extra
    return 0


def log_line_visible(
    level: int,
    show_info: bool,
    show_debug: bool,
    show_errors: bool = True,
) -> bool:
    """Which lines the pane draws. The log file is not affected."""
    if level >= logging.WARNING:
        return show_errors
    if level >= logging.INFO:
        return show_info
    return show_debug


def log_should_stick(follow: bool, value: float, page_size: float, upper: float) -> bool:
    """Keep the newest line in view only when Follow is on and the reader is already there.

    Scrolling up holds the place. A view that is not on screen yet still follows.
    """
    if not follow:
        return False
    if page_size <= 0 or upper <= page_size:
        return True
    return value + page_size >= upper - 24


def follow_target(active: int | None, count: int, lookahead: int = 3) -> int | None:
    """Row to pin at the bottom of a file list.

    The active file stays on screen, and a few later files stay visible under it.
    """
    if active is None or count <= 0 or lookahead < 0:
        return None
    if active < 0 or active >= count:
        return None
    return min(count - 1, active + lookahead)


def format_bytes(size: int) -> str:
    """A short size for a column. Zero and negative stay blank."""
    if size <= 0:
        return "—"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if unit == "B":
            if value < 1024:
                return f"{int(value)} B"
            value /= 1024
            continue
        if value < 1024 or unit == "TB":
            if value >= 100:
                return f"{value:.0f} {unit}"
            if value >= 10:
                return f"{value:.1f} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.0f} TB"


def preferred_column_width(
    samples: list[str],
    *,
    header: str,
    minimum: int,
    maximum: int,
    px_per_char: int = 8,
    padding: int = 36,
) -> int:
    """A starting width from the heading and the text that will sit in the column."""
    longest = header or ""
    for text in samples:
        if text and len(text) > len(longest):
            longest = text
    width = padding + len(longest) * px_per_char
    return max(minimum, min(maximum, width))


def format_file_rate(articles_per_second: float | None, chunks_per_second: float | None) -> str:
    parts: list[str] = []
    if articles_per_second:
        parts.append(f"{articles_per_second * 60:,.0f} art/min")
    if chunks_per_second:
        parts.append(f"{chunks_per_second:,.0f} chk/s")
    return " · ".join(parts) if parts else "—"


def usage_text(*, set_percent: int, chunks: int, server_up: bool, slots_ignored: bool = False) -> str:
    """Set is the slider. It sizes the next embed call. One server stays up."""
    state = "One embed server is running." if server_up else "Embed server: idle."
    text = f"Set {set_percent}% · {chunks} chunks per call. {state}"
    if int(set_percent) > 100:
        text += " Overclock past the detected budget."
    if slots_ignored:
        text += " The slots setting is ignored."
    return text


class RateTracker:
    def __init__(self, window: float = 120.0):
        self.window = window
        self.samples: deque[tuple[float, float]] = deque()

    def push(self, now: float, value: float) -> None:
        self.samples.append((float(now), float(value)))
        cutoff = now - self.window
        while len(self.samples) > 2 and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def per_second(self) -> float | None:
        if len(self.samples) < 2:
            return None
        start_at, start_value = self.samples[0]
        end_at, end_value = self.samples[-1]
        elapsed = end_at - start_at
        if elapsed < 1 or end_value < start_value:
            return None
        return (end_value - start_value) / elapsed


def _fraction(state: str, articles: int, total: int | None) -> float:
    if state == "indexed":
        return 1.0
    if state == "deferred":
        return 0.0
    if total and total > 0:
        return min(1.0, max(0.0, articles / total))
    return 0.0


@dataclass
class FileSnap:
    source: str
    name: str
    origin: str
    status: str
    percent: int
    count: str
    eta: str
    color: str
    state: str
    chunks: int = 0
    chunks_text: str = "—"
    size_bytes: int = 0
    size_text: str = "—"
    articles: int = 0
    articles_per_minute: float | None = None
    chunks_per_second: float | None = None
    rate_text: str = "—"
    eta_seconds: int | None = None


@dataclass
class BoardSnap:
    files: list[FileSnap]
    overall_fraction: float
    overall_text: str
    done_files: int
    total_files: int
    articles_per_second: float | None = None
    skipped: int = 0
    bad: int = 0
    chunks_per_second: float | None = None
    chunks_stored: int = 0


class _Row:
    def __init__(self, row: CatalogRow):
        self.source = row.source
        self.name = row.name
        self.origin = row.origin
        self.kind = row.kind
        self.size_bytes = max(0, int(row.size_bytes))
        self.state = row.state
        self.articles = int(row.articles or 0)
        self.total = row.total_articles
        self.chunks = int(row.chunks or 0)
        self.rate = RateTracker()
        self.chunk_rate = RateTracker()


class ProgressBook:
    def __init__(self, rows: list[CatalogRow]):
        self._lock = threading.Lock()
        self._rows = {row.source: _Row(row) for row in rows}
        self._order = [row.source for row in rows]
        self._bytes = RateTracker()
        self._advanced = 0.0
        self._article_rate = RateTracker()
        self._chunk_advanced = 0.0
        self._chunk_rate = RateTracker()
        self.skipped = 0
        self.bad = 0

    def note_article(self, kind: str) -> None:
        """Count one page that was not stored. A bad page is also skipped."""
        with self._lock:
            if kind == "bad":
                self.bad += 1
                self.skipped += 1
            elif kind == "skipped":
                self.skipped += 1

    def update(
        self,
        source: str,
        *,
        articles: int,
        total: int | None,
        chunks: int,
        state: str,
        now: float | None = None,
    ) -> None:
        moment = time.monotonic() if now is None else now
        with self._lock:
            row = self._rows.get(source)
            if row is None:
                row = _Row(
                    CatalogRow(
                        source=source,
                        name=source.rsplit("/", 1)[-1],
                        origin=origin_for(source),
                        kind="zim" if source.endswith(".zim") else "file",
                        size_bytes=1,
                        state=state,
                    )
                )
                self._rows[source] = row
                self._order.append(source)
            previous = row.articles
            incoming = max(0, int(articles or 0))
            self._advanced += max(0, incoming - previous)
            self._article_rate.push(moment, self._advanced)
            row.articles = incoming
            if total:
                row.total = int(total)
            incoming_chunks = max(row.chunks, int(chunks or 0))
            self._chunk_advanced += incoming_chunks - row.chunks
            self._chunk_rate.push(moment, self._chunk_advanced)
            row.chunks = incoming_chunks
            row.chunk_rate.push(moment, row.chunks)
            row.state = state
            if state == "indexed":
                if not row.total:
                    row.total = max(row.articles, 1)
                row.articles = max(row.articles, row.total or row.articles)
            row.rate.push(moment, row.articles)
            self._bytes.push(moment, self._done_bytes_unlocked())

    def ensure_present(
        self,
        source: str,
        *,
        name: str,
        origin: str,
        kind: str,
        size_bytes: int,
    ) -> bool:
        """Add a file the library scan just found. Leave a row that is already shown."""
        with self._lock:
            row = self._rows.get(source)
            if row is None:
                self._rows[source] = _Row(
                    CatalogRow(
                        source=source,
                        name=name,
                        origin=origin,
                        kind=kind,
                        size_bytes=max(0, int(size_bytes)),
                        state="pending",
                    )
                )
                self._order.append(source)
                return True
            if size_bytes and row.size_bytes <= 1:
                row.size_bytes = int(size_bytes)
            return False

    def drop(self, source: str) -> None:
        with self._lock:
            if source not in self._rows:
                return
            del self._rows[source]
            self._order = [item for item in self._order if item != source]

    def listed(self) -> list[tuple[str, str]]:
        with self._lock:
            return [(source, self._rows[source].state) for source in self._order]

    def mark_all_unindexed(self) -> None:
        with self._lock:
            for row in self._rows.values():
                row.state = "pending"
                row.articles = 0
                row.total = None
                row.chunks = 0
                row.rate = RateTracker()
                row.chunk_rate = RateTracker()
            self._bytes = RateTracker()
            self._advanced = 0.0
            self._article_rate = RateTracker()
            self._chunk_advanced = 0.0
            self._chunk_rate = RateTracker()
            self.skipped = 0
            self.bad = 0

    def snapshot(self, now: float | None = None) -> BoardSnap:
        moment = time.monotonic() if now is None else now
        with self._lock:
            files: list[FileSnap] = []
            done_files = 0
            included_size = 0
            done_bytes = 0.0
            unknown = 0
            for source in self._order:
                row = self._rows[source]
                fraction = _fraction(row.state, row.articles, row.total)
                if row.state == "indexed":
                    done_files += 1
                if row.state != "deferred":
                    weight = max(row.size_bytes, 1)
                    included_size += weight
                    done_bytes += weight * fraction
                    if row.state != "indexed" and not row.total:
                        unknown += 1
                eta = ""
                eta_seconds: int | None = None
                art_rate = row.rate.per_second()
                chk_rate = row.chunk_rate.per_second()
                if row.state == "indexed":
                    count = "indexed"
                    eta_seconds = 0
                elif row.total:
                    count = f"{row.articles:,} / {row.total:,}"
                    remaining = max(0, row.total - row.articles)
                    if art_rate:
                        eta_seconds = int(remaining / art_rate)
                        eta = format_eta(eta_seconds)
                    elif row.state == "running":
                        eta = "measuring"
                    else:
                        eta = ""
                elif row.articles:
                    count = f"{row.articles:,}"
                    eta = "measuring" if row.state == "running" else ""
                else:
                    count = "—"
                    eta = "measuring" if row.state == "running" else ""
                running = row.state == "running"
                files.append(
                    FileSnap(
                        source=row.source,
                        name=row.name,
                        origin=row.origin,
                        status=status_word(row.state),
                        percent=int(round(fraction * 100)),
                        count=count,
                        eta=eta,
                        color=row_color(row.origin, row.state),
                        state=row.state,
                        chunks=row.chunks,
                        chunks_text=f"{row.chunks:,}" if row.chunks else "—",
                        size_bytes=row.size_bytes,
                        size_text=format_bytes(row.size_bytes),
                        articles=row.articles,
                        articles_per_minute=(art_rate * 60) if running and art_rate else None,
                        chunks_per_second=chk_rate if running and chk_rate else None,
                        rate_text=format_file_rate(art_rate if running else None, chk_rate if running else None),
                        eta_seconds=eta_seconds,
                    )
                )
            overall = (done_bytes / included_size) if included_size else 0.0
            rate = self._bytes.per_second()
            remaining_bytes = max(0.0, included_size - done_bytes)
            if overall >= 0.999 and unknown == 0:
                text = "100% · nothing left to index"
                eta_seconds = 0
            elif not rate or remaining_bytes <= 0:
                eta_seconds = None
                text = f"{overall:.0%} of library size · ETA measuring"
            else:
                eta_seconds = remaining_bytes / rate
                text = f"{overall:.0%} of library size · about {format_eta(eta_seconds)}"
            if unknown and overall < 0.999:
                text += f" · {unknown} files not measured yet"
            return BoardSnap(
                files,
                overall,
                text,
                done_files,
                len(self._order),
                self._article_rate.per_second(),
                self.skipped,
                self.bad,
                self._chunk_rate.per_second(),
                sum(item.chunks for item in self._rows.values()),
            )

    def _done_bytes_unlocked(self) -> float:
        total = 0.0
        for row in self._rows.values():
            if row.state == "deferred":
                continue
            weight = max(row.size_bytes, 1)
            total += weight * _fraction(row.state, row.articles, row.total)
        return total


def format_run_stats(phase: str, elapsed: int | None, snap: BoardSnap | None) -> str:
    """The line under the progress bar. Skipped and bad stay hidden until a run."""
    parts: list[str] = []
    active = phase in ("waiting", "running", "done", "stopped", "error")
    if elapsed is not None and active:
        parts.append("Elapsed " + (format_eta(elapsed) if elapsed else "0s"))
    if snap is not None:
        parts.append(f"{snap.done_files} / {snap.total_files} files done")
        if snap.articles_per_second:
            parts.append(f"{snap.articles_per_second * 60:,.0f} articles/min")
        if snap.chunks_per_second:
            parts.append(f"{snap.chunks_per_second * 60:,.0f} chunks/min")
        if snap.chunks_stored:
            parts.append(f"{snap.chunks_stored:,} chunks stored")
        if active:
            parts.append(f"Skipped {snap.skipped:,}")
            parts.append(f"Bad {snap.bad:,}")
    return " · ".join(parts)


# Unknown ETAs sort after every real one.
UNKNOWN_ETA_SECONDS = 10**9


def format_current_file(snap: BoardSnap | None) -> tuple[str, float, str]:
    """Title, bar fraction, and bar text for the file that is indexing."""
    item = None if snap is None else next((row for row in snap.files if row.state == "running"), None)
    if item is None:
        return "No file is indexing.", 0.0, "Idle"
    bits = [f"{item.percent}%", f"{item.count} articles"]
    if item.chunks:
        bits.append(f"{item.chunks:,} chunks")
    if item.size_text and item.size_text != "—":
        bits.append(item.size_text)
    if item.rate_text and item.rate_text != "—":
        bits.append(item.rate_text)
    if item.eta:
        bits.append(item.eta if item.eta == "measuring" else f"ETA {item.eta}")
    return item.name, item.percent / 100.0, " · ".join(bits)
