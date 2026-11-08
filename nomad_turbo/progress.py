"""Progress fractions and ETAs. File size carries the overall bar."""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass

from nomad_turbo.catalog import CatalogRow, origin_for, row_color, status_word
from nomad_turbo.schedule import (
    COL_ARTICLES,
    COL_ARTICLES_SORT,
    COL_ART_MIN,
    COL_ART_MIN_SORT,
    COL_ETA,
    COL_ETA_SORT,
    COL_PROGRESS,
    COL_STATUS,
    file_eta_view,
    file_type_label,
    pace_text,
)


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

# A saved log size in this range is the log text height from the fixed drag bar.
# Larger values are a split measured from the top of the window.
LOG_HEIGHT_MIN = 120
LOG_HEIGHT_MAX = 720
LOG_HEIGHT_DEFAULT = 220


def initial_log_split(saved: int, pane_height: int) -> int:
    """Height of the area above the log bar.

    The window itself stays resizable. This only places the bar inside it.
    """
    height = int(pane_height or 0)
    if height < 160:
        return max(0, height // 2)
    value = int(saved or 0)
    if LOG_HEIGHT_MIN <= value <= LOG_HEIGHT_MAX:
        top = height - value - 16
    elif value > 80:
        top = value
    else:
        top = height - LOG_HEIGHT_DEFAULT - 16
    return max(80, min(top, height - 80))


@dataclass(frozen=True)
class Frame:
    """Size and place to open a window with.

    x and y are None when the window manager should center the window.
    """

    x: int | None
    y: int | None
    width: int
    height: int
    maximized: bool = False


def _title_on_screen(x: int, y: int, width: int, screen_width: int, screen_height: int) -> bool:
    """True when the title bar still has a grab point on the screen.

    An unknown screen (0) keeps the saved place. A window that is only partly
    on screen is kept when its top edge can still be dragged.
    """
    if screen_width <= 0 or screen_height <= 0:
        return True
    if y < -8 or y > screen_height - 24:
        return False
    if x > screen_width - 48:
        return False
    if x + min(int(width), 80) < 8:
        return False
    return True


def restored_frame(
    saved_x: int | None,
    saved_y: int | None,
    saved_width: int,
    saved_height: int,
    *,
    default_width: int,
    default_height: int,
    floor_width: int,
    floor_height: int,
    screen_width: int = 0,
    screen_height: int = 0,
    maximized: bool = False,
) -> Frame:
    """Frame to open with.

    A saved size below the floor uses the default. A saved size larger than
    the screen is shrunk to the screen. A place whose title bar is off the
    screen is centered. Maximized is kept either way, so the unmaximized size
    is still there when the window is restored.
    """
    width = int(saved_width or 0)
    height = int(saved_height or 0)
    if width < floor_width or height < floor_height:
        width = int(default_width)
        height = int(default_height)
    if screen_width > 0:
        width = min(width, max(int(floor_width), int(screen_width)))
    if screen_height > 0:
        height = min(height, max(int(floor_height), int(screen_height)))
    width = max(int(floor_width), width)
    height = max(int(floor_height), height)
    x = None if saved_x is None else int(saved_x)
    y = None if saved_y is None else int(saved_y)
    if x is None or y is None or not _title_on_screen(x, y, width, int(screen_width or 0), int(screen_height or 0)):
        x = None
        y = None
    return Frame(x, y, width, height, bool(maximized))


def restored_log_top(saved_height: int, saved_top: int, pane_height: int) -> int:
    """Height of the area above the log bar, including a dragged split.

    saved_top is the position from a drag. When it is unset, saved_height is
    the older log-text height.
    """
    height = int(pane_height or 0)
    top = int(saved_top or 0)
    if top > 80 and height >= 160:
        return max(80, min(top, height - 80))
    return initial_log_split(saved_height, height)


def list_pane_position(saved: int, pane_height: int, controls_height: int) -> int:
    """Height of the file list. 0 leaves the spare space with the list."""
    height = int(pane_height or 0)
    if height < 96:
        return max(0, height // 2)
    chosen = int(saved or 0)
    if chosen <= 0:
        controls = max(48, int(controls_height or 0))
        chosen = height - controls - 16
    return max(48, min(chosen, height - 48))


# Pixels the wide handle and the control scroller need so the scroller stays mapped.
_LIST_HANDLE = 8
_CONTROL_STRIP = 48
_LIST_FLOOR = 8


def fitted_list_position(
    saved: int,
    pane_height: int,
    controls_height: int,
    list_floor: int = 0,
) -> int:
    """List height that still leaves a mapped control strip.

    A tall pane matches list_pane_position, so spare space stays with the list.
    A short pane keeps the list at least at list_floor when it fits, and always
    leaves the control scroller its own floor. list_floor is the list widget's
    minimum; a smaller position makes GTK draw that minimum past the top of the pane.
    """
    height = int(pane_height or 0)
    if height <= 0:
        return 0
    chosen = list_pane_position(saved, height, controls_height)
    floor = max(_LIST_FLOOR, int(list_floor or 0))
    # room is the list height that still leaves the control strip and the handle.
    room = height - _CONTROL_STRIP - _LIST_HANDLE
    if room < 1:
        return 1
    if room < floor:
        return max(1, min(chosen, room))
    return max(floor, min(chosen, room))


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


def log_should_stick(
    follow: bool,
    value: float,
    page_size: float,
    upper: float,
    *,
    away: bool = False,
) -> bool:
    """Keep the newest line in view when Follow is on and the reader has not left.

    Being behind the end because new lines arrived is still following. ``away``
    is set only when the user scrolled off the end. Follow off does not jump.
    A view that is not on screen yet still follows. The position arguments are
    accepted so callers can pass the adjustment; they do not decide this.
    """
    del value, page_size, upper
    if not follow or away:
        return False
    return True


def host_bar_position(pane_width: int, requested: int, *, main_min: int) -> int:
    """Main-side width of the vertical host bar.

    The bar will not move left of ``main_min``. When the pane is narrower than
    that floor, the main side keeps the pane and the host side shrinks to
    nothing. ``requested`` is the main-side width a drag or a saved split asked
    for. This does not change the window's size request.
    """
    width = max(0, int(pane_width))
    floor = max(0, int(main_min))
    if floor > width:
        floor = width
    position = int(requested)
    if position < floor:
        position = floor
    if position > width:
        position = width
    return position


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


def embed_server_text(running: int) -> str:
    """How many embed servers are up. The chunk count stays in the batch field."""
    count = max(0, int(running))
    noun = "embed server" if count == 1 else "embed servers"
    return f"{count} {noun}"


def ocr_worker_text(shown: int | None, total: int | None = None) -> str:
    """The line under the embed-server count. Idle until a reader reports workers."""
    if shown is None:
        return "OCR workers: idle"
    if total:
        return f"{int(shown)} of {int(total)} OCR workers"
    noun = "OCR worker" if int(shown) == 1 else "OCR workers"
    return f"{int(shown)} {noun}"


def server_status_text(running: int, workers: tuple[int, int] | None) -> str:
    """Embed servers, then the OCR workers currently reading."""
    if not workers:
        return f"{embed_server_text(running)}\n{ocr_worker_text(None)}"
    return f"{embed_server_text(running)}\n{ocr_worker_text(workers[0], workers[1])}"


def usage_text(*, chunks: int, server_up: bool, slots_ignored: bool = False, set_percent: int | None = None) -> str:
    """The chunk count sizes the next embed call. One server stays up.

    `set_percent` is accepted so older callers still run. It is not shown.
    A larger count is not a clock change.
    """
    del set_percent
    state = "One embed server is running." if server_up else "Embed server: idle."
    text = f"{int(chunks)} chunks per call. {state}"
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


def _fraction(
    state: str,
    articles: int,
    total: int | None,
    flight_done: int = 0,
    flight_total: int = 0,
) -> float:
    """Share of the file that is done.

    Articles are the whole steps. While one article is still embedding, its
    chunks fill the step that has not been stored yet.
    """
    if state == "indexed":
        return 1.0
    if state == "deferred":
        return 0.0
    partial = 0.0
    open_article = state == "running" and flight_total > 0 and (not total or articles < int(total))
    if open_article:
        partial = min(1.0, max(0.0, flight_done / flight_total))
    if total and total > 0:
        credit = partial if articles < int(total) else 0.0
        return min(1.0, max(0.0, (articles + credit) / int(total)))
    if open_article:
        return partial
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
    file_type: str = ""
    art_text: str = "—"
    chk_text: str = "—"
    total_articles: int | None = None
    mtime: float | None = None
    walk_index: int = 0


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
    structure_epoch: int = 0


class LibraryItem:
    """A file as the folder tree needs it. Progress fields stay on the book."""

    __slots__ = ("source", "name", "origin", "state", "size_bytes")

    def __init__(self, source: str, name: str, origin: str, state: str, size_bytes: int) -> None:
        self.source = source
        self.name = name
        self.origin = origin
        self.state = state
        self.size_bytes = size_bytes


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
        self.mtime = row.mtime
        self.walk_index = int(row.walk_index or 0)
        self.rate = RateTracker()
        self.chunk_rate = RateTracker()
        self.measured_rate: float | None = None
        # Chunks of the article being embedded. They are not stored yet.
        self.flight_done = 0
        self.flight_total = 0
        self.flight_rate = RateTracker()
        # The file row's own pass. These do not move the library article rate.
        self.label = ""
        self.pass_done = 0
        self.pass_total: int | None = None


# A caller that omits a pass field leaves the row's previous value alone.
_KEEP = object()


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
        self._stats = None
        self._structure_epoch = 0
        self._running = None
        self._running_sources: list[str] = []
        self._reset_totals()
        for source in self._order:
            self._install_contrib(self._rows[source])
        self._remember_running()

    @property
    def structure_epoch(self) -> int:
        """Bumps when a file is added, removed, or changes state. Not on article ticks."""
        return self._structure_epoch

    def _reset_totals(self) -> None:
        self._chunks_stored = 0
        self._done_files = 0
        self._included_size = 0
        self._done_bytes = 0.0
        self._unknown = 0
        self._eta_pending = 0
        self._eta_missing = 0
        self._eta_sum = 0

    def _contrib_of(self, row: _Row) -> tuple:
        """What one file adds to the library bar. The caller holds the book lock."""
        chunks = int(row.chunks or 0)
        done_files = 1 if row.state == "indexed" else 0
        if row.state == "deferred":
            return (chunks, done_files, 0, 0.0, 0, 0, 0, 0)
        weight = max(int(row.size_bytes), 1)
        done_bytes = weight * _fraction(
            row.state,
            row.articles,
            row.total,
            row.flight_done,
            row.flight_total,
        )
        unknown = 1 if row.state != "indexed" and not row.total else 0
        if row.state == "indexed":
            return (chunks, done_files, weight, done_bytes, unknown, 0, 0, 0)
        eta = _eta_seconds(row)
        if eta is None:
            return (chunks, done_files, weight, done_bytes, unknown, 1, 1, 0)
        return (chunks, done_files, weight, done_bytes, unknown, 1, 0, int(eta))

    def _apply_parts(self, parts: tuple, sign: int) -> None:
        self._chunks_stored += sign * parts[0]
        self._done_files += sign * parts[1]
        self._included_size += sign * parts[2]
        self._done_bytes += sign * parts[3]
        self._unknown += sign * parts[4]
        self._eta_pending += sign * parts[5]
        self._eta_missing += sign * parts[6]
        self._eta_sum += sign * parts[7]

    def _clear_contrib(self, row: _Row) -> None:
        old = getattr(row, "_contrib", None)
        if old is None:
            return
        self._apply_parts(old, -1)
        row._contrib = None

    def _install_contrib(self, row: _Row) -> None:
        """Replace this file's share of the bar. The caller holds the book lock."""
        self._clear_contrib(row)
        parts = self._contrib_of(row)
        row._contrib = parts
        self._apply_parts(parts, 1)

    def _remember_running(self) -> None:
        """Every file in progress. The tick paints these and does not walk the rest."""
        sources = [source for source in self._order if self._rows[source].state == "running"]
        self._running_sources = sources
        self._running = self._rows[sources[0]] if sources else None

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
        label=_KEEP,
        pass_done=_KEEP,
        pass_total=_KEEP,
    ) -> None:
        moment = time.monotonic() if now is None else now
        with self._lock:
            row = self._rows.get(source)
            created = row is None
            if created:
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
            previous_state = row.state
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
            # A stored update replaces the in-flight count. The two are not added.
            row.flight_done = 0
            row.flight_total = 0
            row.state = state
            if state == "indexed":
                if not row.total:
                    row.total = max(row.articles, 1)
                row.articles = max(row.articles, row.total or row.articles)
            row.rate.push(moment, row.articles)
            if label is not _KEEP:
                row.label = str(label or "")
            if pass_done is not _KEEP:
                row.pass_done = max(0, int(pass_done or 0))
            if pass_total is not _KEEP:
                row.pass_total = int(pass_total) if pass_total else None
            self._install_contrib(row)
            self._bytes.push(moment, self._done_bytes)
            if created or previous_state != row.state:
                self._structure_epoch += 1
                if created or previous_state == "running" or row.state == "running":
                    self._remember_running()
            remembered = self._progress_fact(row)
        if remembered is not None:
            self._write_progress(remembered)

    def note_flight(self, source: str, done: int, total: int, now: float | None = None) -> None:
        """Chunks of the open article that have vectors. Nothing is stored yet.

        A one-article file would otherwise sit at zero until that article is
        upserted. Checkpoints and saved chunk counts stay on stored articles.
        """
        moment = time.monotonic() if now is None else now
        with self._lock:
            row = self._rows.get(source)
            if row is None or row.state != "running":
                return
            total_chunks = max(0, int(total or 0))
            if total_chunks <= 0:
                return
            done_chunks = min(total_chunks, max(0, int(done or 0)))
            row.flight_done = done_chunks
            row.flight_total = total_chunks
            row.flight_rate.push(moment, done_chunks)
            self._install_contrib(row)
            self._bytes.push(moment, self._done_bytes)

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
                self._install_contrib(self._rows[source])
                self._structure_epoch += 1
                return True
            if size_bytes and row.size_bytes <= 1:
                new_size = int(size_bytes)
                if new_size != row.size_bytes:
                    row.size_bytes = new_size
                    self._install_contrib(row)
                    self._structure_epoch += 1
            return False

    def drop(self, source: str) -> None:
        stats = None
        with self._lock:
            row = self._rows.get(source)
            if row is None:
                return
            was_running = row.state == "running"
            self._clear_contrib(row)
            del self._rows[source]
            self._order = [item for item in self._order if item != source]
            if was_running:
                self._remember_running()
            self._structure_epoch += 1
            stats = self._stats
        if stats is not None and stats.forget(source):
            stats.save()

    def listed(self) -> list[tuple[str, str]]:
        with self._lock:
            return [(source, self._rows[source].state) for source in self._order]

    def state_of(self, source: str) -> str | None:
        """One file's state. None when the library does not list it."""
        with self._lock:
            row = self._rows.get(source)
            if row is None:
                return None
            return row.state

    def mark_all_unindexed(self) -> None:
        with self._lock:
            for row in self._rows.values():
                row.state = "pending"
                row.articles = 0
                row.total = None
                row.chunks = 0
                row.rate = RateTracker()
                row.chunk_rate = RateTracker()
                row.flight_done = 0
                row.flight_total = 0
                row.flight_rate = RateTracker()
                row.measured_rate = None
            self._bytes = RateTracker()
            self._advanced = 0.0
            self._article_rate = RateTracker()
            self._chunk_advanced = 0.0
            self._chunk_rate = RateTracker()
            self.skipped = 0
            self.bad = 0
            self._running = None
            self._running_sources = []
            self._reset_totals()
            for row in self._rows.values():
                row._contrib = None
            for source in self._order:
                self._install_contrib(self._rows[source])
            self._structure_epoch += 1

    def attach_stats(self, stats) -> None:
        with self._lock:
            self._stats = stats

    def restore_saved(self, stats) -> None:
        """Put remembered measurement rates back on rows that are still listed."""
        with self._lock:
            sources = list(self._rows)
        saved_rates: dict[str, float] = {}
        if stats is not None:
            for source in sources:
                saved = stats.get(source)
                if saved is not None and saved.measured_article_rate:
                    saved_rates[source] = float(saved.measured_article_rate)
        with self._lock:
            self._stats = stats
            changed = False
            for source, rate in saved_rates.items():
                row = self._rows.get(source)
                if row is not None and row.measured_rate != rate:
                    row.measured_rate = rate
                    self._install_contrib(row)
                    changed = True
            if changed:
                self._structure_epoch += 1

    def keep_measured_rate(self, source: str) -> None:
        """Remember this file's article pace after Measure leaves it."""
        with self._lock:
            row = self._rows.get(source)
            if row is None:
                return
            rate = row.rate.per_second()
            if rate and row.measured_rate != rate:
                row.measured_rate = rate
                self._structure_epoch += 1
            elif rate:
                row.measured_rate = rate
            if rate:
                self._install_contrib(row)
            eta_seconds = None
            if rate:
                _text, eta_seconds = file_eta_view(
                    state=row.state,
                    articles=row.articles,
                    total=row.total,
                    live_rate=None,
                    measured_rate=rate,
                )
            remembered = self._progress_fact(row)
        if remembered is None:
            return
        self._write_progress(remembered, measured=True, eta_seconds=eta_seconds)
        remembered[0].save()

    def _progress_fact(self, row: _Row):
        """Fields to store. The caller holds the book lock. None when nothing is saved."""
        if self._stats is None:
            return None
        return (
            self._stats,
            row.source,
            row.articles,
            row.total,
            row.chunks,
            row.state,
            row.name,
            row.size_bytes,
            row.mtime,
            file_type_label(row.name),
            row.rate.per_second(),
            row.chunk_rate.per_second(),
        )

    def _write_progress(self, fact, *, measured: bool = False, eta_seconds: int | None = None) -> None:
        stats, source, articles, total, chunks, state, name, size_bytes, mtime, file_type, art_rate, chk_rate = fact
        stats.note_progress(
            source,
            articles=articles,
            total_articles=total,
            chunks=chunks,
            state=state,
            name=name,
            size_bytes=size_bytes,
            mtime=mtime,
            file_type=file_type,
        )
        stats.note_rates(source, art_rate, chk_rate, measured=measured, eta_seconds=eta_seconds if measured else None)

    def snapshot(self, now: float | None = None) -> BoardSnap:
        """Every file. The ordering pass uses this. The window tick uses pulse."""
        moment = time.monotonic() if now is None else now
        with self._lock:
            return self._compose_unlocked(moment, full=True)

    def pulse(self, now: float | None = None) -> BoardSnap:
        """The bars and the file that is indexing. Totals stay on the book."""
        moment = time.monotonic() if now is None else now
        with self._lock:
            return self._compose_unlocked(moment, full=False)

    def file_snap(self, source: str, now: float | None = None) -> FileSnap | None:
        """One file, for the details window. None when the library does not list it."""
        moment = time.monotonic() if now is None else now
        with self._lock:
            row = self._rows.get(source)
            if row is None:
                return None
            return _file_snap(row, moment)

    def files_for(self, sources, now: float | None = None) -> dict[str, FileSnap]:
        """Snaps for the files a folder is showing. Missing sources are left out."""
        moment = time.monotonic() if now is None else now
        with self._lock:
            found = {}
            for source in sources:
                row = self._rows.get(source)
                if row is not None:
                    found[source] = _file_snap(row, moment)
            return found

    def library_items(self) -> tuple[int, list[LibraryItem]]:
        """Folder-tree rows and the epoch they belong to. Progress text stays out."""
        with self._lock:
            items = []
            for source in self._order:
                row = self._rows[source]
                items.append(LibraryItem(source, row.name, row.origin, row.state, int(row.size_bytes)))
            return self._structure_epoch, items

    def _compose_unlocked(self, moment: float, *, full: bool) -> BoardSnap:
        """Caller holds the book lock. full copies every file. A pulse copies the running one.

        Bar totals come from the running share of each file. An article tick updates
        that one share, so the tick does not walk the library.
        """
        files: list[FileSnap] = []
        if full:
            for source in self._order:
                files.append(_file_snap(self._rows[source], moment))
        else:
            for source in self._running_sources:
                row = self._rows.get(source)
                if row is not None and row.state == "running":
                    files.append(_file_snap(row, moment))
        included = self._included_size
        done_bytes = self._done_bytes
        unknown = self._unknown
        overall = (done_bytes / included) if included else 0.0
        rate = self._bytes.per_second()
        remaining_bytes = max(0.0, included - done_bytes)
        if self._eta_missing:
            summed = None
        elif self._eta_pending == 0:
            summed = 0
        else:
            summed = self._eta_sum
        if overall >= 0.999 and unknown == 0:
            text = "100% · nothing left to index"
        elif summed is not None:
            text = f"{overall:.0%} of library size · about {format_eta(summed)}"
        elif not rate or remaining_bytes <= 0:
            text = f"{overall:.0%} of library size · ETA measuring"
        else:
            text = f"{overall:.0%} of library size · about {format_eta(remaining_bytes / rate)}"
        if unknown and overall < 0.999:
            text += f" · {unknown} files not measured yet"
        return BoardSnap(
            files,
            overall,
            text,
            self._done_files,
            len(self._order),
            self._article_rate.per_second(),
            self.skipped,
            self.bad,
            self._chunk_rate.per_second(),
            self._chunks_stored,
            self._structure_epoch,
        )


def _open_article(row: _Row) -> bool:
    """True while the current article has chunks that are not stored yet."""
    return row.state == "running" and row.flight_total > 0 and (not row.total or row.articles < row.total)


def _flight_eta_seconds(row: _Row) -> int | None:
    """Seconds left in the open article, from its chunk pace."""
    if not _open_article(row):
        return None
    rate = row.flight_rate.per_second()
    if not rate:
        return None
    remaining = max(0, row.flight_total - row.flight_done)
    return int(remaining / rate)


def _row_eta(row: _Row) -> tuple[str, int | None]:
    """ETA text and seconds. An open article with no article pace uses chunks.

    A text or OCR pass has its own count. The article total belongs to the
    HTML walk, so it is not what this pass has left.
    """
    running = row.state == "running"
    if running and row.pass_total:
        remaining = max(0, int(row.pass_total) - int(row.pass_done))
        if remaining == 0:
            return format_eta(0), 0
        rate = row.rate.per_second()
        if rate:
            seconds = int(remaining / rate)
            return format_eta(seconds), seconds
        return "measuring", None
    eta, seconds = file_eta_view(
        state=row.state,
        articles=row.articles,
        total=row.total,
        live_rate=row.rate.per_second() if running else None,
        measured_rate=row.measured_rate,
    )
    if seconds is None and running:
        flight = _flight_eta_seconds(row)
        if flight is not None:
            return format_eta(flight), flight
    return eta, seconds


def _eta_seconds(row: _Row) -> int | None:
    """Seconds left for one file. Indexed files report 0 and the sum skips them."""
    _text, seconds = _row_eta(row)
    return seconds


def _file_snap(row: _Row, _moment: float) -> FileSnap:
    """One list row. The caller holds the book lock."""
    fraction = _fraction(row.state, row.articles, row.total, row.flight_done, row.flight_total)
    if row.state == "running" and row.pass_total:
        fraction = min(1.0, max(0.0, row.pass_done / int(row.pass_total)))
    art_rate = row.rate.per_second()
    chk_rate = row.chunk_rate.per_second()
    running = row.state == "running"
    if _open_article(row):
        live_chunks = row.flight_rate.per_second()
        if live_chunks:
            chk_rate = live_chunks
    if row.state == "indexed":
        count = "indexed"
    elif row.state == "running" and row.pass_total:
        count = f"{row.pass_done:,} / {row.pass_total:,}"
    elif row.total:
        count = f"{row.articles:,} / {row.total:,}"
    elif row.articles:
        count = f"{row.articles:,}"
    else:
        count = "—"
    eta, eta_seconds = _row_eta(row)
    art_minute = (art_rate * 60) if running and art_rate else None
    chk_live = chk_rate if running and chk_rate else None
    if _open_article(row):
        chunks_text = f"{row.flight_done:,}"
    elif row.chunks:
        chunks_text = f"{row.chunks:,}"
    else:
        chunks_text = "—"
    return FileSnap(
        source=row.source,
        name=row.name,
        origin=row.origin,
        status=row.label if row.state == "running" and row.label else status_word(row.state),
        percent=int(round(fraction * 100)),
        count=count,
        eta=eta,
        color=row_color(row.origin, row.state),
        state=row.state,
        chunks=row.chunks,
        chunks_text=chunks_text,
        size_bytes=row.size_bytes,
        size_text=format_bytes(row.size_bytes),
        articles=row.articles,
        articles_per_minute=art_minute,
        chunks_per_second=chk_live,
        rate_text=format_file_rate(art_rate if running else None, chk_rate if running else None),
        eta_seconds=eta_seconds,
        file_type=file_type_label(row.name),
        art_text=pace_text(running, art_minute),
        chk_text=pace_text(running, chk_live),
        total_articles=row.total,
        mtime=row.mtime,
        walk_index=row.walk_index,
    )


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


_OCR_LINE = re.compile(r"ocr page (\d+) of (\d+)\b")
_OCR_NOTE = re.compile(r"Page (\d+) of (\d+)")
_OCR_WORKERS = re.compile(r"ocr workers (\d+) of (\d+)")


def ocr_page(line: str) -> tuple[int, int] | None:
    """Page numbers from one OCR progress line. None when the line is not a page."""
    match = _OCR_LINE.search(line or "")
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def ocr_note(name: str, page: int | None = None, total: int | None = None) -> str:
    """Status sentence while a PDF or picture is being read."""
    if page and total:
        return f"Reading {name} with OCR. Page {page} of {total}."
    return f"Reading {name} with OCR."


def _ocr_bar(activity: str) -> tuple[str, float] | None:
    if "with OCR" not in (activity or ""):
        return None
    match = _OCR_NOTE.search(activity)
    if match is None:
        return "Reading with OCR", 0.0
    page, total = int(match.group(1)), int(match.group(2))
    fraction = 0.0 if total <= 0 else min(1.0, page / total)
    return f"OCR page {page:,} of {total:,}", fraction


def ocr_list_cells(activity: str) -> tuple[str, int] | None:
    """Status word and progress percent for the row while OCR is reading it."""
    shown = _ocr_bar(activity)
    if shown is None:
        return None
    _text, fraction = shown
    return "OCR", int(round(fraction * 100))


def ocr_note_matches(activity: str, name: str) -> bool:
    """True when this status sentence is the OCR read of this file.

    A second file can be indexing at the same time. Its row keeps its own
    article and chunk cells.
    """
    text = activity or ""
    if not name or "with OCR" not in text:
        return False
    return text.startswith(f"Reading {name} with OCR")


def ocr_eta_seconds(page: int, total: int, pages_per_minute: float | None) -> int | None:
    """Seconds left in this read, from the page rate. None until a rate exists."""
    if not pages_per_minute or pages_per_minute <= 0 or total <= 0:
        return None
    remaining = max(0, int(total) - int(page))
    return int(remaining / (float(pages_per_minute) / 60.0))


def ocr_pages(activity: str) -> tuple[int, int] | None:
    """Pages read and pages in the file. None until a page has been reported."""
    if "with OCR" not in (activity or ""):
        return None
    match = _OCR_NOTE.search(activity or "")
    if match is None:
        return None
    page, total = int(match.group(1)), int(match.group(2))
    if total <= 0:
        return None
    return page, total


def ocr_workers(line: str) -> tuple[int, int] | None:
    """Active workers and the pool size. None when the line is not a worker count."""
    match = _OCR_WORKERS.search(line or "")
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def remember_ocr_workers(session, line: str) -> None:
    """Keep the latest worker count for the line under the embed servers."""
    found = ocr_workers(line)
    if found is None or session is None:
        return
    setter = getattr(session, "set_ocr_workers", None)
    if callable(setter):
        setter(found[0], found[1])


def ocr_status_level(line: str) -> int:
    """A finished page and the worker count are debug. Failures stay info."""
    if ocr_page(line) or ocr_workers(line):
        return logging.DEBUG
    return logging.INFO


class OcrPace:
    """Pages per minute while one file is being read.

    The same page at the same moment is not sampled twice. A different file,
    or leaving OCR, starts the pace over.
    """

    def __init__(self) -> None:
        self._rate = RateTracker()
        self._page: int | None = None
        self._head = ""

    def reset(self) -> None:
        self._rate = RateTracker()
        self._page = None
        self._head = ""

    def observe(self, activity: str, now: float | None = None) -> float | None:
        """Pages per minute, or None when there is not yet a pace."""
        if "with OCR" not in (activity or ""):
            self.reset()
            return None
        head = (activity or "").split(" Page ", 1)[0]
        if head != self._head:
            self._rate = RateTracker()
            self._page = None
            self._head = head
        pages = ocr_pages(activity)
        if pages is None:
            return None
        page = pages[0]
        moment = time.monotonic() if now is None else now
        last = self._rate.samples[-1][0] if self._rate.samples else None
        if self._page != page or last != moment:
            self._rate.push(moment, page)
            self._page = page
        per = self._rate.per_second()
        if not per:
            return None
        return per * 60.0


def overlay_ocr_columns(values: tuple, activity: str, pages_per_minute: float | None) -> tuple:
    """Status, progress, articles, art/min, and ETA while OCR is reading this row.

    Articles becomes pages done and pages in the file. art/min becomes pages
    per minute. ETA is the pages still to read at that rate. It stays
    measuring until a page rate exists. The column titles stay as they are.
    """
    cells = ocr_list_cells(activity)
    if cells is None:
        return values
    status, percent = cells
    out = list(values)
    out[COL_STATUS] = status
    out[COL_PROGRESS] = percent
    pages = ocr_pages(activity)
    if pages is not None:
        page, total = pages
        out[COL_ARTICLES] = f"{page:,} / {total:,}"
        out[COL_ARTICLES_SORT] = page
        if pages_per_minute:
            out[COL_ART_MIN] = pace_text(True, pages_per_minute)
            out[COL_ART_MIN_SORT] = int(round(pages_per_minute))
        seconds = ocr_eta_seconds(page, total, pages_per_minute)
        if seconds is not None:
            out[COL_ETA] = format_eta(seconds)
            out[COL_ETA_SORT] = seconds
    return tuple(out)


def format_current_file(snap: BoardSnap | None, activity: str = "") -> tuple[str, float, str]:
    """Title, bar fraction, and bar text for the file that is indexing."""
    item = None if snap is None else next((row for row in snap.files if row.state == "running"), None)
    if item is None:
        return "No file is indexing.", 0.0, "Idle"
    shown = _ocr_bar(activity)
    if shown is not None:
        bar, fraction = shown
        return item.name, fraction, bar
    bits = [f"{item.percent}%", f"{item.count} articles"]
    if item.chunks_text and item.chunks_text != "—":
        bits.append(f"{item.chunks_text} chunks")
    elif item.chunks:
        bits.append(f"{item.chunks:,} chunks")
    if item.size_text and item.size_text != "—":
        bits.append(item.size_text)
    if item.rate_text and item.rate_text != "—":
        bits.append(item.rate_text)
    if item.eta:
        bits.append(item.eta if item.eta == "measuring" else f"ETA {item.eta}")
    return item.name, item.percent / 100.0, " · ".join(bits)
