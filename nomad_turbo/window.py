"""Control window. Opening it does not pause NOMAD or delete anything."""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from nomad_turbo import __version__
from nomad_turbo.catalog import wipe_phrase_ok
from nomad_turbo.desktop import icon_dir, write_shortcut
from nomad_turbo.discover import load_install, save_install, state_dir
from nomad_turbo.file_stats import STATS_NAME, FileStats, compose_detail, detail_lines, library_location
from nomad_turbo.passes import bar_color, normalize_ocr_quality
from nomad_turbo.progress import (
    OcrPace,
    ProgressBook,
    format_run_stats,
    ocr_note_matches,
    overlay_ocr_columns,
    LOG_PANE_LINES,
    restored_frame,
    restored_log_top,
    log_line_visible,
    log_lines_to_drop,
    host_bar_position,
    log_should_stick,
    preferred_column_width,
    embed_server_text,
)
from nomad_turbo.qdrant_status import changes_allowed, delete_allowed
from nomad_turbo.repair import format_repair_current
from nomad_turbo.runtime import gpu_rows
from nomad_turbo.scan import scan_library
from nomad_turbo.schedule import (
    COL_ARTICLES,
    COL_ARTICLES_SORT,
    COL_ART_MIN,
    COL_CHK_S,
    COL_CHUNKS,
    COL_COLOR,
    COL_ETA,
    COL_MARK,
    COL_NAME,
    COL_PROGRESS,
    COL_SIZE,
    COL_SOURCE,
    COL_STATUS,
    COL_TYPE,
    INDEX_HEADERS,
    ORDER_CHOICES,
    normalize_order,
    FOLDER,
    FOLDER_HOLD,
    INDEXING_COLOR,
    INDEXING_MARK,
    PRIORITY_COLOR,
    PRIORITY_MARK,
    file_type_label,
    folder_store_row,
    hold_store_row,
    index_store_row,
    library_tree,
    row_mark,
    row_menu_labels,
    then_index_action,
)
from nomad_turbo.session import Session
from nomad_turbo.slots import (
    clamp_embed_chunks,
    host_memory_mib,
    install_embed_chunks,
    pick_gpu,
    safe_call_chunks,
)
from nomad_turbo.tips import TIPS

log = logging.getLogger("nomad_turbo")

# Anyone running this window already has Project NOMAD. The corner link opens this repo.
REPO_URL = "https://github.com/Vibe-Dragon/Nomad-TURBO-Index"
NOMAD_REPO_URL = "https://github.com/Crosstalk-Solutions/project-nomad"
# The README's tested-version sentence uses this same fact.
TESTED_NOMAD_VERSION = "1.35.1"


def about_facts() -> dict[str, str]:
    """Copy for the About page. The window and the tests both read this."""
    tested = f"This version was tested with Project NOMAD {TESTED_NOMAD_VERSION}."
    body = "\n\n".join(
        (
            "Vibe-Dragon",
            "MIT License. Copyright 2026 Vibe-Dragon.",
            "I update this when I get to it. It can fall behind Project NOMAD.",
            (
                "I am not responsible if Turbo messes up your index, "
                "your search results, or anything else on your machine."
            ),
        )
    )
    return {
        "version": __version__,
        "version_line": f"Nomad TURBO Index {__version__}",
        "tested": tested,
        "turbo_repo": REPO_URL,
        "nomad_repo": NOMAD_REPO_URL,
        "turbo_label": "Nomad TURBO Index",
        "nomad_label": "Project NOMAD",
        "body": body,
    }


def window_close_action(indexing: bool) -> str:
    """A running turbo stays up when the window closes. The tray opens it again."""
    return "hide" if indexing else "quit"


def describe_index_controls(app) -> str:
    """Header texts, order choices, Measure, and the row menu. No indexing starts."""
    lines = ["headers:"]
    for column in app.file_tree.get_columns():
        lines.append(column.get_title() or "")
    lines.append("order:")
    model = app.order_combo.get_model()
    iterator = model.get_iter_first()
    while iterator is not None:
        lines.append(str(model.get_value(iterator, 0)))
        iterator = model.iter_next(iterator)
    lines.append("measure:")
    lines.append(app.measure_button.get_label() or "")
    lines.append("then-index:")
    lines.append(app.then_index_button.get_label() or "")
    lines.append("hide:")
    lines.append(app.hide_finished.get_label() or "")
    lines.append("menu:")
    menu = app._file_menu("example.zim")
    for item in menu.get_children():
        lines.append(item.get_label() or "")
    app.session.set_priority("example.zim", True)
    lines.append("menu-priority:")
    marked = app._file_menu("example.zim")
    for item in marked.get_children():
        lines.append(item.get_label() or "")
    app.session.set_priority("example.zim", False)
    return "\n".join(lines) + "\n"


def _hex_rgb(color: str) -> tuple[float, float, float]:
    text = (color or "").strip().lstrip("#")
    if len(text) != 6:
        text = "1d4e89"
    try:
        value = int(text, 16)
    except ValueError:
        value = 0x1D4E89
    return ((value >> 16) & 255) / 255.0, ((value >> 8) & 255) / 255.0, (value & 255) / 255.0


def _round_box(cr, left: float, top: float, width: float, height: float, radius: float) -> None:
    import math

    radius = min(radius, width / 2.0, height / 2.0)
    if radius <= 0:
        cr.rectangle(left, top, width, height)
        return
    cr.new_sub_path()
    cr.arc(left + width - radius, top + radius, radius, -math.pi / 2, 0)
    cr.arc(left + width - radius, top + height - radius, radius, 0, math.pi / 2)
    cr.arc(left + radius, top + height - radius, radius, math.pi / 2, math.pi)
    cr.arc(left + radius, top + radius, radius, math.pi, math.pi * 1.5)
    cr.close_path()


def pass_percent_text(value: int) -> str:
    """The number drawn on a file bar. OCR uses the same bar as the other passes."""
    return f"{max(0, min(100, int(value)))}%"


def paint_pass_bar(cr, area, value: int, color: str) -> None:
    """One file bar. The color is the pass. An empty color uses the ordinary blue."""
    if area is None or cr is None:
        return
    width = int(getattr(area, "width", 0) or 0)
    height = int(getattr(area, "height", 0) or 0)
    if width < 4 or height < 4:
        return
    left = float(getattr(area, "x", 0) or 0) + 2
    wide = float(width - 4)
    bar_h = float(min(14, height - 2))
    top = float(getattr(area, "y", 0) or 0) + (height - bar_h) / 2.0
    red, green, blue = _hex_rgb(color or "#1d4e89")
    _round_box(cr, left, top, wide, bar_h, 3)
    cr.set_source_rgba(red, green, blue, 0.22)
    cr.fill()
    shown = max(0, min(100, int(value)))
    fraction = shown / 100.0
    if fraction > 0:
        _round_box(cr, left, top, max(6.0, min(wide, wide * fraction)), bar_h, 3)
        cr.set_source_rgb(red, green, blue)
        cr.fill()
    label = pass_percent_text(shown)
    cr.select_font_face("Sans")
    cr.set_font_size(max(9.0, min(12.0, bar_h - 1.0)))
    extents = cr.text_extents(label)
    text_x = left + max(0.0, (wide - extents.width) / 2.0 - extents.x_bearing)
    text_y = top + (bar_h - extents.height) / 2.0 - extents.y_bearing
    cr.set_source_rgb(1, 1, 1)
    cr.move_to(text_x, text_y)
    cr.show_text(label)


def _pass_renderer(gtk):
    """A progress cell that can take a different color on each row."""
    kind = getattr(_pass_renderer, "kind", None)
    if kind is None:
        from gi.repository import GObject

        class PassBar(gtk.CellRenderer):
            __gtype_name__ = "NomadTurboPassBar"
            value = GObject.Property(type=int, default=0)
            pass_color = GObject.Property(type=str, default="")

            def __init__(self):
                gtk.CellRenderer.__init__(self)
                self.set_property("xpad", 3)
                self.set_property("ypad", 3)

            def do_get_preferred_height(self, widget):
                return 16, 20

            def do_render(self, cr, widget, background_area, cell_area, flags):
                paint_pass_bar(cr, cell_area, int(self.value or 0), str(self.pass_color or ""))

        _pass_renderer.kind = PassBar
        kind = PassBar
    return kind()


def index_store_types() -> tuple:
    """Article, chunk, and size columns are 64-bit. A ZIM over 2 GB does not fit in 32."""
    from gi.repository import GObject

    wide = GObject.TYPE_INT64
    return (
        str,
        str,
        str,
        str,
        str,
        str,
        int,
        str,
        str,
        str,
        str,
        str,
        str,
        wide,
        wide,
        wide,
        wide,
        wide,
        int,
    )


def repair_store_types() -> tuple:
    """The point-count sort column is 64-bit."""
    from gi.repository import GObject

    wide = GObject.TYPE_INT64
    return (str, str, str, str, int, int, str, str, str, str, wide, int)


def reorder_children(store, parent, wanted_order, key_column: int) -> None:
    """Put one level into wanted_order.

    Gtk.ListStore has reorder(). Gtk.TreeStore on GTK 3 does not. move_before
    does the same job, and a TreeIter stays valid across those moves.
    """
    by_key = {}
    iterator = store.iter_children(parent)
    while iterator is not None:
        by_key[store.get_value(iterator, key_column)] = iterator
        iterator = store.iter_next(iterator)
    successor = None
    for key in reversed(wanted_order):
        iterator = by_key.get(key)
        if iterator is None:
            continue
        store.move_before(iterator, successor)
        successor = iterator


class _MemoryLog(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: deque[tuple[int, int, str]] = deque(maxlen=LOG_PANE_LINES)
        self.seq = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            return
        self.seq += 1
        self.lines.append((self.seq, record.levelno, msg))

    def snapshot(self) -> list[tuple[int, int, str]]:
        self.acquire()
        try:
            return list(self.lines)
        finally:
            self.release()

    def since(self, seq: int) -> list[tuple[int, int, str]]:
        """Lines newer than seq. Empty when the pane is already caught up."""
        self.acquire()
        try:
            if not self.lines or self.lines[-1][0] <= seq:
                return []
            return [item for item in self.lines if item[0] > seq]
        finally:
            self.release()


def _configure_logging(memory: _MemoryLog, install) -> Path:
    from nomad_turbo.logbook import DailyLogHandler, daily_log_path, prepare_collection_logs

    directory = state_dir()
    failed = None
    try:
        prepare_collection_logs(directory, install)
    except Exception as exc:
        failed = exc
    path = daily_log_path(directory)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = DailyLogHandler(directory)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)
    stream.setFormatter(formatter)
    memory.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stream)
    root.addHandler(memory)
    log.setLevel(logging.DEBUG)
    if failed is not None:
        log.warning("could not align log files with the knowledge index: %s", failed)
    return path


def open_window() -> int:
    import gi

    gi.require_version("Gtk", "3.0")
    from gi.repository import GLib, Gtk

    # The taskbar matches this name to nomad-turbo.desktop. Set it before GTK opens the display.
    GLib.set_prgname("nomad-turbo")
    GLib.set_application_name("Nomad TURBO Index")

    if not Gtk.init_check():
        print("could not open the turbo window")
        write_shortcut(False)
        return 1

    try:
        from nomad_turbo.icons import write_icons

        Gtk.Window.set_default_icon_from_file(str(write_icons(icon_dir())["indexing"]))
    except Exception:
        log.warning("could not set the window icon")

    install = load_install()
    memory = _MemoryLog()
    log_path = _configure_logging(memory, install)
    gpu = pick_gpu(gpu_rows())
    host = host_memory_mib()
    session = Session(
        install_embed_chunks(
            install,
            gpu_total_mib=gpu[1] if gpu else None,
            host_total_mib=host[0] if host else 0,
            cpu_count=os.cpu_count() or 1,
        )
    )
    session.set_keep_control(bool(getattr(install, "keep_control", False)))
    app = _Window(install, session, memory, log_path, Gtk, GLib)
    signal.signal(signal.SIGTERM, app.on_signal)
    signal.signal(signal.SIGINT, app.on_signal)
    signal.signal(signal.SIGUSR1, app.on_raise)
    # Opening the window does not start indexing. The desktop icon stays the app icon.
    write_shortcut(False)
    try:
        app.run()
        return 0
    finally:
        write_shortcut(False)
        _clear_own_pid()


def _clear_own_pid() -> None:
    path = state_dir() / "pid"
    try:
        if path.exists() and path.read_text().strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        return


def _wrap_box(*, spacing: int = 8, row_spacing: int = 4):
    """A horizontal row that wraps. Its own minimum height is a single row."""
    import gi

    gi.require_version("Gtk", "3.0")
    from gi.repository import Gdk, Gtk

    if not hasattr(_wrap_box, "kind"):

        class WrapBox(Gtk.Container):
            def __init__(self, spacing, row_spacing):
                super().__init__()
                self.spacing = int(spacing)
                self.row_spacing = int(row_spacing)
                self._kids: list = []
                self.set_has_window(False)
                self.set_hexpand(True)
                self.set_halign(Gtk.Align.FILL)
                self.set_valign(Gtk.Align.START)

            def do_add(self, widget):
                self._kids.append(widget)
                widget.set_parent(self)
                self.queue_resize()

            def do_remove(self, widget):
                if widget in self._kids:
                    self._kids.remove(widget)
                widget.unparent()
                self.queue_resize()

            def do_forall(self, _include_internals, callback):
                for child in list(self._kids):
                    callback(child)

            def do_get_request_mode(self):
                return Gtk.SizeRequestMode.HEIGHT_FOR_WIDTH

            def do_get_preferred_width(self):
                minimum = 0
                natural = 0
                seen = 0
                for child in self._kids:
                    if not child.get_visible():
                        continue
                    child_min, child_nat = child.get_preferred_width()
                    minimum = max(minimum, int(child_min))
                    natural += int(child_nat)
                    seen += 1
                if seen:
                    natural += self.spacing * (seen - 1)
                return minimum, max(minimum, natural)

            def do_get_preferred_height(self):
                height = 0
                for child in self._kids:
                    if not child.get_visible():
                        continue
                    child_min, _child_nat = child.get_preferred_height()
                    height = max(height, int(child_min))
                return height, height

            def do_get_preferred_height_for_width(self, width):
                height = self._layout(int(width), False, 0, 0)
                return height, height

            def do_get_preferred_width_for_height(self, _height):
                return self.do_get_preferred_width()

            def do_size_allocate(self, allocation):
                self.set_allocation(allocation)
                self._layout(int(allocation.width), True, int(allocation.x), int(allocation.y))

            def _child_box(self, child, row_width: int) -> tuple[int, int]:
                child_min, child_nat = child.get_preferred_width()
                minimum = max(0, int(child_min))
                natural = max(minimum, int(child_nat))
                width = natural if natural <= row_width else minimum
                height, _natural_height = child.get_preferred_height_for_width(max(width, 1))
                if int(height) <= 0:
                    height, _natural_height = child.get_preferred_height()
                return width, max(int(height), 0)

            def _layout(self, row_width: int, allocate: bool, origin_x: int, origin_y: int) -> int:
                x = origin_x
                y = origin_y
                row_h = 0
                used = False
                limit = origin_x + max(0, row_width)
                for child in self._kids:
                    if not child.get_visible():
                        continue
                    width, height = self._child_box(child, max(row_width, 1))
                    if used and x > origin_x and x + width > limit:
                        x = origin_x
                        y += row_h + self.row_spacing
                        row_h = 0
                    if allocate:
                        rect = Gdk.Rectangle()
                        rect.x = int(x)
                        rect.y = int(y)
                        rect.width = max(int(width), 0)
                        rect.height = max(int(height), 0)
                        child.size_allocate(rect)
                    x += width + self.spacing
                    row_h = max(row_h, height)
                    used = True
                if not used:
                    return 0
                return (y + row_h) - origin_y

        _wrap_box.kind = WrapBox
    return _wrap_box.kind(spacing, row_spacing)


class _Window:
    def __init__(self, install, session: Session, memory: _MemoryLog, log_path: Path, gtk, glib) -> None:
        self.install = install
        self.session = session
        self.file_stats = FileStats(state_dir() / STATS_NAME)
        try:
            self.file_stats.load()
        except Exception:
            log.exception("could not read saved file stats")
        if hasattr(session, "bind_file_stats"):
            session.bind_file_stats(self.file_stats)
        self.memory = memory
        self.log_path = log_path
        self.gtk = gtk
        self.glib = glib
        self.thread: threading.Thread | None = None
        self.delete_thread: threading.Thread | None = None
        self.repair_thread: threading.Thread | None = None
        self.optimize_thread: threading.Thread | None = None
        self.optimize_starting = False
        self._optimize_pause = None
        self._optimize_live = None
        self._optimize_code = None
        self._tune_hold = None
        self._pace_memory: dict = {}
        self._tune_applying = False
        self._tune_logged = ""
        self.repair_stop = threading.Event()
        self.repair_board = None
        self.repair_starting = False
        self._repair_told = False
        self.qdrant_stop = threading.Event()
        self.qdrant_text = "Checking Qdrant…"
        self.qdrant_ready = False
        self.collection_missing = False
        self.qdrant_vectors = None
        self._qdrant_error = None
        self._qdrant_problem = ""
        self._followed: dict[int, str] = {}
        self._column_manual: dict[str, bool] = {}
        self._column_width: dict[str, int] = {}
        self._columns_by_key: dict[str, list] = {}
        self._fit_meta: list[tuple] = []
        self._sizing = False
        self._open_folders: set[str] = set()
        self._folder_iters: dict[str, object] = {}
        self._file_iters: dict[str, object] = {}
        self._folder_by_key: dict[str, object] = {}
        self._root_folders: tuple = ()
        self._shown_book = None
        self._shown_epoch = None
        self._shown_hide = None
        self._shown_priority = None
        self._tree_dirty = False
        self._running_source = ""
        self._tree_indexing = ""
        self._ocr_pace = OcrPace()
        self._pane_lock = False
        self._then_guard = False
        self._choice_guard = False
        self._log_away = False
        self._log_pinning = 0
        self._log_user = False
        self._log_split_ready = False
        self._log_settle = None
        self._log_placed_for = 0
        self._host_split_ready = False
        self._host_settle = None
        self._pane_styled = False
        self.layout_timer = None
        self._frame_timer = None
        self._frame_applied = False
        # Ignore the window manager's first placement. Map applies the saved frame.
        self._frame_ready_at = time.monotonic() + 10**9
        self._host_guard = False
        self.scanning = False
        self.rows_ready = False
        self.last_seq = 0
        self.filter_dirty = False
        self.live_gpu = pick_gpu(gpu_rows())
        memory = host_memory_mib()
        self.host_total_mib = memory[0] if memory else 0
        self.cpu_count = os.cpu_count() or 1
        self.tick = 0
        self.save_timer = None
        self.hidden = False
        self.raise_requested = False
        self.started_at: float | None = None
        self._finish_told = False
        self.indicator = None
        self._tray_status_enum = None

        window = gtk.Window(title=f"Nomad TURBO Index {__version__}")
        window.set_resizable(True)
        # 1180 by 820 until a size has been saved. A small floor. Portion
        # minimums stay small so this does not pin the window.
        opening = restored_frame(
            getattr(install, "window_x", None),
            getattr(install, "window_y", None),
            int(getattr(install, "window_width", 0) or 0),
            int(getattr(install, "window_height", 0) or 0),
            default_width=1180,
            default_height=820,
            floor_width=480,
            floor_height=280,
            maximized=bool(getattr(install, "window_max", False)),
        )
        window.set_default_size(opening.width, opening.height)
        window.set_size_request(480, 280)
        if opening.x is None or opening.y is None:
            window.set_position(gtk.WindowPosition.CENTER)
        else:
            window.move(opening.x, opening.y)
        try:
            window.set_wmclass("nomad-turbo", "nomad-turbo")
        except Exception:
            log.debug("could not set the window class", exc_info=True)
        self.window = window
        window.connect("delete-event", self.on_delete)
        window.connect("configure-event", self._on_configure)
        window.connect("map-event", self._on_map)
        window.connect("destroy", self._on_destroy)

        root = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=8)
        root.set_border_width(12)
        window.add(root)
        self.root = root

        self.stack = gtk.Stack()
        self.stack.set_transition_type(gtk.StackTransitionType.NONE)
        self.qdrant_label = gtk.Label(label=self.qdrant_text)
        self.qdrant_label.set_xalign(0)
        self.qdrant_label.set_line_wrap(True)
        self.qdrant_label.set_max_width_chars(110)
        self.qdrant_label.set_selectable(True)
        self.qdrant_label.set_tooltip_text(TIPS["qdrant"])
        switcher = gtk.StackSwitcher()
        switcher.set_stack(self.stack)
        switcher.set_halign(gtk.Align.START)
        header = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        header.set_hexpand(True)
        header.pack_start(switcher, False, False, 0)
        self.host_button = gtk.ToggleButton(label="Host system graphs")
        self.host_button.set_tooltip_text(TIPS["host"])
        self.host_button.set_valign(gtk.Align.CENTER)
        self.host_button.set_active(bool(getattr(install, "host_open", False)))
        self.host_button.connect("toggled", self.on_host)
        header.pack_start(self.host_button, False, False, 0)
        order_label = gtk.Label(label="Order")
        order_label.set_tooltip_text(TIPS["order"])
        order_label.set_valign(gtk.Align.CENTER)
        self.order_combo = gtk.ComboBoxText()
        self.order_combo.set_tooltip_text(TIPS["order"])
        self.order_combo.set_valign(gtk.Align.CENTER)
        for choice, label in ORDER_CHOICES:
            self.order_combo.append(choice, label)
        saved_order = normalize_order(getattr(install, "index_order", None))
        self.session.set_order(saved_order)
        self.order_combo.set_active_id(saved_order)
        self.order_combo.connect("changed", self.on_order)
        self.hide_finished = gtk.ToggleButton(label="Hide finished")
        self.hide_finished.set_active(bool(getattr(install, "hide_finished", False)))
        self.hide_finished.set_halign(gtk.Align.START)
        self.hide_finished.set_valign(gtk.Align.CENTER)
        self.hide_finished.set_tooltip_text(TIPS["hide_finished"])
        self.hide_finished.connect("toggled", self.on_hide_finished)
        header.pack_start(order_label, False, False, 0)
        header.pack_start(self.order_combo, False, False, 0)
        header.pack_start(self.hide_finished, False, False, 0)
        self.version_label = gtk.Label(label=__version__)
        self.version_label.set_halign(gtk.Align.END)
        self.version_label.set_tooltip_text(TIPS["version"])
        project = gtk.LinkButton.new_with_label(REPO_URL, "Nomad TURBO Index")
        project.set_halign(gtk.Align.END)
        project.set_tooltip_text(TIPS["project"])
        # The full title stays at the normal window size. A short floor can ellipsize it.
        from gi.repository import Pango

        link_label = project.get_child()
        if link_label is not None:
            link_label.set_ellipsize(Pango.EllipsizeMode.END)
            link_label.set_width_chars(8)
        header.pack_end(project, False, False, 0)
        header.pack_end(self.version_label, False, False, 0)
        # Order names are wide. A short window clips this row instead of
        # growing past the floor the controls were fitted to.
        header_scroll = gtk.ScrolledWindow()
        header_scroll.set_policy(gtk.PolicyType.AUTOMATIC, gtk.PolicyType.NEVER)
        header_scroll.set_shadow_type(gtk.ShadowType.NONE)
        header_scroll.set_min_content_width(1)
        header_scroll.set_propagate_natural_width(False)
        header_scroll.add(header)
        root.pack_start(header_scroll, False, False, 0)
        # Above the pages, not in the log split, so this line does not ride the log bar.
        root.pack_start(self.qdrant_label, False, False, 0)
        index = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=8)
        index.set_vexpand(True)
        self.stack.add_titled(index, "index", "Index")
        self.index_page = index

        self.status = gtk.Label(label="Scanning the library…")
        self.status.set_xalign(0)
        self.status.set_line_wrap(True)
        self.status.set_tooltip_text(TIPS["legend"])
        index.pack_start(self.status, False, False, 0)

        from gi.repository import Pango

        self.status.set_ellipsize(Pango.EllipsizeMode.END)
        self.status.set_lines(2)
        # Hidden until indexing actually waits. A shown bar would steal list space
        # from the short-window layout, and Qdrant publishes no percent to draw.
        self.index_optimize = gtk.ProgressBar()
        self.index_optimize.set_show_text(True)
        self.index_optimize.set_text("Not waiting on an optimization.")
        self.index_optimize.set_hexpand(True)
        self.index_optimize.set_no_show_all(True)
        self.index_optimize.set_tooltip_text(TIPS["index_optimize"])
        index.pack_start(self.index_optimize, False, False, 0)

        def headed(title: str, tip: str, cell, **props):
            column = gtk.TreeViewColumn(title, cell, **props)
            label = gtk.Label(label=title)
            label.set_tooltip_text(tip)
            label.show()
            column.set_widget(label)
            return column

        self.store = gtk.TreeStore(*index_store_types())
        self.now_store = gtk.ListStore(*index_store_types())
        # Headers stay on the current-file row. The folder tree scrolls under it.
        self.now_tree = self._make_index_tree(headed, self.now_store, headers=True, fit=False)
        self.now_tree.set_hexpand(True)
        self.file_tree = self._make_index_tree(headed, self.store, headers=False, fit=True)
        self.file_tree.set_enable_tree_lines(True)
        # Fill a folder before it opens, so the arrow never shows an empty row.
        self.file_tree.connect("test-expand-row", self._on_test_expand)
        self.file_tree.connect("row-expanded", self._on_folder_expanded)
        self.file_tree.connect("row-collapsed", self._on_folder_collapsed)
        scrolled = gtk.ScrolledWindow()
        scrolled.set_policy(gtk.PolicyType.AUTOMATIC, gtk.PolicyType.AUTOMATIC)
        scrolled.set_overlay_scrolling(True)
        scrolled.add(self.file_tree)
        # The header sets the real minimum. A 48px content floor makes the
        # paned slide the list out of a short pane instead of shrinking it.
        scrolled.set_min_content_height(1)
        scrolled.set_vexpand(True)
        scrolled.connect("size-allocate", self._align_current_row)
        self.index_list = scrolled
        # Column minimums are wider than a narrow main pane. The pinned row
        # scrolls with the list so those columns do not push the controls off.
        now_scroll = gtk.ScrolledWindow()
        now_scroll.set_policy(gtk.PolicyType.EXTERNAL, gtk.PolicyType.NEVER)
        now_scroll.set_shadow_type(gtk.ShadowType.NONE)
        now_scroll.set_min_content_width(1)
        now_scroll.set_propagate_natural_width(False)
        now_scroll.set_propagate_natural_height(True)
        now_scroll.set_hexpand(True)
        now_scroll.add(self.now_tree)
        self.now_scroll = now_scroll
        # Shared rows keep this cluster short. The list above it gets the spare height.
        lower = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=4)

        self.overall = gtk.ProgressBar()
        self.overall.set_show_text(True)
        self.overall.set_text("Scanning…")
        self.overall.set_hexpand(True)
        self.overall.set_valign(gtk.Align.CENTER)
        self.overall.set_tooltip_text(TIPS["library_bar"])
        lower.pack_start(self.overall, False, False, 0)
        batch_label = gtk.Label(label="Embed batch")
        batch_label.set_tooltip_text(TIPS["slider"])
        batch_label.set_valign(gtk.Align.CENTER)
        self.chunk_entry = gtk.Entry()
        self.chunk_entry.set_width_chars(6)
        self.chunk_entry.set_text(str(session.get_chunks()))
        self.chunk_entry.set_valign(gtk.Align.CENTER)
        self.chunk_entry.set_tooltip_text(TIPS["slider"])
        self.chunk_entry.connect("changed", self.on_chunks)
        self.recommend_button = gtk.Button(label="Recommended")
        self.recommend_button.set_valign(gtk.Align.CENTER)
        self.recommend_button.set_tooltip_text(TIPS["recommend"])
        self.recommend_button.connect("clicked", self.on_recommend)
        self.server_label = gtk.Label(label=embed_server_text(0))
        self.server_label.set_valign(gtk.Align.CENTER)
        self.server_label.set_halign(gtk.Align.START)
        self.server_label.set_xalign(0)
        self.server_label.set_tooltip_text(TIPS["usage"])

        self.stats = gtk.Label(label="")
        self.stats.set_xalign(0)
        self.stats.set_hexpand(True)
        self.stats.set_line_wrap(True)
        self.stats.set_lines(2)
        self.stats.set_ellipsize(Pango.EllipsizeMode.END)
        self.stats.set_selectable(True)
        self.stats.set_tooltip_text(TIPS["stats"])
        lower.pack_start(self.stats, False, False, 0)

        self.tune_label = gtk.Label(label="A recommendation shows after indexing finishes a few embed calls.")
        self.tune_label.set_xalign(0)
        self.tune_label.set_hexpand(True)
        self.tune_label.set_ellipsize(Pango.EllipsizeMode.END)
        self.tune_label.set_tooltip_text(TIPS["tune"])
        lower.pack_start(self.tune_label, False, False, 0)

        self.keep_check = gtk.ToggleButton(label="Stay on")
        self.keep_check.set_active(bool(getattr(install, "keep_control", False)))
        self.keep_check.set_halign(gtk.Align.START)
        self.keep_check.set_tooltip_text(
            "Stay on after the last file, and index new files as they arrive. " + TIPS["keep"]
        )
        self.keep_check.connect("toggled", self.on_keep)

        self.start_button = gtk.Button(label="Start indexing")
        self.delete_button = gtk.Button(label="Delete index")
        self.rescan_button = gtk.Button(label="Rescan")
        self.stop_button = gtk.Button(label="Stop")
        self.export_button = gtk.Button(label="Export log")
        self.measure_button = gtk.Button(label="Measure")
        self.measure_button.set_tooltip_text(TIPS["measure"])
        self.measure_button.set_valign(gtk.Align.CENTER)
        self.measure_button.connect("clicked", lambda *_args: self.on_measure())
        self.then_index_button = gtk.ToggleButton(label="Then index")
        self.then_index_button.set_tooltip_text(TIPS["then_index"])
        self.then_index_button.set_valign(gtk.Align.CENTER)
        self.then_index_button.connect("toggled", self.on_then_index)
        self.start_button.connect("clicked", lambda *_: self.on_start(False))
        self.delete_button.connect("clicked", self.on_delete_index)
        self.rescan_button.connect("clicked", lambda *_: self.begin_scan())
        self.stop_button.connect("clicked", lambda *_: self.session.stop.set())
        self.export_button.connect("clicked", self.on_export)
        self.start_tip = TIPS["start"]
        self.start_button.set_tooltip_text(TIPS["start"])
        self.stop_button.set_tooltip_text(TIPS["stop"])
        self.rescan_button.set_tooltip_text(TIPS["rescan"])
        self.export_button.set_tooltip_text(TIPS["export"])
        self.export_button.set_valign(gtk.Align.CENTER)
        self.delete_tip = TIPS["delete"]
        self.delete_button.set_tooltip_text(TIPS["delete"])
        self.delete_button.get_style_context().add_class("destructive-action")
        # Run actions, then Measure, then OCR quality, then the embed batch. Delete stays apart.
        self.run_group = self._linked(
            self.start_button,
            self.stop_button,
            self.rescan_button,
            self.keep_check,
        )
        self.measure_group = self._linked(self.measure_button, self.then_index_button)
        chosen_quality = normalize_ocr_quality(getattr(install, "ocr_quality", "medium"))
        self.session.set_ocr_quality(chosen_quality)
        self._ocr_guard = False
        self.ocr_low = gtk.ToggleButton(label="Low")
        self.ocr_medium = gtk.ToggleButton(label="Medium")
        self.ocr_high = gtk.ToggleButton(label="High")
        for name, button in (
            ("low", self.ocr_low),
            ("medium", self.ocr_medium),
            ("high", self.ocr_high),
        ):
            button.set_tooltip_text(TIPS["ocr_quality"])
            button.set_active(name == chosen_quality)
        for name, button in (
            ("low", self.ocr_low),
            ("medium", self.ocr_medium),
            ("high", self.ocr_high),
        ):
            button.connect("toggled", self.on_ocr_quality, name)
        ocr_caption = gtk.Label(label="OCR")
        ocr_caption.set_tooltip_text(TIPS["ocr_quality"])
        self.ocr_group = self._flow(
            ocr_caption,
            self._linked(self.ocr_low, self.ocr_medium, self.ocr_high),
        )
        self.tune_auto = gtk.ToggleButton(label="Auto")
        self.tune_auto.set_tooltip_text(TIPS["tune_auto"])
        self.tune_auto.set_active(bool(getattr(install, "tune_auto", False)))
        self.session.set_tune_auto(bool(getattr(install, "tune_auto", False)))
        self.tune_auto.connect("toggled", self.on_tune_auto)
        self.chunk_entry.set_sensitive(not self.tune_auto.get_active())
        self.embed_group = self._flow(
            self.tune_auto,
            batch_label,
            self.chunk_entry,
            self.recommend_button,
            self.server_label,
        )
        delete_pad = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=0)
        delete_pad.set_margin_start(16)
        delete_pad.pack_start(self.delete_button, False, False, 0)
        self.delete_pad = delete_pad
        buttons = self._flow(
            self.run_group,
            self.measure_group,
            self.ocr_group,
            self.embed_group,
            delete_pad,
        )
        lower.pack_start(buttons, False, False, 0)
        self.index_buttons = buttons
        self.index_controls = lower
        # The list takes extra height. The controls stay at the bottom of the
        # page, against the bar above the log.
        index.pack_start(self.now_scroll, False, False, 0)
        index.pack_start(self.index_list, True, True, 0)
        index.pack_start(self.index_controls, False, False, 0)
        self._build_repair_page(headed)
        self._build_settings_page()
        self._build_about_page()

        self.error_check = gtk.ToggleButton(label="Errors")
        self.info_check = gtk.ToggleButton(label="Info")
        self.debug_check = gtk.ToggleButton(label="Debug")
        self.error_check.set_active(True)
        self.info_check.set_active(True)
        self.debug_check.set_active(False)
        self.error_check.set_tooltip_text(TIPS["errors"])
        self.info_check.set_tooltip_text(TIPS["info"])
        self.debug_check.set_tooltip_text(TIPS["debug"])
        self.info_check.connect("toggled", self.on_filter)
        self.debug_check.connect("toggled", self.on_filter)
        self.error_check.connect("toggled", self.on_filter)
        self.follow_check = gtk.ToggleButton(label="Follow")
        self.follow_check.set_active(bool(getattr(install, "log_follow", True)))
        self.follow_check.set_tooltip_text(TIPS["follow"])
        self.follow_check.connect("toggled", self.on_follow)
        self.log_filters = self._linked(
            self.error_check,
            self.info_check,
            self.debug_check,
            self.follow_check,
        )
        # The log row is the first thing under the bar, so it stays against it.
        # The log text fills the rest. Neither side sets a tall size request,
        # so the window edges stay free.
        log_open = bool(getattr(install, "log_open", True))
        self.log_arrow = gtk.Image.new_from_icon_name(
            "pan-down-symbolic" if log_open else "pan-end-symbolic",
            gtk.IconSize.BUTTON,
        )
        self.log_open_button = gtk.ToggleButton()
        self.log_open_button.set_image(self.log_arrow)
        self.log_open_button.set_relief(gtk.ReliefStyle.NONE)
        self.log_open_button.set_focus_on_click(False)
        self.log_open_button.set_active(log_open)
        self.log_open_button.set_tooltip_text(TIPS["log"])
        self.log_open_button.connect("toggled", self.on_log)
        log_title = gtk.Label(label="Log")
        log_title.set_tooltip_text(TIPS["log"])
        log_title_box = gtk.EventBox()
        log_title_box.add(log_title)
        log_title_box.connect("button-press-event", self._on_log_title)
        self.log_help_button = gtk.Button(label="Log help")
        self.log_help_button.set_tooltip_text(TIPS["log_help"])
        self.log_help_button.set_valign(gtk.Align.CENTER)
        self.log_help_button.connect("clicked", self.on_log_help)
        log_title_group = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=4)
        log_title_group.pack_start(self.log_open_button, False, False, 0)
        log_title_group.pack_start(log_title_box, False, False, 0)
        # Log help stays to the right of Export log.
        self.log_export_group = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        self.log_export_group.pack_start(self.export_button, False, False, 0)
        self.log_export_group.pack_start(self.log_help_button, False, False, 0)
        log_header = self._flow(log_title_group, self.log_filters, self.log_export_group)
        self.log_header = log_header
        self.log_view = gtk.TextView()
        self.log_view.set_editable(False)
        self.log_view.set_monospace(True)
        self.log_view.set_wrap_mode(gtk.WrapMode.WORD_CHAR)
        self.log_buffer = self.log_view.get_buffer()
        log_scroll = gtk.ScrolledWindow()
        log_scroll.set_policy(gtk.PolicyType.AUTOMATIC, gtk.PolicyType.AUTOMATIC)
        log_scroll.set_min_content_height(48)
        log_scroll.set_vexpand(True)
        log_scroll.add(self.log_view)
        self.log_scroll = log_scroll
        self._watch_log_scroll()
        self.log_filters.set_no_show_all(not log_open)
        self.log_filters.set_visible(log_open)
        log_scroll.set_no_show_all(not log_open)
        log_scroll.set_visible(log_open)
        self.stack.set_vexpand(True)
        log_box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=4)
        log_box.set_vexpand(True)
        log_box.pack_start(log_header, False, False, 0)
        log_box.pack_start(log_scroll, True, True, 0)
        self.log_box = log_box
        self.content_pane = self._make_pane()
        # Extra window height goes to the pages. Inside a page the list takes
        # it and the controls stay on the bottom, against this bar. The log
        # row is the top of the other side. The log can shrink. The pages do
        # not shrink below the controls.
        self.content_pane.pack1(self.stack, True, False)
        self.content_pane.pack2(log_box, False, True)
        self._track_log_pane(self.content_pane)
        from nomad_turbo.host_window import HostPanel

        self.host_panel = HostPanel(install, self.gtk, self.glib)
        self.host_pane = self._make_pane(gtk.Orientation.HORIZONTAL)
        self.host_pane.pack1(self.content_pane, True, True)
        self._track_host_pane(self.host_pane)
        if bool(getattr(install, "host_open", False)):
            self.host_pane.pack2(self.host_panel.widget, False, True)
        root.pack_start(self.host_pane, True, True, 0)
        self._build_tray()
        self.refresh_usage()

    def _build_tray(self) -> None:
        try:
            import gi

            gi.require_version("AyatanaAppIndicator3", "0.1")
            from gi.repository import AyatanaAppIndicator3
        except (ImportError, ValueError):
            log.warning("no system tray library, closing the window will stop turbo")
            return
        self._load_icons()
        if not self._idle_icon:
            log.warning("tray has no icons, closing the window will stop turbo")
            return
        indicator = AyatanaAppIndicator3.Indicator.new(
            "nomad-turbo",
            "nomad-turbo",
            AyatanaAppIndicator3.IndicatorCategory.APPLICATION_STATUS,
        )
        indicator.set_icon_full(self._idle_icon, "Nomad TURBO Index")
        indicator.set_attention_icon_full(self._index_icon, "Nomad TURBO Index is running")
        indicator.set_title("Nomad TURBO Index")
        indicator.set_status(AyatanaAppIndicator3.IndicatorStatus.ACTIVE)
        menu = self.gtk.Menu()
        show = self.gtk.MenuItem(label="Show window")
        show.connect("activate", lambda *_args: self.present())
        menu.append(show)
        self.tray_status = self.gtk.MenuItem(label="Idle")
        self.tray_status.set_sensitive(False)
        menu.append(self.tray_status)
        stop = self.gtk.MenuItem(label="Stop indexing")
        stop.connect("activate", lambda *_args: self._stop_all())
        menu.append(stop)
        self.tray_stop = stop
        menu.show_all()
        # Opening the tray menu is the click. Bring the window up with it.
        menu.connect("show", lambda *_args: self.present())
        indicator.set_menu(menu)
        self.indicator = indicator
        self._tray_status_enum = AyatanaAppIndicator3.IndicatorStatus

    def drop_tray(self) -> None:
        """Hide the tray icon created with the window."""
        indicator = getattr(self, "indicator", None)
        self.indicator = None
        status = self._tray_status_enum
        if indicator is None or status is None:
            return
        try:
            indicator.set_status(status.PASSIVE)
        except Exception:
            return

    def _load_icons(self) -> None:
        self._idle_icon = ""
        self._index_icon = ""
        self._repair_icon = ""
        try:
            from nomad_turbo.desktop import icon_dir
            from nomad_turbo.icons import write_icons

            written = write_icons(icon_dir())
        except Exception:
            log.warning("could not write the turbo icons")
            return
        self._idle_icon = str(written["idle"])
        self._index_icon = str(written["indexing"])
        self._repair_icon = str(written["repair"])
        try:
            self.window.set_icon_from_file(self._index_icon)
        except Exception:
            log.debug("could not set the window icon", exc_info=True)

    def _tray_icon(self, indexing: bool, repairing: bool) -> str:
        if repairing and not indexing:
            return self._repair_icon
        if indexing:
            return self._index_icon
        return self._idle_icon

    def present(self) -> None:
        self.hidden = False
        self.window.show()
        self.window.present()
        if getattr(self, "host_button", None) is not None and self.host_button.get_active():
            self.host_panel.start()

    def run(self) -> None:
        self.window.show_all()
        self.begin_scan()
        threading.Thread(target=self._watch_qdrant, name="nomad-turbo-qdrant", daemon=True).start()
        self.glib.timeout_add(500, self.on_tick)
        self.gtk.main()

    def _watch_qdrant(self) -> None:
        from nomad_turbo import segment_guard
        from nomad_turbo.qdrant_status import indexing_qdrant_problem, probe_qdrant

        while not self.qdrant_stop.is_set():
            try:
                view = probe_qdrant(
                    self.install.qdrant_url,
                    self.install.collection,
                    storage=self.install.storage_host,
                    previous_vectors=self.qdrant_vectors,
                )
                self.qdrant_text = view.text
                self.qdrant_ready = view.ready
                self.collection_missing = bool(getattr(view, "collection_missing", False))
                from nomad_turbo.ocr_budget import note_qdrant

                note_qdrant(view.ready, view.collection_status)
                if view.vectors is not None:
                    self.qdrant_vectors = view.vectors
                try:
                    temp_open = segment_guard.optimization_open(self.install)
                except Exception:
                    temp_open = False
                code, bar, detail, pulse = segment_guard.describe_optimizer_state(
                    view.collection_status,
                    view.optimizer_error,
                    temp_open,
                )
                self._optimize_live = (bar, detail, pulse)
                if code != self._optimize_code:
                    self._optimize_code = code
                    level, line = segment_guard.optimize_watch_line(code, detail)
                    log.log(level, line)
                self._note_qdrant_problem(indexing_qdrant_problem(view))
            except Exception as exc:
                message = f"{exc.__class__.__name__}: {exc}"
                if message != self._qdrant_error:
                    self._qdrant_error = message
                    log.warning("qdrant status check failed: %s", message)
                self.qdrant_text = "Qdrant status could not be read. Not ready for changes."
                self.qdrant_ready = False
                self.collection_missing = False
                from nomad_turbo.ocr_budget import note_qdrant

                note_qdrant(False, "")
                self._note_qdrant_problem(f"Qdrant status could not be read. {message}")
            else:
                self._qdrant_error = None
                try:
                    from nomad_turbo.logbook import watch_collection_logs

                    watch_collection_logs(state_dir(), self.install)
                except Exception:
                    log.exception("could not check log files against the knowledge index")
            if self.qdrant_stop.wait(5):
                return

    def _note_qdrant_problem(self, problem: str) -> None:
        """One log line when a run hits a Qdrant problem, and one when it clears."""
        from nomad_turbo.qdrant_status import indexing_problem_line

        try:
            phase = self.session.view().get("phase") or ""
        except Exception:
            phase = ""
        line, self._qdrant_problem = indexing_problem_line(
            self._qdrant_problem,
            phase in ("waiting", "running"),
            problem,
        )
        if not line:
            return
        if line.startswith("Qdrant problem while indexing"):
            log.warning(line)
        else:
            log.info(line)

    def on_signal(self, _signum, _frame) -> None:
        self.qdrant_stop.set()
        self._stop_all()
        self.session.close_requested = True

    def _stop_all(self) -> None:
        self.session.stop.set()
        self.repair_stop.set()

    def _optimize_busy(self) -> bool:
        thread = getattr(self, "optimize_thread", None)
        starting = getattr(self, "optimize_starting", False)
        return bool(starting or (thread is not None and thread.is_alive()))

    def _work_alive(self) -> bool:
        indexing = self.thread is not None and self.thread.is_alive()
        deleting = self.delete_thread is not None and self.delete_thread.is_alive()
        repairing = self.repair_thread is not None and self.repair_thread.is_alive()
        return indexing or deleting or repairing or self.repair_starting or self._optimize_busy()

    def on_raise(self, _signum, _frame) -> None:
        self.raise_requested = True

    def on_host(self, button) -> None:
        if self._host_guard:
            return
        shown = bool(button.get_active())
        try:
            self._show_host(shown)
        except Exception:
            log.exception("could not show the host meters")
            self._host_guard = True
            button.set_active(False)
            self._host_guard = False
            shown = False
            try:
                self._show_host(False)
            except Exception:
                log.exception("could not hide the host meters")
        self.install.host_open = shown
        try:
            save_install(self.install)
        except Exception:
            log.exception("could not save the host setting")

    def _show_host(self, shown: bool) -> None:
        """Show or hide the side pane. The sampler runs only while it is shown."""
        widget = self.host_panel.widget
        pane = self.host_pane
        if shown:
            if widget.get_parent() is None:
                pane.pack2(widget, False, True)
            widget.show_all()
            pane._user_drag = False
            self._host_split_ready = False
            self.host_panel.start()
            self._place_host_split()
            return
        self.host_panel.stop()
        if widget.get_parent() is pane:
            pane.remove(widget)

    def _place_host_split(self) -> None:
        """Keep the saved host width. The pages keep the rest, down to the squeeze limit."""
        pane = getattr(self, "host_pane", None)
        if pane is None or not self.host_button.get_active():
            return
        width = int(pane.get_allocated_width() or 0)
        if width < 80:
            return
        saved = int(getattr(self.install, "host_width", 0) or 0)
        panel = saved if saved >= 180 else 320
        requested = width - panel
        position = host_bar_position(width, requested, main_min=self._main_squeeze())
        self._yield_host(width, self._main_squeeze())
        if abs(int(pane.get_position()) - position) <= 1:
            return
        self._pane_lock = True
        try:
            pane.set_position(int(position))
        finally:
            self._pane_lock = False

    def _mark_host_settled(self, width: int) -> bool:
        self._host_settle = None
        pane = getattr(self, "host_pane", None)
        if pane is None:
            return False
        current = int(pane.get_allocated_width() or 0)
        if abs(current - int(width)) <= 2:
            self._host_split_ready = True
        return False

    def _track_host_pane(self, pane) -> None:
        def on_size(_widget, allocation) -> None:
            if not self.host_button.get_active():
                return
            width = int(allocation.width or 0)
            if width < 80:
                return
            # A drag can leave the main side under the squeeze limit. Put it back.
            if getattr(pane, "_button_down", False) or getattr(pane, "_user_drag", False):
                self._host_split_ready = True
                self._clamp_host_bar(pane)
                return
            if self._host_split_ready:
                self._clamp_host_bar(pane)
                return
            self._place_host_split()
            if self._host_settle:
                self.glib.source_remove(self._host_settle)
            self._host_settle = self.glib.timeout_add(200, self._mark_host_settled, width)

        def on_release(_widget, event) -> bool:
            if getattr(event, "button", 1) == 1:
                pane._button_down = False
            return False

        def on_grab_broken(*_args) -> bool:
            pane._button_down = False
            return False

        def on_position(widget, _pspec) -> None:
            if self._pane_lock or not self.host_button.get_active():
                return
            width = int(widget.get_allocated_width() or 0)
            if width < 80:
                return
            position = self._clamp_host_bar(widget)
            if not getattr(widget, "_user_drag", False):
                return
            panel = width - position
            if panel < 1:
                return
            self.install.host_width = int(panel)
            self._save_layout_soon()

        pane.connect("size-allocate", on_size)
        pane.connect("button-press-event", self._mark_pane_drag)
        pane.connect("button-release-event", on_release)
        pane.connect("grab-broken-event", on_grab_broken)
        pane.connect("notify::position", on_position)

    def _screen_span(self) -> tuple[int, int]:
        screen = self.window.get_screen()
        if screen is None:
            return 0, 0
        return int(screen.get_width() or 0), int(screen.get_height() or 0)

    def _opening_place(self):
        width, height = self._screen_span()
        return restored_frame(
            getattr(self.install, "window_x", None),
            getattr(self.install, "window_y", None),
            int(getattr(self.install, "window_width", 0) or 0),
            int(getattr(self.install, "window_height", 0) or 0),
            default_width=1180,
            default_height=820,
            floor_width=480,
            floor_height=280,
            screen_width=width,
            screen_height=height,
            maximized=bool(getattr(self.install, "window_max", False)),
        )

    def _apply_frame(self) -> bool:
        """Put the saved size and place on the window after it is mapped.

        The first configure from the window manager is not the saved place.
        Applying again on idle beats that placement. The log bar is placed
        after this size, not at the short allocation that comes first.
        """
        place = self._opening_place()
        self.window.resize(place.width, place.height)
        if place.x is not None and place.y is not None:
            self.window.move(place.x, place.y)
        if place.maximized:
            self.window.maximize()
        pane = getattr(self, "content_pane", None)
        if pane is not None and not getattr(pane, "_user_drag", False):
            self._log_split_ready = False
        host = getattr(self, "host_pane", None)
        if host is not None and not getattr(host, "_user_drag", False):
            self._host_split_ready = False
        if getattr(self, "host_button", None) is not None and self.host_button.get_active():
            self.host_panel.start()
            self._place_host_split()
        return False

    def _on_map(self, *_args) -> bool:
        if self._frame_applied:
            return False
        self._frame_applied = True
        self._apply_frame()
        self.glib.idle_add(self._apply_frame)
        self._frame_ready_at = time.monotonic() + 0.8
        self._schedule_frame_save(900)
        return False

    def _on_configure(self, *_args) -> bool:
        if time.monotonic() < self._frame_ready_at:
            return False
        self._remember_frame()
        self._schedule_frame_save(400)
        return False

    def _schedule_frame_save(self, delay: int) -> None:
        if self._frame_timer:
            self.glib.source_remove(self._frame_timer)
        self._frame_timer = self.glib.timeout_add(delay, self._save_frame)

    def _remember_frame(self) -> None:
        window = self.window
        if not window.get_visible():
            return
        try:
            maximized = bool(window.is_maximized())
        except Exception:
            maximized = False
        self.install.window_max = maximized
        if maximized:
            # Keep the size from before maximize. The grown log top belongs to
            # the maximized frame and would crush the log after unmaximize.
            return
        width, height = window.get_size()
        if width >= 480 and height >= 280:
            x, y = window.get_position()
            self.install.window_width = int(width)
            self.install.window_height = int(height)
            self.install.window_x = int(x)
            self.install.window_y = int(y)
        # The log bar moves when the window is resized. Save that top with the
        # frame so the next open keeps the log's thickness, not an old top.
        pane = getattr(self, "content_pane", None)
        button = getattr(self, "log_open_button", None)
        if (
            pane is not None
            and button is not None
            and button.get_active()
            and self._log_split_ready
            and not getattr(pane, "_button_down", False)
        ):
            position = int(pane.get_position())
            if position > 80:
                self.install.log_split = position

    def _save_frame(self) -> bool:
        self._frame_timer = None
        if time.monotonic() < self._frame_ready_at:
            return False
        self._remember_frame()
        try:
            save_install(self.install)
        except Exception:
            log.exception("could not save the window place")
        return False

    def _flush_frame(self) -> None:
        if self._frame_timer:
            self.glib.source_remove(self._frame_timer)
            self._frame_timer = None
        if self.layout_timer:
            self.glib.source_remove(self.layout_timer)
            self.layout_timer = None
        self._frame_ready_at = 0
        self._save_frame()

    def _on_destroy(self, *_args) -> None:
        self._frame_ready_at = time.monotonic() + 10**9
        for name in ("_frame_timer", "layout_timer", "_log_settle", "_host_settle", "_ocr_pulse"):
            timer = getattr(self, name, None)
            if timer:
                self.glib.source_remove(timer)
                setattr(self, name, None)
        panel = getattr(self, "host_panel", None)
        if panel is not None:
            panel.stop()
        stop = getattr(self, "_ocr_install_stop", None)
        if stop is not None:
            stop.set()
        try:
            from nomad_turbo.ocr_setup import stop_install

            stop_install()
        except Exception:
            log.exception("could not stop the GPU reader download")

    def _flush_stats(self, *, wait: bool = False) -> None:
        stats = getattr(self, "file_stats", None)
        if stats is None:
            return
        if wait:
            try:
                stats.save()
            except Exception:
                log.exception("could not save file stats")
            return
        worker = getattr(self, "_stats_thread", None)
        if worker is not None and worker.is_alive():
            return

        def run() -> None:
            try:
                stats.save()
            except Exception:
                log.exception("could not save file stats")

        self._stats_thread = threading.Thread(target=run, name="nomad-turbo-stats", daemon=True)
        self._stats_thread.start()

    def on_delete(self, *_args) -> bool:
        indexing = self._work_alive()
        self._flush_stats(wait=True)
        self._flush_frame()
        if window_close_action(indexing) == "hide" and self.indicator is not None:
            self.hidden = True
            self.window.hide()
            panel = getattr(self, "host_panel", None)
            if panel is not None:
                panel.stop()
            log.info("window closed to the tray. Turbo keeps running.")
            return True
        panel = getattr(self, "host_panel", None)
        if panel is not None:
            panel.stop()
        self._stop_all()
        self.session.close_requested = True
        if indexing:
            if self.repair_thread is not None and self.repair_thread.is_alive():
                self._set_repair_status("Stopping after the current file finishes…")
            else:
                self.status.set_text("Stopping after the current embeds finish…")
            return True
        self._release_log_pane()
        self.qdrant_stop.set()
        self.gtk.main_quit()
        return False

    def begin_scan(self) -> None:
        if self._work_alive():
            return
        self.scanning = True
        self.rows_ready = False
        self.status.set_text("Scanning the library…")
        self.overall.pulse()
        threading.Thread(target=self._scan, name="nomad-turbo-scan", daemon=True).start()

    def _scan(self) -> None:
        try:
            found = scan_library(self.install)
        except Exception as exc:
            log.exception("scan failed")
            self.glib.idle_add(self._scan_failed, str(exc))
            return
        book = ProgressBook(found.rows)
        indexed = sum(1 for row in found.rows if row.state == "indexed")
        self.glib.idle_add(self._scan_ready, book, found, indexed)

    def _scan_failed(self, message: str) -> bool:
        self.scanning = False
        self.status.set_text(f"Scan failed: {message}")
        return False

    def _remember_library(self, book: ProgressBook, rows) -> None:
        """Save this scan and forget files that are no longer in the library."""
        stats = getattr(self, "file_stats", None)
        if stats is None:
            return
        host = getattr(self.install, "storage_host", "") or ""
        root = getattr(self.install, "container_storage", "/app/storage") or "/app/storage"
        present: set[str] = set()
        for row in rows:
            present.add(row.source)
            stats.note_library(
                row.source,
                name=row.name,
                file_type=file_type_label(row.name),
                size_bytes=int(row.size_bytes or 0),
                location=library_location(host, root, row.source),
                mtime=row.mtime,
                articles=int(row.articles or 0),
                total_articles=row.total_articles,
                chunks=int(row.chunks or 0),
                state=row.state,
            )
        stats.prune(present)
        if hasattr(book, "restore_saved"):
            book.restore_saved(stats)
        if hasattr(self.session, "set_priority"):
            for source in stats.priorities():
                if source in present:
                    self.session.set_priority(source, True)
        if hasattr(self.session, "note_zim_date"):
            known = getattr(self.session, "zim_date_known", None)
            for source, raw in stats.zim_dates():
                if source not in present:
                    continue
                if callable(known) and known(source):
                    continue
                self.session.note_zim_date(source, raw)
        stats.save()

    def _scan_ready(self, book: ProgressBook, found, indexed: int) -> bool:
        try:
            self._remember_library(book, found.rows)
        except Exception:
            log.exception("could not restore saved file stats")
        self.session.bind_book(book)
        self.rows_ready = True
        self.scanning = False
        todo = len(found.rows) - indexed
        note = f"{len(found.rows)} files · {indexed} indexed · {todo} still to do."
        if found.active_jobs:
            note += f" NOMAD is still embedding {found.active_jobs} file(s). Start waits for those to finish."
        if found.downloads:
            note += " A download is still in progress."
        self.session.set_note(note)
        self.status.set_text(note)
        log.info(note)
        self.refresh_rows()
        return False

    def _make_index_tree(self, headed, model, *, headers: bool, fit: bool):
        """File columns shared by the pinned row and the folder tree."""
        gtk = self.gtk
        from gi.repository import Pango

        tree = gtk.TreeView(model=model)
        tree.set_headers_visible(headers)
        tree.set_tooltip_column(0)
        name_cell = gtk.CellRendererText()
        name_cell.set_property("ellipsize", Pango.EllipsizeMode.MIDDLE)
        file_column = gtk.TreeViewColumn()
        file_column.set_title(INDEX_HEADERS[0])
        file_label = gtk.Label(label=INDEX_HEADERS[0])
        file_label.set_tooltip_text(TIPS["col_file"])
        file_label.show()
        file_column.set_widget(file_label)
        # The asterisk is its own cell, so ellipsizing the name cannot hide it.
        mark_cell = gtk.CellRendererText()
        mark_cell.set_property("foreground", PRIORITY_COLOR)
        file_column.pack_start(mark_cell, False)
        file_column.pack_start(name_cell, True)
        file_column.add_attribute(mark_cell, "text", COL_MARK)
        file_column.add_attribute(name_cell, "text", COL_NAME)
        file_column.add_attribute(name_cell, "foreground", COL_COLOR)
        if not headers:
            # The folder tree reads the indexing glyph from the row. The pinned row does not.
            file_column.set_cell_data_func(mark_cell, self._paint_tree_mark)
            file_column.set_cell_data_func(name_cell, self._paint_tree_name)
        self._add_column(
            tree,
            "index:File",
            file_column,
            text_index=COL_NAME,
            minimum=180,
            maximum=460,
            fit=fit,
        )
        self._add_column(
            tree,
            "index:File type",
            headed(INDEX_HEADERS[1], TIPS["col_type"], gtk.CellRendererText(), text=COL_TYPE),
            text_index=COL_TYPE,
            minimum=64,
            maximum=140,
            fit=fit,
        )
        self._add_column(
            tree,
            "index:Size",
            headed(INDEX_HEADERS[2], TIPS["col_size"], gtk.CellRendererText(), text=COL_SIZE),
            text_index=COL_SIZE,
            minimum=70,
            maximum=140,
            fit=fit,
        )
        self._add_column(
            tree,
            "index:Status",
            headed(INDEX_HEADERS[3], TIPS["col_status"], gtk.CellRendererText(), text=COL_STATUS),
            text_index=COL_STATUS,
            minimum=80,
            maximum=160,
            fit=fit,
        )
        progress_cell = _pass_renderer(gtk)
        progress_column = headed(
            INDEX_HEADERS[4],
            TIPS["col_progress"],
            progress_cell,
        )
        progress_column.set_cell_data_func(progress_cell, self._paint_pass_bar)
        self._add_column(
            tree,
            "index:Progress",
            progress_column,
            text_index=None,
            minimum=160,
            maximum=280,
            fit=fit,
        )
        self._add_column(
            tree,
            "index:Articles",
            headed(INDEX_HEADERS[5], TIPS["col_articles"], gtk.CellRendererText(), text=COL_ARTICLES),
            text_index=COL_ARTICLES,
            minimum=90,
            maximum=240,
            fit=fit,
        )
        self._add_column(
            tree,
            "index:art/min",
            headed(INDEX_HEADERS[6], TIPS["col_art_min"], gtk.CellRendererText(), text=COL_ART_MIN),
            text_index=COL_ART_MIN,
            minimum=70,
            maximum=140,
            fit=fit,
        )
        self._add_column(
            tree,
            "index:Chunks",
            headed(INDEX_HEADERS[7], TIPS["col_chunks"], gtk.CellRendererText(), text=COL_CHUNKS),
            text_index=COL_CHUNKS,
            minimum=80,
            maximum=160,
            fit=fit,
        )
        self._add_column(
            tree,
            "index:chk/s",
            headed(INDEX_HEADERS[8], TIPS["col_chk_s"], gtk.CellRendererText(), text=COL_CHK_S),
            text_index=COL_CHK_S,
            minimum=70,
            maximum=120,
            fit=fit,
        )
        # Spare width stays on the last column. Growing File moved every later
        # edge, and the folder tree did not grow that column by the same amount.
        self._add_column(
            tree,
            "index:ETA",
            headed(INDEX_HEADERS[9], TIPS["col_eta"], gtk.CellRendererText(), text=COL_ETA),
            text_index=COL_ETA,
            minimum=70,
            maximum=140,
            expand=True,
            fit=fit,
        )
        # A click is not a choice. The file menu is the right-click.
        tree.get_selection().set_mode(gtk.SelectionMode.NONE)
        tree.connect("button-press-event", self.on_file_press)
        return tree

    def _hide_folder_progress(self, _column, cell, model, iterator, _data) -> None:
        cell.set_property("visible", model.get_value(iterator, COL_TYPE) != FOLDER)

    def _paint_pass_bar(self, _column, cell, model, iterator, _data) -> None:
        """Color the file bar from the status word. A folder row has no bar."""
        folder = model.get_value(iterator, COL_TYPE) == FOLDER
        cell.set_property("visible", not folder)
        if folder:
            return
        try:
            value = int(model.get_value(iterator, COL_PROGRESS) or 0)
        except (TypeError, ValueError):
            value = 0
        cell.value = max(0, min(100, value))
        cell.pass_color = bar_color(str(model.get_value(iterator, COL_STATUS) or ""))

    def _paint_tree_mark(self, _column, cell, model, iterator, _data) -> None:
        """The indexing glyph and the priority asterisk. Status text is not either."""
        mark = str(model.get_value(iterator, COL_MARK) or "")
        cell.set_property("text", mark)
        cell.set_property("weight", self._pango_weight(INDEXING_MARK in mark))
        if PRIORITY_MARK in mark:
            cell.set_property("foreground", PRIORITY_COLOR)
        elif INDEXING_MARK in mark:
            cell.set_property("foreground", INDEXING_COLOR)
        else:
            cell.set_property("foreground", "#000000")

    def _paint_tree_name(self, _column, cell, model, iterator, _data) -> None:
        mark = str(model.get_value(iterator, COL_MARK) or "")
        cell.set_property("weight", self._pango_weight(INDEXING_MARK in mark))

    def _pango_weight(self, bold: bool):
        from gi.repository import Pango

        return Pango.Weight.BOLD if bold else Pango.Weight.NORMAL

    def _saved_width(self, key: str) -> int | None:
        raw = getattr(self.install, "column_widths", None) or {}
        if not isinstance(raw, dict):
            return None
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        if value < 40:
            return None
        return value

    def _add_column(
        self,
        tree,
        key: str,
        column,
        *,
        text_index: int | None,
        minimum: int,
        maximum: int,
        expand: bool = False,
        sort_index: int | None = None,
        fit: bool = True,
    ) -> None:
        """A resizable column. A saved width is kept. Otherwise the first rows size it."""
        column.set_resizable(True)
        column.set_sizing(self.gtk.TreeViewColumnSizing.FIXED)
        column.set_min_width(minimum)
        if expand:
            column.set_expand(True)
        if sort_index is not None:
            column.set_sort_column_id(sort_index)
        self._columns_by_key.setdefault(key, []).append(column)
        widget = column.get_widget()
        if widget is not None:
            tip = widget.get_tooltip_text() or ""
            extra = " Drag the edge to resize."
            if sort_index is not None:
                extra += " Click the heading to sort."
            if "Drag the edge" not in tip:
                widget.set_tooltip_text((tip + extra).strip())
        saved = self._saved_width(key)
        width = saved if saved is not None else minimum
        self._sizing = True
        try:
            column.set_fixed_width(width)
        finally:
            self._sizing = False
        self._column_width[key] = width
        self._column_manual[key] = saved is not None
        column.connect("notify::width", self._on_column_width, key)
        column.connect("clicked", self._on_column_clicked, tree)
        if text_index is not None and fit:
            self._fit_meta.append((key, column, text_index, minimum, maximum))
        tree.append_column(column)

    def _on_column_clicked(self, _column, tree) -> None:
        """Sorting moves rows. Look up the active file again on the next tick."""
        self._followed.pop(id(tree), None)

    def _on_column_width(self, column, _pspec, key: str) -> None:
        if self._sizing or column.get_expand():
            return
        # Allocated width includes leftover space. Only a real drag changes fixed width.
        width = int(column.get_fixed_width() or 0)
        if width < 40 or width == self._column_width.get(key):
            return
        self._column_manual[key] = True
        self._column_width[key] = width
        self._sizing = True
        try:
            self._mirror_fixed_width(column, key, width)
        finally:
            self._sizing = False
        raw = getattr(self.install, "column_widths", None) or {}
        widths = dict(raw) if isinstance(raw, dict) else {}
        widths[key] = width
        self.install.column_widths = widths
        self._save_layout_soon()

    def _refit_columns(self, tree) -> None:
        """Widen columns the user has not dragged, using the text now in the list."""
        model = tree.get_model()
        if model is None or not self._fit_meta:
            return
        samples: dict[int, list[str]] = {}
        interesting = [
            item
            for item in self._fit_meta
            if item[1].get_tree_view() == tree and not self._column_manual.get(item[0]) and not item[1].get_expand()
        ]
        if not interesting:
            return
        for row in self._model_sample(model, 80):
            for _key, _column, text_index, _minimum, _maximum in interesting:
                samples.setdefault(text_index, []).append(str(row[text_index] or ""))
        self._sizing = True
        try:
            for key, column, text_index, minimum, maximum in interesting:
                title = column.get_title() or ""
                width = preferred_column_width(
                    samples.get(text_index, []),
                    header=title,
                    minimum=minimum,
                    maximum=maximum,
                )
                if abs(width - int(column.get_fixed_width() or 0)) <= 8:
                    continue
                column.set_fixed_width(width)
                self._column_width[key] = width
                self._mirror_fixed_width(column, key, width)
        finally:
            self._sizing = False

    def _mirror_fixed_width(self, source, key: str, width: int) -> None:
        """Keep the pinned row and the folder tree on the same column edges."""
        for other in self._columns_by_key.get(key, ()):
            if other is source:
                continue
            if int(other.get_fixed_width() or 0) == int(width):
                continue
            other.set_fixed_width(int(width))

    def _model_sample(self, model, limit: int) -> list:
        """Breadth-first rows, so both library roots contribute to column width."""
        rows = []
        level = []
        iterator = model.iter_children(None)
        while iterator is not None:
            level.append(iterator)
            iterator = model.iter_next(iterator)
        while level and len(rows) < limit:
            nxt = []
            for iterator in level:
                if len(rows) >= limit:
                    break
                rows.append(model[iterator])
                child = model.iter_children(iterator)
                while child is not None:
                    nxt.append(child)
                    child = model.iter_next(child)
            level = nxt
        return rows

    def _apply_shared_widths(self) -> None:
        """Put the saved fixed width on both trees. The last column keeps the leftover."""
        if self._sizing:
            return
        self._sizing = True
        try:
            for key, width in self._column_width.items():
                for column in self._columns_by_key.get(key, ()):
                    if column.get_expand():
                        continue
                    if int(column.get_fixed_width() or 0) != int(width):
                        column.set_fixed_width(int(width))
        finally:
            self._sizing = False

    def _align_current_row(self, *_args) -> None:
        """Line the pinned row up with the tree when a scrollbar takes width."""
        self._apply_shared_widths()
        bar = self.index_list.get_vscrollbar()
        width = 0
        if bar is not None and bar.get_visible() and not self.index_list.get_overlay_scrolling():
            width = max(0, int(bar.get_allocated_width()))
        if self.now_tree.get_margin_end() != width:
            self.now_tree.set_margin_end(width)
        self._follow_list_scroll()

    def _follow_list_scroll(self, *_args) -> None:
        """The pinned headings scroll with the folder tree. They do not widen the page."""
        now_scroll = getattr(self, "now_scroll", None)
        listing = getattr(self, "index_list", None)
        if now_scroll is None or listing is None:
            return
        src = listing.get_hadjustment()
        dst = now_scroll.get_hadjustment()
        if src is None or dst is None:
            return
        if getattr(self, "_now_hscroll_adj", None) is not src:
            self._now_hscroll_adj = src
            src.connect("value-changed", self._follow_list_scroll)
        value = float(src.get_value())
        if abs(float(dst.get_value()) - value) > 0.5:
            dst.set_value(value)

    def _on_test_expand(self, _tree, iterator, _path) -> bool:
        """Load this folder's rows before the arrow opens it."""
        key = str(self.store.get_value(iterator, COL_SOURCE))
        if key == FOLDER_HOLD:
            return True
        book = self.session.get_book() if hasattr(self.session, "get_book") else None
        if book is None:
            return True
        if self._ensure_library(book, self._hide_finished(), getattr(book, "structure_epoch", None)):
            self._tree_dirty = True
        folder = self._folder_by_key.get(key)
        if folder is None:
            return True
        self._sync_visible(iterator, self._child_entries(folder), self._priorities(), self._activity())
        return False

    def _on_folder_expanded(self, _tree, iterator, _path) -> None:
        key = str(self.store.get_value(iterator, COL_SOURCE))
        if key == FOLDER_HOLD:
            return
        self._open_folders.add(key)
        self._restore_open_folders()
        self._refit_columns(self.file_tree)

    def _on_folder_collapsed(self, _tree, iterator, _path) -> None:
        key = str(self.store.get_value(iterator, COL_SOURCE))
        self._open_folders.discard(key)
        self._drop_children(iterator)
        self._hold(iterator)

    def _save_layout_soon(self) -> None:
        if self.layout_timer:
            self.glib.source_remove(self.layout_timer)
        self.layout_timer = self.glib.timeout_add(400, self._save_layout)

    def _save_layout(self) -> bool:
        self.layout_timer = None
        try:
            save_install(self.install)
        except Exception:
            log.exception("could not save the window layout")
        return False

    def _style_panes(self) -> None:
        if self._pane_styled:
            return
        self._pane_styled = True
        provider = self.gtk.CssProvider()
        provider.load_from_data(
            b"""
            paned separator,
            paned separator.wide {
                background-color: alpha(@theme_fg_color, 0.55);
                background-image: none;
                border: none;
                margin: 0;
                padding: 0;
                min-height: 6px;
                min-width: 6px;
            }
            paned separator:hover,
            paned separator:active,
            paned separator.wide:hover,
            paned separator.wide:active {
                background-color: alpha(@theme_fg_color, 0.85);
            }
            .nomad-segment {
                border: 1px solid alpha(@theme_fg_color, 0.55);
                border-radius: 4px;
                padding: 0;
            }
            .nomad-segment button {
                border: none;
                border-radius: 0;
                box-shadow: none;
                margin: 0;
            }
            .nomad-segment button:first-child {
                border-right: 1px solid alpha(@theme_fg_color, 0.35);
            }
            progressbar,
            progressbar trough {
                min-width: 48px;
            }
            progressbar text {
                color: #ffffff;
                text-shadow: none;
            }
            .nomad-segment button:checked {
                background-image: none;
                background-color: alpha(@theme_selected_bg_color, 0.45);
            }
            """
        )
        from gi.repository import Gdk

        screen = Gdk.Screen.get_default()
        if screen is None:
            return
        self.gtk.StyleContext.add_provider_for_screen(
            screen,
            provider,
            self.gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def _linked(self, *children):
        """One segmented control. The children sit flush inside a shared outline."""
        box = self.gtk.Box(orientation=self.gtk.Orientation.HORIZONTAL, spacing=0)
        box.get_style_context().add_class("linked")
        box.get_style_context().add_class("nomad-segment")
        box.set_halign(self.gtk.Align.START)
        box.set_valign(self.gtk.Align.CENTER)
        for child in children:
            child.set_valign(self.gtk.Align.CENTER)
            child.set_margin_start(0)
            child.set_margin_end(0)
            box.pack_start(child, False, False, 0)
        return box

    def _flow(self, *children, spacing: int = 8):
        """A row that wraps whole children when the main area gets narrow.

        The minimum height stays one row, so a short window is not forced open
        by the wrapped height. The wrapped height is what a narrow width asks
        for, and the list above gives up that space.
        """
        flow = _wrap_box(spacing=spacing, row_spacing=4)
        for child in children:
            flow.add(child)
        return flow

    def _bind_choice(self, chosen, other) -> None:
        """Two buttons, exactly one selected. Choosing one releases the other."""

        def on_toggle(button, mate) -> None:
            if self._choice_guard:
                return
            self._choice_guard = True
            try:
                if button.get_active():
                    mate.set_active(False)
                elif not mate.get_active():
                    button.set_active(True)
            finally:
                self._choice_guard = False

        chosen.connect("toggled", on_toggle, other)
        other.connect("toggled", on_toggle, chosen)

    def _preferred_min_width(self, widget) -> int:
        if widget is None:
            return 0
        try:
            minimum, _natural = widget.get_preferred_width()
        except Exception:
            return 0
        return max(0, int(minimum or 0))

    def _main_squeeze(self) -> int:
        """Narrowest main-pane width that still fits every control, wrapped."""
        widths = [
            self._preferred_min_width(widget)
            for widget in (
                getattr(self, "run_group", None),
                getattr(self, "measure_group", None),
                getattr(self, "ocr_group", None),
                getattr(self, "embed_group", None),
                getattr(self, "delete_pad", None),
                getattr(self, "repair_choice", None),
                getattr(self, "repair_actions", None),
                getattr(self, "optimize_segments", None),
                getattr(self, "log_filters", None),
                getattr(self, "log_export_group", None),
                getattr(self, "settings_page", None),
            )
        ]
        # A hidden page still sets the stack's minimum. Settings is wider than
        # the Index button groups, and the bar has to stop at that width.
        pane = self._preferred_min_width(getattr(self, "content_pane", None))
        return max(max(widths or [0]) + 4, pane)

    def _yield_host(self, pane_width: int, main_min: int) -> None:
        """Let the host side shrink when the window is already at its floor."""
        panel = getattr(self, "host_panel", None)
        widget = None if panel is None else getattr(panel, "widget", None)
        if widget is None:
            return
        room = int(pane_width) - int(main_min)
        request = 220 if room >= 220 else max(1, room)
        current = widget.get_size_request()
        if int(current.width) != int(request):
            widget.set_size_request(int(request), -1)

    def _clamp_host_bar(self, pane) -> int:
        """Stop the vertical bar at the squeeze limit. Returns the main width."""
        width = int(pane.get_allocated_width() or 0)
        if width < 1:
            return 0
        main_min = self._main_squeeze()
        self._yield_host(width, main_min)
        position = host_bar_position(width, int(pane.get_position()), main_min=main_min)
        if abs(int(pane.get_position()) - position) > 1:
            self._pane_lock = True
            try:
                pane.set_position(int(position))
            finally:
                self._pane_lock = False
        return position

    def _make_pane(self, orientation=None):
        """A visible split. The log bar is vertical. The host bar runs the other way."""
        self._style_panes()
        if orientation is None:
            orientation = self.gtk.Orientation.VERTICAL
        pane = self.gtk.Paned(orientation=orientation)
        pane.set_wide_handle(True)
        pane.set_vexpand(True)
        pane.set_hexpand(True)
        return pane


    def _mark_pane_drag(self, pane, event) -> bool:
        # A window resize also moves the bar. Only a drag on the bar is saved.
        if getattr(event, "button", 1) == 1:
            pane._user_drag = True
            pane._button_down = True
        return False

    def _place_log_split(self, pane, height: int) -> None:
        if height < 160 or not self.log_open_button.get_active():
            return
        position = restored_log_top(
            int(getattr(self.install, "log_pane", 0) or 0),
            int(getattr(self.install, "log_split", 0) or 0),
            height,
        )
        self._pane_lock = True
        try:
            pane.set_position(position)
        finally:
            self._pane_lock = False

    def _mark_log_split_settled(self, height: int) -> bool:
        self._log_settle = None
        pane = getattr(self, "content_pane", None)
        if pane is None:
            return False
        current = int(pane.get_allocated_height() or 0)
        if abs(current - int(height)) <= 2:
            self._log_split_ready = True
        return False

    def _collapse_log(self) -> None:
        """Leave only the log row under the bar while the text is hidden."""
        pane = getattr(self, "content_pane", None)
        header = getattr(self, "log_header", None)
        if pane is None or header is None or self.log_open_button.get_active():
            return
        height = int(pane.get_allocated_height() or 0)
        if height < 40:
            return
        minimum, _natural = header.get_preferred_height()
        keep = max(int(minimum or 0), 24)
        position = max(0, height - keep)
        if abs(int(pane.get_position()) - position) <= 2:
            return
        self._pane_lock = True
        try:
            pane.set_position(position)
        finally:
            self._pane_lock = False

    def _track_log_pane(self, pane) -> None:
        def on_size(_widget, allocation) -> None:
            if not self.log_open_button.get_active():
                self._collapse_log()
                return
            # A drag owns the bar. After that, a window resize keeps the log's
            # pixel height and the drag handler records the new top.
            if getattr(pane, "_button_down", False) or getattr(pane, "_user_drag", False):
                self._log_split_ready = True
                return
            if self._log_split_ready:
                return
            height = int(allocation.height or 0)
            # The first allocation is often short. Placing once at that size
            # leaves the log stuck there after the window grows.
            if height < 160:
                return
            self._place_log_split(pane, height)
            self._log_placed_for = height
            if self._log_settle:
                self.glib.source_remove(self._log_settle)
            self._log_settle = self.glib.timeout_add(200, self._mark_log_split_settled, height)

        def on_release(_widget, event) -> bool:
            if getattr(event, "button", 1) == 1:
                pane._button_down = False
            return False

        def on_grab_broken(*_args) -> bool:
            pane._button_down = False
            return False

        def on_position(widget, _pspec) -> None:
            if self._pane_lock or not getattr(widget, "_user_drag", False):
                return
            if not self.log_open_button.get_active():
                return
            position = int(widget.get_position())
            if position < 80:
                return
            self.install.log_split = position
            self._save_layout_soon()

        pane.connect("size-allocate", on_size)
        pane.connect("button-press-event", self._mark_pane_drag)
        pane.connect("button-release-event", on_release)
        pane.connect("grab-broken-event", on_grab_broken)
        pane.connect("notify::position", on_position)

    def on_follow(self, button) -> None:
        following = bool(button.get_active())
        self.install.log_follow = following
        try:
            save_install(self.install)
        except Exception:
            log.exception("could not save the log follow setting")
        if following:
            self._log_away = False
            end = self.log_buffer.get_end_iter()
            self.log_buffer.place_cursor(end)
            self._settle_log(bottom=True)

    def _on_log_title(self, _widget, event) -> bool:
        if getattr(event, "button", 1) != 1:
            return False
        self.log_open_button.set_active(not self.log_open_button.get_active())
        return True

    def on_log(self, button) -> None:
        expanded = bool(button.get_active())
        icon = "pan-down-symbolic" if expanded else "pan-end-symbolic"
        self.log_arrow.set_from_icon_name(icon, self.gtk.IconSize.BUTTON)
        self.install.log_open = expanded
        self.log_filters.set_no_show_all(not expanded)
        self.log_filters.set_visible(expanded)
        self.log_scroll.set_no_show_all(not expanded)
        self.log_scroll.set_visible(expanded)
        if expanded:
            self._place_log_split(self.content_pane, int(self.content_pane.get_allocated_height() or 0))
        else:
            self._collapse_log()
        try:
            save_install(self.install)
        except Exception:
            log.exception("could not save the log setting")

    def on_keep(self, button) -> None:
        value = bool(button.get_active())
        self.session.set_keep_control(value)
        self.install.keep_control = value
        try:
            save_install(self.install)
        except Exception:
            log.exception("could not save the stay-on setting")

    def on_ocr_quality(self, button, name: str) -> None:
        """One of Low, Medium, or High stays on. The choice is saved."""
        if getattr(self, "_ocr_guard", False):
            return
        self._ocr_guard = True
        try:
            buttons = (
                ("low", self.ocr_low),
                ("medium", self.ocr_medium),
                ("high", self.ocr_high),
            )
            if button.get_active():
                for _other_name, other in buttons:
                    if other is not button and other.get_active():
                        other.set_active(False)
                chosen = name
            elif any(other.get_active() for _other_name, other in buttons):
                return
            else:
                button.set_active(True)
                chosen = name
            chosen = normalize_ocr_quality(chosen)
            self.session.set_ocr_quality(chosen)
            self.install.ocr_quality = chosen
            try:
                save_install(self.install)
            except Exception:
                log.exception("could not save the ocr quality")
        finally:
            self._ocr_guard = False

    def on_order(self, combo) -> None:
        """Choose which unfinished file comes next. The one already running stays."""
        choice = normalize_order(combo.get_active_id())
        self.session.set_order(choice)
        self.install.index_order = choice
        try:
            save_install(self.install)
        except Exception:
            log.exception("could not save the index order")
        self.refresh_rows()

    def on_hide_finished(self, button) -> None:
        """Show or hide indexed files. The setting is kept for the next open."""
        self.install.hide_finished = bool(button.get_active())
        try:
            save_install(self.install)
        except Exception:
            log.exception("could not save the hide-finished setting")
        self.refresh_rows()

    def on_measure(self, *_args, then_index: bool = False) -> None:
        """Sample each file that has not been started. Started idle, the pass stops the way Stop does.

        A file that already has progress keeps its ETA and is left alone.
        A pass started during a run keeps indexing. Then index is a separate
        half: it does not start this pass.
        """
        del then_index
        if hasattr(self.session, "measuring") and self.session.measuring():
            return
        running = (
            self.session.view()["phase"] in ("waiting", "running")
            and self.thread is not None
            and self.thread.is_alive()
        )
        if running:
            self.session.start_measure(from_idle=False, then_index=False)
            self.session.set_note("Measuring files that have not been started, about five minutes each.")
            return
        if not self.rows_ready or not self.qdrant_ready or self._work_alive():
            return
        self._measure_pending = True
        self.session.start_measure(from_idle=True, then_index=False)
        if not self.on_start(False):
            self._measure_pending = False
            self.session.set_measuring(False)

    def on_then_index(self, button) -> None:
        """Arm the measurement already running. Idle does not start a worker."""
        if self._then_guard:
            return
        measuring = bool(self.session.measuring()) if hasattr(self.session, "measuring") else False
        if then_index_action(measuring=measuring) != "arm":
            self._set_then_index_shown(False)
            return
        armed = bool(self.session.arm_then_index()) if hasattr(self.session, "arm_then_index") else False
        self._set_then_index_shown(armed)

    def _set_then_index_shown(self, armed: bool) -> None:
        button = getattr(self, "then_index_button", None)
        if button is None or not hasattr(button, "set_active"):
            return
        if bool(button.get_active()) == bool(armed):
            return
        self._then_guard = True
        try:
            button.set_active(bool(armed))
        finally:
            self._then_guard = False

    def on_file_press(self, tree, event):
        if getattr(event, "button", 1) != 3:
            return False
        info = tree.get_path_at_pos(int(event.x), int(event.y))
        if not info:
            return False
        path = info[0]
        row = tree.get_model()[path]
        if row[COL_TYPE] == FOLDER:
            return False
        source = row[0]
        menu = self._file_menu(str(source))
        # GTK drops the menu if nothing in Python still holds it.
        self._row_menu = menu
        if hasattr(menu, "popup_at_pointer"):
            menu.popup_at_pointer(event)
        else:
            menu.popup(None, None, None, None, event.button, event.time)
        return True

    def _file_menu(self, source: str):
        priority = source in self.session.priorities()
        labels = row_menu_labels(priority=priority)
        menu = self.gtk.Menu()
        now = self.gtk.MenuItem()
        now.set_label(labels[0])
        now.set_tooltip_text(TIPS["index_now"])
        now.connect("activate", lambda *_args, src=source: self.on_index_now(src))
        menu.append(now)
        mark = self.gtk.MenuItem()
        mark.set_label(labels[1])
        mark.set_tooltip_text(TIPS["priority"])
        mark.connect("activate", lambda *_args, src=source, on=not priority: self.on_priority(src, on))
        menu.append(mark)
        details = self.gtk.MenuItem()
        details.set_label(labels[2])
        details.set_tooltip_text(TIPS["details"])
        details.connect("activate", lambda *_args, src=source: self.on_details(src))
        menu.append(details)
        menu.show_all()
        return menu

    def file_detail_rows(self, source: str) -> list[tuple[str, str]]:
        """The facts the details window shows for one file."""
        live = None
        book = self.session.get_book() if hasattr(self.session, "get_book") else None
        if book is not None and hasattr(book, "file_snap"):
            try:
                live = book.file_snap(source)
            except Exception:
                live = None
        elif book is not None and hasattr(book, "snapshot"):
            try:
                snap = book.snapshot()
            except Exception:
                snap = None
            if snap is not None:
                live = next((item for item in snap.files if item.source == source), None)
        stats = getattr(self, "file_stats", None)
        saved = stats.get(source) if stats is not None else None
        priority = source in self.session.priorities() if hasattr(self.session, "priorities") else None
        zim = None
        if hasattr(self.session, "zim_date_known") and self.session.zim_date_known(source):
            zim = self.session.zim_date(source) or ""
        location = library_location(
            getattr(self.install, "storage_host", "") or "",
            getattr(self.install, "container_storage", "/app/storage") or "/app/storage",
            source,
        )
        stat = compose_detail(
            source,
            saved,
            name=live.name if live is not None else "",
            file_type=live.file_type if live is not None else "",
            size_bytes=live.size_bytes if live is not None else 0,
            location=location,
            mtime=live.mtime if live is not None else None,
            zim_date=zim,
            articles=live.articles if live is not None else None,
            total_articles=live.total_articles if live is not None else None,
            chunks=live.chunks if live is not None else None,
            state=live.state if live is not None else "",
            priority=priority,
            have_live=live is not None,
        )
        return detail_lines(stat)

    def on_details(self, source: str) -> None:
        """Show the saved facts for one file. This does not start indexing."""
        gtk = self.gtk
        rows = self.file_detail_rows(source)
        title = next((value for label, value in rows if label == "Name" and value and value != "—"), "File details")
        dialog = gtk.Dialog(
            title=title,
            parent=self.window,
            flags=gtk.DialogFlags.MODAL,
        )
        dialog.set_default_size(520, 460)
        dialog.add_button("Close", gtk.ResponseType.CLOSE)
        scroll = gtk.ScrolledWindow()
        scroll.set_policy(gtk.PolicyType.NEVER, gtk.PolicyType.AUTOMATIC)
        grid = gtk.Grid(column_spacing=16, row_spacing=6)
        grid.set_border_width(12)
        for index, (label, value) in enumerate(rows):
            caption = gtk.Label(label=label)
            caption.set_xalign(0)
            shown = gtk.Label(label=value)
            shown.set_xalign(0)
            shown.set_selectable(True)
            shown.set_line_wrap(True)
            shown.set_max_width_chars(48)
            grid.attach(caption, 0, index, 1, 1)
            grid.attach(shown, 1, index, 1, 1)
        scroll.add(grid)
        dialog.get_content_area().pack_start(scroll, True, True, 0)
        dialog.show_all()
        dialog.run()
        dialog.destroy()

    def on_index_now(self, source: str) -> None:
        """Leave the embed call already running, then index this file."""
        self.session.request_index_now(source)
        running = self.thread is not None and self.thread.is_alive()
        if running:
            self.session.set_note("That file is next.")
            return
        self.on_start(False)

    def on_priority(self, source: str, on: bool) -> None:
        self.session.set_priority(source, bool(on))
        stats = getattr(self, "file_stats", None)
        if stats is not None:
            stats.set_priority(source, bool(on))
            try:
                stats.save()
            except Exception:
                log.exception("could not save the priority for %s", source)
        self.refresh_rows()

    def on_tune_auto(self, button) -> None:
        active = bool(button.get_active())
        self.session.set_tune_auto(active)
        self.chunk_entry.set_sensitive(not active)
        self._tune_hold = None
        self.install.tune_auto = active
        try:
            save_install(self.install)
        except Exception:
            log.exception("could not save the auto batch setting")
        self._apply_tune()

    def on_chunks(self, entry) -> None:
        """Apply a typed chunk count as the digits change. The next call uses it."""
        if self._tune_applying:
            return
        text = entry.get_text().strip()
        if not text.isdigit():
            return
        typed = int(text)
        if typed < 1:
            return
        count = clamp_embed_chunks(typed)
        if count != typed:
            self._tune_applying = True
            try:
                entry.set_text(str(count))
            finally:
                self._tune_applying = False
        self.session.set_chunks(count)
        self._tune_hold = None
        self._queue_chunk_save(count)
        self.refresh_usage()

    def on_recommend(self, *_args) -> None:
        """Set the chunk count to the safe call for the computer running Turbo."""
        gpu = self._index_memory()
        gpu_total = gpu[1] if gpu else None
        count = safe_call_chunks(
            gpu_total_mib=gpu_total,
            host_total_mib=self.host_total_mib,
            cpu_count=self.cpu_count,
        )
        self._tune_hold = None
        self._show_chunk_count(count)
        self._queue_chunk_save(count)
        self.refresh_usage()

    def _show_chunk_count(self, count: int) -> None:
        count = clamp_embed_chunks(count)
        self.session.set_chunks(count)
        shown = str(count)
        if self.chunk_entry.get_text().strip() != shown:
            self._tune_applying = True
            try:
                self.chunk_entry.set_text(shown)
            finally:
                self._tune_applying = False

    def _queue_chunk_save(self, count: int) -> None:
        if self.save_timer:
            self.glib.source_remove(self.save_timer)
        self.save_timer = self.glib.timeout_add(400, self._save_chunks, int(count))

    def _save_chunks(self, count: int) -> bool:
        self.save_timer = None
        try:
            self.install.embed_chunks = int(count)
            save_install(self.install)
        except Exception:
            log.exception("could not save the embed batch size")
        return False

    def on_filter(self, *_args) -> None:
        self.filter_dirty = True

    def on_start(self, wipe: bool) -> bool:
        if not self.rows_ready or self._work_alive() or not self.qdrant_ready:
            return False
        # Start indexing is a full run. Measure sets the flag, then calls this.
        # A sample stopped earlier must not stay on and turn this run into samples.
        if not getattr(self, "_measure_pending", False) and hasattr(self.session, "set_measuring"):
            self.session.set_measuring(False)
        self._measure_pending = False
        self.session.stop.clear()
        self.session.wipe_confirmed = bool(wipe)
        self.session.set_phase("waiting")
        self.started_at = time.monotonic()
        self._finish_told = False
        self.thread = threading.Thread(target=self._run, args=(wipe,), name="nomad-turbo")
        self.thread.start()
        return True

    def on_delete_index(self, *_args) -> None:
        if not self.rows_ready or self._work_alive() or not delete_allowed(
            self.qdrant_ready, getattr(self, "collection_missing", False)
        ):
            return
        gtk = self.gtk
        dialog = gtk.MessageDialog(
            parent=self.window,
            flags=0,
            message_type=gtk.MessageType.WARNING,
            buttons=gtk.ButtonsType.NONE,
            text="Delete the whole knowledge index?",
        )
        dialog.format_secondary_text(
            "This deletes every stored vector and every ingest row. "
            "ZIM files and uploaded documents stay on disk. "
            "Indexing does not start. Click Start indexing when you want to rebuild. "
            "Ingest stays on Manual, so NOMAD does not start embedding on its own. "
            "NOMAD finishes its current embed batches first.\n\n"
            "Type DELETE to confirm."
        )
        entry = gtk.Entry()
        dialog.get_content_area().add(entry)
        dialog.add_button("Cancel", gtk.ResponseType.CANCEL)
        confirm = dialog.add_button("Delete", gtk.ResponseType.OK)
        confirm.set_sensitive(False)
        entry.connect("changed", lambda widget: confirm.set_sensitive(wipe_phrase_ok(widget.get_text())))
        dialog.show_all()
        response = dialog.run()
        typed = entry.get_text()
        dialog.destroy()
        if response == gtk.ResponseType.OK and wipe_phrase_ok(typed):
            log.warning("index delete confirmed")
            self.session.stop.clear()
            self.session.set_phase("deleting")
            self.session.set_note("Deleting the knowledge index. Indexing will not start.")
            self.status.set_text("Deleting the knowledge index. Indexing will not start.")
            self.delete_thread = threading.Thread(target=self._delete_only, name="nomad-turbo-delete")
            self.delete_thread.start()

    def _reload_indexer(self):
        """Reload the indexer and the modules it imports.

        Delete and Start read engine.py again so a fix applies without
        reopening the window. A name added after this window opened is
        missing from the copy already in memory, and the reload then dies
        before the index is touched. The providing module is reloaded
        first. progress follows schedule. passes follows packing. slots
        is first because pool binds it at import.
        """
        import importlib

        import nomad_turbo.chunking as chunking
        import nomad_turbo.decisions as decisions
        import nomad_turbo.discover as discover
        import nomad_turbo.engine as engine
        import nomad_turbo.logbook as logbook
        import nomad_turbo.packing as packing
        import nomad_turbo.passes as passes
        import nomad_turbo.pool as pool
        import nomad_turbo.progress as progress
        import nomad_turbo.runtime as runtime
        import nomad_turbo.schedule as schedule
        import nomad_turbo.session as session_mod
        import nomad_turbo.slots as slots
        import nomad_turbo.store as store

        importlib.reload(slots)
        importlib.reload(discover)
        importlib.reload(session_mod)
        importlib.reload(store)
        importlib.reload(runtime)
        importlib.reload(schedule)
        importlib.reload(progress)
        importlib.reload(decisions)
        importlib.reload(chunking)
        importlib.reload(packing)
        importlib.reload(passes)
        importlib.reload(pool)
        importlib.reload(logbook)
        importlib.reload(engine)
        return engine

    def _delete_only(self) -> None:
        try:
            engine = self._reload_indexer()
            engine.delete_index_only(self.session.stop, session=self.session)
        except Exception as exc:
            log.exception("index delete failed")
            self.session.set_phase("error", str(exc))
            self.session.set_note(str(exc))
            return
        note = self.session.view().get("note") or "Knowledge index deleted. Indexing has not started."
        self.glib.idle_add(self._show, self.status, note)

    def on_log_help(self, *_args) -> None:
        from nomad_turbo.logbook import log_help_text

        gtk = self.gtk
        dialog = gtk.Dialog(
            title="What the log lines mean",
            parent=self.window,
            flags=gtk.DialogFlags.MODAL,
        )
        dialog.set_default_size(560, 480)
        dialog.add_button("Close", gtk.ResponseType.CLOSE)
        scroll = gtk.ScrolledWindow()
        scroll.set_policy(gtk.PolicyType.AUTOMATIC, gtk.PolicyType.AUTOMATIC)
        view = gtk.TextView()
        view.set_editable(False)
        view.set_cursor_visible(False)
        view.set_wrap_mode(gtk.WrapMode.WORD)
        view.set_left_margin(12)
        view.set_right_margin(12)
        view.set_top_margin(12)
        view.set_bottom_margin(12)
        view.get_buffer().set_text(log_help_text())
        scroll.add(view)
        dialog.get_content_area().pack_start(scroll, True, True, 0)
        dialog.show_all()
        dialog.run()
        dialog.destroy()

    def on_export(self, *_args) -> None:
        from nomad_turbo.logbook import export_logs

        gtk = self.gtk
        chooser = gtk.FileChooserDialog(
            title="Export log",
            parent=self.window,
            action=gtk.FileChooserAction.SAVE,
            buttons=(
                gtk.STOCK_CANCEL,
                gtk.ResponseType.CANCEL,
                gtk.STOCK_SAVE,
                gtk.ResponseType.OK,
            ),
        )
        chooser.set_current_name("nomad-turbo-logs.zip")
        chooser.set_do_overwrite_confirmation(True)
        response = chooser.run()
        target = chooser.get_filename()
        chooser.destroy()
        if response != gtk.ResponseType.OK or not target:
            return
        try:
            export_logs(state_dir(), Path(target))
        except OSError as exc:
            self.status.set_text(f"Could not export the log: {exc}")
            return
        self.status.set_text(f"Logs saved to {target}")

    def _run(self, wipe: bool) -> None:
        import importlib

        import nomad_turbo.decisions as decisions
        import nomad_turbo.schedule as schedule
        import nomad_turbo.session as session_mod
        import nomad_turbo.tune as tune

        # Start loads the modules again so a fix applies without reopening the window.
        # tune is reloaded so Auto picks up a batch-rule fix on this Start.
        # session is reloaded before engine, then the live object takes the new class.
        # The object was built when the window opened. Leaving it on the old class
        # makes the first recorded embed call abort the file.
        engine = self._reload_indexer()
        importlib.reload(tune)
        if self.session.__class__ is not session_mod.Session:
            try:
                self.session.__class__ = session_mod.Session
            except TypeError:
                log.warning("session recorder kept the copy loaded when the window opened")
        execute = engine.execute

        write_shortcut(True)
        try:
            execute(self.session.stop, session=self.session, wipe=wipe, manage_icon=False)
            saved_policy = None
            policy_path = engine.state_dir() / "policy"
            if policy_path.is_file():
                saved_policy = policy_path.read_text(encoding="utf-8").strip()
            if self.session.stop.is_set():
                self.session.set_phase("stopped")
                self.session.set_note(decisions.stop_status_note(saved_policy))
            else:
                self.session.set_phase("done")
                self.session.set_note(decisions.finished_status_note(saved_policy))
        except Exception as exc:
            # Stop during an embed raises RuntimeError("stopped"). That is a stop, not a failure.
            if str(exc) == "stopped":
                saved_policy = None
                policy_path = engine.state_dir() / "policy"
                if policy_path.is_file():
                    saved_policy = policy_path.read_text(encoding="utf-8").strip()
                log.info("stopped")
                self.session.set_phase("stopped")
                self.session.set_note(decisions.stop_status_note(saved_policy))
            else:
                log.exception("turbo failed")
                self.session.set_phase("error", str(exc))
                self.session.set_note(str(exc))
        finally:
            write_shortcut(False)
            self.session.wipe_confirmed = False
            schedule.release_measure(self.session)

    def on_tick(self) -> bool:
        self.tick += 1
        if self.tick % 4 == 0:
            self._flush_stats()
        snap = None
        view = {"phase": "", "note": "", "error": ""}
        try:
            if self.tick % 4 == 0:
                self.live_gpu = pick_gpu(gpu_rows())
            if self.raise_requested:
                self.raise_requested = False
                self.present()
            view = self.session.view()
            if view["phase"] in ("waiting", "running", "done", "stopped", "error", "deleting") and view["note"]:
                self._show(self.status, view["error"] or view["note"])
            if self.scanning:
                self.overall.pulse()
            else:
                snap = self.refresh_rows()
            self.refresh_stats(snap, view)
            self.refresh_usage()
            self.refresh_repair()
            self.refresh_qdrant()
            self.refresh_tray(view, snap)
            self._paint_index_optimize(view.get("optimize_hold") or "")
            self._paint_repair_optimize()
            self._tell_if_finished(view)
            self._buttons(view["phase"])
        except Exception:
            if not getattr(self, "_paint_failed", False):
                self._paint_failed = True
                log.exception("window refresh failed")
        try:
            self.refresh_log()
        except Exception:
            if not getattr(self, "_log_failed", False):
                self._log_failed = True
                log.exception("log pane failed")
        if self.session.close_requested and not self._work_alive():
            self._flush_frame()
            self._release_log_pane()
            self.qdrant_stop.set()
            self.gtk.main_quit()
            return False
        return True

    def _buttons(self, phase: str) -> None:
        running = phase in ("waiting", "running") and self.thread is not None and self.thread.is_alive()
        deleting = self.delete_thread is not None and self.delete_thread.is_alive()
        file_repairing = (self.repair_thread is not None and self.repair_thread.is_alive()) or self.repair_starting
        optimizing = self._optimize_busy()
        repairing = file_repairing or optimizing
        library_idle = self.rows_ready and not running and not repairing and not deleting
        changes = changes_allowed(self.qdrant_ready)
        can_delete = delete_allowed(self.qdrant_ready, getattr(self, "collection_missing", False))
        self.start_button.set_sensitive(library_idle and changes)
        self.delete_button.set_sensitive(library_idle and can_delete)
        self.rescan_button.set_sensitive(library_idle)
        measuring = bool(self.session.measuring()) if hasattr(self.session, "measuring") else False
        measure_ready = (
            self.rows_ready and changes and not measuring and not repairing and not deleting and (library_idle or running)
        )
        self.measure_button.set_sensitive(measure_ready)
        # Then index only takes effect during a measurement. It stays off until then.
        self.then_index_button.set_sensitive(measuring and not repairing and not deleting)
        armed = bool(measuring and hasattr(self.session, "measure_then_index") and self.session.measure_then_index())
        self._set_then_index_shown(armed)
        self._gate_tip(self.start_button, self.start_tip)
        if can_delete:
            self.delete_button.set_tooltip_text(self.delete_tip)
        else:
            self.delete_button.set_tooltip_text(TIPS["qdrant_gate"])
        self.stop_button.set_sensitive(running or deleting)
        if hasattr(self, "repair_start"):
            idle_repair = not running and not repairing and not deleting and not self.scanning and changes
            self.repair_scan.set_sensitive(idle_repair)
            self.repair_start.set_sensitive(idle_repair)
            self._gate_tip(self.repair_scan, self.repair_scan_tip)
            self._gate_tip(self.repair_start, self.repair_start_tip)
            self.repair_stop_button.set_sensitive(file_repairing)
            self.repair_quick.set_sensitive(not repairing)
            self.repair_complete.set_sensitive(not repairing)
            if hasattr(self, "optimize_segments"):
                self.optimize_segments.set_sensitive(idle_repair)
                self._gate_tip(self.optimize_segments, self.optimize_segments_tip)

    def refresh_rows(self):
        book = self.session.get_book()
        if book is None:
            self._running_source = ""
            return None
        # pulse reads the running file and the bar totals. Those totals stay on
        # the book, so this tick does not walk the library. A closed folder is one row.
        if hasattr(book, "pulse"):
            snap = book.pulse()
        else:
            snap = book.snapshot()
        priorities = self._priorities()
        activity = self._activity()
        hide = self._hide_finished()
        rebuilt = self._ensure_library(book, hide, getattr(snap, "structure_epoch", None))
        if rebuilt or self._tree_dirty:
            self._sync_visible(None, self._root_entries(), priorities, activity)
            self._open_folders.intersection_update(self._folder_by_key)
            self._restore_open_folders()
            self._refit_columns(self.file_tree)
            self._tree_dirty = False
        elif frozenset(priorities) != self._shown_priority:
            self._paint_marks(priorities)
        self._shown_priority = frozenset(priorities)
        running = [item for item in snap.files if item.state == "running"][:2]
        self._running_source = running[0].source if running else ""
        self._paint_running(running, priorities, activity)
        self.overall.set_fraction(min(1.0, max(0.0, snap.overall_fraction)))
        self._show(self.overall, snap.overall_text)
        return snap

    def _hide_finished(self) -> bool:
        if not hasattr(self, "hide_finished"):
            return False
        return bool(self.hide_finished.get_active())

    def _priorities(self) -> set:
        if hasattr(self.session, "priorities"):
            return self.session.priorities()
        return set()

    def _activity(self) -> str:
        if hasattr(self.session, "view"):
            return self.session.view().get("note") or ""
        return ""

    def _ocr_activity(self) -> str:
        """Page progress. Falls back to the status line when nothing else is reading."""
        getter = getattr(self.session, "ocr_activity", None)
        if callable(getter):
            text = getter() or ""
            if text:
                return text
        return self._activity()

    def _ensure_library(self, book, hide: bool, seen_epoch) -> bool:
        """Rebuild the folder cache when the library membership changes."""
        if self._shown_book is book and self._shown_hide == hide and self._shown_epoch == seen_epoch:
            return False
        if hasattr(book, "library_items"):
            epoch, items = book.library_items()
        else:
            snap = book.snapshot()
            epoch = getattr(snap, "structure_epoch", 0)
            items = snap.files
        folders = library_tree(items, hide_finished=hide)
        self._root_folders = folders
        self._folder_by_key = {}
        self._index_folders(folders)
        self._shown_book = book
        self._shown_epoch = epoch
        self._shown_hide = hide
        return True

    def _index_folders(self, folders) -> None:
        for folder in folders:
            self._folder_by_key[folder.key] = folder
            self._index_folders(folder.folders)

    def _root_entries(self) -> list:
        return [("folder", folder.key, folder) for folder in self._root_folders]

    def _child_entries(self, folder) -> list:
        entries = [("folder", child.key, child) for child in folder.folders]
        entries.extend(("file", source, source) for source in folder.files)
        return entries

    def _file_values(self, item, priorities, activity: str, *, tree: bool = False) -> tuple:
        values = index_store_row(item, priority=item.source in priorities)
        if tree:
            mark = row_mark(priority=item.source in priorities, indexing=item.state == "running")
            values = (*values[:COL_MARK], mark, *values[COL_MARK + 1 :])
        if item.state == "running":
            note = self._ocr_activity()
            if ocr_note_matches(note, item.name):
                pace = None
                observer = getattr(self, "_ocr_pace", None)
                if observer is not None:
                    pace = observer.observe(note)
                values = overlay_ocr_columns(values, note, pace)
        return values

    def _sync_now_rows(self, items, priorities, activity: str) -> None:
        """Files in progress stay above the folder tree, outside the scroll.

        CPU OCR can be reading one file while the GPU indexes another. Both
        stay here. One file still uses the single row.
        """
        if not items:
            if len(self.now_store):
                self.now_store.clear()
            return
        sources = [item.source for item in items]
        current = [self.now_store[index][COL_SOURCE] for index in range(len(self.now_store))]
        if current != sources:
            self.now_store.clear()
            for item in items:
                self.now_store.append(self._file_values(item, priorities, activity))
            return
        iterator = self.now_store.get_iter_first()
        for item in items:
            if iterator is None:
                break
            self._set_row(self.now_store, iterator, self._file_values(item, priorities, activity))
            iterator = self.now_store.iter_next(iterator)

    def _paint_running(self, items, priorities, activity: str) -> None:
        if items is None or getattr(items, "state", None) == "running":
            items = [items] if items is not None else []
        self._sync_now_rows(items, priorities, activity)
        current = [item.source for item in items]
        previous = list(getattr(self, "_tree_indexing_sources", None) or [])
        if not previous:
            one = getattr(self, "_tree_indexing", "") or ""
            previous = [one] if one else []
        for source in previous:
            if source not in current:
                self._rewrite_tree_file(source, priorities, activity)
        self._tree_indexing_sources = current
        self._tree_indexing = current[0] if current else ""
        for item in items:
            iterator = self._file_iters.get(item.source)
            if iterator is None:
                continue
            self._set_row(self.store, iterator, self._file_values(item, priorities, activity, tree=True))

    def _rewrite_tree_file(self, source: str, priorities, activity: str) -> None:
        """Drop the indexing mark once that file is no longer the one indexing."""
        iterator = self._file_iters.get(source)
        if iterator is None:
            return
        book = self.session.get_book() if hasattr(self.session, "get_book") else None
        item = None
        if book is not None and hasattr(book, "files_for"):
            item = book.files_for([source]).get(source)
        if item is None:
            mark = row_mark(priority=source in priorities, indexing=False)
            if self.store.get_value(iterator, COL_MARK) != mark:
                self.store.set_value(iterator, COL_MARK, mark)
            return
        self._set_row(self.store, iterator, self._file_values(item, priorities, activity, tree=True))

    def _paint_marks(self, priorities) -> None:
        running = set(getattr(self, "_tree_indexing_sources", None) or [])
        if not running:
            one = getattr(self, "_tree_indexing", "") or ""
            if one:
                running.add(one)
        for source, iterator in self._file_iters.items():
            mark = row_mark(priority=source in priorities, indexing=source in running)
            if self.store.get_value(iterator, COL_MARK) != mark:
                self.store.set_value(iterator, COL_MARK, mark)

    def _sync_visible(self, parent, entries, priorities, activity: str) -> None:
        """Keep one open level in place. A closed folder keeps a single hidden child."""
        wanted = {key for _kind, key, _payload in entries}
        found = self._child_map(parent)
        for key, iterator in list(found.items()):
            if key not in wanted:
                self._forget_subtree(iterator)
                self.store.remove(iterator)
        found = self._child_map(parent)
        book = self.session.get_book() if hasattr(self.session, "get_book") else None
        file_sources = [key for kind, key, _payload in entries if kind == "file"]
        snaps = {}
        if file_sources and book is not None and hasattr(book, "files_for"):
            snaps = book.files_for(file_sources)
        for kind, key, payload in entries:
            if kind == "file":
                item = snaps.get(key)
                if item is None:
                    continue
                values = self._file_values(item, priorities, activity, tree=True)
            else:
                values = folder_store_row(payload)
            iterator = found.get(key)
            if iterator is None:
                iterator = self.store.append(parent, values)
                found[key] = iterator
            else:
                self._set_row(self.store, iterator, values)
            if kind == "folder":
                self._folder_iters[key] = iterator
                path = self.store.get_path(iterator)
                if path is not None and self.file_tree.row_expanded(path):
                    self._sync_visible(iterator, self._child_entries(payload), priorities, activity)
                else:
                    self._hold(iterator)
            else:
                self._file_iters[key] = iterator
        order = []
        iterator = self.store.iter_children(parent)
        while iterator is not None:
            order.append(self.store.get_value(iterator, COL_SOURCE))
            iterator = self.store.iter_next(iterator)
        wanted_order = [key for _kind, key, _payload in entries]
        if order != wanted_order and len(order) == len(wanted_order) and set(order) == set(wanted_order):
            reorder_children(self.store, parent, wanted_order, COL_SOURCE)

    def _hold(self, parent) -> None:
        """One hidden child, so a closed folder still has an arrow."""
        child = self.store.iter_children(parent)
        if child is None:
            self.store.append(parent, hold_store_row())
            return
        key = self.store.get_value(child, COL_SOURCE)
        if key == FOLDER_HOLD and self.store.iter_next(child) is None:
            return
        self._drop_children(parent)
        self.store.append(parent, hold_store_row())

    def _drop_children(self, parent) -> None:
        child = self.store.iter_children(parent)
        while child is not None:
            self._forget_subtree(child)
            if not self.store.remove(child):
                break

    def _forget_subtree(self, iterator) -> None:
        child = self.store.iter_children(iterator)
        while child is not None:
            self._forget_subtree(child)
            child = self.store.iter_next(child)
        key = self.store.get_value(iterator, COL_SOURCE)
        self._file_iters.pop(key, None)
        self._folder_iters.pop(key, None)

    def _child_map(self, parent) -> dict:
        found = {}
        iterator = self.store.iter_children(parent)
        while iterator is not None:
            found[self.store.get_value(iterator, COL_SOURCE)] = iterator
            iterator = self.store.iter_next(iterator)
        return found

    def _restore_open_folders(self) -> None:
        """Re-open folders the user left open. A closed parent stays closed."""
        pending = True
        while pending:
            pending = False
            for key in list(self._open_folders):
                iterator = self._folder_iters.get(key)
                if iterator is None:
                    continue
                path = self.store.get_path(iterator)
                if path is None or self.file_tree.row_expanded(path):
                    continue
                if path.get_depth() > 1:
                    parent_path = path.copy()
                    if not parent_path.up() or not self.file_tree.row_expanded(parent_path):
                        continue
                self.file_tree.expand_row(path, False)
                pending = True

    def _iters_by_source(self, store):
        """Row iterators stay valid when a sort column moves the row."""
        found = {}
        iterator = store.get_iter_first()
        while iterator is not None:
            found[store.get_value(iterator, 0)] = iterator
            iterator = store.iter_next(iterator)
        return found

    def _set_row(self, store, iterator, values) -> None:
        for column, value in enumerate(values):
            if store.get_value(iterator, column) != value:
                store.set_value(iterator, column, value)

    def _follow_files(self, tree, source: str) -> None:
        """Bring the active file into view when it changes, near the top of the list."""
        if not source or not tree.get_realized():
            return
        if self._followed.get(id(tree)) == source:
            return
        model = tree.get_model()
        if model is None:
            return
        for index, row in enumerate(model):
            if row[0] != source:
                continue
            self._followed[id(tree)] = source
            tree.scroll_to_cell(self.gtk.TreePath.new_from_indices([index]), None, True, 0.25, 0.0)
            return

    def _show(self, widget, text: str) -> None:
        if widget.get_text() != text:
            widget.set_text(text)

    def refresh_stats(self, snap, view: dict) -> None:
        elapsed = None
        if self.started_at:
            elapsed = max(0, int(time.monotonic() - self.started_at))
        text = format_run_stats(view["phase"], elapsed, snap)
        self._show(self.stats, text)
        self.stats.set_tooltip_text(f"{text}\n{TIPS['stats']}".strip() if text else TIPS["stats"])

    def refresh_tray(self, view: dict, snap) -> None:
        if self.indicator is None:
            return
        indexing = view["phase"] in ("waiting", "running") and self.thread is not None and self.thread.is_alive()
        repairing = self.repair_thread is not None and self.repair_thread.is_alive()
        status = self._tray_status_enum
        icon = self._tray_icon(indexing, repairing)
        if icon:
            self.indicator.set_icon_full(icon, "Nomad TURBO Index")
            self.indicator.set_attention_icon_full(icon, "Nomad TURBO Index")
        if self.hidden and (indexing or repairing):
            self.indicator.set_status(status.ATTENTION)
            if indexing:
                label = str(view.get("chunks", view["percent"]))
            elif getattr(self.repair_board, "kind", "") == "scan":
                label = "scan"
            else:
                label = "repair"
            self.indicator.set_label(label, "100%")
        else:
            self.indicator.set_status(status.ACTIVE)
            self.indicator.set_label(str(view.get("chunks", view["percent"])) if indexing else "", "chunks")
        if repairing and hasattr(self, "tray_stop"):
            scanning = getattr(self.repair_board, "kind", "") == "scan"
            self.tray_stop.set_label("Stop scan" if scanning else "Stop repair")
        elif hasattr(self, "tray_stop"):
            self.tray_stop.set_label("Stop indexing")
        repair_snap = self.repair_board.snapshot() if repairing and self.repair_board is not None else None
        if repair_snap is not None:
            self.tray_status.set_label(repair_snap.overall[:120])
        elif snap is not None:
            self.tray_status.set_label(snap.overall_text[:120])

    def _tell_if_finished(self, view: dict) -> None:
        if not self.hidden or self._finish_told:
            return
        repair = self.repair_board.snapshot() if self.repair_board is not None else None
        repair_done = repair is not None and repair.phase in ("done", "stopped", "error") and not repair.busy
        if view["phase"] not in ("done", "stopped", "error") and not repair_done:
            return
        if repair_done and not self._repair_told:
            self._repair_told = True
            self._notify(repair.note or "Repair stopped.")
            return
        self._finish_told = True
        self._notify(view["note"] or "Turbo stopped.")

    def _notify(self, body: str) -> None:
        if shutil.which("notify-send"):
            subprocess.run(
                ["notify-send", "NOMAD Turbo", body],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def refresh_qdrant(self) -> None:
        self._show(self.qdrant_label, self.qdrant_text)

    def refresh_usage(self) -> None:
        view = self.session.view()
        running = view["active_slots"] if view["phase"] in ("waiting", "running") else 0
        workers = None
        getter = getattr(self.session, "ocr_workers", None)
        if callable(getter):
            workers = getter()
        from nomad_turbo.progress import server_status_text

        text = server_status_text(running, workers)
        self._show(self.server_label, text)
        self.server_label.set_tooltip_text(f"{TIPS['usage']} {text}".strip())
        self._apply_tune()

    def _apply_tune(self) -> None:
        """Show the live batch advice. Auto changes the chunk count only after finished calls earn it."""
        from nomad_turbo.tune import EmbedSample, content_kind, tune_advice

        if self._tune_applying or not hasattr(self, "tune_label"):
            return
        view = self.session.view()
        running = view["phase"] in ("waiting", "running") and self.thread is not None and self.thread.is_alive()
        gpu = self._index_memory()
        free = None
        if gpu and gpu[1]:
            free = max(0, int(gpu[1]) - int(gpu[2]))
        samples = []
        for item in self.session.embed_calls():
            size, chunks, seconds, ok = item[:4]
            kind = item[4] if len(item) > 4 else ""
            article_size = item[5] if len(item) > 5 else 0
            samples.append(EmbedSample(int(size), int(chunks), float(seconds), bool(ok), str(kind), int(article_size)))
        live_kind = ""
        source = getattr(self, "_running_source", "") or ""
        if source:
            live_kind = content_kind(source)
        count = int(view.get("chunks", view["percent"]))
        advice = tune_advice(
            samples,
            count,
            auto=bool(self.tune_auto.get_active()) if hasattr(self, "tune_auto") else False,
            running=running,
            optimizing=bool(view.get("optimize_hold")),
            gpu_free_mib=free,
            hold=self._tune_hold,
            call_count=self.session.embed_call_count(),
            now=time.monotonic(),
            memory=self._pace_memory,
            kind=live_kind,
        )
        self._tune_hold = advice.hold
        self._show(self.tune_label, advice.text)
        self.tune_label.set_tooltip_text(f"{TIPS['tune']} {advice.text}".strip())
        if not advice.move or advice.count == count:
            return
        if advice.text != self._tune_logged:
            self._tune_logged = advice.text
            log.info("embed batch: %s", advice.text)
        self._show_chunk_count(advice.count)
        self._queue_chunk_save(advice.count)

    def refresh_log(self) -> None:
        show_info = self.info_check.get_active()
        show_debug = self.debug_check.get_active()
        show_errors = self.error_check.get_active()
        if self.filter_dirty:
            lines = self.memory.since(0)
        else:
            lines = self.memory.since(self.last_seq)
            if not lines:
                self._trim_log_pane()
                return
        adjustment = self.log_scroll.get_vadjustment()
        follow = self.follow_check.get_active()
        stick = log_should_stick(
            follow,
            adjustment.get_value(),
            adjustment.get_page_size(),
            adjustment.get_upper(),
            away=bool(getattr(self, "_log_away", False)),
        )
        # New lines move the adjustment. That is not the user leaving the end.
        self._log_pinning += 1
        try:
            if self.filter_dirty:
                self.log_buffer.set_text("")
                self.last_seq = 0
                self.filter_dirty = False
                stick = follow
            # Inserting at the end moves the cursor, and the text view scrolls to
            # that cursor on its own. Hold a mark when Follow is off, then put the
            # cursor back on it so the new lines do not pull the page down.
            anchor = None
            if not stick:
                anchor = self.log_buffer.create_mark(None, self._log_anchor_iter(), True)
            added = False
            for seq, level, msg in lines:
                if seq <= self.last_seq:
                    continue
                self.last_seq = seq
                if not log_line_visible(level, show_info, show_debug, show_errors):
                    continue
                self.log_buffer.insert(self.log_buffer.get_end_iter(), msg + "\n")
                added = True
            self._trim_log_pane()
            if not added:
                if anchor is not None:
                    self.log_buffer.delete_mark(anchor)
                return
            if stick or anchor is None or anchor.get_deleted():
                self.log_buffer.place_cursor(self.log_buffer.get_end_iter())
                self._settle_log(bottom=True)
            else:
                self.log_buffer.place_cursor(self.log_buffer.get_iter_at_mark(anchor))
                self._settle_log(bottom=False, mark=anchor)
        finally:
            self._log_pinning -= 1

    def _trim_log_pane(self) -> None:
        drop = log_lines_to_drop(self.log_buffer.get_line_count())
        if drop <= 0:
            return
        start = self.log_buffer.get_start_iter()
        cut = self.log_buffer.get_iter_at_line(drop)
        self.log_buffer.delete(start, cut)

    def _release_log_pane(self) -> None:
        """Drop the on-screen log before quit so a long run can close."""
        buffer = getattr(self, "log_buffer", None)
        if buffer is None:
            return
        try:
            buffer.set_text("")
        except Exception:
            return

    def _log_anchor_iter(self):
        try:
            rect = self.log_view.get_visible_rect()
            ok, iterator = self.log_view.get_iter_at_location(rect.x, rect.y)
        except Exception:
            return self.log_buffer.get_start_iter()
        if not ok:
            return self.log_buffer.get_start_iter()
        return iterator

    def _settle_log(self, *, bottom: bool, mark=None) -> None:
        def apply_pin() -> bool:
            self._log_pinning += 1
            try:
                adjustment = self.log_scroll.get_vadjustment()
                if bottom and self.follow_check.get_active() and not self._log_away:
                    adjustment.set_value(max(0.0, adjustment.get_upper() - adjustment.get_page_size()))
                elif mark is not None and not mark.get_deleted():
                    self.log_view.scroll_to_mark(mark, 0.0, True, 0.0, 0.0)
                    self.log_buffer.delete_mark(mark)
            finally:
                self._log_pinning -= 1
            return False

        self.glib.idle_add(apply_pin, priority=self.glib.PRIORITY_LOW)

    def _log_near_end(self) -> bool:
        adjustment = self.log_scroll.get_vadjustment()
        value = float(adjustment.get_value())
        page = float(adjustment.get_page_size())
        upper = float(adjustment.get_upper())
        if page <= 0 or upper <= page + 1:
            return True
        return value + page >= upper - 24

    def _finish_log_user(self) -> bool:
        if not self._log_pinning and self.follow_check.get_active():
            self._log_away = not self._log_near_end()
        self._log_user = False
        return False

    def _arm_log_user(self, *_args) -> bool:
        self._log_user = True
        return False

    def _on_log_user_scroll(self, *_args) -> bool:
        self._log_user = True
        self.glib.idle_add(self._finish_log_user)
        return False

    def _on_log_key(self, _widget, event) -> bool:
        key = int(getattr(event, "keyval", 0) or 0)
        # Up, Down, Page Up, Page Down, Home, End, space. Other keys are not a scroll.
        if key in (32, 65360, 65361, 65362, 65363, 65364, 65365, 65366, 65367):
            return self._on_log_user_scroll()
        return False

    def _watch_log_scroll(self) -> None:
        """A scroll the user made holds the place. Lines that arrive do not."""

        def on_value(_adjustment) -> None:
            if self._log_pinning or not self._log_user:
                return
            if not self.follow_check.get_active():
                return
            self._log_away = not self._log_near_end()

        self.log_scroll.get_vadjustment().connect("value-changed", on_value)
        self.log_view.connect("scroll-event", self._on_log_user_scroll)
        self.log_scroll.connect("scroll-event", self._on_log_user_scroll)
        self.log_view.connect("key-press-event", self._on_log_key)
        bar = self.log_scroll.get_vscrollbar()
        if bar is not None:
            bar.connect("button-press-event", self._arm_log_user)
            bar.connect("button-release-event", self._on_log_user_scroll)

    def _build_repair_page(self, headed) -> None:
        gtk = self.gtk
        page = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=8)
        page.set_vexpand(True)
        self.repair_status = gtk.Label(label="Ready. Scan changes nothing until you press Start repair.")
        self.repair_status.set_xalign(0)
        self.repair_status.set_line_wrap(True)
        self.repair_status.set_tooltip_text(TIPS["repair_page"])
        page.pack_start(self.repair_status, False, False, 0)

        from gi.repository import Pango

        self.repair_status.set_ellipsize(Pango.EllipsizeMode.END)
        self.repair_status.set_lines(4)

        self.repair_store = gtk.ListStore(*repair_store_types())
        tree = gtk.TreeView(model=self.repair_store)
        tree.set_headers_visible(True)
        tree.set_tooltip_column(8)
        name_cell = gtk.CellRendererText()
        name_cell.set_property("ellipsize", Pango.EllipsizeMode.MIDDLE)
        self._add_column(
            tree,
            "repair:File",
            headed(
                "File",
                TIPS["repair_col_file"],
                name_cell,
                text=1,
                foreground=9,
            ),
            text_index=1,
            minimum=180,
            maximum=460,
            expand=True,
            sort_index=1,
        )
        problem_cell = gtk.CellRendererText()
        problem_cell.set_property("ellipsize", Pango.EllipsizeMode.END)
        self._add_column(
            tree,
            "repair:Problem",
            headed("Problem", TIPS["repair_col_problem"], problem_cell, text=2),
            text_index=2,
            minimum=140,
            maximum=420,
            expand=True,
            sort_index=2,
        )
        self._add_column(
            tree,
            "repair:Points",
            headed(
                "Points",
                TIPS["repair_col_points"],
                gtk.CellRendererText(),
                text=3,
            ),
            text_index=3,
            minimum=80,
            maximum=180,
            sort_index=10,
        )
        self._add_column(
            tree,
            "repair:Progress",
            headed(
                "Progress",
                TIPS["repair_col_progress"],
                gtk.CellRendererProgress(),
                value=4,
                pulse=5,
            ),
            text_index=None,
            minimum=160,
            maximum=280,
            sort_index=4,
        )
        self._add_column(
            tree,
            "repair:Status",
            headed(
                "Status",
                TIPS["repair_col_status"],
                gtk.CellRendererText(),
                text=6,
            ),
            text_index=6,
            minimum=90,
            maximum=220,
            sort_index=6,
        )
        self._add_column(
            tree,
            "repair:Elapsed",
            headed("Elapsed", TIPS["repair_col_elapsed"], gtk.CellRendererText(), text=7),
            text_index=7,
            minimum=70,
            maximum=140,
            sort_index=11,
        )
        scrolled = gtk.ScrolledWindow()
        scrolled.set_policy(gtk.PolicyType.AUTOMATIC, gtk.PolicyType.AUTOMATIC)
        scrolled.add(tree)
        scrolled.set_min_content_height(1)
        scrolled.set_min_content_width(1)
        scrolled.set_propagate_natural_width(False)
        scrolled.set_vexpand(True)
        self.repair_tree = tree
        self.repair_list = scrolled
        lower = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=4)

        self.repair_current_label = gtk.Label(label="No file is being repaired.")
        self.repair_current_label.set_xalign(0)
        self.repair_current_label.set_hexpand(True)
        self.repair_current_label.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
        self.repair_current_label.set_width_chars(10)
        self.repair_current_label.set_valign(gtk.Align.CENTER)
        self.repair_current_label.set_tooltip_text(TIPS["repair_file_now"])
        self.repair_current_bar = gtk.ProgressBar()
        self.repair_current_bar.set_show_text(True)
        self.repair_current_bar.set_text("Idle")
        self.repair_current_bar.set_hexpand(True)
        self.repair_current_bar.set_valign(gtk.Align.CENTER)
        self.repair_current_bar.set_tooltip_text(TIPS["repair_file_bar"])
        self.repair_overall = gtk.ProgressBar()
        self.repair_overall.set_show_text(True)
        self.repair_overall.set_text("Idle")
        self.repair_overall.set_hexpand(True)
        self.repair_overall.set_valign(gtk.Align.CENTER)
        self.repair_overall.set_tooltip_text(TIPS["repair_overall_bar"])
        self.repair_current_bar.set_size_request(48, -1)
        self.repair_overall.set_size_request(48, -1)
        repair_bars = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        repair_bars.pack_start(self.repair_current_label, True, True, 0)
        repair_bars.pack_start(self.repair_current_bar, True, True, 0)
        repair_bars.pack_start(self.repair_overall, True, True, 0)
        lower.pack_start(repair_bars, False, False, 0)

        self.repair_stats = gtk.Label(label="Press Scan to check the index without changing it.")
        self.repair_stats.set_xalign(0)
        self.repair_stats.set_line_wrap(True)
        self.repair_stats.set_lines(6)
        self.repair_stats.set_ellipsize(Pango.EllipsizeMode.END)
        self.repair_stats.set_max_width_chars(110)
        self.repair_stats.set_selectable(True)
        self.repair_stats.override_font(Pango.FontDescription("Monospace"))
        self.repair_stats.set_tooltip_text(TIPS["repair_stats"])
        lower.pack_start(self.repair_stats, False, False, 0)

        self.repair_specs = gtk.Label(
            label="Each row's spec is kept up to date while that file is repaired. Hover a row to read it."
        )
        self.repair_specs.set_xalign(0)
        self.repair_specs.set_hexpand(True)
        self.repair_specs.set_ellipsize(Pango.EllipsizeMode.END)
        self.repair_specs.set_selectable(True)
        self.repair_specs.set_tooltip_text(self.repair_specs.get_text())
        lower.pack_start(self.repair_specs, False, False, 0)

        self.repair_quick = gtk.ToggleButton(label="Quick repair")
        self.repair_complete = gtk.ToggleButton(label="Complete repair")
        self.repair_quick.set_active(True)
        self.repair_quick.set_tooltip_text(TIPS["repair_quick"])
        self.repair_complete.set_tooltip_text(TIPS["repair_complete"])
        self._bind_choice(self.repair_quick, self.repair_complete)
        self.repair_choice = self._linked(self.repair_quick, self.repair_complete)
        self.repair_scan = gtk.Button(label="Scan")
        self.repair_start = gtk.Button(label="Start repair")
        self.repair_stop_button = gtk.Button(label="Stop")
        self.repair_scan_tip = TIPS["repair_scan"]
        self.repair_scan.set_tooltip_text(TIPS["repair_scan"])
        self.repair_start_tip = TIPS["repair_start"]
        self.repair_start.set_tooltip_text(TIPS["repair_start"])
        self.repair_stop_button.set_tooltip_text(TIPS["repair_stop"])
        self.repair_scan.connect("clicked", lambda *_args: self.on_repair_scan())
        self.repair_start.connect("clicked", lambda *_args: self.on_repair_start())
        self.repair_stop_button.connect("clicked", lambda *_args: self.repair_stop.set())
        self.repair_stop_button.set_sensitive(False)
        self.optimize_segments = gtk.Button(label="Optimize stuck segments")
        self.optimize_segments_tip = TIPS["repair_optimize"]
        self.optimize_segments.set_tooltip_text(TIPS["repair_optimize"])
        self.optimize_segments.connect("clicked", lambda *_args: self.on_optimize_segments())
        self.repair_actions = self._linked(self.repair_scan, self.repair_start, self.repair_stop_button)
        buttons = self._flow(self.repair_choice, self.repair_actions)
        buttons.set_tooltip_text(TIPS["repair_page"])
        lower.pack_start(buttons, False, False, 0)
        # Own row so the file-repair buttons keep their width on a short window.
        optimize_row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        optimize_row.pack_start(self.optimize_segments, False, False, 0)
        self.repair_optimize_bar = gtk.ProgressBar()
        self.repair_optimize_bar.set_show_text(True)
        self.repair_optimize_bar.set_text("Not optimizing")
        self.repair_optimize_bar.set_hexpand(True)
        self.repair_optimize_bar.set_valign(gtk.Align.CENTER)
        self.repair_optimize_bar.set_tooltip_text(TIPS["repair_optimize_bar"])
        optimize_row.pack_start(self.repair_optimize_bar, True, True, 0)
        lower.pack_start(optimize_row, False, False, 0)
        self.repair_buttons = buttons
        self.repair_controls = lower
        page.pack_start(self.repair_list, True, True, 0)
        page.pack_start(self.repair_controls, False, False, 0)
        self.stack.add_titled(page, "repair", "Repair")
        self.repair_page = page

    def _index_memory(self):
        """The card indexing will use. None when indexing is on the CPU."""
        from nomad_turbo.devices import pick_index

        devices = getattr(self, "_devices", None)
        if not devices:
            return self.live_gpu
        choice = self.session.index_device() if hasattr(self.session, "index_device") else "auto"
        device = pick_index(devices, choice)
        if device.kind != "nvidia":
            return None
        return (device.index, device.total_mib, device.used_mib)

    def _build_settings_page(self) -> None:
        """CPU or a card for indexing, and CPU or a card for OCR."""
        from nomad_turbo.devices import list_devices, normalize_device_choice

        gtk = self.gtk
        page = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=8)
        page.set_vexpand(True)
        scroll = gtk.ScrolledWindow()
        scroll.set_policy(gtk.PolicyType.NEVER, gtk.PolicyType.AUTOMATIC)
        inner = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=12)
        inner.set_margin_top(4)
        inner.set_margin_bottom(8)
        heading = gtk.Label()
        heading.set_markup("<span size='large' weight='bold'>Settings</span>")
        heading.set_xalign(0)
        inner.pack_start(heading, False, False, 0)

        self._devices = list_devices()
        self._ocr_probe = {}
        self._device_guard = False
        self._device_touched = False
        self.session.set_index_device(normalize_device_choice(getattr(self.install, "index_device", "auto")))
        self.session.set_ocr_device(normalize_device_choice(getattr(self.install, "ocr_device", "auto")))

        self.index_cpu = gtk.ToggleButton(label="CPU")
        self.index_gpu = gtk.ToggleButton(label="GPU")
        self.ocr_cpu = gtk.ToggleButton(label="CPU")
        self.ocr_gpu = gtk.ToggleButton(label="GPU")
        for button, tip in (
            (self.index_cpu, TIPS["settings_index"]),
            (self.index_gpu, TIPS["settings_index"]),
            (self.ocr_cpu, TIPS["settings_ocr"]),
            (self.ocr_gpu, TIPS["settings_ocr"]),
        ):
            button.set_tooltip_text(tip)
        self.index_combo = gtk.ComboBoxText()
        self.index_combo.set_tooltip_text(TIPS["settings_index"])
        self.ocr_combo = gtk.ComboBoxText()
        self.ocr_combo.set_tooltip_text(TIPS["settings_ocr"])
        nvidia = [device for device in self._devices if device.kind == "nvidia"]
        cards = [device for device in self._devices if device.kind != "cpu"]
        for device in nvidia:
            self.index_combo.append(device.id, device.name)
        for device in cards:
            self.ocr_combo.append(device.id, device.name)
        if not nvidia:
            self.index_gpu.set_sensitive(False)
        # An NVIDIA card stays locked until its reader is installed and checked.
        # Another card can still be chosen; the sentence says when it falls back.
        if not cards or nvidia:
            self.ocr_gpu.set_sensitive(False)

        self.ocr_install = gtk.Button(label="Install GPU reader")
        self.ocr_install.set_tooltip_text(TIPS["settings_install"])
        self.ocr_install.set_sensitive(False)
        self.ocr_install.connect("clicked", self.on_ocr_install)
        self.ocr_install_bar = gtk.ProgressBar()
        self.ocr_install_bar.set_show_text(False)
        self.ocr_install_bar.set_hexpand(True)
        self.ocr_install_note = gtk.Label(label="Checking the GPU reader…")
        self.ocr_install_note.set_xalign(0)
        self.ocr_install_note.set_line_wrap(True)
        self.ocr_install_note.set_max_width_chars(80)
        self.ocr_install_note.set_selectable(True)
        self.ocr_install_note.set_tooltip_text(TIPS["settings_install"])
        install_row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        install_row.pack_start(self.ocr_install, False, False, 0)
        install_row.pack_start(self.ocr_install_bar, True, True, 0)
        self.ocr_install_row = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=4)
        self.ocr_install_row.pack_start(install_row, False, False, 0)
        self.ocr_install_row.pack_start(self.ocr_install_note, False, False, 0)
        self.ocr_install_bar.hide()
        self._ocr_installing = False
        self._ocr_install_stop = threading.Event()
        self._ocr_pulse = None
        if not nvidia:
            self.ocr_install_row.hide()

        self.device_sentence = gtk.Label(label="")
        self.device_sentence.set_xalign(0)
        self.device_sentence.set_line_wrap(True)
        self.device_sentence.set_max_width_chars(80)
        self.device_sentence.set_selectable(True)
        self.device_sentence.set_tooltip_text(TIPS["settings_engine"])

        inner.pack_start(self._device_block("Indexing", TIPS["settings_index"], self.index_cpu, self.index_gpu, self.index_combo), False, False, 0)
        inner.pack_start(self._device_block("OCR", TIPS["settings_ocr"], self.ocr_cpu, self.ocr_gpu, self.ocr_combo), False, False, 0)
        inner.pack_start(self.ocr_install_row, False, False, 0)
        inner.pack_start(self.device_sentence, False, False, 0)
        scroll.add(inner)
        page.pack_start(scroll, True, True, 0)
        self.stack.add_titled(page, "settings", "Settings")
        self.settings_page = page
        self._paint_devices()
        self.index_cpu.connect("toggled", self.on_device_mode, "index", "cpu")
        self.index_gpu.connect("toggled", self.on_device_mode, "index", "gpu")
        self.ocr_cpu.connect("toggled", self.on_device_mode, "ocr", "cpu")
        self.ocr_gpu.connect("toggled", self.on_device_mode, "ocr", "gpu")
        self.index_combo.connect("changed", self.on_device_card, "index")
        self.ocr_combo.connect("changed", self.on_device_card, "ocr")
        threading.Thread(target=self._load_ocr_probe, name="nomad-turbo-ocr-probe", daemon=True).start()

    def _device_block(self, title, tip, cpu, gpu, combo):
        gtk = self.gtk
        block = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=4)
        caption = gtk.Label(label=title)
        caption.set_xalign(0)
        caption.set_tooltip_text(tip)
        line = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        card = gtk.Label(label="Card")
        card.set_tooltip_text(tip)
        line.pack_start(self._linked(cpu, gpu), False, False, 0)
        line.pack_start(card, False, False, 0)
        line.pack_start(combo, False, False, 0)
        block.pack_start(caption, False, False, 0)
        block.pack_start(line, False, False, 0)
        return block

    def _paint_devices(self) -> None:
        from nomad_turbo.devices import assignment_sentence, resolve_assignment

        if getattr(self, "_device_guard", False):
            return
        self._device_guard = True
        try:
            assignment = resolve_assignment(
                self._devices,
                self.session.index_device(),
                self.session.ocr_device(),
                getattr(self, "_ocr_probe", {}) or {},
            )
            index_gpu = assignment.index.kind == "nvidia" and self.index_gpu.get_sensitive()
            self.index_cpu.set_active(not index_gpu)
            self.index_gpu.set_active(index_gpu)
            if assignment.index.kind == "nvidia":
                self.index_combo.set_active_id(assignment.index.id)
            elif self.index_combo.get_active() < 0:
                self.index_combo.set_active(0)
            self.index_combo.set_sensitive(bool(index_gpu))
            ocr_choice = self.session.ocr_device()
            if ocr_choice == "cpu" or not self.ocr_gpu.get_sensitive():
                ocr_gpu = False
            elif ocr_choice == "auto":
                ocr_gpu = assignment.ocr.kind != "cpu"
            else:
                ocr_gpu = True
            self.ocr_cpu.set_active(not ocr_gpu)
            self.ocr_gpu.set_active(ocr_gpu)
            if ocr_choice.startswith(("nvidia:", "amd:", "intel:")):
                self.ocr_combo.set_active_id(ocr_choice)
            elif assignment.ocr.kind != "cpu":
                self.ocr_combo.set_active_id(assignment.ocr.id)
            elif self.ocr_combo.get_active() < 0:
                self.ocr_combo.set_active(0)
            self.ocr_combo.set_sensitive(bool(ocr_gpu))
            self.device_sentence.set_text(assignment_sentence(assignment))
        finally:
            self._device_guard = False

    def _save_devices(self) -> None:
        self.install.index_device = self.session.index_device()
        self.install.ocr_device = self.session.ocr_device()
        try:
            save_install(self.install)
        except Exception:
            log.exception("could not save the device settings")

    def on_device_mode(self, button, which: str, kind: str) -> None:
        """CPU or GPU. The last one stays on. GPU uses the card in the list."""
        if getattr(self, "_device_guard", False):
            return
        cpu = self.index_cpu if which == "index" else self.ocr_cpu
        gpu = self.index_gpu if which == "index" else self.ocr_gpu
        combo = self.index_combo if which == "index" else self.ocr_combo
        if not button.get_active():
            if not cpu.get_active() and not gpu.get_active():
                self._device_guard = True
                button.set_active(True)
                self._device_guard = False
            return
        self._device_touched = True
        self._device_guard = True
        try:
            cpu.set_active(kind == "cpu")
            gpu.set_active(kind == "gpu" and gpu.get_sensitive())
            if kind == "gpu" and gpu.get_sensitive():
                if combo.get_active() < 0:
                    combo.set_active(0)
                chosen = combo.get_active_id() or "cpu"
            else:
                chosen = "cpu"
            if which == "index":
                self.session.set_index_device(chosen)
            else:
                self.session.set_ocr_device(chosen)
        finally:
            self._device_guard = False
        self._paint_devices()
        self._save_devices()

    def on_device_card(self, combo, which: str) -> None:
        if getattr(self, "_device_guard", False):
            return
        gpu = self.index_gpu if which == "index" else self.ocr_gpu
        if not gpu.get_active():
            return
        chosen = combo.get_active_id()
        if not chosen:
            return
        self._device_touched = True
        if which == "index":
            self.session.set_index_device(chosen)
        else:
            self.session.set_ocr_device(chosen)
        self._paint_devices()
        self._save_devices()

    def on_ocr_install(self, *_args) -> None:
        if self._ocr_installing:
            self._ocr_install_stop.set()
            self.ocr_install.set_sensitive(False)
            self.ocr_install_note.set_text("Stopping the download.")
            return
        gtk = self.gtk
        dialog = gtk.MessageDialog(
            parent=self.window,
            flags=0,
            message_type=gtk.MessageType.QUESTION,
            buttons=gtk.ButtonsType.NONE,
            text="Install the GPU reader?",
        )
        dialog.format_secondary_text(
            "This downloads PyTorch and EasyOCR into your home folder. "
            "The download is a few gigabytes and needs a network connection. "
            "GPU OCR turns on when the install finishes and a small check on the card succeeds. "
            "The English models download the first time a page is read."
        )
        dialog.add_button("Cancel", gtk.ResponseType.CANCEL)
        dialog.add_button("Install", gtk.ResponseType.OK)
        dialog.show_all()
        response = dialog.run()
        dialog.destroy()
        if response != gtk.ResponseType.OK:
            return
        self._ocr_installing = True
        self._ocr_install_stop = threading.Event()
        self.ocr_install.set_label("Cancel")
        self.ocr_gpu.set_sensitive(False)
        self.ocr_install_bar.show()
        self.ocr_install_note.set_text("Starting the download.")
        self._paint_devices()
        if self._ocr_pulse:
            self.glib.source_remove(self._ocr_pulse)
        self._ocr_pulse = self.glib.timeout_add(120, self._pulse_ocr_install)
        threading.Thread(target=self._install_ocr_reader, name="nomad-turbo-ocr-install", daemon=True).start()

    def _pulse_ocr_install(self) -> bool:
        if not self._ocr_installing:
            self._ocr_pulse = None
            return False
        self.ocr_install_bar.pulse()
        return True

    def _show_ocr_install_line(self, text: str) -> bool:
        if text:
            self.ocr_install_note.set_text(text)
        return False

    def _install_ocr_reader(self) -> None:
        from nomad_turbo.ocr_engine import probe_info, probe_map
        from nomad_turbo.ocr_setup import install_gpu_reader

        error = install_gpu_reader(lambda text: self.glib.idle_add(self._show_ocr_install_line, text), self._ocr_install_stop)
        probes = {}
        if not error and not self._ocr_install_stop.is_set():
            try:
                probe_info(force=True)
                probes = probe_map(self._devices)
            except Exception as exc:
                log.exception("could not check the GPU OCR runtime")
                error = str(exc)
        self.glib.idle_add(self._finish_ocr_install, error, probes)

    def _finish_ocr_install(self, error: str, probes: dict) -> bool:
        from nomad_turbo.ocr_setup import reader_ready

        self._ocr_installing = False
        if self._ocr_pulse:
            self.glib.source_remove(self._ocr_pulse)
            self._ocr_pulse = None
        self.ocr_install_bar.hide()
        if probes:
            self._ocr_probe = probes
        ready = reader_ready(self._devices, self._ocr_probe)
        self._apply_ocr_lock(ready)
        self.ocr_install.set_label("Install GPU reader")
        if error:
            self.ocr_install_note.set_text(error)
            self.ocr_install.set_sensitive(True)
            self.ocr_install.show()
        elif ready:
            self.ocr_install_note.set_text("GPU reader installed. Choose GPU to use it on the next file.")
            self.ocr_install.hide()
        else:
            self.ocr_install_note.set_text("The download finished, but the card could not run the reader.")
            self.ocr_install.set_sensitive(True)
            self.ocr_install.show()
        return False

    def _apply_ocr_lock(self, ready: bool) -> None:
        nvidia = any(device.kind == "nvidia" for device in self._devices)
        cards = any(device.kind != "cpu" for device in self._devices)
        if ready or not nvidia:
            self.ocr_gpu.set_sensitive(bool(cards))
        else:
            self.ocr_gpu.set_sensitive(False)
        self._paint_devices()

    def _load_ocr_probe(self) -> None:
        from nomad_turbo.ocr_engine import ocr_python, probe_info, probe_map

        if not any(device.kind == "nvidia" for device in self._devices):
            return
        if not ocr_python():
            self.glib.idle_add(self._finish_ocr_probe, {}, "")
            return
        try:
            probe_info(force=True)
            probes = probe_map(self._devices)
            error = ""
        except Exception as exc:
            log.exception("could not check the GPU OCR runtime")
            probes = {}
            error = str(exc)
        self.glib.idle_add(self._finish_ocr_probe, probes, error)

    def _finish_ocr_probe(self, probes: dict, error: str) -> bool:
        from nomad_turbo.ocr_setup import reader_ready

        if self._ocr_installing:
            return False
        if probes:
            self._ocr_probe = probes
        ready = reader_ready(self._devices, self._ocr_probe)
        self._apply_ocr_lock(ready)
        if error:
            self.ocr_install_note.set_text(error)
            self.ocr_install.set_sensitive(True)
            self.ocr_install.show()
        elif ready:
            self.ocr_install_note.set_text("GPU reader installed. Choose GPU to use it on the next file.")
            self.ocr_install.hide()
        else:
            self.ocr_install_note.set_text("GPU reader is not installed.")
            self.ocr_install.set_label("Install GPU reader")
            self.ocr_install.set_sensitive(True)
            self.ocr_install.show()
        return False

    def _build_about_page(self) -> None:
        gtk = self.gtk
        facts = about_facts()
        page = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=8)
        page.set_vexpand(True)
        scroll = gtk.ScrolledWindow()
        scroll.set_policy(gtk.PolicyType.NEVER, gtk.PolicyType.AUTOMATIC)
        inner = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=10)
        inner.set_margin_top(4)
        inner.set_margin_bottom(4)

        heading = gtk.Label()
        heading.set_markup("<span size='large' weight='bold'>About</span>")
        heading.set_xalign(0)
        self.about_heading = heading

        version = gtk.Label(label=facts["version_line"])
        version.set_xalign(0)
        version.set_selectable(True)
        self.about_version = version

        tested = gtk.Label(label=facts["tested"])
        tested.set_xalign(0)
        tested.set_line_wrap(True)
        tested.set_selectable(True)
        self.about_tested = tested

        body = gtk.Label(label=facts["body"])
        body.set_xalign(0)
        body.set_yalign(0)
        body.set_line_wrap(True)
        body.set_selectable(True)
        body.set_max_width_chars(80)
        self.about_body = body

        turbo = gtk.LinkButton.new_with_label(facts["turbo_repo"], facts["turbo_label"])
        turbo.set_halign(gtk.Align.START)
        turbo.set_tooltip_text(TIPS["project"])
        nomad = gtk.LinkButton.new_with_label(facts["nomad_repo"], facts["nomad_label"])
        nomad.set_halign(gtk.Align.START)
        nomad.set_tooltip_text(TIPS["nomad_repo"])
        self.about_turbo_link = turbo
        self.about_nomad_link = nomad

        for widget in (heading, version, tested, body, turbo, nomad):
            inner.pack_start(widget, False, False, 0)
        scroll.add(inner)
        page.pack_start(scroll, True, True, 0)
        self.stack.add_titled(page, "about", "About")
        self.about_page = page

    def _set_repair_status(self, text: str) -> None:
        self._show(self.repair_status, text)
        note = text.strip()
        guide = TIPS["repair_page"]
        self.repair_status.set_tooltip_text(f"{note}\n\n{guide}" if note else guide)

    def _gate_tip(self, button, normal: str) -> None:
        if self.qdrant_ready:
            button.set_tooltip_text(normal)
        else:
            button.set_tooltip_text(TIPS["qdrant_gate"])

    def on_optimize_segments(self) -> None:
        """Force optimization of stuck segments. The file-repair list can be empty."""
        from nomad_turbo import segment_guard

        if self._work_alive() or self.scanning or not changes_allowed(self.qdrant_ready):
            why = []
            if self._work_alive():
                why.append("indexing, repair, or another optimize is still running")
            if self.scanning:
                why.append("a library scan is still running")
            if not changes_allowed(self.qdrant_ready):
                why.append("Qdrant is not ready for changes")
            segment_guard.report_optimize(
                "not started",
                "The button was pressed but " + ", and ".join(why) + ". The collection was left unchanged.",
            )
            return
        self.optimize_starting = True
        segment_guard.report_optimize(
            "checking",
            "Reading segment sizes. Points and the collection stay.",
        )
        self._set_repair_status("Checking stuck segments. The result is written to the log.")
        self._show(self.repair_optimize_bar, "Checking stuck segments")
        self.repair_optimize_bar.set_fraction(0)
        self.stack.set_visible_child_name("repair")
        self.optimize_thread = threading.Thread(
            target=self._optimize_segments,
            name="nomad-turbo-optimize",
            daemon=True,
        )
        self.optimize_thread.start()

    def _optimize_segments(self) -> None:
        import importlib

        import nomad_turbo.segment_guard as segment_guard

        try:
            importlib.reload(segment_guard)
            facts = segment_guard.measure_repair_facts(self.install)
            result = segment_guard.apply_force_optimization(
                facts,
                segment_guard.LiveOptimizeBoundary(self.install),
            )
        except Exception as exc:
            log.exception("optimize stuck segments failed")
            result = segment_guard.ForceResult(
                False,
                False,
                "Optimize stuck segments failed: "
                f"{exc}. The collection was left unchanged.",
            )
        segment_guard.report_optimize(
            "started" if result.ok else ("failed" if result.requested else "refused"),
            result.reason,
        )
        self.glib.idle_add(self._apply_optimize_result, result)

    def _apply_optimize_result(self, result) -> bool:
        from nomad_turbo.segment_guard import optimize_bar_after_click

        self._set_repair_status(result.reason)
        text, pulse = optimize_bar_after_click(result.ok, result.requested, result.reason)
        bar = self.repair_optimize_bar
        if pulse:
            bar.set_fraction(0)
            bar.pulse()
        else:
            bar.set_fraction(0)
        self._show(bar, text)
        bar.set_tooltip_text(result.reason)
        self.optimize_starting = False
        return False

    def _paint_index_optimize(self, reason: str) -> None:
        from nomad_turbo.segment_guard import pause_caption, track_optimize_pause

        bar = getattr(self, "index_optimize", None)
        if bar is None:
            return
        now = time.monotonic()
        pause = track_optimize_pause(self._optimize_pause, reason, now)
        changed = self._optimize_pause is None or pause.generation != self._optimize_pause.generation
        self._optimize_pause = pause
        if not pause.active and pause.generation == 0:
            if bar.get_visible():
                bar.hide()
            return
        if not bar.get_visible():
            bar.show()
        text, fraction = pause_caption(pause, now)
        if changed:
            bar.set_fraction(0)
        if fraction is None:
            bar.pulse()
        else:
            bar.set_fraction(min(1.0, max(0.0, fraction)))
        self._show(bar, text)

    def _paint_repair_optimize(self) -> None:
        if self._optimize_busy():
            return
        live = self._optimize_live
        bar = getattr(self, "repair_optimize_bar", None)
        if live is None or bar is None:
            return
        text, detail, pulse = live
        if pulse:
            bar.pulse()
        else:
            bar.set_fraction(0)
        self._show(bar, text)
        if detail:
            bar.set_tooltip_text(detail)

    def on_repair_scan(self) -> None:
        if self._work_alive() or self.scanning or not self.qdrant_ready:
            return
        mode = "complete" if self.repair_complete.get_active() else "quick"
        self.repair_stop = threading.Event()
        self._repair_told = False
        self.repair_starting = True
        self._set_repair_status("Starting scan…")
        self.stack.set_visible_child_name("repair")
        self.repair_thread = threading.Thread(
            target=self._repair_scan,
            args=(mode,),
            name="nomad-turbo-repair-scan",
            daemon=True,
        )
        self.repair_thread.start()

    def _repair_scan(self, mode: str) -> None:
        import importlib

        import nomad_turbo.repair as repair

        # Scan does not pause NOMAD. The desktop icon does not change.
        try:
            self._reload_indexer()
            importlib.reload(repair)
            board = repair.RepairBoard()
            self.repair_board = board
            repair.run_scan(self.repair_stop, board, mode, repair.live_deps(self.install))
        except Exception as exc:
            log.exception("scan failed")
            board = self.repair_board
            if board is not None:
                board.set_phase("error", str(exc))
            else:
                self.glib.idle_add(self._set_repair_status, str(exc))
        finally:
            self.repair_starting = False

    def on_repair_start(self) -> None:
        if self._work_alive() or self.scanning or not self.qdrant_ready:
            return
        mode = "complete" if self.repair_complete.get_active() else "quick"
        self.repair_stop = threading.Event()
        self._repair_told = False
        self.repair_starting = True
        self._set_repair_status("Starting repair…")
        self.stack.set_visible_child_name("repair")
        self.repair_thread = threading.Thread(
            target=self._repair,
            args=(mode,),
            name="nomad-turbo-repair",
            daemon=True,
        )
        self.repair_thread.start()

    def _repair(self, mode: str) -> None:
        import importlib

        import nomad_turbo.repair as repair

        # The desktop icon does not change. Stop is in this window and the tray.
        try:
            self._reload_indexer()
            importlib.reload(repair)
            board = repair.RepairBoard()
            self.repair_board = board
            repair.run_repair(self.repair_stop, board, mode, repair.live_deps(self.install))
        except Exception as exc:
            log.exception("repair failed")
            board = self.repair_board
            if board is not None:
                board.set_phase("error", str(exc))
            else:
                self.glib.idle_add(self._set_repair_status, str(exc))
        finally:
            self.repair_starting = False

    def refresh_repair(self) -> None:
        board = self.repair_board
        if board is None or not hasattr(self, "repair_store"):
            return
        snap = board.snapshot()
        if snap.note:
            self._set_repair_status(snap.note)
        seen = {row.source for row in snap.rows}
        for index in range(len(self.repair_store) - 1, -1, -1):
            if self.repair_store[index][0] not in seen:
                del self.repair_store[index]
        existing = self._iters_by_source(self.repair_store)
        for item in snap.rows:
            tooltip = f"{item.source}\n{item.detail}" if item.source else item.detail
            values = (
                item.source,
                item.name,
                item.problem,
                item.points,
                item.percent,
                item.pulse,
                item.status,
                item.elapsed,
                tooltip,
                item.color,
                int(item.points_value),
                int(item.elapsed_seconds),
            )
            if item.source in existing:
                self._set_row(self.repair_store, existing[item.source], values)
            else:
                self.repair_store.append(values)
        source = next((row.source for row in snap.rows if row.pulse >= 0 and row.source), "")
        self._refit_columns(self.repair_tree)
        self._follow_files(self.repair_tree, source)
        title, fraction, text = format_repair_current(snap)
        self._show(self.repair_current_label, title)
        self.repair_current_label.set_tooltip_text(title if source else TIPS["repair_file_now"])
        self.repair_current_bar.set_fraction(min(1.0, max(0.0, fraction)))
        self._show(self.repair_current_bar, text)
        if snap.phase in ("locking", "scanning") and snap.fraction <= 0:
            self.repair_overall.pulse()
        else:
            self.repair_overall.set_fraction(min(1.0, max(0.0, snap.fraction)))
        self._show(self.repair_overall, snap.overall)
        if snap.stats:
            stats_text = snap.stats
        elif snap.phase in ("idle", ""):
            stats_text = "Press Scan to check the index without changing it."
        else:
            stats_text = ""
        self._show(self.repair_stats, stats_text)
        self.repair_stats.set_tooltip_text(
            f"{stats_text}\n{TIPS['repair_stats']}".strip() if stats_text else TIPS["repair_stats"]
        )
        active = next((row for row in snap.rows if row.pulse >= 0), None)
        if active is None and snap.rows:
            active = snap.rows[-1]
        if active is not None:
            spec = f"{active.name}: {active.detail}"
            self._show(self.repair_specs, spec)
            self.repair_specs.set_tooltip_text(spec)
