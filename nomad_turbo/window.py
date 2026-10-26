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

from nomad_turbo.catalog import wipe_phrase_ok
from nomad_turbo.desktop import icon_dir, write_shortcut
from nomad_turbo.discover import load_install, save_install, state_dir
from nomad_turbo.progress import (
    UNKNOWN_ETA_SECONDS,
    ProgressBook,
    format_current_file,
    format_run_stats,
    LOG_PANE_LINES,
    list_pane_position,
    restored_log_top,
    log_line_visible,
    log_lines_to_drop,
    log_should_stick,
    preferred_column_width,
    usage_text,
)
from nomad_turbo.qdrant_status import changes_allowed
from nomad_turbo.repair import format_repair_current
from nomad_turbo.runtime import gpu_rows
from nomad_turbo.scan import scan_library
from nomad_turbo.session import Session
from nomad_turbo.slots import (
    clamp_boost_percent,
    embed_call_budget,
    host_memory_mib,
    install_boost_percent,
    pick_gpu,
    reserve_ratio_from_percent,
)
from nomad_turbo.tips import TIPS

log = logging.getLogger("nomad_turbo")

# People who open this window can reach the project Turbo is built for.
PROJECT_URL = "https://github.com/Crosstalk-Solutions/project-nomad"


def window_close_action(indexing: bool) -> str:
    """A running turbo stays up when the window closes. The tray opens it again."""
    return "hide" if indexing else "quit"


def index_store_types() -> tuple:
    """Article, chunk, and size columns are 64-bit. A ZIM over 2 GB does not fit in 32."""
    from gi.repository import GObject

    wide = GObject.TYPE_INT64
    return (str, str, str, str, int, str, str, str, str, str, str, wide, wide, wide, int, int)


def repair_store_types() -> tuple:
    """The point-count sort column is 64-bit."""
    from gi.repository import GObject

    wide = GObject.TYPE_INT64
    return (str, str, str, str, int, int, str, str, str, str, wide, int)


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


def _configure_logging(memory: _MemoryLog) -> Path:
    path = state_dir() / "session.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, mode="w")
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
    log_path = _configure_logging(memory)
    session = Session(install_boost_percent(install))
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


class _Window:
    def __init__(self, install, session: Session, memory: _MemoryLog, log_path: Path, gtk, glib) -> None:
        self.install = install
        self.session = session
        self.memory = memory
        self.log_path = log_path
        self.gtk = gtk
        self.glib = glib
        self.thread: threading.Thread | None = None
        self.delete_thread: threading.Thread | None = None
        self.repair_thread: threading.Thread | None = None
        self.repair_stop = threading.Event()
        self.repair_board = None
        self.repair_starting = False
        self._repair_told = False
        self.qdrant_stop = threading.Event()
        self.qdrant_text = "Checking Qdrant…"
        self.qdrant_ready = False
        self._qdrant_error = None
        self._followed: dict[int, str] = {}
        self._column_manual: dict[str, bool] = {}
        self._column_width: dict[str, int] = {}
        self._fit_meta: list[tuple] = []
        self._sizing = False
        self._pane_lock = False
        self._log_split_ready = False
        self._pane_styled = False
        self.layout_timer = None
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

        window = gtk.Window(title="Nomad TURBO Index")
        window.set_resizable(True)
        window.set_default_size(1180, 820)
        # A small floor. Portion minimums stay small so this does not pin the window.
        window.set_size_request(480, 280)
        window.set_position(gtk.WindowPosition.CENTER)
        try:
            window.set_wmclass("nomad-turbo", "nomad-turbo")
        except Exception:
            log.debug("could not set the window class", exc_info=True)
        self.window = window
        window.connect("delete-event", self.on_delete)

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
        project = gtk.LinkButton.new_with_label(PROJECT_URL, "Project NOMAD")
        project.set_halign(gtk.Align.END)
        project.set_tooltip_text(TIPS["project"])
        header.pack_end(project, False, False, 0)
        root.pack_start(header, False, False, 0)
        index = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=8)
        index.set_vexpand(True)
        self.stack.add_titled(index, "index", "Index")
        self.index_page = index

        self.status = gtk.Label(label="Scanning the library…")
        self.status.set_xalign(0)
        self.status.set_line_wrap(True)
        index.pack_start(self.status, False, False, 0)

        from gi.repository import Pango

        def headed(title: str, tip: str, cell, **props):
            column = gtk.TreeViewColumn(title, cell, **props)
            label = gtk.Label(label=title)
            label.set_tooltip_text(tip)
            label.show()
            column.set_widget(label)
            return column

        self.store = gtk.ListStore(*index_store_types())
        tree = gtk.TreeView(model=self.store)
        tree.set_headers_visible(True)
        tree.set_tooltip_column(0)
        name_cell = gtk.CellRendererText()
        name_cell.set_property("ellipsize", Pango.EllipsizeMode.MIDDLE)
        self._add_column(
            tree,
            "index:File",
            headed(
                "File",
                TIPS["col_file"],
                name_cell,
                text=1,
                foreground=10,
            ),
            text_index=1,
            minimum=180,
            maximum=460,
            expand=True,
            sort_index=1,
        )
        self._add_column(
            tree,
            "index:Kind",
            headed("Kind", TIPS["col_kind"], gtk.CellRendererText(), text=2),
            text_index=2,
            minimum=64,
            maximum=120,
            sort_index=2,
        )
        self._add_column(
            tree,
            "index:Status",
            headed("Status", TIPS["col_status"], gtk.CellRendererText(), text=3),
            text_index=3,
            minimum=80,
            maximum=160,
            sort_index=3,
        )
        self._add_column(
            tree,
            "index:Progress",
            headed(
                "Progress",
                TIPS["col_progress"],
                gtk.CellRendererProgress(),
                value=4,
            ),
            text_index=None,
            minimum=160,
            maximum=280,
            sort_index=4,
        )
        self._add_column(
            tree,
            "index:Articles",
            headed("Articles", TIPS["col_articles"], gtk.CellRendererText(), text=5),
            text_index=5,
            minimum=90,
            maximum=240,
            sort_index=11,
        )
        self._add_column(
            tree,
            "index:Chunks",
            headed("Chunks", TIPS["col_chunks"], gtk.CellRendererText(), text=6),
            text_index=6,
            minimum=80,
            maximum=160,
            sort_index=12,
        )
        self._add_column(
            tree,
            "index:Size",
            headed("Size", TIPS["col_size"], gtk.CellRendererText(), text=7),
            text_index=7,
            minimum=70,
            maximum=120,
            sort_index=13,
        )
        self._add_column(
            tree,
            "index:Rate",
            headed("Rate", TIPS["col_rate"], gtk.CellRendererText(), text=8),
            text_index=8,
            minimum=120,
            maximum=280,
            sort_index=14,
        )
        self._add_column(
            tree,
            "index:ETA",
            headed("ETA", TIPS["col_eta"], gtk.CellRendererText(), text=9),
            text_index=9,
            minimum=70,
            maximum=140,
            sort_index=15,
        )
        scrolled = gtk.ScrolledWindow()
        scrolled.set_policy(gtk.PolicyType.AUTOMATIC, gtk.PolicyType.AUTOMATIC)
        scrolled.add(tree)
        scrolled.set_min_content_height(48)
        scrolled.set_vexpand(True)
        self.file_tree = tree
        lower = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=8)

        legend = gtk.Label(label="Blue names are ZIM files. Amber names are files you uploaded. Drag a column edge to resize it. Drag a bar to resize the list, the controls, or the log. Click a heading to sort.")
        legend.set_xalign(0)
        legend.set_line_wrap(True)
        lower.pack_start(legend, False, False, 0)

        self.current_label = gtk.Label(label="No file is indexing.")
        self.current_label.set_xalign(0)
        self.current_label.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
        self.current_label.set_tooltip_text(TIPS["file_now"])
        lower.pack_start(self.current_label, False, False, 0)
        self.current_bar = gtk.ProgressBar()
        self.current_bar.set_show_text(True)
        self.current_bar.set_text("Idle")
        self.current_bar.set_tooltip_text(TIPS["file_bar"])
        lower.pack_start(self.current_bar, False, False, 0)

        self.overall = gtk.ProgressBar()
        self.overall.set_show_text(True)
        self.overall.set_text("Scanning…")
        self.overall.set_tooltip_text(TIPS["library_bar"])
        lower.pack_start(self.overall, False, False, 0)

        self.stats = gtk.Label(label="")
        self.stats.set_xalign(0)
        self.stats.set_line_wrap(True)
        self.stats.set_selectable(True)
        self.stats.set_tooltip_text(TIPS["stats"])
        lower.pack_start(self.stats, False, False, 0)

        gpu = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=4)
        slider_row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        slider_label = gtk.Label(label="Embed batch")
        slider_label.set_tooltip_text(TIPS["slider"])
        slider_row.pack_start(slider_label, False, False, 0)
        self.scale = gtk.Scale.new_with_range(gtk.Orientation.HORIZONTAL, 10, 110, 5)
        self.scale.set_value(session.get_percent())
        self.scale.set_hexpand(True)
        self.scale.set_draw_value(True)
        self.scale.set_value_pos(gtk.PositionType.RIGHT)
        for mark in (10, 100, 110):
            self.scale.add_mark(mark, gtk.PositionType.BOTTOM, f"{mark}%")
        self.scale.set_tooltip_text(TIPS["slider"])
        self.scale.connect("format-value", lambda _scale, value: f"{int(round(value))}%")
        self.scale.connect("value-changed", self.on_slider)
        slider_row.pack_start(self.scale, True, True, 0)
        gpu.pack_start(slider_row, False, False, 0)
        self.usage = gtk.Label(label="")
        self.usage.set_xalign(0)
        self.usage.set_line_wrap(True)
        self.usage.set_tooltip_text(TIPS["usage"])
        gpu.pack_start(self.usage, False, False, 0)
        self.card = gtk.Label(label="")
        self.card.set_xalign(0)
        self.card.set_tooltip_text(TIPS["card"])
        gpu.pack_start(self.card, False, False, 0)
        self.keep_check = gtk.CheckButton(
            label="Stay on after the last file, and index new files as they arrive"
        )
        self.keep_check.set_active(bool(getattr(install, "keep_control", False)))
        self.keep_check.set_halign(gtk.Align.START)
        self.keep_check.set_tooltip_text(TIPS["keep"])
        self.keep_check.connect("toggled", self.on_keep)
        gpu.pack_start(self.keep_check, False, False, 0)
        lower.pack_start(gpu, False, False, 0)

        buttons = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        self.start_button = gtk.Button(label="Start indexing")
        self.delete_button = gtk.Button(label="Delete index")
        self.rescan_button = gtk.Button(label="Rescan")
        self.stop_button = gtk.Button(label="Stop")
        self.export_button = gtk.Button(label="Export log")
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
        self.delete_tip = TIPS["delete"]
        self.delete_button.set_tooltip_text(TIPS["delete"])
        self.delete_button.get_style_context().add_class("destructive-action")
        for button in (self.start_button, self.stop_button, self.rescan_button, self.export_button):
            buttons.pack_start(button, False, False, 0)
        buttons.pack_end(self.delete_button, False, False, 0)
        lower.pack_start(buttons, False, False, 0)
        self.index_pane = self._make_pane()
        # The list takes extra space and keeps a small floor. The controls keep
        # the height from their drag bar and can shrink so the window can too.
        self.index_pane.pack1(scrolled, True, False)
        self.index_pane.pack2(self._portion(lower), False, True)
        self._track_pane(self.index_pane, "index_pane")
        index.pack_start(self.index_pane, True, True, 0)
        self._build_repair_page(headed)

        log_head = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        self.error_check = gtk.CheckButton(label="Errors")
        self.info_check = gtk.CheckButton(label="Info")
        self.debug_check = gtk.CheckButton(label="Debug")
        self.error_check.set_active(True)
        self.info_check.set_active(True)
        self.debug_check.set_active(True)
        self.error_check.set_tooltip_text(TIPS["errors"])
        self.info_check.set_tooltip_text(TIPS["info"])
        self.debug_check.set_tooltip_text(TIPS["debug"])
        self.info_check.connect("toggled", self.on_filter)
        self.debug_check.connect("toggled", self.on_filter)
        self.error_check.connect("toggled", self.on_filter)
        self.follow_check = gtk.CheckButton(label="Follow")
        self.follow_check.set_active(bool(getattr(install, "log_follow", True)))
        self.follow_check.set_tooltip_text(TIPS["follow"])
        self.follow_check.connect("toggled", self.on_follow)
        log_head.pack_start(self.error_check, False, False, 0)
        log_head.pack_start(self.info_check, False, False, 0)
        log_head.pack_start(self.debug_check, False, False, 0)
        log_head.pack_start(self.follow_check, False, False, 0)
        self.log_filters = log_head
        # The Log row is the last thing above the split, so the bar sits under it.
        # The log text is the other side. Neither side sets a tall size request,
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
        log_header = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        log_header.pack_start(self.log_open_button, False, False, 0)
        log_header.pack_start(log_title_box, False, False, 0)
        log_header.pack_start(log_head, False, False, 0)
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
        self.log_filters.set_no_show_all(not log_open)
        self.log_filters.set_visible(log_open)
        log_scroll.set_no_show_all(not log_open)
        log_scroll.set_visible(log_open)
        self.stack.set_vexpand(True)
        top = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=8)
        top.pack_start(self.stack, True, True, 0)
        top.pack_start(self.qdrant_label, False, False, 0)
        top.pack_start(log_header, False, False, 0)
        self.content_pane = self._make_pane()
        # Extra window height goes to the pages. The log keeps its dragged size
        # until the pages are at their floor, and the log itself can still shrink.
        self.content_pane.pack1(top, True, False)
        self.content_pane.pack2(log_scroll, False, True)
        self._track_log_pane(self.content_pane)
        root.pack_start(self.content_pane, True, True, 0)
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

    def run(self) -> None:
        self.window.show_all()
        self.begin_scan()
        threading.Thread(target=self._watch_qdrant, name="nomad-turbo-qdrant", daemon=True).start()
        self.glib.timeout_add(500, self.on_tick)
        self.gtk.main()

    def _watch_qdrant(self) -> None:
        from nomad_turbo.qdrant_status import probe_qdrant

        while not self.qdrant_stop.is_set():
            try:
                view = probe_qdrant(self.install.qdrant_url, self.install.collection)
                self.qdrant_text = view.text
                self.qdrant_ready = view.ready
            except Exception as exc:
                message = f"{exc.__class__.__name__}: {exc}"
                if message != self._qdrant_error:
                    self._qdrant_error = message
                    log.warning("qdrant status check failed: %s", message)
                self.qdrant_text = "Qdrant status could not be read. Not ready for changes."
                self.qdrant_ready = False
            else:
                self._qdrant_error = None
            if self.qdrant_stop.wait(5):
                return

    def on_signal(self, _signum, _frame) -> None:
        self.qdrant_stop.set()
        self._stop_all()
        self.session.close_requested = True

    def _stop_all(self) -> None:
        self.session.stop.set()
        self.repair_stop.set()

    def _work_alive(self) -> bool:
        indexing = self.thread is not None and self.thread.is_alive()
        deleting = self.delete_thread is not None and self.delete_thread.is_alive()
        repairing = self.repair_thread is not None and self.repair_thread.is_alive()
        return indexing or deleting or repairing or self.repair_starting

    def on_raise(self, _signum, _frame) -> None:
        self.raise_requested = True

    def on_delete(self, *_args) -> bool:
        indexing = self._work_alive()
        if window_close_action(indexing) == "hide" and self.indicator is not None:
            self.hidden = True
            self.window.hide()
            log.info("window closed to the tray. Turbo keeps running.")
            return True
        self._stop_all()
        self.session.close_requested = True
        if indexing:
            if self.repair_thread is not None and self.repair_thread.is_alive():
                self.repair_status.set_text("Stopping after the current file finishes…")
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

    def _scan_ready(self, book: ProgressBook, found, indexed: int) -> bool:
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
    ) -> None:
        """A resizable column. A saved width is kept. Otherwise the first rows size it."""
        column.set_resizable(True)
        column.set_sizing(self.gtk.TreeViewColumnSizing.FIXED)
        column.set_min_width(minimum)
        if expand:
            column.set_expand(True)
        if sort_index is not None:
            column.set_sort_column_id(sort_index)
        widget = column.get_widget()
        if widget is not None:
            tip = widget.get_tooltip_text() or ""
            extra = " Drag the edge to resize. Click the heading to sort."
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
        if text_index is not None:
            self._fit_meta.append((key, column, text_index, minimum, maximum))
        tree.append_column(column)

    def _on_column_clicked(self, _column, tree) -> None:
        """Sorting moves rows. Look up the active file again on the next tick."""
        self._followed.pop(id(tree), None)

    def _on_column_width(self, column, _pspec, key: str) -> None:
        if self._sizing:
            return
        width = int(column.get_fixed_width() or column.get_width() or 0)
        if width < 40 or width == self._column_width.get(key):
            return
        self._column_manual[key] = True
        self._column_width[key] = width
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
        limit = min(80, len(model))
        interesting = [
            item for item in self._fit_meta if item[1].get_tree_view() == tree and not self._column_manual.get(item[0])
        ]
        if not interesting:
            return
        for index in range(limit):
            row = model[index]
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
        finally:
            self._sizing = False

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
            paned separator {
                background-color: alpha(@theme_fg_color, 0.45);
                background-image: none;
                border: none;
                min-height: 14px;
                min-width: 14px;
            }
            paned separator:hover,
            paned separator:active {
                background-color: alpha(@theme_fg_color, 0.85);
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

    def _make_pane(self):
        """A tall, visible split. Both sides can shrink so the window can too."""
        self._style_panes()
        pane = self.gtk.Paned(orientation=self.gtk.Orientation.VERTICAL)
        pane.set_wide_handle(True)
        pane.set_vexpand(True)
        return pane

    def _portion(self, child):
        """Scroll the controls when a split is dragged smaller than their text."""
        scrolled = self.gtk.ScrolledWindow()
        scrolled.set_policy(self.gtk.PolicyType.AUTOMATIC, self.gtk.PolicyType.AUTOMATIC)
        scrolled.set_min_content_height(48)
        scrolled.set_propagate_natural_height(True)
        scrolled.set_vexpand(True)
        scrolled.add(child)
        return scrolled

    def _mark_pane_drag(self, pane, event) -> bool:
        # A window resize also moves the bar. Only a drag on the bar is saved.
        if getattr(event, "button", 1) == 1:
            pane._user_drag = True
        return False

    def _track_pane(self, pane, key: str) -> None:
        saved = int(getattr(self.install, key, 0) or 0)

        def on_size(_widget, allocation) -> None:
            if allocation.height < 80 or getattr(pane, "_split_ready", False):
                return
            pane._split_ready = True
            controls = 0
            child = pane.get_child2()
            if saved <= 0 and child is not None:
                _minimum, controls = child.get_preferred_height()
            self._pane_lock = True
            try:
                pane.set_position(list_pane_position(saved, allocation.height, controls))
            finally:
                self._pane_lock = False

        def on_position(widget, _pspec) -> None:
            if self._pane_lock or not getattr(widget, "_user_drag", False):
                return
            position = int(widget.get_position())
            if position < 48:
                return
            setattr(self.install, key, position)
            self._save_layout_soon()

        pane.connect("size-allocate", on_size)
        pane.connect("button-press-event", self._mark_pane_drag)
        pane.connect("notify::position", on_position)

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

    def _track_log_pane(self, pane) -> None:
        def on_size(_widget, allocation) -> None:
            if self._log_split_ready or allocation.height < 160:
                return
            self._log_split_ready = True
            self._place_log_split(pane, allocation.height)

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
        pane.connect("notify::position", on_position)

    def on_follow(self, button) -> None:
        following = bool(button.get_active())
        self.install.log_follow = following
        try:
            save_install(self.install)
        except Exception:
            log.exception("could not save the log follow setting")
        if following:
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
        if expanded and self._log_split_ready:
            self._place_log_split(self.content_pane, int(self.content_pane.get_allocated_height() or 0))
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

    def on_slider(self, scale) -> None:
        percent = clamp_boost_percent(int(round(scale.get_value())))
        self.session.set_percent(percent)
        self.refresh_usage()
        if self.save_timer:
            self.glib.source_remove(self.save_timer)
        self.save_timer = self.glib.timeout_add(400, self._save_percent, percent)

    def _save_percent(self, percent: int) -> bool:
        self.save_timer = None
        try:
            self.install.boost_percent = percent
            self.install.reserve_ratio = reserve_ratio_from_percent(min(percent, 100))
            save_install(self.install)
        except Exception:
            log.exception("could not save the embed batch size")
        return False

    def on_filter(self, *_args) -> None:
        self.filter_dirty = True

    def on_start(self, wipe: bool) -> None:
        if not self.rows_ready or self._work_alive() or not self.qdrant_ready:
            return
        self.session.stop.clear()
        self.session.wipe_confirmed = bool(wipe)
        self.session.set_phase("waiting")
        self.started_at = time.monotonic()
        self._finish_told = False
        self.thread = threading.Thread(target=self._run, args=(wipe,), name="nomad-turbo")
        self.thread.start()

    def on_delete_index(self, *_args) -> None:
        if not self.rows_ready or self._work_alive() or not self.qdrant_ready:
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

    def _delete_only(self) -> None:
        import importlib

        import nomad_turbo.engine as engine
        import nomad_turbo.store as store

        importlib.reload(store)
        importlib.reload(engine)
        try:
            engine.delete_index_only(self.session.stop, session=self.session)
        except Exception as exc:
            log.exception("index delete failed")
            self.session.set_phase("error", str(exc))
            self.session.set_note(str(exc))
            return
        note = self.session.view().get("note") or "Knowledge index deleted. Indexing has not started."
        self.glib.idle_add(self._show, self.status, note)

    def on_export(self, *_args) -> None:
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
        chooser.set_current_name("nomad-turbo-log.txt")
        chooser.set_do_overwrite_confirmation(True)
        response = chooser.run()
        target = chooser.get_filename()
        chooser.destroy()
        if response != gtk.ResponseType.OK or not target:
            return
        try:
            shutil.copyfile(self.log_path, target)
        except OSError as exc:
            self.status.set_text(f"Could not export the log: {exc}")
            return
        self.status.set_text(f"Log saved to {target}")

    def _run(self, wipe: bool) -> None:
        import importlib

        import nomad_turbo.chunking as chunking
        import nomad_turbo.decisions as decisions
        import nomad_turbo.engine as engine
        import nomad_turbo.packing as packing
        import nomad_turbo.pool as pool
        import nomad_turbo.runtime as runtime
        import nomad_turbo.slots as slots
        import nomad_turbo.store as store

        # Start loads the modules again so a fix applies without reopening the window.
        # slots is first because pool and engine bind its functions at import.
        # chunking is before packing because packing binds the fit helpers at import.
        importlib.reload(slots)
        importlib.reload(store)
        importlib.reload(runtime)
        importlib.reload(decisions)
        importlib.reload(chunking)
        importlib.reload(packing)
        importlib.reload(pool)
        importlib.reload(engine)
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
            log.exception("turbo failed")
            self.session.set_phase("error", str(exc))
            self.session.set_note(str(exc))
        finally:
            write_shortcut(False)
            self.session.wipe_confirmed = False

    def on_tick(self) -> bool:
        self.tick += 1
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
            self._release_log_pane()
            self.qdrant_stop.set()
            self.gtk.main_quit()
            return False
        return True

    def _buttons(self, phase: str) -> None:
        running = phase in ("waiting", "running") and self.thread is not None and self.thread.is_alive()
        deleting = self.delete_thread is not None and self.delete_thread.is_alive()
        repairing = (self.repair_thread is not None and self.repair_thread.is_alive()) or self.repair_starting
        library_idle = self.rows_ready and not running and not repairing and not deleting
        changes = changes_allowed(self.qdrant_ready)
        self.start_button.set_sensitive(library_idle and changes)
        self.delete_button.set_sensitive(library_idle and changes)
        self.rescan_button.set_sensitive(library_idle)
        self._gate_tip(self.start_button, self.start_tip)
        self._gate_tip(self.delete_button, self.delete_tip)
        self.stop_button.set_sensitive(running or deleting)
        if hasattr(self, "repair_start"):
            idle_repair = not running and not repairing and not deleting and not self.scanning and changes
            self.repair_scan.set_sensitive(idle_repair)
            self.repair_start.set_sensitive(idle_repair)
            self._gate_tip(self.repair_scan, self.repair_scan_tip)
            self._gate_tip(self.repair_start, self.repair_start_tip)
            self.repair_stop_button.set_sensitive(repairing)
            self.repair_quick.set_sensitive(not repairing)
            self.repair_complete.set_sensitive(not repairing)

    def refresh_rows(self):
        book = self.session.get_book()
        if book is None:
            return None
        snap = book.snapshot()
        seen = {item.source for item in snap.files}
        for index in range(len(self.store) - 1, -1, -1):
            if self.store[index][0] not in seen:
                del self.store[index]
        existing = self._iters_by_source(self.store)
        for item in snap.files:
            kind = "Upload" if item.origin == "upload" else "ZIM"
            rate_sort = int(item.articles_per_minute or 0)
            eta_sort = UNKNOWN_ETA_SECONDS if item.eta_seconds is None else int(item.eta_seconds)
            values = (
                item.source,
                item.name,
                kind,
                item.status,
                item.percent,
                item.count,
                item.chunks_text,
                item.size_text,
                item.rate_text,
                item.eta,
                item.color,
                int(item.articles),
                int(item.chunks),
                int(item.size_bytes),
                rate_sort,
                eta_sort,
            )
            if item.source in existing:
                self._set_row(self.store, existing[item.source], values)
            else:
                self.store.append(values)
        source = next((item.source for item in snap.files if item.state == "running"), "")
        self._refit_columns(self.file_tree)
        self._follow_files(self.file_tree, source)
        title, fraction, text = format_current_file(snap)
        self._show(self.current_label, title)
        self.current_label.set_tooltip_text(title if source else TIPS["file_now"])
        self.current_bar.set_fraction(min(1.0, max(0.0, fraction)))
        self._show(self.current_bar, text)
        self.overall.set_fraction(min(1.0, max(0.0, snap.overall_fraction)))
        self._show(self.overall, snap.overall_text)
        return snap

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
        self._show(self.stats, format_run_stats(view["phase"], elapsed, snap))

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
                label = f"{view['percent']}%"
            elif getattr(self.repair_board, "kind", "") == "scan":
                label = "scan"
            else:
                label = "repair"
            self.indicator.set_label(label, "100%")
        else:
            self.indicator.set_status(status.ACTIVE)
            self.indicator.set_label(f"{view['percent']}%" if indexing else "", "100%")
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
        percent = view["percent"]
        gpu_total = self.live_gpu[1] if self.live_gpu else None
        chunks = embed_call_budget(
            percent,
            gpu_total_mib=gpu_total,
            host_total_mib=self.host_total_mib,
            cpu_count=self.cpu_count,
        )
        text = usage_text(
            set_percent=percent,
            chunks=chunks,
            server_up=view["active_slots"] > 0 and view["phase"] in ("waiting", "running"),
            slots_ignored=bool(self.install.slots),
        )
        self._show(self.usage, text)
        gpu = self.live_gpu
        if gpu:
            _index, total, used = gpu
            card = f"Card memory in use: {used:,} / {total:,} MiB"
        elif view["total_mib"]:
            card = "Card memory in use: waiting for a reading."
        else:
            card = "No GPU detected. The embed server will use the CPU."
        self._show(self.card, card)

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
        )
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
            adjustment = self.log_scroll.get_vadjustment()
            if bottom:
                adjustment.set_value(max(0.0, adjustment.get_upper() - adjustment.get_page_size()))
            elif mark is not None and not mark.get_deleted():
                self.log_view.scroll_to_mark(mark, 0.0, True, 0.0, 0.0)
                self.log_buffer.delete_mark(mark)
            return False

        self.glib.idle_add(apply_pin, priority=self.glib.PRIORITY_LOW)

    def _build_repair_page(self, headed) -> None:
        gtk = self.gtk
        page = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=8)
        page.set_vexpand(True)
        self.repair_status = gtk.Label(
            label=(
                "Scan checks the library and the index and changes nothing. "
                "Start repair removes index entries for library files that are gone. "
                "NOMAD keeps embedding until you press Start repair."
            )
        )
        self.repair_status.set_xalign(0)
        self.repair_status.set_line_wrap(True)
        page.pack_start(self.repair_status, False, False, 0)

        modes = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=16)
        self.repair_quick = gtk.RadioButton.new_with_label_from_widget(None, "Quick repair")
        self.repair_complete = gtk.RadioButton.new_with_label_from_widget(self.repair_quick, "Complete repair")
        self.repair_quick.set_tooltip_text(TIPS["repair_quick"])
        self.repair_complete.set_tooltip_text(TIPS["repair_complete"])
        modes.pack_start(self.repair_quick, False, False, 0)
        modes.pack_start(self.repair_complete, False, False, 0)
        page.pack_start(modes, False, False, 0)

        from gi.repository import Pango

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
        scrolled.set_min_content_height(48)
        scrolled.set_vexpand(True)
        self.repair_tree = tree
        lower = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=8)

        self.repair_current_label = gtk.Label(label="No file is being repaired.")
        self.repair_current_label.set_xalign(0)
        self.repair_current_label.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
        self.repair_current_label.set_tooltip_text(TIPS["repair_file_now"])
        lower.pack_start(self.repair_current_label, False, False, 0)
        self.repair_current_bar = gtk.ProgressBar()
        self.repair_current_bar.set_show_text(True)
        self.repair_current_bar.set_text("Idle")
        self.repair_current_bar.set_tooltip_text(TIPS["repair_file_bar"])
        lower.pack_start(self.repair_current_bar, False, False, 0)

        self.repair_overall = gtk.ProgressBar()
        self.repair_overall.set_show_text(True)
        self.repair_overall.set_text("Idle")
        self.repair_overall.set_tooltip_text(TIPS["repair_overall_bar"])
        lower.pack_start(self.repair_overall, False, False, 0)

        self.repair_stats = gtk.Label(label="Press Scan to check the index without changing it.")
        self.repair_stats.set_xalign(0)
        self.repair_stats.set_line_wrap(True)
        self.repair_stats.set_max_width_chars(110)
        self.repair_stats.set_selectable(True)
        self.repair_stats.override_font(Pango.FontDescription("Monospace"))
        self.repair_stats.set_tooltip_text(TIPS["repair_stats"])
        lower.pack_start(self.repair_stats, False, False, 0)

        self.repair_specs = gtk.Label(label="Each row's spec is kept up to date while that file is repaired. Hover a row to read it.")
        self.repair_specs.set_xalign(0)
        self.repair_specs.set_line_wrap(True)
        self.repair_specs.set_selectable(True)
        lower.pack_start(self.repair_specs, False, False, 0)

        buttons = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
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
        buttons.pack_start(self.repair_scan, False, False, 0)
        buttons.pack_start(self.repair_start, False, False, 0)
        buttons.pack_start(self.repair_stop_button, False, False, 0)
        lower.pack_start(buttons, False, False, 0)

        hint = gtk.Label(
            label=(
                "Scan changes nothing. Quick compares files, ingest rows, and checkpoints. "
                "Complete also lists sources in the index when Qdrant is answering. "
                "Start repair pauses file embedding, then deletes points for files that are gone. "
                "Points for a file that is still on disk stay there. "
                "The tray icon turns blue while a repair is running. Stop is here and in the tray."
            )
        )
        hint.set_xalign(0)
        hint.set_line_wrap(True)
        lower.pack_start(hint, False, False, 0)
        self.repair_pane = self._make_pane()
        self.repair_pane.pack1(scrolled, True, False)
        self.repair_pane.pack2(self._portion(lower), False, True)
        self._track_pane(self.repair_pane, "repair_pane")
        page.pack_start(self.repair_pane, True, True, 0)
        self.stack.add_titled(page, "repair", "Repair")

    def _gate_tip(self, button, normal: str) -> None:
        if self.qdrant_ready:
            button.set_tooltip_text(normal)
        else:
            button.set_tooltip_text(TIPS["qdrant_gate"])

    def on_repair_scan(self) -> None:
        if self._work_alive() or self.scanning or not self.qdrant_ready:
            return
        mode = "complete" if self.repair_complete.get_active() else "quick"
        self.repair_stop = threading.Event()
        self._repair_told = False
        self.repair_starting = True
        self.repair_status.set_text("Starting scan…")
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

        import nomad_turbo.decisions as decisions
        import nomad_turbo.engine as engine
        import nomad_turbo.repair as repair
        import nomad_turbo.runtime as runtime
        import nomad_turbo.store as store

        # Scan does not pause NOMAD. The desktop icon does not change.
        try:
            importlib.reload(decisions)
            importlib.reload(store)
            importlib.reload(runtime)
            importlib.reload(engine)
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
                self.glib.idle_add(self.repair_status.set_text, str(exc))
        finally:
            self.repair_starting = False

    def on_repair_start(self) -> None:
        if self._work_alive() or self.scanning or not self.qdrant_ready:
            return
        mode = "complete" if self.repair_complete.get_active() else "quick"
        self.repair_stop = threading.Event()
        self._repair_told = False
        self.repair_starting = True
        self.repair_status.set_text("Starting repair…")
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

        import nomad_turbo.decisions as decisions
        import nomad_turbo.engine as engine
        import nomad_turbo.repair as repair
        import nomad_turbo.runtime as runtime
        import nomad_turbo.store as store

        # The desktop icon does not change. Stop is in this window and the tray.
        try:
            importlib.reload(decisions)
            importlib.reload(store)
            importlib.reload(runtime)
            importlib.reload(engine)
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
                self.glib.idle_add(self.repair_status.set_text, str(exc))
        finally:
            self.repair_starting = False

    def refresh_repair(self) -> None:
        board = self.repair_board
        if board is None or not hasattr(self, "repair_store"):
            return
        snap = board.snapshot()
        if snap.note:
            self._show(self.repair_status, snap.note)
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
            self._show(self.repair_stats, snap.stats)
        elif snap.phase in ("idle", ""):
            self._show(self.repair_stats, "Press Scan to check the index without changing it.")
        else:
            self._show(self.repair_stats, "")
        active = next((row for row in snap.rows if row.pulse >= 0), None)
        if active is None and snap.rows:
            active = snap.rows[-1]
        if active is not None:
            self._show(self.repair_specs, f"{active.name}: {active.detail}")
