"""The Models page. The window stays thin. Decisions live in chat_models.

Install, test, and delete wait while indexing is running. A menu name,
the order, the thinking default, and the job can still be saved.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import urllib.request
from pathlib import Path

from nomad_turbo.chat_models import (
    BENCHMARK_MODEL,
    INDEX_WAIT,
    PARAM_WAIT,
    THINK_OFF_PROMPT,
    THINK_ON_PROMPT,
    MenuPatchError,
    OllamaError,
    action_block,
    assign_job,
    bytes_needed,
    catalog_path,
    chat_names,
    chat_payload,
    clean_label,
    clean_token,
    cleanup_command,
    context_advice,
    context_sentence,
    create_payload,
    delete_block_reason,
    delete_model,
    discover_ollama_base,
    file_preview,
    forked_reason,
    format_bytes,
    gguf_modelfile,
    incoming_paths,
    in_use_sentence,
    kv_bytes_for,
    largest_nvidia_mib,
    merge_live,
    next_chat_model,
    parse_parameter_billions,
    page_menu,
    parse_show,
    parse_tags,
    plan_install,
    get_json,
    post_json,
    post_stream,
    progress_from_status,
    publish_menu,
    read_catalog,
    registry_manifest_url,
    reply_ok,
    rewrite_modelfile,
    sampling_from_entries,
    settings_changed,
    space_error,
    stage_commands,
    sum_manifest_bytes,
    write_catalog,
)
from nomad_turbo.tips import TIPS

log = logging.getLogger("nomad_turbo")


class _split_holder:
    kind = None


class ModelPage:
    """One worker at a time. Widget updates go through the UI thread."""

    def __init__(self, install, session, gtk, glib, parent) -> None:
        self.install = install
        self.session = session
        self.gtk = gtk
        self.glib = glib
        self.parent = parent
        self._busy = False
        self._loaded = False
        self._guard = False
        self._generation = 0
        self._catalog = {"models": {}}
        self._tags: list[dict] = []
        self._shows: dict[str, dict] = {}
        self._base = "http://127.0.0.1:11434"
        self._tasks = ""
        self._last = ""
        self._cap = ""
        self._vram = None
        self.widget = self._build()

    def _build(self):
        gtk = self.gtk
        page = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=8)
        page.set_vexpand(True)
        scroll = gtk.ScrolledWindow()
        scroll.set_policy(gtk.PolicyType.NEVER, gtk.PolicyType.AUTOMATIC)
        inner = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=6)
        inner.set_margin_top(4)
        inner.set_margin_bottom(8)
        inner.set_margin_start(2)
        inner.set_margin_end(2)

        heading = gtk.Label()
        heading.set_markup("<span size='large' weight='bold'>Models</span>")
        heading.set_xalign(0)
        heading.set_tooltip_text(TIPS["models_page"])
        inner.pack_start(heading, False, False, 0)

        self.status = self._wrapped("Models are read when this page is opened.", TIPS["models_page"])
        inner.pack_start(self.status, False, False, 0)
        self.index_note = self._note(TIPS["models_install"])
        inner.pack_start(self.index_note, False, False, 0)
        embed = getattr(self.install, "embed_model", "nomic-embed-text:v1.5")
        self.embed_label = self._wrapped(f"Indexing uses {embed}. It is not in this list.", TIPS["models_embed"])
        inner.pack_start(self.embed_label, False, False, 0)

        chosen = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=6)
        chosen.set_hexpand(True)
        install = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=6)
        install.set_hexpand(True)
        inner.pack_start(self._split(chosen, install), False, False, 0)

        self.model_combo = gtk.ComboBoxText()
        self._narrow_combo(self.model_combo)
        self.model_combo.set_hexpand(True)
        self.model_combo.set_tooltip_text(TIPS["models_list"])
        self.model_combo.connect("changed", self.on_model_changed)
        self.refresh_button = gtk.Button(label="Refresh")
        self.refresh_button.set_tooltip_text(TIPS["models_refresh"])
        self.refresh_button.connect("clicked", self.on_refresh)
        chosen.pack_start(self._row(self.model_combo, self.refresh_button), False, False, 0)

        self.detail = self._note(TIPS["models_detail"])
        chosen.pack_start(self.detail, False, False, 0)
        self.in_use = self._note(TIPS["models_in_use"])
        chosen.pack_start(self.in_use, False, False, 0)

        self.name_entry = self._entry(24, TIPS["models_name"])
        self.show_check = gtk.CheckButton(label="Show in menu")
        self.show_check.set_tooltip_text(TIPS["models_show"])
        self.order_entry = self._entry(4, TIPS["models_order"], expand=False)
        chosen.pack_start(self._caption_row("Menu name", self.name_entry, TIPS["models_name"]), False, False, 0)
        chosen.pack_start(self._row(self.show_check, self._order_box()), False, False, 0)

        self.job_chat = gtk.RadioButton.new_with_label_from_widget(None, "Chat")
        self.job_tasks = gtk.RadioButton.new_with_label_from_widget(self.job_chat, "Tasks")
        self.job_chat.set_tooltip_text(TIPS["models_chat"])
        self.job_tasks.set_tooltip_text(TIPS["models_tasks"])
        self.job_chat.connect("toggled", self.on_job)
        self.job_tasks.connect("toggled", self.on_job)
        self.photos = self._note(TIPS["models_photos"])
        self.tools = self._note(TIPS["models_tools"])
        # These sit on the button row. Extra width stays with the fields above.
        self.photos.set_hexpand(False)
        self.tools.set_hexpand(False)
        chosen.pack_start(
            self._row(self.job_chat, self.job_tasks, self.photos, self.tools),
            False,
            False,
            0,
        )
        self.think_follow = gtk.ToggleButton(label="Chat decides")
        self.think_on = gtk.ToggleButton(label="On")
        self.think_off = gtk.ToggleButton(label="Off")
        self.think_follow.set_tooltip_text(TIPS["models_think_follow"])
        self.think_on.set_tooltip_text(TIPS["models_think_on"])
        self.think_off.set_tooltip_text(TIPS["models_think_off"])
        self.think_follow.set_active(True)
        for button in (self.think_follow, self.think_on, self.think_off):
            button.connect("toggled", self.on_think)
        chosen.pack_start(
            self._caption_row("Thinking", self._linked(self.think_follow, self.think_on, self.think_off), TIPS["models_think"]),
            False,
            False,
            0,
        )
        self.job_note = self._note(TIPS["models_job"])
        self.think_note = self._note(TIPS["models_think"])
        chosen.pack_start(self.job_note, False, False, 0)
        chosen.pack_start(self.think_note, False, False, 0)

        self.context_combo = gtk.ComboBoxText()
        self.context_combo.set_hexpand(True)
        self.context_combo.set_tooltip_text(TIPS["models_context"])
        self.context_combo.append("auto", "Auto")
        self.context_combo.set_active_id("auto")
        self.context_label = self._note(TIPS["models_context"])
        chosen.pack_start(
            self._caption_row("Context", self.context_combo, TIPS["models_context"], self.context_label),
            False,
            False,
            0,
        )

        self.temp_entry = self._entry(8, TIPS["models_temperature"])
        self.top_p_entry = self._entry(8, TIPS["models_top_p"])
        self.top_k_entry = self._entry(8, TIPS["models_top_k"])
        self.min_p_entry = self._entry(8, TIPS["models_min_p"])
        chosen.pack_start(
            self._row(
                self._caption_row("Temperature", self.temp_entry, TIPS["models_temperature"], caption_width=0),
                self._caption_row("Top p", self.top_p_entry, TIPS["models_top_p"], caption_width=0),
            ),
            False,
            False,
            0,
        )
        chosen.pack_start(
            self._row(
                self._caption_row("Top k", self.top_k_entry, TIPS["models_top_k"], caption_width=0),
                self._caption_row("Min p", self.min_p_entry, TIPS["models_min_p"], caption_width=0),
            ),
            False,
            False,
            0,
        )

        self.renderer_entry = self._entry(24, TIPS["models_renderer"])
        self.parser_entry = self._entry(24, TIPS["models_parser"])
        advanced_box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=6)
        advanced_box.pack_start(self._caption_row("Renderer", self.renderer_entry, TIPS["models_renderer"]), False, False, 0)
        advanced_box.pack_start(self._caption_row("Parser", self.parser_entry, TIPS["models_parser"]), False, False, 0)
        advanced = gtk.Expander(label="Advanced")
        advanced.set_tooltip_text(TIPS["models_advanced"])
        advanced.add(advanced_box)
        chosen.pack_start(advanced, False, False, 0)

        self.save_button = gtk.Button(label="Save")
        self.test_button = gtk.Button(label="Test")
        self.delete_button = gtk.Button(label="Delete")
        self.save_button.set_tooltip_text(TIPS["models_save"])
        self.test_button.set_tooltip_text(TIPS["models_test"])
        self.delete_button.set_tooltip_text(TIPS["models_delete"])
        self.save_button.connect("clicked", self.on_save)
        self.test_button.connect("clicked", self.on_test)
        self.delete_button.connect("clicked", self.on_delete)
        chosen.pack_start(self._row(self.save_button, self.test_button, self.delete_button), False, False, 0)

        install_heading = gtk.Label()
        install_heading.set_markup("<span weight='bold'>Install</span>")
        install_heading.set_xalign(0)
        install_heading.set_tooltip_text(TIPS["models_install"])
        install.pack_start(install_heading, False, False, 0)
        self.tag_entry = self._entry(24, TIPS["models_tag"])
        self.tag_entry.connect("changed", self.on_install_changed)
        install.pack_start(self._caption_row("Library tag", self.tag_entry, TIPS["models_tag"]), False, False, 0)
        self.gguf_entry = self._entry(24, TIPS["models_gguf"])
        self.gguf_entry.connect("changed", self.on_install_changed)
        self.gguf_browse = gtk.Button(label="Browse")
        self.gguf_browse.set_tooltip_text(TIPS["models_gguf_browse"])
        self.gguf_browse.connect("clicked", self.on_browse, self.gguf_entry)
        install.pack_start(self._caption_row("GGUF", self.gguf_entry, TIPS["models_gguf"], self.gguf_browse), False, False, 0)
        self.mmproj_entry = self._entry(24, TIPS["models_projector"])
        self.mmproj_entry.connect("changed", self.on_install_changed)
        self.mmproj_browse = gtk.Button(label="Browse")
        self.mmproj_browse.set_tooltip_text(TIPS["models_projector_browse"])
        self.mmproj_browse.connect("clicked", self.on_browse, self.mmproj_entry)
        install.pack_start(
            self._caption_row("Projector", self.mmproj_entry, TIPS["models_projector"], self.mmproj_browse),
            False,
            False,
            0,
        )
        self.install_name = self._entry(24, TIPS["models_new_name"])
        self.install_name.connect("changed", self.on_install_changed)
        install.pack_start(self._caption_row("New name", self.install_name, TIPS["models_new_name"]), False, False, 0)

        self.fork_check = gtk.CheckButton(label="Install this forked file anyway")
        self.fork_check.set_tooltip_text(TIPS["models_fork"])
        self.fork_check.set_no_show_all(True)
        self.fork_check.connect("toggled", self.on_install_changed)
        self.fork_note = self._note(TIPS["models_fork"])
        install.pack_start(self.fork_check, False, False, 0)
        install.pack_start(self.fork_note, False, False, 0)

        self.install_button = gtk.Button(label="Install")
        self.install_button.set_tooltip_text(TIPS["models_install"])
        self.install_button.connect("clicked", self.on_install)
        install.pack_start(self.install_button, False, False, 0)

        self.progress = gtk.ProgressBar()
        self.progress.set_show_text(True)
        self.progress.set_no_show_all(True)
        self.progress.set_hexpand(True)
        self.progress.set_tooltip_text(TIPS["models_install"])
        install.pack_start(self.progress, False, False, 0)

        scroll.add(inner)
        page.pack_start(scroll, True, True, 0)
        return page

    def _order_box(self):
        return self._caption_row("Order", self.order_entry, TIPS["models_order"])

    def _linked(self, *children):
        """One segmented control, the same shape as the other either/or buttons."""
        box = self.gtk.Box(orientation=self.gtk.Orientation.HORIZONTAL, spacing=0)
        box.get_style_context().add_class("linked")
        box.set_halign(self.gtk.Align.START)
        for child in children:
            child.set_valign(self.gtk.Align.CENTER)
            box.pack_start(child, False, False, 0)
        return box

    def on_think(self, button) -> None:
        """One of Chat decides, On, or Off stays pressed."""
        if self._guard:
            return
        buttons = (self.think_follow, self.think_on, self.think_off)
        self._guard = True
        try:
            if button.get_active():
                for other in buttons:
                    if other is not button and other.get_active():
                        other.set_active(False)
            elif not any(other.get_active() for other in buttons):
                button.set_active(True)
        finally:
            self._guard = False

    def _wrapped(self, text: str, tip: str):
        label = self.gtk.Label(label=text)
        label.set_xalign(0)
        label.set_line_wrap(True)
        label.set_max_width_chars(48)
        label.set_hexpand(True)
        label.set_selectable(True)
        label.set_tooltip_text(tip)
        return label

    def _note(self, tip: str):
        label = self._wrapped("", tip)
        label.set_no_show_all(True)
        return label

    def _set_note(self, label, text: str) -> None:
        text = text or ""
        label.set_text(text)
        if text.strip():
            label.set_no_show_all(False)
            label.show()
        else:
            label.hide()
            label.set_no_show_all(True)

    def _entry(self, width: int, tip: str, expand: bool = True):
        entry = self.gtk.Entry()
        # Paths and Ollama names stay on a short row. Sampling and order are narrower.
        if width == 24:
            entry.set_width_chars(24)
        else:
            entry.set_width_chars(width)
        entry.set_hexpand(expand)
        entry.set_tooltip_text(tip)
        return entry

    def _row(self, *children):
        row = self.gtk.Box(orientation=self.gtk.Orientation.HORIZONTAL, spacing=8)
        row.set_hexpand(True)
        for child in children:
            row.pack_start(child, bool(child.get_hexpand()), True, 0)
        return row

    def _caption_row(self, caption: str, widget, tip: str, *extras, caption_width: int = 12):
        label = self.gtk.Label(label=caption)
        label.set_tooltip_text(tip)
        label.set_xalign(0)
        if caption_width:
            label.set_width_chars(caption_width)
        row = self._row(label, widget, *extras)
        return row

    def _split(self, left, right, spacing: int = 24):
        """Two columns when the page is wide. The minimum width stays one column."""
        import gi

        gi.require_version("Gtk", "3.0")
        from gi.repository import Gdk

        gtk = self.gtk
        if _split_holder.kind is None:

            class Split(gtk.Box):
                """Two columns when the page is wide. The minimum width stays one column.

                A plain box would make the page as wide as both columns. Gtk.Box already
                knows how to add, remove, and destroy its children.
                """

                def __init__(self, spacing):
                    super().__init__(orientation=gtk.Orientation.HORIZONTAL, spacing=int(spacing))
                    self._gap = int(spacing)
                    self.set_hexpand(True)
                    self.set_halign(gtk.Align.FILL)
                    self.set_valign(gtk.Align.START)

                def do_get_request_mode(self):
                    return gtk.SizeRequestMode.HEIGHT_FOR_WIDTH

                def _visible(self):
                    return [child for child in self.get_children() if child.get_visible()]

                def do_get_preferred_width(self):
                    kids = self._visible()
                    if not kids:
                        return 0, 0
                    mins = []
                    nats = []
                    for child in kids:
                        child_min, child_nat = child.get_preferred_width()
                        mins.append(int(child_min))
                        nats.append(max(int(child_min), int(child_nat)))
                    gap = self._gap * (len(kids) - 1)
                    return max(mins), sum(nats) + gap

                def do_get_preferred_height_for_width(self, width):
                    height = self._layout(int(width), False, 0, 0)
                    return height, height

                def do_get_preferred_height(self):
                    return self.do_get_preferred_height_for_width(self.do_get_preferred_width()[0])

                def do_get_preferred_width_for_height(self, _height):
                    return self.do_get_preferred_width()

                def do_size_allocate(self, allocation):
                    # The box records the CSS allocation. The columns are placed after that.
                    if int(allocation.width) < 1 or int(allocation.height) < 1:
                        self.set_allocation(allocation)
                        return
                    gtk.Box.do_size_allocate(self, allocation)
                    self._layout(int(allocation.width), True, int(allocation.x), int(allocation.y))

                def _layout(self, width, allocate, origin_x, origin_y):
                    kids = self._visible()
                    if not kids:
                        return 0
                    mins = []
                    for child in kids:
                        child_min, _child_nat = child.get_preferred_width()
                        mins.append(max(0, int(child_min)))
                    gap = self._gap * (len(kids) - 1) if len(kids) > 1 else 0
                    side_by_side = len(kids) > 1 and width >= sum(mins) + gap
                    if side_by_side:
                        spare = max(0, width - gap - sum(mins))
                        widths = [item + spare // len(kids) for item in mins]
                        widths[-1] += spare - (spare // len(kids)) * len(kids)
                        heights = []
                        for child, child_width in zip(kids, widths):
                            child_h, _nat = child.get_preferred_height_for_width(max(child_width, 1))
                            heights.append(max(int(child_h), 0))
                        if allocate:
                            x = origin_x
                            for child, child_width, child_h in zip(kids, widths, heights):
                                rect = Gdk.Rectangle()
                                rect.x = int(x)
                                rect.y = int(origin_y)
                                rect.width = int(child_width)
                                rect.height = int(child_h)
                                child.size_allocate(rect)
                                x += child_width + self._gap
                        return max(heights) if heights else 0
                    y = origin_y
                    for index, child in enumerate(kids):
                        child_h, _nat = child.get_preferred_height_for_width(max(width, 1))
                        child_h = max(int(child_h), 0)
                        if allocate:
                            rect = Gdk.Rectangle()
                            rect.x = int(origin_x)
                            rect.y = int(y)
                            rect.width = int(max(width, 0))
                            rect.height = child_h
                            child.size_allocate(rect)
                        y += child_h
                        if index < len(kids) - 1:
                            y += self._gap
                    return y - origin_y

            _split_holder.kind = Split
        box = _split_holder.kind(spacing)
        box.add(left)
        box.add(right)
        return box

    def _narrow_combo(self, combo) -> None:
        try:
            import gi

            gi.require_version("Pango", "1.0")
            from gi.repository import Pango

            cell = combo.get_cells()[0]
            cell.set_property("ellipsize", Pango.EllipsizeMode.MIDDLE)
            cell.set_property("width-chars", 28)
            cell.set_property("max-width-chars", 32)
        except Exception:
            log.debug("model combo could not ellipsize", exc_info=True)
        try:
            combo.set_popup_fixed_width(False)
        except Exception:
            pass
        combo.set_size_request(260, -1)

    def repair_menu_async(self) -> None:
        threading.Thread(target=self._repair_menu, name="nomad-turbo-models-menu", daemon=True).start()

    def _repair_menu(self) -> None:
        try:
            note = publish_menu(self.install, read_catalog(catalog_path()))
        except Exception as exc:
            log.exception("chat menu")
            note = str(exc)
        self.glib.idle_add(self._repair_done, note)

    def _repair_done(self, note: str) -> bool:
        if not self._loaded:
            self.status.set_text(note)
        return False

    def on_visible(self, stack, *_args) -> None:
        if stack.get_visible_child_name() != "models":
            return
        self._paint_index_note()
        if not self._loaded and not self._busy:
            self._begin_load()

    def on_refresh(self, *_args) -> None:
        self._begin_load()

    def on_model_changed(self, *_args) -> None:
        if self._guard:
            return
        name = self._selected()
        self._fill_menu(name)
        if name and name not in self._shows and not self._busy:
            self._start("Reading the model…", lambda _report: self._fetch_show(name, self._generation))

    def on_job(self, button) -> None:
        if self._guard or not button.get_active():
            return
        name = self._selected()
        if not name:
            return
        job = "benchmark" if name == BENCHMARK_MODEL else "tasks" if button is self.job_tasks else "chat"
        try:
            assign_job(
                self._catalog,
                name,
                job,
                parameter_size=self._parameter_size(name),
                size_bytes=self._size_of(name),
            )
        except ValueError as exc:
            self._guard = True
            try:
                if self._tasks == name:
                    self.job_tasks.set_active(True)
                else:
                    self.job_chat.set_active(True)
            finally:
                self._guard = False
            self.status.set_text(str(exc))
            return
        self._paint_locks(name)

    def on_install_changed(self, *_args) -> None:
        reason = forked_reason(
            self.tag_entry.get_text(),
            self.gguf_entry.get_text(),
            self.mmproj_entry.get_text(),
            self.install_name.get_text(),
        )
        if reason:
            self.fork_check.set_no_show_all(False)
            self.fork_check.show()
            self._set_note(self.fork_note, reason)
        else:
            self.fork_check.set_active(False)
            self.fork_check.hide()
            self.fork_check.set_no_show_all(True)
            self._set_note(self.fork_note, "")
        self._paint_index_note()

    def on_browse(self, _button, entry) -> None:
        gtk = self.gtk
        chooser = gtk.FileChooserDialog(
            title="GGUF file",
            parent=self.parent,
            action=gtk.FileChooserAction.OPEN,
            buttons=(
                gtk.STOCK_CANCEL,
                gtk.ResponseType.CANCEL,
                gtk.STOCK_OPEN,
                gtk.ResponseType.OK,
            ),
        )
        filt = gtk.FileFilter()
        filt.set_name("GGUF")
        filt.add_pattern("*.gguf")
        chooser.add_filter(filt)
        response = chooser.run()
        filename = chooser.get_filename()
        chooser.destroy()
        if response == gtk.ResponseType.OK and filename:
            entry.set_text(filename)

    def on_save(self, *_args) -> None:
        name = self._selected()
        if not name or self._busy:
            return
        if not self._loaded:
            self.status.set_text("Models are still loading.")
            return
        try:
            form = self._read_form(name)
        except ValueError as exc:
            self.status.set_text(str(exc))
            return
        phase = self.session.view().get("phase")
        catalog = json.loads(json.dumps(self._catalog))
        self._start("Saving…", lambda report: self._save(name, form, phase, catalog, report))

    def on_test(self, *_args) -> None:
        name = self._selected()
        if not name or self._busy:
            return
        phase = self.session.view().get("phase")
        block = action_block("test", phase)
        if block:
            self.status.set_text(block)
            return
        show = self._shows.get(name) or {}
        base = self._base
        self._start("Testing…", lambda report: self._test(name, phase, bool(show.get("thinking")), base, report))

    def on_delete(self, *_args) -> None:
        name = self._selected()
        if not name or self._busy:
            return
        reason = delete_block_reason(name, tasks_model=self._tasks)
        if reason:
            self.status.set_text(reason)
            return
        phase = self.session.view().get("phase")
        block = action_block("delete", phase)
        if block:
            self.status.set_text(block)
            return
        if not self._confirm_delete(name):
            return
        catalog = json.loads(json.dumps(self._catalog))
        installed = [item["name"] for item in self._tags]
        self._start(
            "Deleting…",
            lambda report: self._delete(name, phase, catalog, installed, self._tasks, self._last, self._base),
        )

    def on_install(self, *_args) -> None:
        if self._busy or not self._install_allowed():
            reason = action_block("install", self.session.view().get("phase"))
            if reason:
                self.status.set_text(reason)
            return
        phase = self.session.view().get("phase")
        installed = [item["name"] for item in self._tags]
        form = {
            "tag": self.tag_entry.get_text(),
            "gguf": self.gguf_entry.get_text(),
            "mmproj": self.mmproj_entry.get_text(),
            "name": self.install_name.get_text(),
            "confirm": bool(self.fork_check.get_active()),
        }
        catalog = json.loads(json.dumps(self._catalog))
        models_dir = getattr(self.install, "models_dir", "")
        self._start(
            "Installing…",
            lambda report: self._install(form, phase, installed, self._base, models_dir, catalog, report),
        )

    def _confirm_delete(self, name: str) -> bool:
        gtk = self.gtk
        dialog = gtk.MessageDialog(
            transient_for=self.parent,
            modal=True,
            destroy_with_parent=True,
            message_type=gtk.MessageType.WARNING,
            buttons=gtk.ButtonsType.NONE,
            text=f"Delete {name}?",
        )
        extra = in_use_sentence(name, self._last, self._tasks) or "The model is removed from Ollama."
        if self._last == name:
            nxt = next_chat_model(self._catalog, [item["name"] for item in self._tags], name)
            if nxt:
                extra += f" The assistant will use {nxt}."
        dialog.format_secondary_text(extra)
        dialog.add_button("Cancel", gtk.ResponseType.CANCEL)
        dialog.add_button("Delete", gtk.ResponseType.OK)
        response = dialog.run()
        dialog.destroy()
        return response == gtk.ResponseType.OK

    def _selected(self) -> str:
        return self.model_combo.get_active_id() or ""

    def _tag(self, name: str) -> dict:
        for item in self._tags:
            if item["name"] == name:
                return item
        return {}

    def _parameter_size(self, name: str) -> str:
        show = self._shows.get(name) or {}
        return show.get("parameter_size") or self._tag(name).get("parameter_size") or ""

    def _size_of(self, name: str) -> int:
        return int(self._tag(name).get("size") or 0)

    def _read_form(self, name: str) -> dict:
        label = clean_label(self.name_entry.get_text())
        order_text = self.order_entry.get_text().strip()
        if not order_text.isdigit() or int(order_text) > 9999:
            raise ValueError("Order is a whole number up to 9999.")
        if name == BENCHMARK_MODEL:
            job = "benchmark"
        elif self.job_tasks.get_active():
            job = "tasks"
        else:
            job = "chat"
        think = "on" if self.think_on.get_active() else "off" if self.think_off.get_active() else "follow"
        ctx = self.context_combo.get_active_id() or "auto"
        clear_num_ctx = ctx == "auto"
        num_ctx = None if clear_num_ctx else int(ctx)
        return {
            "label": label,
            "order": int(order_text),
            "hidden": not self.show_check.get_active(),
            "show_sensitive": self.show_check.get_sensitive(),
            "think": think,
            "job": job,
            "sampling": sampling_from_entries(
                {
                    "temperature": self.temp_entry.get_text(),
                    "top_p": self.top_p_entry.get_text(),
                    "top_k": self.top_k_entry.get_text(),
                    "min_p": self.min_p_entry.get_text(),
                }
            ),
            "num_ctx": num_ctx,
            "clear_num_ctx": clear_num_ctx,
            "num_ctx_set": name in self._shows,
            "renderer": clean_token(self.renderer_entry.get_text(), "Renderer"),
            "parser": clean_token(self.parser_entry.get_text(), "Parser"),
            "parameter_size": self._parameter_size(name),
            "size": self._size_of(name),
            "details_ready": name in self._shows,
            "show": self._shows.get(name) or {},
        }

    def _begin_load(self) -> None:
        if self._busy:
            return
        self._generation += 1
        token = self._generation
        self._start("Reading models…", lambda _report: self._load(token))

    def _load(self, token: int) -> dict:
        base = discover_ollama_base(self.install.ollama_container)
        payload = get_json(base, "/api/tags", timeout=30)
        tags = chat_names(parse_tags(payload))
        tasks = last = cap = ""
        kv_error = ""
        try:
            from nomad_turbo.store import get_kv

            tasks = get_kv(self.install, "ai.tasksModel") or ""
            last = get_kv(self.install, "chat.lastModel") or ""
            cap = get_kv(self.install, "ai.contextWindow") or ""
        except Exception as exc:
            kv_error = str(exc)
        vram = None
        try:
            from nomad_turbo.devices import list_devices

            vram = largest_nvidia_mib(list_devices())
        except Exception:
            log.exception("video memory")
        catalog = merge_live(read_catalog(catalog_path()), [item["name"] for item in tags], tasks or None)
        return {
            "kind": "load",
            "token": token,
            "base": base,
            "tags": tags,
            "catalog": catalog,
            "tasks": tasks,
            "last": last,
            "cap": cap,
            "vram": vram,
            "kv_error": kv_error,
        }

    def _fetch_show(self, name: str, token: int) -> dict:
        shown = parse_show(post_json(self._base, "/api/show", {"model": name}, timeout=60))
        return {"kind": "show", "token": token, "show_for": name, "show": shown}

    def _save(self, name: str, form: dict, phase: object, catalog: dict, report) -> dict:
        del report
        if name not in catalog["models"]:
            catalog = merge_live(catalog, [name], self._tasks or None)
        pref = catalog["models"][name]
        pref["label"] = form["label"]
        pref["order"] = form["order"]
        pref["think"] = form["think"]
        if form["show_sensitive"]:
            pref["hidden"] = form["hidden"]
        catalog = assign_job(
            catalog,
            name,
            form["job"],
            parameter_size=form["parameter_size"],
            size_bytes=form["size"],
        )
        write_catalog(catalog_path(), catalog)
        notes = ["Saved."]
        if form["job"] == "tasks":
            from nomad_turbo.store import set_kv

            set_kv(self.install, "ai.tasksModel", name)
        try:
            notes.append(publish_menu(self.install, catalog))
        except MenuPatchError as exc:
            notes.append(str(exc))
        wrote = False
        if form["details_ready"] and name != BENCHMARK_MODEL:
            show = form["show"]
            changed = settings_changed(
                modelfile=show.get("modelfile") or "",
                parameters=show.get("parameters") or "",
                renderer=show.get("renderer") or "",
                parser=show.get("parser") or "",
                sampling=form["sampling"],
                num_ctx=form["num_ctx"],
                num_ctx_set=form["num_ctx_set"],
                clear_num_ctx=form["clear_num_ctx"],
                renderer_edit=form["renderer"],
                parser_edit=form["parser"],
            )
            if changed:
                block = action_block("parameters", phase)
                if block:
                    notes.append(PARAM_WAIT)
                else:
                    notes.append(self._write_parameters(name, form))
                    wrote = True
        tasks = name if form["job"] == "tasks" else self._tasks
        return {"kind": "note", "note": " ".join(notes), "catalog": catalog, "tasks": tasks, "drop_show": name if wrote else ""}

    def _write_parameters(self, name: str, form: dict) -> str:
        show = form["show"]
        modelfile = show.get("modelfile") or ""
        if not modelfile.strip():
            return "This model has no modelfile to update."
        rewritten = rewrite_modelfile(
            modelfile,
            sampling=form["sampling"],
            num_ctx=None if form["clear_num_ctx"] else form["num_ctx"],
            clear_num_ctx=form["clear_num_ctx"],
            renderer=form["renderer"],
            parser=form["parser"],
        )
        for event in post_stream(self._base, "/api/create", create_payload(name, rewritten), timeout=None):
            status, fraction = progress_from_status(event)
            self._report(status or "Writing the model…", fraction)
        return "Context and sampling were written. The model was reloaded."

    def _test(self, name: str, phase: object, thinking: bool, base: str, report) -> dict:
        del report
        block = action_block("test", phase)
        if block:
            return {"kind": "note", "note": block}
        self._report("Thinking off…", None)
        off = post_json(base, "/api/chat", chat_payload(name, THINK_OFF_PROMPT, False), timeout=180)
        ok, text = reply_ok(off)
        parts = ["Thinking off: " + _short(text)]
        if not ok:
            return {"kind": "note", "note": parts[0]}
        if thinking:
            self._report("Thinking on…", None)
            on = post_json(base, "/api/chat", chat_payload(name, THINK_ON_PROMPT, True), timeout=180)
            ok, text = reply_ok(on)
            parts.append("Thinking on: " + _short(text))
        return {"kind": "note", "note": " ".join(parts)}

    def _delete(self, name, phase, catalog, installed, tasks, last, base) -> dict:
        block = action_block("delete", phase)
        if block:
            return {"kind": "note", "note": block}
        reason = delete_block_reason(name, tasks_model=tasks)
        if reason:
            return {"kind": "note", "note": reason}
        delete_model(base, name)
        note = f"Deleted {name}."
        if last == name:
            nxt = next_chat_model(catalog, installed, name)
            if nxt:
                from nomad_turbo.store import set_kv

                set_kv(self.install, "chat.lastModel", nxt)
                note += f" The assistant will use {nxt}."
        return {"kind": "note", "note": note, "reload": True}

    def _install(self, form, phase, installed, base, models_dir, catalog, report) -> dict:
        del report
        block = action_block("install", phase)
        if block:
            return {"kind": "note", "note": block}
        plan = plan_install(
            tag=form["tag"],
            gguf=form["gguf"],
            mmproj=form["mmproj"],
            name=form["name"],
            confirm=form["confirm"],
            installed=installed,
        )
        if plan["kind"] == "pull":
            self._report(self._library_preview(plan["name"]), None)
            for event in post_stream(base, "/api/pull", {"model": plan["name"], "stream": True}, timeout=None):
                status, fraction = progress_from_status(event)
                self._report(status or "Downloading…", fraction)
            name = plan["name"]
        else:
            name = self._create_gguf(plan, models_dir, base)
        fresh = name not in installed and name not in catalog.get("models", {})
        catalog = merge_live(catalog, installed + [name], self._tasks or None)
        if fresh:
            orders = [int(pref["order"]) for pref in catalog["models"].values()] or [0]
            catalog["models"][name]["hidden"] = True
            catalog["models"][name]["order"] = max(orders) + 1
            catalog["models"][name]["job"] = "chat"
        write_catalog(catalog_path(), catalog)
        try:
            menu = publish_menu(self.install, catalog)
        except MenuPatchError as exc:
            menu = str(exc)
        hidden = fresh
        note = f"Installed {name}. " + menu
        if hidden:
            note += " It stays off the menu until Show in menu is turned on."
        return {"kind": "note", "note": note, "reload": True, "select": name}

    def _library_preview(self, tag: str) -> str:
        billions = parse_parameter_billions("", tag)
        hint = f" The name says about {billions:g}B." if billions else ""
        url = registry_manifest_url(tag)
        if not url:
            return "Size appears as the download starts." + hint
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/vnd.docker.distribution.manifest.v2+json",
                    "User-Agent": "nomad-turbo",
                },
            )
            with urllib.request.urlopen(request, timeout=8) as response:
                manifest = json.loads(response.read().decode("utf-8"))
            return format_bytes(sum_manifest_bytes(manifest)) + " to download." + hint
        except Exception:
            return "Size appears as the download starts." + hint

    def _create_gguf(self, plan: dict, models_dir: str, base: str) -> str:
        gguf = Path(plan["gguf"])
        if not gguf.is_file():
            raise ValueError("The GGUF file was not found.")
        mmproj = Path(plan["mmproj"]) if plan.get("mmproj") else None
        if mmproj is not None and not mmproj.is_file():
            raise ValueError("The projector file was not found.")
        if not models_dir:
            raise ValueError("The Ollama models folder was not found.")
        sizes = [gguf.stat().st_size]
        if mmproj is not None:
            sizes.append(mmproj.stat().st_size)
        free = shutil.disk_usage(models_dir).free
        short = space_error(free, bytes_needed(sizes))
        if short:
            raise ValueError(short)
        self._report(file_preview(gguf), None)
        host, container = incoming_paths(models_dir, plan["name"])
        pairs = [(str(gguf), str(host / "model.gguf"))]
        model_from = container + "/model.gguf"
        proj_from = ""
        if mmproj is not None:
            pairs.append((str(mmproj), str(host / "mmproj.gguf")))
            proj_from = container + "/mmproj.gguf"
        self._run_commands(stage_commands(pairs))
        try:
            modelfile = gguf_modelfile(model_from, proj_from)
            for event in post_stream(base, "/api/create", create_payload(plan["name"], modelfile), timeout=None):
                status, fraction = progress_from_status(event)
                self._report(status or "Creating the model…", fraction)
        except Exception as exc:
            raise RuntimeError(f"{exc} The copied files are still in {host}.") from exc
        try:
            self._run_commands([cleanup_command(str(host))])
            subprocess.run(["sudo", "-n", "rmdir", str(host.parent)], check=False)
        except Exception:
            log.exception("incoming cleanup for %s", host)
        return plan["name"]

    def _run_commands(self, commands: list[list[str]]) -> None:
        for command in commands:
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip()
                raise RuntimeError(detail or "The file copy failed.")

    def _start(self, label: str, work) -> None:
        if self._busy:
            return
        self._busy = True
        self._paint_busy(True, label)
        threading.Thread(target=self._run, args=(work,), name="nomad-turbo-models", daemon=True).start()

    def _run(self, work) -> None:
        try:
            result = work(self._report)
        except Exception as exc:
            log.exception("models")
            result = str(exc)
        self.glib.idle_add(self._finish, result)

    def _report(self, text: str, fraction) -> None:
        self.glib.idle_add(self._paint_progress, text or "", fraction)

    def _paint_progress(self, text: str, fraction) -> bool:
        self.progress.set_no_show_all(False)
        self.progress.show()
        self.progress.set_text(text)
        if fraction is None:
            self.progress.pulse()
        else:
            self.progress.set_fraction(float(fraction))
        return False

    def _paint_busy(self, busy: bool, label: str) -> None:
        for button in (
            self.save_button,
            self.test_button,
            self.delete_button,
            self.install_button,
            self.refresh_button,
            self.gguf_browse,
            self.mmproj_browse,
        ):
            button.set_sensitive(not busy)
        if busy:
            self._paint_progress(label, None)
        else:
            self.progress.hide()
            self.progress.set_no_show_all(True)
            self._paint_index_note()

    def _finish(self, result) -> bool:
        self._busy = False
        self._paint_busy(False, "")
        if isinstance(result, dict) and result.get("kind") == "load":
            if result.get("token") != self._generation:
                return False
            self._apply_load(result)
            return False
        if isinstance(result, dict) and result.get("kind") == "show":
            if result.get("token") != self._generation:
                return False
            name = result.get("show_for") or ""
            self._shows[name] = result.get("show") or {}
            if self._selected() == name:
                self._fill_details(name)
            return False
        if isinstance(result, dict):
            select = result.get("select") or ""
            if result.get("catalog") is not None:
                self._catalog = result["catalog"]
                if not result.get("reload"):
                    self._fill_combo(select or self._selected())
            if result.get("tasks") is not None and result.get("kind") == "note" and "tasks" in result:
                self._tasks = result.get("tasks") or self._tasks
            if result.get("drop_show"):
                self._shows.pop(result["drop_show"], None)
            self.status.set_text(str(result.get("note") or ""))
            if result.get("reload"):
                self._pending_select = select
                self._begin_load()
            return False
        self.status.set_text(str(result))
        return False

    def _apply_load(self, result: dict) -> None:
        self._loaded = True
        self._base = result["base"]
        self._tags = result["tags"]
        self._catalog = result["catalog"]
        self._tasks = result.get("tasks") or ""
        self._last = result.get("last") or ""
        self._cap = result.get("cap") or ""
        self._vram = result.get("vram")
        self._shows = {}
        names = [item["name"] for item in self._tags]
        preferred = getattr(self, "_pending_select", "") or self._last
        self._pending_select = ""
        ordered = page_menu(self._catalog, names)
        if preferred not in ordered:
            preferred = ordered[0] if ordered else ""
        self._fill_combo(preferred)
        note = f"{len(self._tags)} models."
        if result.get("kv_error"):
            note += " " + str(result["kv_error"])
        self.status.set_text(note)
        self._fill_menu(preferred)
        if preferred:
            self._start("Reading the model…", lambda _report: self._fetch_show(preferred, self._generation))

    def _fill_combo(self, preferred: str) -> None:
        tags = getattr(self, "_tags", None) or []
        if not tags:
            return
        names = page_menu(self._catalog, [item["name"] for item in tags])
        current = preferred or self._selected()
        self._guard = True
        try:
            self.model_combo.remove_all()
            for name in names:
                label = self._catalog.get("models", {}).get(name, {}).get("label") or name
                self.model_combo.append(name, label)
            if current:
                self.model_combo.set_active_id(current)
        finally:
            self._guard = False

    def _fill_menu(self, name: str) -> None:
        pref = self._catalog.get("models", {}).get(name) or {}
        self._guard = True
        try:
            self.name_entry.set_text(pref.get("label") or "")
            self.order_entry.set_text(str(pref.get("order") if pref else ""))
            job = pref.get("job") or "chat"
            if job == "tasks":
                self.job_tasks.set_active(True)
            else:
                self.job_chat.set_active(True)
            think = pref.get("think") or "follow"
            self.think_follow.set_active(think == "follow")
            self.think_on.set_active(think == "on")
            self.think_off.set_active(think == "off")
            shown = not bool(pref.get("hidden")) and job not in ("tasks", "benchmark")
            self.show_check.set_active(shown)
        finally:
            self._guard = False
        self._paint_locks(name)
        tag = self._tag(name)
        bits = [name] if name else []
        if tag.get("parameter_size"):
            bits.append(str(tag["parameter_size"]))
        if tag.get("quantization"):
            bits.append(str(tag["quantization"]))
        if tag.get("size"):
            bits.append(format_bytes(int(tag["size"])))
        self._set_note(self.detail, " · ".join(bits))
        self._set_note(self.in_use, in_use_sentence(name, self._last, self._tasks) if name else "")
        if name in self._shows:
            self._fill_details(name)
        else:
            self._clear_details()

    def _clear_details(self) -> None:
        self._set_note(self.photos, "")
        self._set_note(self.tools, "")
        self._set_note(self.think_note, "")
        self._set_note(self.context_label, "")
        for entry in (self.temp_entry, self.top_p_entry, self.top_k_entry, self.min_p_entry, self.renderer_entry, self.parser_entry):
            entry.set_text("")

    def _fill_details(self, name: str) -> None:
        show = self._shows.get(name) or {}
        self._set_note(self.photos, "Photos: yes" if show.get("vision") else "Photos: no")
        self._set_note(self.tools, "Tools: yes" if show.get("tools") else "Tools: no")
        can_think = bool(show.get("thinking"))
        for button in (self.think_follow, self.think_on, self.think_off):
            button.set_sensitive(can_think)
        self._set_note(self.think_note, "" if can_think else "This model does not think.")
        samplers = show.get("samplers") or {}
        self._guard = True
        try:
            self.temp_entry.set_text(samplers.get("temperature", ""))
            self.top_p_entry.set_text(samplers.get("top_p", ""))
            self.top_k_entry.set_text(samplers.get("top_k", ""))
            self.min_p_entry.set_text(samplers.get("min_p", ""))
            self.renderer_entry.set_text(show.get("renderer") or "")
            self.parser_entry.set_text(show.get("parser") or "")
            self._fill_context(name, show)
        finally:
            self._guard = False
        self._paint_locks(name)

    def _fill_context(self, name: str, show: dict) -> None:
        cap = int(self._cap) if str(self._cap).isdigit() else None
        model_bytes = self._size_of(name)
        advice = context_advice(
            model_max=show.get("context_length"),
            num_ctx=show.get("num_ctx"),
            user_cap=cap,
            vram_mib=self._vram,
            model_bytes=model_bytes,
            kv_bytes_per_token=kv_bytes_for(show.get("model_info"), show.get("parameter_size"), name),
        )
        self._set_note(self.context_label, context_sentence(advice, show.get("context_length"), cap))
        current = show.get("num_ctx")
        self.context_combo.remove_all()
        self.context_combo.append("auto", "Auto")
        rungs = list(advice["rungs"])
        if current and int(current) not in rungs:
            rungs.append(int(current))
        for rung in rungs:
            self.context_combo.append(str(rung), str(rung))
        self.context_combo.set_active_id(str(current) if current else "auto")

    def _paint_locks(self, name: str) -> None:
        benchmark = name == BENCHMARK_MODEL
        only_tasks = bool(name) and name == self._tasks
        self.job_chat.set_sensitive(not benchmark and not only_tasks)
        self.job_tasks.set_sensitive(not benchmark)
        tasks_selected = self.job_tasks.get_active() and not benchmark
        self.show_check.set_sensitive(not benchmark and not tasks_selected and not only_tasks)
        if benchmark or tasks_selected:
            self._set_note(
                self.job_note,
                "Benchmark. Llama 3.1 8B stays on this job."
                if benchmark
                else "Tasks stays off the menu. Choose another tasks model before this one returns to chat."
                if only_tasks
                else "Tasks stays off the menu.",
            )
        else:
            self._set_note(self.job_note, "")
        locked = benchmark
        self.context_combo.set_sensitive(not locked)
        for entry in (self.temp_entry, self.top_p_entry, self.top_k_entry, self.min_p_entry, self.renderer_entry, self.parser_entry):
            entry.set_sensitive(not locked)

    def _paint_index_note(self) -> None:
        waiting = action_block("install", self.session.view().get("phase"))
        self._set_note(self.index_note, waiting or "")
        if self._busy:
            return
        self.test_button.set_sensitive(not waiting)
        self.delete_button.set_sensitive(not waiting)
        self.refresh_button.set_sensitive(True)
        self.gguf_browse.set_sensitive(True)
        self.mmproj_browse.set_sensitive(True)
        self.install_button.set_sensitive(self._install_allowed())
        self.save_button.set_sensitive(True)

    def _install_allowed(self) -> bool:
        if self._busy or action_block("install", self.session.view().get("phase")):
            return False
        reason = forked_reason(
            self.tag_entry.get_text(),
            self.gguf_entry.get_text(),
            self.mmproj_entry.get_text(),
            self.install_name.get_text(),
        )
        if reason and not self.fork_check.get_active():
            return False
        return True


def _short(text: str) -> str:
    text = " ".join(str(text or "").split())
    if len(text) > 180:
        return text[:177] + "..."
    return text
