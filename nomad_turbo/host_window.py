"""Separate window for the host meters.

The Index and Repair pages stay as they are. This frame resizes on its own
and remembers its size and place.
"""

from __future__ import annotations

import logging

from nomad_turbo.discover import save_install
from nomad_turbo.host import (
    HostPoint,
    HostSampler,
    cpu_caption,
    disk_caption,
    gpu_caption,
    plot_segments,
    ram_caption,
    vram_caption,
)
from nomad_turbo.tips import TIPS

log = logging.getLogger("nomad_turbo")

_CPU = (0.35, 0.68, 0.95)
_RAM = (0.35, 0.80, 0.72)
_GPU = (0.92, 0.62, 0.28)
_VRAM = (0.86, 0.75, 0.32)


def _stroke(cr, segments, rgb, bottom: int, fill: bool) -> None:
    cr.set_line_width(1.6)
    for segment in segments:
        if len(segment) < 2:
            if segment:
                cr.set_source_rgb(*rgb)
                cr.arc(segment[0][0], segment[0][1], 1.6, 0, 6.3)
                cr.fill()
            continue
        if fill:
            cr.set_source_rgba(rgb[0], rgb[1], rgb[2], 0.22)
            cr.move_to(segment[0][0], segment[0][1])
            for x, y in segment:
                cr.line_to(x, y)
            cr.line_to(segment[-1][0], bottom)
            cr.line_to(segment[0][0], bottom)
            cr.close_path()
            cr.fill()
        cr.set_source_rgb(*rgb)
        cr.move_to(*segment[0])
        for point in segment[1:]:
            cr.line_to(*point)
        cr.stroke()


def _paint_series(cr, width: int, height: int, values, low: float, high: float, rgb, fill: bool) -> None:
    cr.set_source_rgba(1, 1, 1, 0.05)
    cr.rectangle(0, 0, width, height)
    cr.fill()
    segments = plot_segments(values, width, height, low, high)
    _stroke(cr, segments, rgb, height, fill)


class HostWindow:
    """CPU, RAM, GPU, vRAM, and the disks the library uses. Closed on its own."""

    def __init__(self, install, gtk, glib, on_close) -> None:
        self.install = install
        self.gtk = gtk
        self.glib = glib
        self.on_close = on_close
        self._closed = False
        self._timer = None
        self._save_timer = None
        self._points: list[HostPoint] = []
        self.sampler = HostSampler(install.storage_host)
        window = gtk.Window(title="Nomad TURBO Host")
        window.set_resizable(True)
        width = int(getattr(install, "host_width", 0) or 0)
        height = int(getattr(install, "host_height", 0) or 0)
        window.set_default_size(width or 460, height or 720)
        window.set_size_request(300, 420)
        self.window = window
        root = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=6)
        root.set_border_width(12)
        window.add(root)
        self.cpu_label = self._caption(gtk, "CPU")
        self.ram_label = self._caption(gtk, "RAM")
        self.gpu_label = self._caption(gtk, "GPU")
        self.vram_label = self._caption(gtk, "vRAM —")
        self.disk_label = gtk.Label(label="Disk")
        self.disk_label.set_xalign(0)
        self.disk_label.set_yalign(0)
        self.disk_label.set_line_wrap(True)
        self.disk_label.set_max_width_chars(48)
        self.disk_label.set_tooltip_text(TIPS["host_disk"])
        self.cpu_area = self._graph(gtk)
        self.ram_area = self._graph(gtk)
        self.gpu_area = self._graph(gtk)
        self.vram_area = self._graph(gtk)
        for label, area in (
            (self.cpu_label, self.cpu_area),
            (self.ram_label, self.ram_area),
            (self.gpu_label, self.gpu_area),
            (self.vram_label, self.vram_area),
        ):
            root.pack_start(label, False, False, 0)
            root.pack_start(area, True, True, 0)
        root.pack_start(self.disk_label, False, False, 0)
        self.cpu_area.connect("draw", self._draw_cpu)
        self.ram_area.connect("draw", self._draw_ram)
        self.gpu_area.connect("draw", self._draw_gpu)
        self.vram_area.connect("draw", self._draw_vram)
        window.connect("delete-event", self._on_delete)
        window.connect("configure-event", self._on_configure)
        saved_x = getattr(install, "host_x", None)
        saved_y = getattr(install, "host_y", None)
        if saved_x is not None and saved_y is not None:
            window.move(int(saved_x), int(saved_y))
        self.sampler.start()
        self._timer = glib.timeout_add(500, self._refresh)

    def _caption(self, gtk, text: str):
        from gi.repository import Pango

        label = gtk.Label(label=text)
        label.set_xalign(0)
        label.set_ellipsize(Pango.EllipsizeMode.END)
        return label

    def _graph(self, gtk):
        area = gtk.DrawingArea()
        area.set_hexpand(True)
        area.set_vexpand(True)
        area.set_size_request(-1, 48)
        return area

    def present(self) -> None:
        self.window.present()

    def _latest(self) -> HostPoint | None:
        return self._points[-1] if self._points else None

    def _values(self, name: str) -> list:
        return [getattr(point, name) for point in self._points]

    def _draw_cpu(self, _widget, cr) -> bool:
        width, height = self._size(self.cpu_area)
        _paint_series(cr, width, height, self._values("cpu"), 0, 100, _CPU, True)
        return False

    def _draw_ram(self, _widget, cr) -> bool:
        width, height = self._size(self.ram_area)
        _paint_series(cr, width, height, self._values("ram_percent"), 0, 100, _RAM, True)
        return False

    def _draw_gpu(self, _widget, cr) -> bool:
        width, height = self._size(self.gpu_area)
        _paint_series(cr, width, height, self._values("gpu"), 0, 100, _GPU, True)
        return False

    def _draw_vram(self, _widget, cr) -> bool:
        width, height = self._size(self.vram_area)
        _paint_series(cr, width, height, self._values("vram_percent"), 0, 100, _VRAM, True)
        return False

    def _size(self, widget) -> tuple[int, int]:
        return int(widget.get_allocated_width() or 0), int(widget.get_allocated_height() or 0)

    def _refresh(self) -> bool:
        if self._closed:
            return False
        self._points = self.sampler.snapshot()
        latest = self._latest()
        self.cpu_label.set_text(cpu_caption(latest))
        self.ram_label.set_text(ram_caption(latest))
        self.gpu_label.set_text(gpu_caption(latest))
        self.vram_label.set_text(vram_caption(latest))
        self.disk_label.set_text(disk_caption(latest))
        for area in (self.cpu_area, self.ram_area, self.gpu_area, self.vram_area):
            area.queue_draw()
        return True

    def _on_configure(self, *_args) -> bool:
        if self._closed:
            return False
        if self._save_timer:
            self.glib.source_remove(self._save_timer)
        self._save_timer = self.glib.timeout_add(400, self._save_place)
        return False

    def _save_place(self) -> bool:
        self._save_timer = None
        width, height = self.window.get_size()
        x, y = self.window.get_position()
        if width < 200 or height < 200:
            return False
        self.install.host_width = int(width)
        self.install.host_height = int(height)
        self.install.host_x = int(x)
        self.install.host_y = int(y)
        try:
            save_install(self.install)
        except Exception:
            log.exception("could not save the host window")
        return False

    def _on_delete(self, *_args) -> bool:
        self._shutdown()
        return False

    def close(self) -> None:
        self._shutdown()
        self.window.destroy()

    def _shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._timer:
            self.glib.source_remove(self._timer)
            self._timer = None
        if self._save_timer:
            self.glib.source_remove(self._save_timer)
            self._save_timer = None
        self._save_place()
        self.sampler.stop()
        callback = self.on_close
        self.on_close = None
        if callback is not None:
            callback()
