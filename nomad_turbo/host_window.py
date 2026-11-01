"""Host meters for the side pane.

The header button shows and hides this pane. Sampling runs only while it
is shown. The main window remembers the pane's width.
"""

from __future__ import annotations

from nomad_turbo.host import (
    HostPoint,
    HostSampler,
    cpu_caption,
    disk_line,
    disk_series,
    gpu_caption,
    plot_segments,
    ram_caption,
    rate_ceiling,
    vram_caption,
)
from nomad_turbo.tips import TIPS

_CPU = (0.35, 0.68, 0.95)
_RAM = (0.35, 0.80, 0.72)
_GPU = (0.92, 0.62, 0.28)
_VRAM = (0.86, 0.75, 0.32)
_READ = (0.35, 0.62, 0.95)
_WRITE = (0.90, 0.42, 0.38)


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


class HostPanel:
    """CPU, RAM, GPU, vRAM, and the disks the library uses."""

    def __init__(self, install, gtk, glib) -> None:
        self.install = install
        self.gtk = gtk
        self.glib = glib
        self._timer = None
        self._points: list[HostPoint] = []
        self.sampler = HostSampler(install.storage_host)
        root = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=6)
        root.set_margin_start(8)
        root.set_margin_end(4)
        root.set_hexpand(False)
        root.set_vexpand(True)
        # Wide enough to read a disk line. The bar beside the pages sets the rest.
        root.set_size_request(220, -1)
        self.widget = root
        self.cpu_label = self._caption(gtk, "CPU")
        self.ram_label = self._caption(gtk, "RAM")
        self.gpu_label = self._caption(gtk, "GPU")
        self.vram_label = self._caption(gtk, "vRAM —")
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
        self._disk_widgets: list = []
        self._disk_rows: dict = {}
        self._disk_labels: tuple[str, ...] = ()
        self._disk_series: dict = {}
        self._rebuild_disks(())
        self.cpu_area.connect("draw", self._draw_cpu)
        self.ram_area.connect("draw", self._draw_ram)
        self.gpu_area.connect("draw", self._draw_gpu)
        self.vram_area.connect("draw", self._draw_vram)

    def start(self) -> None:
        """Read the host while the pane is on screen."""
        if self._timer is not None:
            self.sampler.start()
            return
        self.sampler.start()
        self._timer = self.glib.timeout_add(500, self._refresh)

    def stop(self) -> None:
        """Pause readings when the pane is hidden or the window closes."""
        if self._timer is not None:
            self.glib.source_remove(self._timer)
            self._timer = None
        self.sampler.stop()

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

    def _draw_disk(self, widget, cr, label: str) -> bool:
        width, height = self._size(widget)
        reads, writes = self._disk_series.get(label, ([], []))
        cr.set_source_rgba(1, 1, 1, 0.05)
        cr.rectangle(0, 0, width, height)
        cr.fill()
        high = rate_ceiling(list(reads) + list(writes))
        _stroke(cr, plot_segments(reads, width, height, 0, high), _READ, height, True)
        _stroke(cr, plot_segments(writes, width, height, 0, high), _WRITE, height, False)
        return False

    def _rebuild_disks(self, labels: tuple[str, ...]) -> None:
        """One caption and one graph per disk. An empty reading stays one line."""
        for widget in self._disk_widgets:
            self.widget.remove(widget)
            widget.destroy()
        self._disk_widgets = []
        self._disk_rows = {}
        self._disk_labels = labels
        if not labels:
            caption = self._caption(self.gtk, "Disk")
            caption.set_tooltip_text(TIPS["host_disk"])
            self.widget.pack_start(caption, False, False, 0)
            caption.show()
            self._disk_widgets.append(caption)
            return
        for name in labels:
            caption = self._caption(self.gtk, name)
            caption.set_tooltip_text(TIPS["host_disk"])
            area = self._graph(self.gtk)
            area.connect("draw", self._draw_disk, name)
            self.widget.pack_start(caption, False, False, 0)
            self.widget.pack_start(area, True, True, 0)
            caption.show()
            area.show()
            self._disk_widgets.extend((caption, area))
            self._disk_rows[name] = (caption, area)

    def _size(self, widget) -> tuple[int, int]:
        return int(widget.get_allocated_width() or 0), int(widget.get_allocated_height() or 0)

    def _refresh(self) -> bool:
        if self._timer is None:
            return False
        self._points = self.sampler.snapshot()
        latest = self._latest()
        self.cpu_label.set_text(cpu_caption(latest))
        self.ram_label.set_text(ram_caption(latest))
        self.gpu_label.set_text(gpu_caption(latest))
        self.vram_label.set_text(vram_caption(latest))
        labels = tuple(disk.label for disk in latest.disks) if latest and latest.disks else ()
        if labels != self._disk_labels:
            self._rebuild_disks(labels)
        by_label = {disk.label: disk for disk in (latest.disks if latest else ())}
        self._disk_series = {
            name: (reads, writes) for name, reads, writes in disk_series(self._points)
        }
        for name, (caption, area) in self._disk_rows.items():
            disk = by_label.get(name)
            if disk is not None:
                caption.set_text(disk_line(disk))
            area.queue_draw()
        for area in (self.cpu_area, self.ram_area, self.gpu_area, self.vram_area):
            area.queue_draw()
        return True
