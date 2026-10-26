"""Older share dialog. The Index window slider is the control that ships.

Cancel leaves turbo off. This dialog stops at 100%. The window slider is
what offers the 110% overclock. Neither one starts indexing by itself.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable


def display_available() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def ask_gpu_share(default_percent: int, describe: Callable[[int], str]) -> tuple[str, int | None]:
    """Return (status, percent).

    status is ok, cancel, unavailable, or error.
    unavailable means there is no display, so the caller keeps the saved share.
    """
    if not display_available():
        return "unavailable", None
    try:
        return _ask_gtk(default_percent, describe)
    except Exception:
        try:
            return _ask_zenity(default_percent, describe(default_percent))
        except Exception:
            return "error", None


def _clamp_percent(percent: float) -> int:
    return min(100, max(10, int(round(float(percent)))))


def _ask_gtk(default_percent: int, describe: Callable[[int], str]) -> tuple[str, int | None]:
    import gi

    gi.require_version("Gtk", "3.0")
    from gi.repository import Gtk

    if not Gtk.init_check():
        return "error", None
    dialog = Gtk.Dialog(title="NOMAD Turbo")
    dialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
    dialog.add_button("Start indexing", Gtk.ResponseType.OK)
    dialog.set_default_response(Gtk.ResponseType.OK)
    dialog.set_default_size(480, -1)
    dialog.set_position(Gtk.WindowPosition.CENTER)
    dialog.set_keep_above(True)
    box = dialog.get_content_area()
    box.set_border_width(12)
    box.set_spacing(10)

    title = Gtk.Label(label="How much of this GPU should indexing use?")
    title.set_xalign(0)
    title.set_line_wrap(True)
    hint = Gtk.Label(
        label=(
            "100% is the whole card except a small overhead, so the GPU does not "
            "run out of memory. Use that only when NOMAD chat will sit idle. "
            "A lower number leaves the rest free. The process count is a guess "
            "until the first model loads, then it is measured to fit this share. "
            "Cancel leaves turbo off."
        )
    )
    hint.set_xalign(0)
    hint.set_line_wrap(True)
    hint.set_max_width_chars(52)
    scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 10, 100, 5)
    scale.set_value(_clamp_percent(default_percent))
    scale.set_hexpand(True)
    scale.set_draw_value(True)
    scale.set_value_pos(Gtk.PositionType.RIGHT)
    scale.connect("format-value", lambda _scale, value: f"{int(round(value))}%")
    for mark in (10, 50, 100):
        scale.add_mark(mark, Gtk.PositionType.BOTTOM, f"{mark}%")
    detail = Gtk.Label()
    detail.set_xalign(0)
    detail.set_line_wrap(True)
    detail.set_max_width_chars(52)

    def refresh(*_args):
        percent = _clamp_percent(scale.get_value())
        try:
            detail.set_text(describe(percent))
        except Exception as exc:
            detail.set_text(str(exc))

    scale.connect("value-changed", refresh)
    refresh()
    for widget in (title, hint, scale, detail):
        box.add(widget)
    dialog.show_all()
    response = dialog.run()
    percent = _clamp_percent(scale.get_value())
    dialog.destroy()
    if response != Gtk.ResponseType.OK:
        return "cancel", None
    return "ok", percent


def _ask_zenity(default_percent: int, preview: str) -> tuple[str, int | None]:
    zenity = shutil.which("zenity")
    if not zenity:
        return "error", None
    text = "How much of this GPU should indexing use?\n\n" + preview + "\n\nCancel leaves turbo off."
    result = subprocess.run(
        [
            zenity,
            "--scale",
            "--title=NOMAD Turbo",
            "--text",
            text,
            "--min-value=10",
            "--max-value=100",
            f"--value={_clamp_percent(default_percent)}",
            "--step=5",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return "cancel", None
    try:
        return "ok", _clamp_percent(int(result.stdout.strip() or default_percent))
    except ValueError:
        return "error", None
