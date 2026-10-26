"""One desktop shortcut. The picture stays put. The tray shows the run state."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from nomad_turbo.icons import write_icons


def desktop_dir() -> Path:
    try:
        out = subprocess.check_output(["xdg-user-dir", "DESKTOP"], text=True, stderr=subprocess.DEVNULL).strip()
        if out:
            return Path(out)
    except (OSError, subprocess.CalledProcessError):
        pass
    return Path.home() / "Desktop"


def shortcut_path() -> Path:
    return desktop_dir() / "NOMAD Turbo.desktop"


def applications_dir() -> Path:
    return Path.home() / ".local" / "share" / "applications"


def desktop_text(icon: Path, command: str) -> str:
    """Launcher text. StartupWMClass matches the window so the taskbar can show the icon."""
    name = "Nomad TURBO Index"
    comment = "Open the indexing window. This icon does not start or stop indexing."
    return (
        "\n".join(
            [
                "[Desktop Entry]",
                "Version=1.0",
                "Type=Application",
                "Name=" + name,
                "Comment=" + comment,
                f"Exec={command}",
                "Icon=" + str(icon),
                "Terminal=false",
                "StartupNotify=false",
                "StartupWMClass=nomad-turbo",
                "Categories=Education;",
            ]
        )
        + "\n"
    )


def icon_dir() -> Path:
    path = Path.home() / ".local" / "share" / "nomad-turbo" / "icons"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _icon() -> Path:
    """The app icon. Indexing and repair change the tray, not this file."""
    return write_icons(icon_dir())["indexing"]


def installed_command(bin_path: Path, source_path: Path) -> Path:
    """Command a desktop entry should run.

    The Debian package installs /usr/bin/nomad-turbo. A source checkout uses
    the launcher next to this package. That fallback is the only time the
    command lives beside the source tree.
    """
    if bin_path.is_file():
        return bin_path
    return source_path


def wrapper_path() -> Path:
    return installed_command(
        Path("/usr/bin/nomad-turbo"),
        Path(__file__).resolve().parent.parent / "nomad-turbo",
    )


def _desktop_env() -> dict[str, str]:
    """Cinnamon reads the icon through the session bus, not only the file."""
    env = os.environ.copy()
    if not env.get("DBUS_SESSION_BUS_ADDRESS"):
        bus = Path(f"/run/user/{os.getuid()}/bus")
        if bus.exists():
            env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus}"
    if not env.get("DISPLAY"):
        env["DISPLAY"] = ":0"
    return env


def write_shortcut(active: bool = False) -> Path:
    """Write the launcher. The flag is ignored so older callers cannot turn it into a stop button."""
    del active
    icon = _icon()
    path = shortcut_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # A second file with the visible name would sit beside this one. Cinnamon shows both.
    stale = path.parent / "Nomad TURBO Index.desktop"
    if stale != path and stale.exists():
        stale.unlink()
    body = desktop_text(icon, f"{wrapper_path()} toggle")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(body)
    temporary.chmod(0o755)
    os.replace(temporary, path)
    # Nemo-desktop keeps painting the old icon across an atomic replace.
    # A second write hits the file it is already watching.
    path.write_text(body)
    if shutil.which("gio"):
        subprocess.run(
            ["gio", "set", "-t", "string", str(path), "metadata::custom-icon", icon.as_uri()],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_desktop_env(),
        )
    app_dir = applications_dir()
    app_dir.mkdir(parents=True, exist_ok=True)
    app_path = app_dir / "nomad-turbo.desktop"
    app_path.write_text(body)
    app_path.chmod(0o755)
    if shutil.which("update-desktop-database"):
        subprocess.run(
            ["update-desktop-database", str(app_dir)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return path


def install_autostart(command: str) -> None:
    folder = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "autostart"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "nomad-turbo-recover.desktop"
    path.write_text(
        "\n".join(
            [
                "[Desktop Entry]",
                "Type=Application",
                "Name=NOMAD Turbo recover",
                "Comment=If turbo died, hand embedding back to NOMAD",
                f"Exec={command} recover",
                "Terminal=false",
                "X-GNOME-Autostart-enabled=true",
                "",
            ]
        )
    )
