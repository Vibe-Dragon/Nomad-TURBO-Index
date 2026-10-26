"""App icons. The desktop icon stays put. The tray swaps color for indexing and repair."""

from __future__ import annotations

from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"
KINDS = {
    "idle": "turbo-idle.png",
    "indexing": "turbo-index.png",
    "repair": "turbo-repair.png",
}


def icon_bytes(kind: str) -> bytes:
    """PNG bytes for idle, indexing, or repair."""
    path = ASSETS / KINDS[kind]
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG"):
        raise RuntimeError(f"{path.name} is not a png")
    return data


def write_icons(directory: Path) -> dict[str, Path]:
    """Write the three tray icons. The old on/off names stay as the app icon."""
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for kind, name in KINDS.items():
        dest = directory / name
        dest.write_bytes(icon_bytes(kind))
        written[kind] = dest
    fire = icon_bytes("indexing")
    for legacy in ("turbo-on.png", "turbo-off.png"):
        (directory / legacy).write_bytes(fire)
    return written
