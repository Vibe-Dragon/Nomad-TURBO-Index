"""Color for the two lines above the pages.

Green is ready: indexing, or idle with Qdrant answering.
Blue is a pause that is still working, such as saving records after Stop.
Amber means leave Qdrant running.
Red needs a person.
"""

from __future__ import annotations

READY = "ready"
WORKING = "working"
WATCH = "watch"
ALERT = "alert"

COLORS = {
    READY: "#1f7a3a",
    WORKING: "#1d4e89",
    WATCH: "#b86e00",
    ALERT: "#a32020",
}

STOP_HANDOFF_NOTE = (
    "Stopping. Updating the knowledge-base records. "
    "The file list stays still until that finishes."
)


def markup(text: str, level: str) -> str:
    """One colored line. The plain sentence stays readable by a screen reader."""
    safe = (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    color = COLORS.get(level, COLORS[READY])
    return f'<span foreground="{color}">{safe}</span>'


def qdrant_level(
    text: str,
    *,
    ready: bool,
    collection_status: str = "",
    optimizer_error: str = "",
    collection_missing: bool = False,
    unreadable: bool = False,
) -> str:
    """How much attention the Qdrant line needs."""
    if unreadable:
        return ALERT
    error = (optimizer_error or "").strip()
    if error and error.lower() != "ok":
        return ALERT
    if (collection_status or "").strip().lower() == "red":
        return ALERT
    first = (text or "").split("\n", 1)[0].strip().lower()
    if "could not be read" in first or "health check" in first:
        return ALERT
    if not ready and any(word in first for word in ("exited", "missing", "dead", "restarting")):
        return ALERT
    if collection_missing or "not there" in first:
        return WATCH
    if not ready:
        return WATCH
    if (collection_status or "").strip().lower() in ("yellow", "grey"):
        return WATCH
    return READY


def note_level(note: str) -> str | None:
    """The attention level of a status sentence. None when it is ordinary progress."""
    text = (note or "").strip().lower()
    if not text:
        return None
    if text.startswith("stopping.") or "knowledge-base records" in text:
        return WORKING
    if "do not restart" in text or "still writing" in text or "disk is busy" in text:
        return WATCH
    if "paused so a segment" in text or "new points are waiting" in text:
        return WATCH
    if (
        text.startswith("waiting for")
        or "download is in progress" in text
        or "just finished" in text
        or "staying on" in text
        or "queue is empty" in text
    ):
        return WORKING
    if text.startswith("deleting") or text.startswith("delete cancelled"):
        return WORKING
    if text.startswith("stopped."):
        return WORKING
    if text.startswith("finished.") or text.startswith("nothing left to index"):
        return READY
    return None


def activity_status(
    *,
    phase: str = "",
    note: str = "",
    error: str = "",
    optimize_hold: str = "",
    scanning: bool = False,
    repairing: bool = False,
    repair_note: str = "",
    qdrant_ready: bool = True,
    qdrant_attention: str = READY,
    work_alive: bool = False,
) -> tuple[str, str]:
    """The line under Qdrant, and its color, while the file list is still."""
    phase = (phase or "").strip()
    note = (note or "").strip()
    error = (error or "").strip()
    if phase == "error" or error:
        return error or note or "Turbo hit an error.", ALERT
    if scanning:
        return "Scanning the library.", WORKING
    if repairing:
        repair_note = (repair_note or "").strip()
        if "error" in repair_note.lower() or "failed" in repair_note.lower():
            return repair_note or "Repair needs attention.", ALERT
        return repair_note or "Repair is working through the list.", WORKING
    if phase == "deleting" or note.lower().startswith("deleting"):
        return note or "Deleting the knowledge index.", WORKING
    if optimize_hold:
        return note or "Indexing is paused so a segment can be optimized.", WATCH
    held = note_level(note)
    if held:
        return note, held
    if phase == "waiting":
        return note or "Waiting for NOMAD to finish its current embed batches.", WORKING
    if (
        phase in ("waiting", "running")
        and work_alive
        and not qdrant_ready
        and qdrant_attention in (WATCH, ALERT)
    ):
        return (
            "Waiting until Qdrant can take changes. The file list stays still while that write finishes.",
            qdrant_attention,
        )
    if phase == "running" and work_alive:
        return "Indexing.", READY
    if phase == "stopped":
        return note or "Stopped.", WORKING
    if phase == "done":
        return note or "Finished.", READY
    if not qdrant_ready and qdrant_attention in (WATCH, ALERT):
        return "Waiting until Qdrant can take changes.", qdrant_attention
    return "Idle.", READY
