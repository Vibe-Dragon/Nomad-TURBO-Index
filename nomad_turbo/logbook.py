"""Session log files and the text for the log-help window.

Each window start keeps the previous session and drops anything older.
The checkboxes do not change what these files record.
"""

from __future__ import annotations

import os
from pathlib import Path

CURRENT_NAME = "session.log"
PREVIOUS_NAME = "session-previous.log"
CONSOLE_NAME = "turbo.log"


def prepare_session_logs(directory: Path) -> tuple[Path, Path]:
    """Make this start a new session log and keep only the one before it.

    Returns the new session path and the previous session path. The console
    log is cleared so it does not grow across sessions. An open append handle
    writes the next line at the start of that file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    current = directory / CURRENT_NAME
    previous = directory / PREVIOUS_NAME
    if current.exists() and current.stat().st_size > 0:
        os.replace(current, previous)
    elif current.exists():
        current.unlink()
    for path in directory.glob("session*.log"):
        if path.name not in {CURRENT_NAME, PREVIOUS_NAME}:
            path.unlink(missing_ok=True)
    console = directory / CONSOLE_NAME
    if console.exists():
        try:
            os.truncate(console, 0)
        except OSError:
            pass
    return current, previous


def log_help_text() -> str:
    """What the lines Turbo writes on purpose mean. An unforeseen error is not in this list."""
    sections = (
        (
            "Where a line goes",
            "Errors, Info, and Debug only change which lines the pane draws. "
            "The log file still records every line, including the ones those boxes hide. "
            "Follow only scrolls the pane.",
        ),
        (
            "Sessions",
            "Opening the window starts a new session log. The log that was already there becomes the last session. "
            "Older session logs are deleted. Export log asks whether to save the current session or that last one. "
            "If the window crashed, open it again and export the last session. That file is the run that stopped.",
        ),
        (
            "Optimization status",
            "These are one sentence, written when the state changes. "
            "\"Optimize: not running.\" means Qdrant is not building a segment. "
            "\"Optimize: a segment build is open. No percent reported.\" means a build is already going, "
            "Qdrant has not given a percent, and this check did not start another one. "
            "\"Optimize: collection is yellow. No build is open.\" means some segments are not finished and nothing was sent. "
            "\"Optimize: waiting for an update. Nothing was sent.\" means Qdrant is waiting and this check did not send the update. "
            "\"Optimize: status unknown. Nothing was changed.\" means Qdrant did not say. "
            "An optimizer error is a warning and keeps Qdrant's full message.",
        ),
        (
            "Optimize stuck segments",
            "Pressing the button writes \"Optimize stuck segments: checking.\" "
            "Then one of refused, started, failed, or not started. "
            "A refusal or a failure keeps the reason: already in progress, segment files could not be read, "
            "nothing is stuck, not enough free space, or not enough spare memory. The collection is left unchanged. "
            "The same reason is on the Repair status line. The bar's tip still has the longer wording.",
        ),
        (
            "A pause while indexing",
            "\"Indexing is paused so a segment can be optimized.\" A plain segment is past the 4 GiB vector cap, "
            "or free space is under twice the segment that would be copied. New points wait. Points already stored stay. "
            "\"Knowledge-base writes resumed.\" means that pause ended. "
            "\"A ZIM download is in progress\" and \"The disk is busy\" are the other reasons new points wait.",
        ),
        (
            "Embed batch",
            "\"<n> chunks per call at <percent>%\" is the size of the next embed call. "
            "\"embed batch:\" is Auto, or the recommendation, saying why the slider moved. "
            "\"one embed server is up\" means the server started. One server stays up. "
            "\"embedder not ready yet\" means the model is still loading. "
            "\"embedder call failed\" means that call will be tried again.",
        ),
        (
            "Indexing",
            "\"indexing <file> from article\" is the file being written. "
            "\"skipping <file>, points already stored\" means that file was already in the index. "
            "\"qdrant write timed out. Trying that batch again.\" means the write is retried, and the line includes the error. "
            "\"qdrant is not taking writes. Staying on <file>\" means Turbo is waiting on that article. "
            "\"Qdrant problem while indexing\" is logged once when a run hits a problem. "
            "\"Qdrant problem cleared. The collection is answering again.\" means it recovered. "
            "A yellow collection by itself is not that problem.",
        ),
        (
            "Repair",
            "\"scan found\" counts files that would be deleted and notes that would not. "
            "\"repair deleting points\" and \"repaired\" are one file each. "
            "\"repair could not delete points\" and \"repair failed\" keep the error. "
            "\"a repair delete may still be running\" means the embed queue stays paused.",
        ),
        (
            "Errors that are not listed here",
            "A warning or a traceback from something this list does not name is the program reporting what it hit. "
            "The line and the traceback are the detail. This window cannot explain those in advance.",
        ),
    )
    blocks = []
    for title, body in sections:
        blocks.append(f"{title}\n{body}")
    return "\n\n".join(blocks)
