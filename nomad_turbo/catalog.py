"""What the library contains, before turbo changes anything."""

from __future__ import annotations

from dataclasses import dataclass


ZIM_COLOR = "#1d4e89"
ZIM_DONE_COLOR = "#7f93a8"
UPLOAD_COLOR = "#b45309"
UPLOAD_DONE_COLOR = "#c4a574"

# Typed by a person into the delete dialog. Nothing calls a wipe without this.
WIPE_PHRASE = "DELETE"


@dataclass
class CatalogRow:
    source: str
    name: str
    origin: str
    kind: str
    size_bytes: int
    state: str
    articles: int = 0
    total_articles: int | None = None
    chunks: int = 0


def origin_for(source: str) -> str:
    """User uploads live under kb_uploads. Everything else in the library is a ZIM."""
    path = source.replace("\\", "/")
    if "/kb_uploads/" in f"/{path}/" or path.endswith("/kb_uploads"):
        return "upload"
    return "zim"


def catalog_state(
    *,
    ingest_state: str | None,
    resume_articles: int,
    checkpoint_done: bool,
    has_continuation: bool,
) -> str:
    """indexed, partial, or pending. A continuation means the file is not finished."""
    if checkpoint_done and not has_continuation:
        return "indexed"
    if int(resume_articles or 0) > 0 or has_continuation:
        return "partial"
    if ingest_state == "indexed":
        return "indexed"
    return "pending"


def row_color(origin: str, state: str) -> str:
    done = state == "indexed"
    if origin == "upload":
        return UPLOAD_DONE_COLOR if done else UPLOAD_COLOR
    return ZIM_DONE_COLOR if done else ZIM_COLOR


def status_word(state: str) -> str:
    return {
        "indexed": "indexed",
        "partial": "partial",
        "pending": "waiting",
        "running": "indexing",
        "deferred": "needs NOMAD",
        "failed": "failed",
    }.get(state, state)


def wipe_phrase_ok(text: str) -> bool:
    return text.strip() == WIPE_PHRASE


def wipe_allowed(requested: bool, confirmed: bool) -> bool:
    """Both the button and the typed phrase are required."""
    return bool(requested) and bool(confirmed)
