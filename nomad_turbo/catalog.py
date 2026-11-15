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
    mtime: float | None = None
    walk_index: int = 0
    # Pages are stored and PDFs from this ZIM are still queued. The stored
    # state stays partial, so Hide finished does not drop the row.
    waiting_ocr: bool = False


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


def waiting_on_ocr(state: str, *, kind: str = "", phase: str = "", done: bool = False) -> bool:
    """A ZIM whose pages are stored while its PDFs are still queued.

    The file list keeps the stored state. This only changes the status word.
    A file still being read, and one that failed, keep their own words.
    """
    if done or state in ("running", "indexed", "failed", "deferred", "pending"):
        return False
    if kind != "zim":
        return False
    return phase in ("queued", "ocr")


def status_word(state: str, *, waiting_ocr: bool = False) -> str:
    if waiting_ocr and state not in ("running", "indexed", "failed", "deferred"):
        return "Waiting OCR"
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
