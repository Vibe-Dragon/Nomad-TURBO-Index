"""Embed queue and OCR queue for the Queue window.

The indexing loop asks `pick_next` for the next file. This module only
decides which rows that window shows, and how a drag on one side rewrites
the waiting list. It does not start a read or an embed.
"""

from __future__ import annotations

import json

from nomad_turbo.decisions import file_is_embeddable
from nomad_turbo.discover import state_dir
from nomad_turbo.schedule import (
    choose_embed_source,
    files_for_list_order,
    ordered_files,
    rank_embed_waiting,
)


def apply_manual(default: list[str], manual: list[str] | None) -> list[str]:
    """Put `manual` first. Sources it does not name keep their order at the end.

    An empty manual leaves `default` alone. Running files are not dropped
    here; the caller leaves those out of both lists.
    """
    current = [source for source in default if source]
    if not manual:
        return current
    rank: dict[str, int] = {}
    for index, source in enumerate(manual):
        if source and source not in rank:
            rank[source] = index
    named = [source for source in current if source in rank]
    named.sort(key=rank.__getitem__)
    rest = [source for source in current if source not in rank]
    return named + rest


def locked_moved(previous: list[str], current: list[str], locked: set[str]) -> bool:
    """True when a file that is already running changed rows."""
    if not locked:
        return False
    old = {source: index for index, source in enumerate(previous)}
    new = {source: index for index, source in enumerate(current)}
    return any(old.get(source) != new.get(source) for source in locked)


def waiting_order(sources: list[str], running: set[str], manual: list[str] | None) -> list[str]:
    """Waiting sources, in the saved order. A file that is running is skipped."""
    waiting = [source for source in sources if source and source not in running]
    return apply_manual(waiting, manual)


def rewrite_side(order: list[str], side: set[str] | list[str], new_side: list[str]) -> list[str]:
    """Permute one side and leave every other source in its slot.

    `[zimA, pdf1, zimB, pdf2, md]` with the PDFs swapped is
    `[zimA, pdf2, zimB, pdf1, md]`. Sources on that side that the drag did
    not name keep their relative order in the remaining slots.
    """
    side_set = {source for source in side if source}
    positions = [index for index, source in enumerate(order) if source in side_set]
    ranked: list[str] = []
    seen: set[str] = set()
    for source in new_side:
        if source in side_set and source not in seen:
            ranked.append(source)
            seen.add(source)
    rest = [source for source in order if source in side_set and source not in seen]
    filled = ranked + rest
    if len(filled) == len(positions):
        result = list(order)
        for pos, source in zip(positions, filled):
            result[pos] = source
        return result
    result = []
    cursor = 0
    for source in order:
        if source in side_set:
            if cursor < len(filled):
                result.append(filled[cursor])
                cursor += 1
        else:
            result.append(source)
    result.extend(filled[cursor:])
    return result


# The Queue window paints this many rows per column. The saved order is the full list.
QUEUE_SHOW = 200

_MISSING_TITLE = frozenset({"", "null", "none", "undefined"})


def display_title(title: str, *fallbacks: str) -> str:
    """Name to show. A ZIM entry titled null is the file name."""
    raw = str(title or "").strip()
    if raw.casefold() not in _MISSING_TITLE:
        return raw
    for candidate in fallbacks:
        text = str(candidate or "").replace("\\", "/").strip()
        leaf = text.replace("\n", "/").rsplit("/", 1)[-1].strip()
        if leaf.casefold() not in _MISSING_TITLE:
            return leaf
    return raw or "pdf"


def shown_slots(slots, limit: int = QUEUE_SHOW):
    """The in-progress row and the files after it, capped for the window.

    The running row is already first. A drag of this slice is merged back
    into the full waiting list, so the rows past the cap stay in order.
    """
    rows = tuple(slots)
    if len(rows) <= limit:
        return rows
    return rows[:limit]


def queue_caption(title: str, shown: int, total: int) -> str:
    """Column heading. The count appears when some of the queue is not listed."""
    if total > shown:
        return f"{title} · {shown} of {total}"
    return title


def embedding_file_view(job, *, status: str = "text", chunks: int = 0) -> dict | None:
    """The document the GPU is storing when the row above the tree is not that file.

    A document inside a ZIM also has a row under Internal Zim Documents.
    This pinned row is the one being read or stored. An upload is already
    the library row above the tree, so this stays empty for an upload.
    """
    source = str(getattr(job, "source", "") or "")
    if "\n" not in source:
        return None
    article = str(getattr(job, "article_path", "") or "")
    title = display_title(str(getattr(job, "title", "") or ""), article, source)
    return {
        "source": source,
        "title": title,
        "article_path": article,
        "kind": str(getattr(job, "kind", "") or "") or "pdf",
        "status": status or "text",
        "chunks": max(0, int(chunks or 0)),
        "articles": 0,
        "total_articles": None,
    }


def _stored_face(embedding):
    """The pinned row for a document that is being embedded. None when unset."""
    if not isinstance(embedding, dict) or not embedding.get("source"):
        return None
    item = _face_for(embedding, locked=True)
    item.status = str(embedding.get("status") or "text")
    item.state = "running"
    chunks = max(0, int(embedding.get("chunks") or 0))
    item.chunks = chunks
    item.chunks_text = f"{chunks:,}" if chunks else "—"
    return item


def pinned_files(running, lane, embedding=None) -> list:
    """Rows above the folder tree.

    The file the GPU is storing stays first. That is a document inside a ZIM
    when `embedding` names one. The document OCR is reading stays second,
    including a PDF inside a ZIM. A library file that is already that read
    is not listed twice. Library order must not put the read in the embedding row.
    """
    rows = [item for item in (running or []) if getattr(item, "state", "") == "running"]
    embedding_row = next((item for item in rows if running_column(item) == "embed"), None)
    reading_row = next((item for item in rows if running_column(item) == "ocr"), None)
    lead = embedding_row or reading_row or (rows[0] if rows else None)
    stored = _stored_face(embedding)
    if stored is not None and not any(getattr(item, "source", "") == stored.source for item in rows):
        lead = stored
    reading = lane.get("reading") if isinstance(lane, dict) else None
    source = str(reading.get("source") or "") if isinstance(reading, dict) else ""
    pinned = [lead] if lead is not None else []
    if not source:
        for item in rows:
            if lead is None or getattr(item, "source", "") != getattr(lead, "source", ""):
                pinned.append(item)
                break
        return pinned[:2]
    if any(getattr(item, "source", "") == source for item in pinned):
        return pinned[:2]
    match = next((item for item in rows if getattr(item, "source", "") == source), None)
    second = match if match is not None else _face_for(reading, locked=True)
    if not pinned or getattr(pinned[0], "source", "") == getattr(second, "source", ""):
        return [second]
    return [pinned[0], second]


def is_visual_name(name: str, file_type: str = "") -> bool:
    """True for a PDF or a picture. Those wait on the OCR side."""
    kind = file_is_embeddable(name or "")
    if kind in ("pdf", "image"):
        return True
    return (file_type or "") == "pdf"


def running_column(item) -> str | None:
    """Which column holds a file that is already being processed.

    Recognizing a page is the OCR column. Storing text, including text that
    was just recognized, is the embedding column.
    """
    if getattr(item, "state", "") != "running":
        return None
    status = str(getattr(item, "status", "") or "").casefold()
    if status.startswith("ocr") and "text" not in status:
        return "ocr"
    return "embed"


def queue_path():
    return state_dir() / "queue.json"


def _names(value) -> list[str] | None:
    if not isinstance(value, list):
        return None
    found: list[str] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, str) and item and item not in seen:
            found.append(item)
            seen.add(item)
    return found or None


def load_queues() -> tuple[list[str] | None, list[str] | None]:
    """Saved waiting lists. Missing or empty lists are None."""
    path = queue_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    return _names(data.get("embed")), _names(data.get("ocr"))


def save_queues(embed: list[str] | None, ocr: list[str] | None) -> None:
    """Remember both waiting lists. Clearing both removes the file."""
    path = queue_path()
    payload = {
        "embed": list(embed) if embed else None,
        "ocr": list(ocr) if ocr else None,
    }
    if payload["embed"] is None and payload["ocr"] is None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class QueueFace:
    """A queue row that is not its own library file, such as a PDF inside a ZIM."""

    __slots__ = (
        "source",
        "name",
        "origin",
        "status",
        "percent",
        "count",
        "eta",
        "color",
        "state",
        "chunks",
        "chunks_text",
        "size_bytes",
        "size_text",
        "articles",
        "articles_per_minute",
        "chunks_per_second",
        "rate_text",
        "eta_seconds",
        "file_type",
        "art_text",
        "chk_text",
        "total_articles",
        "mtime",
        "walk_index",
    )

    def __init__(self, source: str, name: str, *, status: str, state: str, file_type: str, articles: int = 0, total_articles: int | None = None, chunks: int = 0):
        self.source = source
        self.name = name
        self.origin = "Upload"
        self.status = status
        self.state = state
        self.file_type = file_type or "pdf"
        self.articles = int(articles or 0)
        self.total_articles = int(total_articles) if total_articles else None
        self.chunks = int(chunks or 0)
        self.chunks_text = f"{self.chunks:,}" if self.chunks else "—"
        self.size_bytes = 0
        self.size_text = "—"
        self.articles_per_minute = None
        self.chunks_per_second = None
        self.rate_text = "—"
        self.art_text = "—"
        self.chk_text = "—"
        self.eta = ""
        self.eta_seconds = None
        self.color = ""
        self.mtime = None
        self.walk_index = 0
        if self.total_articles:
            self.percent = int(round(100 * self.articles / self.total_articles))
            self.count = f"{self.articles:,} / {self.total_articles:,}"
        elif self.articles:
            self.percent = 0
            self.count = f"{self.articles:,}"
        else:
            self.percent = 0
            self.count = "—"


class QueueSlot:
    __slots__ = ("source", "locked", "item")

    def __init__(self, source: str, locked: bool, item) -> None:
        self.source = source
        self.locked = locked
        self.item = item


class QueueBoards:
    """The two columns, plus the waiting lists a drag rewrites."""

    __slots__ = (
        "embed",
        "ocr",
        "sequential",
        "embed_waiting",
        "ocr_waiting",
        "parallel",
        "visual",
    )

    def __init__(
        self,
        embed: list[QueueSlot],
        ocr: list[QueueSlot],
        sequential: list[str],
        *,
        parallel: bool,
        visual: list[str],
    ) -> None:
        self.embed = tuple(embed)
        self.ocr = tuple(ocr)
        self.sequential = tuple(sequential)
        self.embed_waiting = tuple(slot.source for slot in embed if not slot.locked)
        self.ocr_waiting = tuple(slot.source for slot in ocr if not slot.locked)
        self.parallel = parallel
        self.visual = tuple(visual)


def _open_items(files) -> list:
    opened = []
    for item in files or []:
        state = getattr(item, "state", "")
        if state in ("indexed", "deferred"):
            continue
        if not getattr(item, "source", ""):
            continue
        opened.append(item)
    return opened


def _rank_waiting(items, choice: str, priorities: set[str], dates: dict | None) -> list:
    rows = files_for_list_order(items, priorities=priorities, dates=dates)
    ordered = ordered_files(rows, choice)
    first = [item for item in ordered if item.priority]
    rest = [item for item in ordered if not item.priority]
    return first + rest


def _by_source(files) -> dict[str, object]:
    return {item.source: item for item in files if getattr(item, "source", "")}


def _slot(source: str, locked: bool, item) -> QueueSlot:
    return QueueSlot(source, locked, item)


def _face_for(job: dict, *, locked: bool):
    source = str(job.get("source") or "")
    article = str(job.get("article_path") or "")
    title = display_title(str(job.get("title") or ""), article, source)
    kind = str(job.get("kind") or "")
    if kind not in ("pdf", "image"):
        kind = "pdf" if is_visual_name(title, kind) or not kind else kind
    total = job.get("total_articles")
    return QueueFace(
        source,
        title,
        status="OCR" if locked else "Waiting",
        state="running" if locked else "pending",
        file_type="pdf" if kind == "pdf" else (kind or "pdf"),
        articles=int(job.get("articles") or 0),
        total_articles=int(total) if total else None,
        chunks=int(job.get("chunks") or 0),
    )


def _item_for(source: str, snaps: dict[str, object], job: dict | None, *, locked: bool):
    found = snaps.get(source)
    if found is not None:
        return found
    if job is not None:
        return _face_for(job, locked=locked)
    return QueueFace(source, source, status="OCR" if locked else "Waiting", state="running" if locked else "pending", file_type="pdf")


def _jobs(lane: dict, key: str) -> list[dict]:
    rows = lane.get(key) or []
    if isinstance(rows, dict):
        return [rows]
    found = []
    for row in rows:
        if isinstance(row, dict) and row.get("source"):
            found.append(row)
    return found


def _with_embedding(boards: QueueBoards, embedding) -> QueueBoards:
    """Keep the document being stored at the top of the embed column."""
    item = _stored_face(embedding)
    if item is None:
        return boards
    embed = [slot for slot in boards.embed if slot.source != item.source]
    embed.insert(0, _slot(item.source, True, item))
    boards.embed = tuple(embed)
    boards.embed_waiting = tuple(source for source in boards.embed_waiting if source != item.source)
    return boards


def build_queue(
    files,
    *,
    choice: str,
    priorities: set[str] | None = None,
    dates: dict | None = None,
    embed_order: list[str] | None = None,
    ocr_order: list[str] | None = None,
    lane: dict | None = None,
    embedding=None,
) -> QueueBoards:
    """Left column, right column, and the waiting sequence a drag rewrites.

    With no OCR lane, both columns are one sequence: PDFs and pictures on
    the right, everything else on the left. With a lane, the right column is
    that lane, including a PDF inside a ZIM. On the left, prioritized files
    come first, text that has finished OCR comes next, and the selected
    order follows. A saved queue still comes before that.
    """
    marks = set(priorities or ())
    opened = _open_items(files)
    snaps = _by_source(opened)
    if lane is not None:
        boards = _parallel_boards(opened, snaps, choice, marks, dates, embed_order, ocr_order, lane)
    else:
        boards = _sequential_boards(opened, snaps, choice, marks, dates, embed_order, ocr_order)
    return _with_embedding(boards, embedding)


def _sequential_boards(opened, snaps, choice, priorities, dates, embed_order, ocr_order) -> QueueBoards:
    running = [item for item in opened if running_column(item)]
    waiting = [item for item in opened if running_column(item) is None]
    ranked = _rank_waiting(waiting, choice, priorities, dates)
    default = [item.source for item in ranked]
    visual_ids = [source for source in default if is_visual_name(_name_of(snaps.get(source)), _type_of(snaps.get(source)))]
    if embed_order:
        sequence = apply_manual(default, embed_order)
    elif ocr_order:
        sequence = rewrite_side(default, visual_ids, apply_manual(visual_ids, ocr_order))
    else:
        sequence = default
    embed: list[QueueSlot] = []
    ocr: list[QueueSlot] = []
    for item in running:
        slot = _slot(item.source, True, item)
        if running_column(item) == "ocr":
            ocr.append(slot)
        else:
            embed.append(slot)
    visual_waiting = []
    for source in sequence:
        item = snaps.get(source)
        if item is None:
            continue
        slot = _slot(source, False, item)
        if is_visual_name(getattr(item, "name", ""), getattr(item, "file_type", "")):
            ocr.append(slot)
            visual_waiting.append(source)
        else:
            embed.append(slot)
    return QueueBoards(embed, ocr, sequence, parallel=False, visual=visual_waiting)


def _name_of(item) -> str:
    return str(getattr(item, "name", "") or "")


def _type_of(item) -> str:
    return str(getattr(item, "file_type", "") or "")


def _parallel_boards(opened, snaps, choice, priorities, dates, embed_order, ocr_order, lane: dict) -> QueueBoards:
    reading = lane.get("reading") if isinstance(lane.get("reading"), dict) else None
    queued = _jobs(lane, "queued")
    ready = _jobs(lane, "ready")
    reading_source = str(reading.get("source") or "") if reading else ""
    queued_ids = [str(job["source"]) for job in queued if str(job["source"]) != reading_source]
    ready_ids = [str(job["source"]) for job in ready if str(job["source"]) not in {reading_source, *queued_ids}]
    lane_ids = {reading_source, *queued_ids, *ready_ids}

    running_embed = []
    for item in opened:
        column = running_column(item)
        if column == "embed":
            running_embed.append(item)
        elif column == "ocr" and not reading_source:
            reading_source = item.source
            reading = {"source": item.source, "title": getattr(item, "name", "")}

    ready_set = set(ready_ids)
    embeddable = []
    for item in opened:
        if running_column(item):
            continue
        if item.source in lane_ids and item.source not in ready_set:
            continue
        visual = is_visual_name(getattr(item, "name", ""), getattr(item, "file_type", ""))
        if visual and item.source not in ready_set:
            continue
        embeddable.append(item)
    program = [item.source for item in _rank_waiting(embeddable, choice, priorities, dates)]
    embed_seen = {item.source for item in running_embed}
    embed_default = []
    for source in rank_embed_waiting(program, ready_ids, priorities):
        if source in embed_seen:
            continue
        embed_seen.add(source)
        embed_default.append(source)
    embed_waiting = apply_manual(embed_default, embed_order)
    ocr_waiting = apply_manual(queued_ids, ocr_order)

    by_job = {str(job["source"]): job for job in queued}
    by_job.update({str(job["source"]): job for job in ready})
    if reading:
        by_job[str(reading.get("source") or "")] = reading

    embed = [_slot(item.source, True, item) for item in running_embed]
    embed_placed = {slot.source for slot in embed}
    for source in embed_waiting:
        if source in embed_placed:
            continue
        embed_placed.add(source)
        embed.append(_slot(source, False, _item_for(source, snaps, by_job.get(source), locked=False)))

    ocr: list[QueueSlot] = []
    if reading_source:
        ocr.append(_slot(reading_source, True, _item_for(reading_source, snaps, by_job.get(reading_source), locked=True)))
    for source in ocr_waiting:
        if source == reading_source:
            continue
        ocr.append(_slot(source, False, _item_for(source, snaps, by_job.get(source), locked=False)))
    return QueueBoards(embed, ocr, [], parallel=True, visual=list(ocr_waiting))
