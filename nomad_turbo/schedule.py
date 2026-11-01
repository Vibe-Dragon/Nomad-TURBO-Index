"""Index order, next-file choice, measure timing, and the file-list cells.

The Index page and the indexing loop both call these. A test should call
them too, rather than keeping a second copy of the rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from nomad_turbo.decisions import file_is_embeddable

# How long Measure stays on one file before the in-flight embed call ends it.
MEASURE_SECONDS = 300

INDEX_HEADERS = (
    "File",
    "File type",
    "Size",
    "Status",
    "Progress",
    "Articles",
    "art/min",
    "Chunks",
    "chk/s",
    "ETA",
)

# List-store columns. Sizes and counts are 64-bit so a multi-gigabyte ZIM fits.
COL_SOURCE = 0
COL_MARK = 1
COL_NAME = 2
COL_TYPE = 3
COL_SIZE = 4
COL_STATUS = 5
COL_PROGRESS = 6
COL_ARTICLES = 7
COL_ART_MIN = 8
COL_CHUNKS = 9
COL_CHK_S = 10
COL_ETA = 11
COL_COLOR = 12
COL_SIZE_SORT = 13
COL_ARTICLES_SORT = 14
COL_ART_MIN_SORT = 15
COL_CHUNKS_SORT = 16
COL_CHK_SORT = 17
COL_ETA_SORT = 18

PRIORITY_MARK = "*"
PRIORITY_COLOR = "#c0392b"
EM_DASH = "—"

# id, label. Library order is the walk the indexer already used.
ORDER_CHOICES: tuple[tuple[str, str], ...] = (
    ("library", "Library order"),
    ("size_desc", "Largest first"),
    ("size_asc", "Smallest first"),
    ("date_desc", "Newest date first"),
    ("date_asc", "Oldest date first"),
    ("articles_desc", "Most articles first"),
    ("articles_asc", "Fewest articles first"),
    ("completed_desc", "Most complete first"),
    ("completed_asc", "Least complete first"),
    ("name_asc", "Name A to Z"),
    ("name_desc", "Name Z to A"),
    ("eta_asc", "Shortest ETA first"),
    ("eta_desc", "Longest ETA first"),
    ("chunks_desc", "Most chunks first"),
    ("chunks_asc", "Fewest chunks first"),
    ("type_asc", "File type A to Z"),
    ("type_desc", "File type Z to A"),
)

ORDER_IDS = frozenset(choice for choice, _label in ORDER_CHOICES)

# Reasons that checkpoint the open file and keep its points.
SWITCH_REASONS = frozenset({"now", "priority", "measure", "switch"})


def file_type_label(name: str) -> str:
    """Extension the indexer accepts, without a dot.

    txt, md, and rtf stay distinct. An image keeps its own extension.
    The words ZIM and Upload are origins, not types.
    """
    kind = file_is_embeddable(name or "")
    if not kind:
        return ""
    lower = (name or "").lower()
    dot = lower.rfind(".")
    ext = lower[dot + 1 :] if dot >= 0 else ""
    if kind in ("image", "text"):
        return ext
    return kind


def pace_text(running: bool, value: float | None) -> str:
    """A rate cell. Only the file being indexed, and only when a rate exists."""
    if not running or not value:
        return EM_DASH
    return f"{value:,.0f}"


def priority_mark(priority: bool) -> str:
    return PRIORITY_MARK if priority else ""


def index_store_row(item, *, priority: bool) -> tuple:
    """One file-list row, in the same order as the list store."""
    art = getattr(item, "articles_per_minute", None)
    chk = getattr(item, "chunks_per_second", None)
    eta_seconds = getattr(item, "eta_seconds", None)
    from nomad_turbo.progress import UNKNOWN_ETA_SECONDS

    return (
        item.source,
        priority_mark(priority),
        item.name,
        getattr(item, "file_type", "") or file_type_label(item.name),
        item.size_text,
        item.status,
        int(item.percent),
        item.count,
        getattr(item, "art_text", None) or pace_text(item.state == "running", art),
        item.chunks_text,
        getattr(item, "chk_text", None) or pace_text(item.state == "running", chk),
        item.eta or "",
        item.color,
        int(item.size_bytes),
        int(item.articles),
        int(round(art or 0)),
        int(item.chunks),
        int(round(chk or 0)),
        UNKNOWN_ETA_SECONDS if eta_seconds is None else int(eta_seconds),
    )


def order_choice_labels() -> list[str]:
    return [label for _choice, label in ORDER_CHOICES]


def normalize_order(choice: str | None) -> str:
    if choice in ORDER_IDS:
        return str(choice)
    return "library"


class _Rev:
    """Sort descending while the following key stays ascending."""

    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __lt__(self, other: _Rev) -> bool:
        return self.value > other.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Rev) and self.value == other.value


def parse_zim_date(text: str | None) -> float | None:
    """Unix seconds for a ZIM Date. Empty or unreadable text is None.

    Accepts a year, a year and month, or a calendar day, and an ISO
    datetime whose date comes first.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    head = raw.split("T", 1)[0].split(" ", 1)[0]
    parsed = None
    if len(head) >= 10:
        parsed = _parse_date(head[:10], "%Y-%m-%d")
    if parsed is None and len(head) >= 7:
        parsed = _parse_date(head[:7], "%Y-%m")
    if parsed is None and len(head) >= 4:
        parsed = _parse_date(head[:4], "%Y")
    if parsed is None:
        return None
    return parsed.replace(tzinfo=timezone.utc).timestamp()


def _parse_date(text: str, fmt: str) -> datetime | None:
    try:
        return datetime.strptime(text, fmt)
    except ValueError:
        return None


def sort_timestamp(*, file_type: str, zim_date: str | None, mtime: float | None) -> float | None:
    """When this file was published, for ordering.

    A zim uses its archive Date. If that cannot be read, it uses the file's
    modification time, which is also what every other type uses.
    """
    if file_type == "zim":
        parsed = parse_zim_date(zim_date)
        if parsed is not None:
            return parsed
    if mtime is None:
        return None
    return float(mtime)


@dataclass(frozen=True)
class OrderFile:
    source: str
    name: str
    file_type: str
    size_bytes: int
    mtime: float | None
    zim_date: str | None
    articles_total: int | None
    completed: float
    chunks: int
    measured_eta: float | None
    state: str
    walk_index: int
    priority: bool = False


def _tie(item: OrderFile) -> tuple[str, str]:
    return (item.name.casefold(), item.source)


def ordered_files(files: list[OrderFile], choice: str) -> list[OrderFile]:
    """Unfinished or not, in the selected order. Ties break by file name."""
    key = _order_key(normalize_order(choice))
    return sorted(files, key=key)


def ordered_sources(files: list[OrderFile], choice: str) -> list[str]:
    return [item.source for item in ordered_files(files, choice)]


def files_for_list_order(items, *, priorities: set[str], dates: dict[str, str] | None = None) -> list[OrderFile]:
    """Order rows from the file list snapshot, including cached ZIM Dates."""
    known = dates or {}
    rows: list[OrderFile] = []
    for index, item in enumerate(items):
        total = getattr(item, "total_articles", None)
        articles = int(getattr(item, "articles", 0) or 0)
        state = item.state
        if state == "indexed":
            completed = 1.0
        elif total:
            completed = min(1.0, max(0.0, articles / int(total)))
        else:
            completed = 0.0
        eta = getattr(item, "eta_seconds", None)
        if state == "indexed":
            eta = None
        name = item.name
        rows.append(
            OrderFile(
                source=item.source,
                name=name,
                file_type=getattr(item, "file_type", "") or file_type_label(name),
                size_bytes=int(getattr(item, "size_bytes", 0) or 0),
                mtime=getattr(item, "mtime", None),
                zim_date=known.get(item.source) or None,
                articles_total=int(total) if total else None,
                completed=completed,
                chunks=int(getattr(item, "chunks", 0) or 0),
                measured_eta=float(eta) if eta is not None else None,
                state=state,
                walk_index=int(getattr(item, "walk_index", index) or 0),
                priority=item.source in priorities,
            )
        )
    return rows


def list_display_order(
    files: list[OrderFile],
    choice: str,
    *,
    current: str | None = None,
    hide_finished: bool = False,
) -> list[str]:
    """The file list, in the same order the indexer uses.

    The file in progress is first. Prioritized unfinished files follow, in
    the selected order, then the other unfinished files. Indexed files stay
    at the bottom unless they are hidden. A prioritized file that is already
    indexed stays with the finished files.
    """
    current_item = next(
        (item for item in files if item.source == current and item.state != "indexed"),
        None,
    )
    finished = [item for item in files if item.state == "indexed"]
    rest = [item for item in files if item.state != "indexed" and item is not current_item]
    ranked = ordered_files(rest, choice)
    ordered: list[OrderFile] = []
    if current_item is not None:
        ordered.append(current_item)
    ordered.extend(item for item in ranked if item.priority)
    ordered.extend(item for item in ranked if not item.priority)
    if not hide_finished:
        ordered.extend(ordered_files(finished, choice))
    return [item.source for item in ordered]


def _order_key(choice: str):
    if choice == "size_desc":
        return lambda item: (-item.size_bytes, _tie(item))
    if choice == "size_asc":
        return lambda item: (item.size_bytes, _tie(item))
    if choice == "date_desc":
        return lambda item: _date_key(item, newest_first=True)
    if choice == "date_asc":
        return lambda item: _date_key(item, newest_first=False)
    if choice == "articles_desc":
        return lambda item: _known_key(item.articles_total, high_first=True, item=item)
    if choice == "articles_asc":
        return lambda item: _known_key(item.articles_total, high_first=False, item=item)
    if choice == "completed_desc":
        return lambda item: (-float(item.completed), _tie(item))
    if choice == "completed_asc":
        return lambda item: (float(item.completed), _tie(item))
    if choice == "name_desc":
        return lambda item: (_Rev(item.name.casefold()), item.source)
    if choice == "name_asc":
        return lambda item: (item.name.casefold(), item.source)
    if choice == "eta_desc":
        return lambda item: _known_key(item.measured_eta, high_first=True, item=item)
    if choice == "eta_asc":
        return lambda item: _known_key(item.measured_eta, high_first=False, item=item)
    if choice == "chunks_desc":
        return lambda item: (-int(item.chunks), _tie(item))
    if choice == "chunks_asc":
        return lambda item: (int(item.chunks), _tie(item))
    if choice == "type_desc":
        return lambda item: (_Rev(item.file_type.casefold()), _tie(item))
    if choice == "type_asc":
        return lambda item: (item.file_type.casefold(), _tie(item))
    return lambda item: (item.walk_index, _tie(item))


def _date_key(item: OrderFile, *, newest_first: bool) -> tuple:
    stamp = sort_timestamp(file_type=item.file_type, zim_date=item.zim_date, mtime=item.mtime)
    return _known_key(stamp, high_first=newest_first, item=item)


def _known_key(value: float | int | None, *, high_first: bool, item: OrderFile) -> tuple:
    """Known values first. Missing values stay after them in both directions."""
    if value is None:
        return (1, 0, _tie(item))
    number = -float(value) if high_first else float(value)
    return (0, number, _tie(item))


def _open_file(item: OrderFile, *, index_now: str | None, skip: set[str], defer: set[str], honor_defer: bool) -> bool:
    if item.state in ("indexed", "deferred"):
        return False
    if item.source in skip and item.source != index_now:
        return False
    if honor_defer and item.source in defer and item.source != index_now:
        return False
    return True


@dataclass(frozen=True)
class Pick:
    target: str | None
    switch: bool
    reason: str


def pick_next(
    files: list[OrderFile],
    *,
    choice: str,
    current: str | None,
    index_now: str | None,
    skip: set[str] | frozenset[str] | None = None,
    defer: set[str] | frozenset[str] | None = None,
) -> Pick:
    """The file to index next.

    Index now wins until that file finishes. Prioritized unfinished files
    come next, in the selected order, and a prioritized file already running
    is not dropped for a different one. Changing the order does not leave
    the file already in progress. `defer` is the set Measure has already sampled.
    """
    skipped = set(skip or ())
    deferred = set(defer or ())

    def eligible(item: OrderFile, *, honor_defer: bool) -> bool:
        return _open_file(item, index_now=index_now, skip=skipped, defer=deferred, honor_defer=honor_defer)

    target = None
    reason = "none"
    if index_now:
        hit = next((item for item in files if item.source == index_now and eligible(item, honor_defer=False)), None)
        if hit is not None and hit.state not in ("indexed", "deferred"):
            target = hit.source
            reason = "now"
    pending = [item for item in files if eligible(item, honor_defer=True)]
    if target is None and pending:
        ordered = ordered_files(pending, choice)
        prioritized = [item for item in ordered if item.priority]
        pool = prioritized or ordered
        target = pool[0].source
        reason = "priority" if prioritized else "order"
    if target is None:
        return Pick(None, False, "none")
    if current is None or target == current:
        return Pick(target, False, reason)
    current_row = next((item for item in files if item.source == current), None)
    current_open = current_row is not None and current_row.state not in ("indexed", "deferred") and current not in skipped
    if not current_open:
        return Pick(target, False, reason)
    if reason == "order":
        return Pick(current, False, "stay")
    if reason == "priority" and current_row is not None and current_row.priority:
        return Pick(current, False, "stay")
    return Pick(target, True, reason)


def row_menu_labels(*, priority: bool) -> tuple[str, str, str]:
    """Right-click actions. The second one sets or clears priority."""
    return ("Index now", "Clear priority" if priority else "Set priority", "Details")


def sample_expired(started: float, now: float, *, limit: float = MEASURE_SECONDS) -> bool:
    """True once a measure sample has had its 300 seconds."""
    return float(now) - float(started) >= float(limit)


def open_index_now(session, files: list[OrderFile] | None = None) -> str | None:
    """Index now when that file is still unfinished. Otherwise None.

    Indexed, deferred, and missing files are not selectable. A failed file
    still is, so the user can retry it. With no list and no library yet,
    the request is kept.
    """
    target = session.index_now() if session is not None and hasattr(session, "index_now") else None
    if not target:
        return None
    if files is not None:
        hit = next((item for item in files if item.source == target), None)
        if hit is not None:
            if hit.state in ("indexed", "deferred"):
                return None
            return target
    listed = None
    book = session.get_book() if hasattr(session, "get_book") else None
    if book is not None and hasattr(book, "listed"):
        listed = book.listed()
    if listed:
        for source, state in listed:
            if source == target:
                if state in ("indexed", "deferred"):
                    return None
                return target
        return None
    if files is not None:
        return None
    return target


def engine_switch_reason(
    session,
    source: str,
    sample_started: float | None,
    now: float,
    armed_epoch: int = 0,
    files: list[OrderFile] | None = None,
) -> str | None:
    """Why the file in progress should be left after the embed call already running.

    None means stay. Order changes are not a reason. A prioritized file is
    not left for another prioritized file. Index now on this same file runs
    to completion, including during Measure. A request that names an indexed,
    deferred, or missing file is cleared so later files keep going.
    """
    if session is None:
        return None
    target = open_index_now(session, files)
    if session.index_now() and target is None:
        session.clear_index_now()
    if target and target != source:
        return "now"
    wants = getattr(session, "wants_priority_switch", None)
    if callable(wants) and wants(source, armed_epoch):
        return "priority"
    if sample_started is None:
        clock = getattr(session, "measure_clock", None)
        if callable(clock):
            sample_started = clock()
    measuring = getattr(session, "measuring", None)
    if sample_started is not None and callable(measuring) and measuring() and target != source:
        if sample_expired(sample_started, now):
            return "measure"
    return None


def interrupt_outcome(*, reason: str, finished: bool) -> str:
    """What to do with the file when work on it stops.

    `replaced` is the file-disappeared path and deletes points.
    A switch checkpoints and keeps them. Stop does that too.
    """
    if reason == "replaced":
        return "drop"
    if finished:
        return "indexed"
    return "partial"


def measure_start_chunks(*, auto: bool, user_chunks: int, recommended: int) -> int:
    """Batch size a measure sample starts at.

    Auto starts at the same hardware recommendation as the Recommended
    control, and may move afterward. Otherwise the user's count is kept.
    """
    if auto:
        return int(recommended)
    return int(user_chunks)


def apply_sample_batch(session, *, auto: bool, recommended: int) -> int:
    """Set the next embed call for one measure sample. Manual does not move it."""
    count = measure_start_chunks(auto=auto, user_chunks=session.get_chunks(), recommended=recommended)
    if auto:
        session.set_chunks(count)
        return count
    return session.get_chunks()


def measure_ending(*, from_idle: bool, pass_complete: bool) -> str:
    """What to do when Measure has visited every unfinished file.

    `stop` ends the run the way Stop does. `resume` keeps indexing.
    Empty means the pass is not finished.
    """
    if not pass_complete:
        return ""
    return "stop" if from_idle else "resume"


def file_eta_view(
    *,
    state: str,
    articles: int,
    total: int | None,
    live_rate: float | None,
    measured_rate: float | None,
) -> tuple[str, int | None]:
    """ETA text and seconds.

    A measured article rate is used even when another file is current,
    as long as this file's remaining article count is known.
    """
    from nomad_turbo.progress import format_eta

    if state == "indexed":
        return "", 0
    rate = live_rate if state == "running" and live_rate else None
    if rate is None:
        rate = measured_rate
    if total:
        remaining = max(0, int(total) - int(articles))
        if rate:
            seconds = int(remaining / rate)
            return format_eta(seconds), seconds
        if state == "running":
            return "measuring", None
        return "", None
    if state == "running":
        return "measuring", None
    return "", None


def sum_known_etas(rows: list[tuple[str, int | None]]) -> int | None:
    """Seconds left for the library, from per-file ETAs.

    None when any unfinished file has no ETA yet. Indexed and deferred
    files are not part of the sum. The caller then keeps the size-based ETA.
    """
    pending = False
    total = 0
    for state, eta in rows:
        if state in ("indexed", "deferred"):
            continue
        pending = True
        if eta is None:
            return None
        total += int(eta)
    if not pending:
        return 0
    return total
