"""Facts remembered for each library file.

Speeds, the last measured ETA, dates, and priority stay after Turbo is
closed. A file that is no longer in the library is removed.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from nomad_turbo.schedule import EM_DASH


STATS_NAME = "file-stats.json"


@dataclass
class FileStat:
    source: str
    name: str = ""
    file_type: str = ""
    size_bytes: int = 0
    location: str = ""
    mtime: float | None = None
    zim_date: str = ""
    zim_known: bool = False
    articles: int = 0
    total_articles: int | None = None
    chunks: int = 0
    state: str = ""
    priority: bool = False
    measured_article_rate: float | None = None
    measured_chunk_rate: float | None = None
    last_article_rate: float | None = None
    last_chunk_rate: float | None = None
    min_article_rate: float | None = None
    max_article_rate: float | None = None
    min_chunk_rate: float | None = None
    max_chunk_rate: float | None = None
    measured_eta_seconds: int | None = None


def _known_fields() -> set[str]:
    return {item.name for item in fields(FileStat)}


def library_location(storage_host: str, container_storage: str, source: str) -> str:
    """Host path of a library file. Other sources stay as they are."""
    root = (container_storage or "").rstrip("/")
    if root and source.startswith(root + "/"):
        relative = source[len(root) + 1 :]
        if relative.startswith("zim/") or relative.startswith("kb_uploads/"):
            return str(Path(storage_host) / relative)
    return source


def format_downloaded(mtime: float | None) -> str:
    if not mtime:
        return EM_DASH
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(mtime)))


def format_article_pace(per_second: float | None) -> str:
    """Articles per minute. The stored rate is articles per second."""
    if not per_second or per_second <= 0:
        return EM_DASH
    return f"{per_second * 60:,.0f}"


def format_chunk_pace(per_second: float | None) -> str:
    if not per_second or per_second <= 0:
        return EM_DASH
    return f"{per_second:,.1f}"


def format_articles(articles: int, total: int | None) -> str:
    if total:
        return f"{int(articles):,} / {int(total):,}"
    if articles:
        return f"{int(articles):,}"
    return EM_DASH


def format_chunks(chunks: int) -> str:
    if not chunks:
        return EM_DASH
    return f"{int(chunks):,}"


def detail_lines(stat: FileStat) -> list[tuple[str, str]]:
    """Every saved fact, in the order the details window shows them."""
    from nomad_turbo.catalog import status_word
    from nomad_turbo.progress import format_bytes, format_eta

    name = stat.name or (stat.source.rsplit("/", 1)[-1] if stat.source else "")
    zim = stat.zim_date if stat.zim_date else EM_DASH
    eta = EM_DASH if stat.measured_eta_seconds is None else format_eta(stat.measured_eta_seconds)
    return [
        ("Name", name or EM_DASH),
        ("File type", stat.file_type or EM_DASH),
        ("Size", format_bytes(int(stat.size_bytes or 0))),
        ("Location", stat.location or stat.source or EM_DASH),
        ("Date downloaded", format_downloaded(stat.mtime)),
        ("ZIM date", zim),
        ("Status", status_word(stat.state) if stat.state else EM_DASH),
        ("Priority", "prioritized" if stat.priority else "not prioritized"),
        ("Articles", format_articles(stat.articles, stat.total_articles)),
        ("Chunks", format_chunks(stat.chunks)),
        ("Last art/min", format_article_pace(stat.last_article_rate)),
        ("Minimum art/min", format_article_pace(stat.min_article_rate)),
        ("Maximum art/min", format_article_pace(stat.max_article_rate)),
        ("Last chk/s", format_chunk_pace(stat.last_chunk_rate)),
        ("Minimum chk/s", format_chunk_pace(stat.min_chunk_rate)),
        ("Maximum chk/s", format_chunk_pace(stat.max_chunk_rate)),
        ("Last measured ETA", eta),
    ]


def detail_text(stat: FileStat) -> str:
    return "\n".join(f"{label}: {value}" for label, value in detail_lines(stat))


def compose_detail(
    source: str,
    saved: FileStat | None,
    *,
    name: str = "",
    file_type: str = "",
    size_bytes: int = 0,
    location: str = "",
    mtime: float | None = None,
    zim_date: str | None = None,
    articles: int | None = None,
    total_articles: int | None = None,
    chunks: int | None = None,
    state: str = "",
    priority: bool | None = None,
    have_live: bool = False,
) -> FileStat:
    """Live library facts cover the saved copy. Speeds stay remembered."""
    row = FileStat(**asdict(saved)) if saved is not None else FileStat(source=source)
    row.source = source
    if name:
        row.name = name
    if file_type:
        row.file_type = file_type
    if size_bytes:
        row.size_bytes = int(size_bytes)
    if location:
        row.location = location
    if mtime:
        row.mtime = float(mtime)
    if zim_date is not None:
        row.zim_date = zim_date
        row.zim_known = True
    if have_live:
        if articles is not None:
            row.articles = int(articles)
        if total_articles:
            row.total_articles = int(total_articles)
        if chunks is not None:
            row.chunks = int(chunks)
        if state:
            row.state = state
    if priority is not None:
        row.priority = bool(priority)
    return row


def _apply_rate(row: FileStat, kind: str, rate: float | None, measured: bool) -> bool:
    if not rate or rate <= 0:
        return False
    rate = float(rate)
    changed = False
    last_name = f"last_{kind}_rate"
    min_name = f"min_{kind}_rate"
    max_name = f"max_{kind}_rate"
    if getattr(row, last_name) != rate:
        setattr(row, last_name, rate)
        changed = True
    current_min = getattr(row, min_name)
    if current_min is None or rate < current_min:
        setattr(row, min_name, rate)
        changed = True
    current_max = getattr(row, max_name)
    if current_max is None or rate > current_max:
        setattr(row, max_name, rate)
        changed = True
    if measured:
        measured_name = f"measured_{kind}_rate"
        if getattr(row, measured_name) != rate:
            setattr(row, measured_name, rate)
            changed = True
    return changed


def _partial_name(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest() + ".json"


def _dump(row: FileStat) -> dict:
    data = asdict(row)
    data.pop("source", None)
    return data


def _row_from(source: str, item: dict) -> FileStat:
    known = _known_fields()
    values = {key: item[key] for key in item if key in known and key != "source"}
    row = FileStat(source=source, **values)
    if item.get("zim_date"):
        row.zim_known = True
    if row.measured_eta_seconds is not None:
        row.measured_eta_seconds = int(row.measured_eta_seconds)
    return row


class FileStats:
    """One JSON file of per-file facts. Missing library files are deleted."""

    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()
        self._io = threading.Lock()
        self._rows: dict[str, FileStat] = {}
        self._touched: set[str] = set()
        self._removed = False

    def load(self) -> None:
        path = self.path
        if path is None:
            return
        rows: dict[str, FileStat] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return
            files = raw.get("files") if isinstance(raw, dict) else None
            if isinstance(files, dict):
                for source, item in files.items():
                    if not isinstance(source, str) or not isinstance(item, dict):
                        continue
                    rows[source] = _row_from(source, item)
        self._overlay_partial(rows)
        with self._lock:
            self._rows = rows
            self._touched.clear()
            self._removed = False

    def save(self) -> None:
        """Write the facts. One changed file does not rewrite the whole set."""
        with self._io:
            with self._lock:
                if self.path is None or (not self._touched and not self._removed):
                    return
                path = self.path
                one = None
                if not self._removed and len(self._touched) == 1:
                    source = next(iter(self._touched))
                    row = self._rows.get(source)
                    if row is not None:
                        one = (source, asdict(row))
                if one is None:
                    copied = [(source, asdict(row)) for source, row in self._rows.items()]
                else:
                    copied = None
                self._touched.clear()
                self._removed = False
            try:
                if one is None:
                    self._write_full(path, copied)
                else:
                    self._write_one(path, one[0], one[1])
            except OSError:
                with self._lock:
                    if one is None:
                        self._removed = True
                    else:
                        self._touched.add(one[0])
                raise

    def _partial_dir(self, path: Path) -> Path:
        return path.with_name(path.name + ".d")

    def _overlay_partial(self, rows: dict[str, FileStat]) -> None:
        """Facts saved for one file since the last full write."""
        path = self.path
        if path is None:
            return
        folder = self._partial_dir(path)
        if not folder.is_dir():
            return
        for child in sorted(folder.glob("*.json")):
            try:
                item = json.loads(child.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(item, dict):
                continue
            source = item.get("source")
            fields = item.get("fields")
            if not isinstance(source, str) or not isinstance(fields, dict):
                continue
            rows[source] = _row_from(source, fields)

    def _write_full(self, path: Path, copied: list[tuple[str, dict]]) -> None:
        rows = {}
        for source, data in copied:
            data.pop("source", None)
            rows[source] = data
        text = json.dumps({"files": rows}, indent=2, sort_keys=True) + "\n"
        self._atomic_write(path, text)
        self._clear_partial(path)

    def _write_one(self, path: Path, source: str, data: dict) -> None:
        data.pop("source", None)
        folder = self._partial_dir(path)
        folder.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"source": source, "fields": data}, indent=2, sort_keys=True) + "\n"
        self._atomic_write(folder / _partial_name(source), text)

    def _atomic_write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)

    def _clear_partial(self, path: Path) -> None:
        folder = self._partial_dir(path)
        if not folder.is_dir():
            return
        for child in folder.glob("*"):
            if child.is_file():
                child.unlink()

    def get(self, source: str) -> FileStat | None:
        with self._lock:
            row = self._rows.get(source)
            if row is None:
                return None
            return FileStat(**asdict(row))

    def forget(self, source: str) -> bool:
        with self._lock:
            if source not in self._rows:
                return False
            del self._rows[source]
            self._touched.discard(source)
            self._removed = True
            return True

    def prune(self, present: set[str]) -> list[str]:
        """Drop facts for files that are not in the library anymore."""
        with self._lock:
            removed = [source for source in self._rows if source not in present]
            for source in removed:
                del self._rows[source]
            if removed:
                for source in removed:
                    self._touched.discard(source)
                self._removed = True
            return removed

    def priorities(self) -> set[str]:
        with self._lock:
            return {source for source, row in self._rows.items() if row.priority}

    def zim_dates(self) -> list[tuple[str, str]]:
        with self._lock:
            return [(source, row.zim_date) for source, row in self._rows.items() if row.zim_known]

    def note_library(
        self,
        source: str,
        *,
        name: str = "",
        file_type: str = "",
        size_bytes: int = 0,
        location: str = "",
        mtime: float | None = None,
        articles: int = 0,
        total_articles: int | None = None,
        chunks: int = 0,
        state: str = "",
    ) -> None:
        """Copy identity and progress from a library scan. Speeds stay."""
        with self._lock:
            row = self._ensure(source)
            if name:
                self._assign(row, "name", name)
            if file_type:
                self._assign(row, "file_type", file_type)
            # A size of one byte is the placeholder for a row that has not been stated.
            if size_bytes and int(size_bytes) > 1:
                self._assign(row, "size_bytes", int(size_bytes))
            if location:
                self._assign(row, "location", location)
            if mtime:
                self._assign(row, "mtime", float(mtime))
            self._assign(row, "articles", int(articles or 0))
            if total_articles:
                self._assign(row, "total_articles", int(total_articles))
            self._assign(row, "chunks", int(chunks or 0))
            if state:
                self._assign(row, "state", state)

    def note_progress(
        self,
        source: str,
        *,
        articles: int,
        total_articles: int | None,
        chunks: int,
        state: str,
        name: str = "",
        size_bytes: int = 0,
        mtime: float | None = None,
        file_type: str = "",
    ) -> None:
        self.note_library(
            source,
            name=name,
            file_type=file_type,
            size_bytes=size_bytes,
            mtime=mtime,
            articles=articles,
            total_articles=total_articles,
            chunks=chunks,
            state=state,
        )

    def note_rates(
        self,
        source: str,
        article_rate: float | None,
        chunk_rate: float | None,
        *,
        measured: bool = False,
        eta_seconds: int | None = None,
    ) -> None:
        with self._lock:
            if not article_rate and not chunk_rate and eta_seconds is None:
                return
            row = self._ensure(source)
            changed = _apply_rate(row, "article", article_rate, measured)
            changed = _apply_rate(row, "chunk", chunk_rate, measured) or changed
            if measured and eta_seconds is not None and row.measured_eta_seconds != int(eta_seconds):
                row.measured_eta_seconds = int(eta_seconds)
                changed = True
            if changed:
                self._touch(source)

    def note_measurement(
        self,
        source: str,
        *,
        article_rate: float | None,
        chunk_rate: float | None,
        eta_seconds: int | None,
        articles: int | None = None,
        total_articles: int | None = None,
        chunks: int | None = None,
        state: str = "",
    ) -> None:
        if articles is not None or chunks is not None or state:
            self.note_progress(
                source,
                articles=int(articles or 0),
                total_articles=total_articles,
                chunks=int(chunks or 0),
                state=state,
            )
        self.note_rates(
            source,
            article_rate,
            chunk_rate,
            measured=True,
            eta_seconds=eta_seconds,
        )

    def note_zim_date(self, source: str, raw: str | None) -> bool:
        value = "" if raw is None else str(raw)
        with self._lock:
            row = self._ensure(source)
            if row.zim_known and row.zim_date == value:
                return False
            row.zim_date = value
            row.zim_known = True
            self._touch(source)
            return True

    def set_priority(self, source: str, on: bool) -> None:
        with self._lock:
            row = self._ensure(source)
            on = bool(on)
            if row.priority == on:
                return
            row.priority = on
            self._touch(source)

    def _touch(self, source: str) -> None:
        """The caller holds the library lock."""
        self._touched.add(source)

    def _ensure(self, source: str) -> FileStat:
        row = self._rows.get(source)
        if row is None:
            row = FileStat(source=source)
            self._rows[source] = row
            self._touch(source)
        return row

    def _assign(self, row: FileStat, key: str, value) -> None:
        if getattr(row, key) != value:
            setattr(row, key, value)
            self._touch(row.source)
