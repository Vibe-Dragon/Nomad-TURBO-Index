"""Log files for the current copy of the knowledge index.

One file per local day, under logs/YYYY-MM-DD.log. The files stay until that
Qdrant collection is deleted, or until a file's first line is older than the
collection. Export writes every kept file into one zip. The log buttons do
not change what these files record.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import zipfile
from datetime import datetime
from pathlib import Path
from typing import TextIO

LEGACY_NAMES = ("session-previous.log", "session.log")
DAILY_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}\.log$")
_STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:,(\d+))?")
_align_lock = threading.Lock()
_seen_lock = threading.Lock()
# directory -> ("present", birth_epoch) or ("missing", 0) or ("unknown", 0)
_seen: dict[str, tuple] = {}


def daily_log_path(directory: Path, moment: datetime | None = None) -> Path:
    """Path of the log for one local day. The file is created when a line is written."""
    when = moment or datetime.now()
    return Path(directory) / "logs" / f"{when.strftime('%Y-%m-%d')}.log"


def collection_directory(storage_host: str, collection: str) -> Path:
    """Host folder of one collection. NOMAD storage plus Qdrant's own directory."""
    return Path(storage_host) / "qdrant" / "collections" / collection


def parse_stamp(line: str) -> datetime | None:
    """Local time at the start of a log line. None when the line has no timestamp."""
    match = _STAMP.match(line or "")
    if match is None:
        return None
    try:
        moment = datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    if match.group(3):
        frac = (match.group(3) + "000000")[:6]
        moment = moment.replace(microsecond=int(frac))
    return moment


def birth_epoch(path: Path) -> int | None:
    """Unix seconds the directory was created. None when birth time is unavailable.

    The directory's modified time moves when Qdrant writes. Birth time is the
    moment this copy of the collection was created.
    """
    if not path.exists():
        return None
    try:
        proc = subprocess.run(
            ["stat", "-c", "%W", str(path)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    raw = (proc.stdout or "").strip()
    if proc.returncode != 0 or not raw.isdigit():
        return None
    seconds = int(raw)
    if seconds <= 0:
        return None
    return seconds


def log_files(directory: Path) -> list[Path]:
    """Daily logs, oldest day first, then any leftover session logs."""
    directory = Path(directory)
    found: list[Path] = []
    folder = directory / "logs"
    if folder.is_dir():
        for path in sorted(folder.glob("*.log")):
            if path.is_file() and DAILY_NAME.fullmatch(path.name):
                found.append(path)
    seen = {path.resolve() for path in found}
    legacy: list[Path] = []
    for path in sorted(directory.glob("session*.log")):
        if path.is_file() and path.resolve() not in seen:
            legacy.append(path)
    return found + legacy


def first_stamp(path: Path) -> datetime | None:
    """First parseable timestamp in a file. None when no line has one."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stamp = parse_stamp(line)
                if stamp is not None:
                    return stamp
    except OSError:
        return None
    return None


def export_logs(directory: Path, target: Path) -> int:
    """Write every kept log into one zip. Returns how many files were packed."""
    _flush_handlers()
    files = [path for path in log_files(directory) if path.is_file() and path.stat().st_size > 0]
    target = Path(target)
    if target.parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    return len(files)


class DailyLogHandler(logging.Handler):
    """Append to logs/YYYY-MM-DD.log and switch files at local midnight."""

    def __init__(self, directory: Path, clock=None) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.clock = clock or datetime.now
        self._day = ""
        self._stream = None
        self._path: Path | None = None

    def current_path(self) -> Path | None:
        return self._path

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            self.handleError(record)
            return
        self.acquire()
        try:
            stream = self._ensure_stream()
            stream.write(msg + "\n")
            stream.flush()
        except Exception:
            self.handleError(record)
        finally:
            self.release()

    def flush(self) -> None:
        self.acquire()
        try:
            if self._stream is not None:
                self._stream.flush()
        finally:
            self.release()

    def close(self) -> None:
        self.acquire()
        try:
            self._close_stream()
            self._day = ""
        finally:
            self.release()
            super().close()

    def _ensure_stream(self) -> TextIO:
        day = self.clock().strftime("%Y-%m-%d")
        if self._stream is not None and day == self._day:
            return self._stream
        self._close_stream()
        folder = self.directory / "logs"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{day}.log"
        stream = path.open("a", encoding="utf-8")
        self._stream = stream
        self._path = path
        self._day = day
        return stream

    def _close_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            stream.flush()
            stream.close()
        except OSError:
            pass


def prepare_collection_logs(directory: Path, install) -> None:
    """Keep logs that belong to this collection. Drop the ones that do not.

    A Qdrant timeout does not delete logs. A confirmed missing collection does.
    A file whose first line is older than the collection directory is removed.
    Lines that remain are folded into one file per day.
    """
    if _foreign_indexer(directory):
        return
    with _align_lock:
        _prepare_unlocked(directory, install)


def align_running_logs(directory: Path, install) -> None:
    """Apply the collection clock and point the open process at today's file."""
    if _foreign_indexer(directory):
        return
    with _align_lock:
        formatter, level = _detach_file_handlers()
        try:
            _prepare_unlocked(directory, install)
        finally:
            _attach_daily_handler(directory, formatter, level)


def clear_collection_logs(directory: Path) -> None:
    """Delete kept logs and open a new file for the day. Used when the collection is deleted."""
    if _foreign_indexer(directory):
        return
    with _align_lock:
        formatter, level = _detach_file_handlers()
        try:
            for path in log_files(directory):
                path.unlink(missing_ok=True)
            _remember(directory, "missing", 0)
        finally:
            _attach_daily_handler(directory, formatter, level)


def watch_collection_logs(directory: Path, install) -> None:
    """Notice a deleted or reborn collection. A timeout does not clear logs."""
    directory = Path(directory)
    path = collection_directory(install.storage_host, install.collection)
    epoch = birth_epoch(path) if path.exists() else None
    key = _clock_key(directory)
    with _seen_lock:
        seen = _seen.get(key)
    if epoch is not None:
        if seen == ("present", epoch):
            return
        align_running_logs(directory, install)
        return
    if path.exists():
        # The folder is there, but its birth time could not be read. Keep the logs.
        return
    if seen == ("missing", 0):
        return
    from nomad_turbo.store import collection_copy_state

    # The folder is gone. A timeout still means the collection may exist, so logs stay.
    state = collection_copy_state(install)
    if state != "missing":
        return
    align_running_logs(directory, install)


def _prepare_unlocked(directory: Path, install) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if install is None:
        _adopt_legacy(directory)
        return
    kind, epoch = _read_clock(install)
    if kind == "unknown":
        _adopt_legacy(directory)
        return
    if kind == "missing":
        for path in log_files(directory):
            path.unlink(missing_ok=True)
        _remember(directory, "missing", 0)
        return
    born = datetime.fromtimestamp(epoch)
    for path in list(log_files(directory)):
        stamp = first_stamp(path)
        if stamp is not None and stamp < born:
            path.unlink(missing_ok=True)
    _adopt_legacy(directory)
    _remember(directory, "present", epoch)


def _read_clock(install) -> tuple[str, int]:
    """present plus birth epoch, missing, or unknown. Unknown must not clear logs."""
    path = collection_directory(install.storage_host, install.collection)
    if not path.exists():
        from nomad_turbo.store import collection_copy_state

        state = collection_copy_state(install)
        if state == "missing":
            return "missing", 0
        return "unknown", 0
    epoch = birth_epoch(path)
    if epoch is None:
        return "unknown", 0
    return "present", epoch


def _adopt_legacy(directory: Path) -> None:
    """Move session.log leftovers into the day file their lines belong to."""
    legacy = [path for path in log_files(directory) if path.parent == Path(directory)]
    if not legacy:
        return
    folder = directory / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    for path in legacy:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size == 0:
            path.unlink(missing_ok=True)
            continue
        if _fold_into_days(folder, path):
            path.unlink(missing_ok=True)


def _fold_into_days(folder: Path, path: Path) -> bool:
    """Append each line to that line's day. False when the file has no timestamp."""
    pending: list[str] = []
    handles: dict[str, TextIO] = {}
    current = ""
    folded = False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stamp = parse_stamp(line)
                if stamp is None:
                    if not current:
                        pending.append(line)
                    else:
                        handles[current].write(line)
                    continue
                day = stamp.strftime("%Y-%m-%d")
                if day not in handles:
                    target = (folder / f"{day}.log").open("a", encoding="utf-8")
                    handles[day] = target
                    for old in pending:
                        target.write(old)
                    pending.clear()
                current = day
                handles[day].write(line)
                folded = True
    finally:
        for target in handles.values():
            target.close()
    return folded


def _clock_key(directory: Path) -> str:
    try:
        return str(Path(directory).resolve())
    except OSError:
        return str(directory)


def _remember(directory: Path, kind: str, epoch: int) -> None:
    with _seen_lock:
        _seen[_clock_key(directory)] = (kind, epoch)


def _foreign_indexer(directory: Path) -> bool:
    """True when another live Turbo process owns this state directory.

    A short command must not move the log out from under the open window.
    """
    try:
        from nomad_turbo.discover import state_dir

        if Path(directory).resolve() != state_dir().resolve():
            return False
    except Exception:
        return False
    pid_path = Path(directory) / "pid"
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    if pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _file_handlers() -> list[logging.Handler]:
    root = logging.getLogger()
    return [
        handler
        for handler in list(root.handlers)
        if isinstance(handler, (DailyLogHandler, logging.FileHandler))
    ]


def _detach_file_handlers() -> tuple[logging.Formatter | None, int]:
    root = logging.getLogger()
    found = _file_handlers()
    for handler in found:
        handler.acquire()
    formatter = None
    level = logging.DEBUG
    try:
        for handler in found:
            if handler.formatter is not None:
                formatter = handler.formatter
            level = handler.level
            root.removeHandler(handler)
            handler.close()
    finally:
        for handler in found:
            handler.release()
    return formatter, level


def _attach_daily_handler(directory: Path, formatter: logging.Formatter | None, level: int) -> DailyLogHandler:
    root = logging.getLogger()
    handler = DailyLogHandler(directory)
    handler.setLevel(level)
    if formatter is not None:
        handler.setFormatter(formatter)
    root.addHandler(handler)
    return handler


def _flush_handlers() -> None:
    for handler in list(logging.getLogger().handlers):
        try:
            handler.flush()
        except Exception:
            pass


def log_help_text() -> str:
    """What the lines Turbo writes on purpose mean. An unforeseen error is not in this list."""
    sections = (
        (
            "Where a line goes",
            "Errors, Info, and Debug only change which lines the pane draws. "
            "Debug starts off, so article-by-article lines stay out of the pane. "
            "The log file still records every line, including the ones those boxes hide. "
            "Follow only scrolls the pane.",
        ),
        (
            "Logs",
            "Log files are kept for the current copy of the knowledge index. "
            "Each day has its own file, and a new file starts at midnight. "
            "Opening the window continues today's file. "
            "Deleting the collection, from Turbo or from Project NOMAD, clears these files. "
            "If the first line of a log is older than the current collection, that log is cleared when Turbo starts. "
            "Export log saves the kept files in one zip. "
            "The buttons only hide lines in the pane. The files still record every line.",
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
            "\"<n> chunks per call\" is the size of the next embed call. It is a debug line. "
            "\"embed batch:\" is Auto, or the recommendation, saying why the chunk count moved. "
            "\"one embed server is up\" means the server started. One server stays up. "
            "\"embed sequences\" is that server adding or dropping a second sequence. "
            "Each sequence keeps 2048 tokens. "
            "\"embedder not ready yet\" means the model is still loading. "
            "\"embedder call failed\" means that call will be tried again.",
        ),
        (
            "Indexing",
            "\"indexing <file> from article <n>\" is where this file continues. "
            "The parentheses are article paths already stored when that list was read. "
            "A file that already has many points continues from its checkpoint, and the line gives that point count. "
            "Those paths are not listed again. "
            "\"point count unavailable\" means Qdrant did not answer the count. "
            "\"skipping <file>, points already stored\" means that file was already in the index. "
            "\"qdrant write timed out. Trying that batch again.\" means the write is retried, and the line includes the error. "
            "\"qdrant is not taking writes. Staying on <file>\" means Turbo is waiting on that article. "
            "\"Qdrant problem while indexing\" is logged once when a run hits a problem. "
            "\"Qdrant problem cleared. The collection is answering again.\" means it recovered. "
            "A yellow collection by itself is not that problem. "
            "A finished OCR page is a debug line, \"ocr page <n> of <total>\". "
            "\"ocr workers <n> of <total>\" is a debug line too. "
            "Debug starts off, so those lines stay in the log file and out of the pane. "
            "Other OCR lines stay on the info list.",
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
