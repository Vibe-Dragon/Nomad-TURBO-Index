"""What the Qdrant container is doing, and whether it can take changes.

Health is a short check. A collection that does not answer is not ready,
even when the process health check has already passed.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from nomad_turbo.store import _health_ok, _request


@dataclass(frozen=True)
class QdrantView:
    text: str
    ready: bool
    vectors: int | None = None
    collection_status: str = ""
    optimizer_error: str = ""


def activity_from_text(text: str) -> str:
    """The newest Qdrant log line, without the timestamp and level."""
    chosen = ""
    for line in text.splitlines():
        line = line.strip()
        if line:
            chosen = line
    if not chosen:
        return ""
    for token in ("ERROR", "WARN", "INFO", "DEBUG", "TRACE"):
        marker = f" {token} "
        if marker not in chosen:
            continue
        rest = chosen.split(marker, 1)[1]
        if ": " in rest:
            rest = rest.split(": ", 1)[1]
        chosen = rest.strip()
        break
    return chosen[:220]


def uptime_text(started_at: str, now: datetime | None = None) -> str:
    """How long the container has been up. Empty when the timestamp is missing."""
    raw = (started_at or "").strip()
    if not raw or raw.startswith("0001-"):
        return ""
    stamp = raw.replace("Z", "+00:00")
    # Docker prints nanoseconds. datetime accepts microseconds.
    if "." in stamp:
        head, tail = stamp.split(".", 1)
        digits = []
        rest = []
        for char in tail:
            if char.isdigit() and not rest:
                digits.append(char)
            else:
                rest.append(char)
        stamp = head + "." + "".join(digits[:6]) + "".join(rest)
    try:
        started = datetime.fromisoformat(stamp)
    except ValueError:
        return ""
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    moment = now or datetime.now(timezone.utc)
    seconds = max(0, int((moment - started).total_seconds()))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def indexing_qdrant_problem(view: QdrantView) -> str:
    """A problem worth logging during a run. Empty when the collection can take writes.

    Yellow, grey, and an open segment build are optimization states. They are
    not a failure. An optimizer error, a red collection, or a collection that
    is not ready is.
    """
    error = (view.optimizer_error or "").strip()
    if error and error.lower() != "ok":
        return f"Optimizer error: {error}."
    if (view.collection_status or "").strip().lower() == "red":
        return "Qdrant reports the collection is in an error state."
    if not view.ready:
        headline = (view.text or "").split("\n", 1)[0].strip()
        return headline or "Qdrant is not ready for changes."
    return ""


def indexing_problem_line(previous: str, indexing: bool, problem: str) -> tuple[str, str]:
    """The next log line, and the problem to remember.

    Idle runs remember nothing, so the same problem is logged again when
    indexing starts. A repeat of the same problem during a run is quiet.
    """
    if not indexing:
        return "", ""
    problem = (problem or "").strip()
    previous = (previous or "").strip()
    if problem and problem != previous:
        return f"Qdrant problem while indexing: {problem}", problem
    if previous and not problem:
        return "Qdrant problem cleared. The collection is answering again.", ""
    return "", problem


def changes_allowed(ready: bool) -> bool:
    """Indexing, delete, and repair wait until the collection answered."""
    return bool(ready)


def read_vector_count(raw: bytes) -> int | None:
    """The vector counter Qdrant stores at the front of a segment status file."""
    if len(raw) < 8:
        return None
    count = int.from_bytes(raw[:8], "little")
    if count < 0:
        return None
    return count


def open_segment_vectors(storage: str, collection: str) -> int | None:
    """How many vectors the busiest segment holds. None when the files cannot be read.

    This is a local file read. It does not call Qdrant, so it still works
    while a write is holding the collection.
    """
    root = Path(storage) / "qdrant" / "collections" / collection
    if not root.is_dir():
        return None
    newest: Path | None = None
    newest_mtime = -1.0
    try:
        matches = root.glob("*/segments/*/vector_storage/vectors/status.dat")
    except OSError:
        return None
    for status in matches:
        try:
            mtime = status.stat().st_mtime
        except OSError:
            continue
        if mtime >= newest_mtime:
            newest = status
            newest_mtime = mtime
    if newest is None:
        return None
    try:
        return read_vector_count(newest.read_bytes())
    except OSError:
        return None


def growth_line(previous: int | None, current: int | None) -> str:
    """Say whether the open segment is still taking vectors."""
    if current is None:
        return "The index files could not be read. Leave Qdrant running."
    if previous is not None and current > previous:
        return (
            f"The open segment grew by {current - previous:,} vectors "
            f"and now holds {current:,}. That write is still moving."
        )
    if previous is not None:
        return (
            f"The open segment still holds {current:,} vectors. "
            "Leave Qdrant running until the collection answers."
        )
    return f"The open segment holds {current:,} vectors."


def describe_qdrant(
    *,
    running: bool,
    status: str,
    health_ok: bool,
    ready_ok: bool,
    collection_ok: bool,
    collection: str,
    activity: str,
    uptime: str,
    vectors: int | None = None,
    previous_vectors: int | None = None,
    collection_status: str = "",
    optimizer_error: str = "",
) -> QdrantView:
    """One block of text for the window. ready is true only when changes can be sent."""
    state = status or "missing"
    hold_detail = ""
    if not running:
        headline = f"Qdrant is {state}. Not ready for changes."
        ready = False
    elif not health_ok:
        headline = "Qdrant is running and has not passed the health check. Not ready for changes."
        ready = False
    elif not ready_ok:
        headline = "Qdrant is running. The ready check has not passed. Not ready for changes."
        ready = False
    elif not collection_ok:
        headline = "Qdrant is still writing. Do not restart it or delete the index."
        hold_detail = growth_line(previous_vectors, vectors)
        ready = False
    else:
        headline = "Qdrant is ready for changes."
        ready = True
    parts = [f"Container {state}."]
    if uptime:
        parts.append(f"Up {uptime}.")
    parts.append("Health check passed." if health_ok else "Health check has not passed.")
    parts.append("Ready check passed." if ready_ok else "Ready check has not passed.")
    if collection_ok:
        parts.append(f"Collection {collection} answered.")
    elif health_ok and ready_ok:
        parts.append(f"Collection {collection} did not answer. A restart would replay this write.")
    else:
        parts.append(f"Collection {collection} was not asked while the process is still starting.")
    if hold_detail:
        parts.append(hold_detail)
    if activity:
        parts.append(f"Last log: {activity}")
    color = (collection_status or "").strip().lower()
    if optimizer_error:
        parts.append(f"Optimizer error: {optimizer_error}. Nothing was changed by this check.")
    elif color == "yellow":
        parts.append("An optimization is still outstanding. Do not restart Qdrant.")
    elif color == "grey":
        parts.append("Optimizations are waiting for an update before they run.")
    elif color == "red":
        parts.append("Qdrant reports a collection error. Nothing was changed by this check.")
    return QdrantView(
        text=headline + "\n" + " ".join(parts),
        ready=ready,
        vectors=vectors,
        collection_status=color,
        optimizer_error=optimizer_error,
    )


def _plain_ok(url: str, timeout: float = 2) -> bool:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (TimeoutError, urllib.error.URLError, OSError):
        return False


def _docker(*args: str, timeout: float = 5) -> str:
    try:
        return subprocess.check_output(["docker", *args], text=True, stderr=subprocess.STDOUT, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return ""


def find_qdrant_container() -> str:
    listing = _docker("ps", "-a", "--format", "{{.Names}}\t{{.Image}}")
    for line in listing.splitlines():
        name, sep, image = line.partition("\t")
        if sep and "qdrant" in image:
            return name.strip()
    return ""


def _collection_info(url: str, timeout: float = 3) -> tuple[bool, str, str]:
    """(answered, status color, optimizer error). A timeout is not an error state.

    Repair reloads store.py, so the StoreError raised here may be a new class.
    Catching Exception still treats that timeout as "did not answer".
    """
    try:
        body = _request(url, timeout=timeout)
    except Exception:
        return False, "", ""
    result = body.get("result") if isinstance(body, dict) else None
    if not isinstance(result, dict):
        return True, "", ""
    status = str(result.get("status") or "")
    opt = result.get("optimizer_status")
    error = ""
    if isinstance(opt, dict):
        error = str(opt.get("error") or "")
    elif isinstance(opt, str) and opt.lower() not in ("", "ok"):
        error = opt
    return True, status, error


def _collection_answered(url: str, timeout: float = 3) -> bool:
    """False when the collection is missing or too busy to answer. Never raises."""
    answered, _status, _error = _collection_info(url, timeout=timeout)
    return answered


def probe_qdrant(
    base_url: str,
    collection: str,
    container: str | None = None,
    storage: str | None = None,
    previous_vectors: int | None = None,
) -> QdrantView:
    """Ask Docker and Qdrant. A starting collection is not poked with a long call."""
    name = container if container is not None else find_qdrant_container()
    status = "missing"
    started = ""
    if name:
        raw = _docker("inspect", "-f", "{{.State.Status}}|{{.State.StartedAt}}", name).strip()
        status, sep, started = raw.partition("|")
        if not sep:
            status = raw or "missing"
    running = status == "running"
    base = base_url.rstrip("/")
    health_ok = _health_ok(base) if running else False
    ready_ok = _plain_ok(f"{base}/readyz") if health_ok else False
    collection_status = ""
    optimizer_error = ""
    if ready_ok:
        collection_ok, collection_status, optimizer_error = _collection_info(f"{base}/collections/{collection}")
    else:
        collection_ok = False
    vectors = None
    if storage and running and health_ok and ready_ok and not collection_ok:
        vectors = open_segment_vectors(storage, collection)
    activity = ""
    if name and running:
        activity = activity_from_text(_docker("logs", "--tail", "40", name))
    return describe_qdrant(
        running=running,
        status=status,
        health_ok=health_ok,
        ready_ok=ready_ok,
        collection_ok=collection_ok,
        collection=collection,
        activity=activity,
        uptime=uptime_text(started) if running else "",
        vectors=vectors,
        previous_vectors=previous_vectors,
        collection_status=collection_status,
        optimizer_error=optimizer_error,
    )
