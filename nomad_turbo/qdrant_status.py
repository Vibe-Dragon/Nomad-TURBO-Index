"""What the Qdrant container is doing, and whether it can take changes.

Health is a short check. A collection that does not answer is not ready,
even when the process health check has already passed.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

from nomad_turbo.store import _health_ok, _request


@dataclass(frozen=True)
class QdrantView:
    text: str
    ready: bool


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


def changes_allowed(ready: bool) -> bool:
    """Indexing, delete, and repair wait until the collection answered."""
    return bool(ready)


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
) -> QdrantView:
    """One block of text for the window. ready is true only when changes can be sent."""
    state = status or "missing"
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
        headline = "Qdrant is up and the collection is not answering yet. Not ready for changes."
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
        parts.append(f"Collection {collection} did not answer.")
    else:
        parts.append(f"Collection {collection} was not asked while the process is still starting.")
    if activity:
        parts.append(f"Last log: {activity}")
    return QdrantView(text=headline + "\n" + " ".join(parts), ready=ready)


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


def _collection_answered(url: str, timeout: float = 3) -> bool:
    """False when the collection is missing or too busy to answer. Never raises.

    Repair reloads store.py, so the StoreError raised here may be a new class.
    Catching Exception still treats that timeout as "did not answer".
    """
    try:
        _request(url, timeout=timeout)
    except Exception:
        return False
    return True


def probe_qdrant(base_url: str, collection: str, container: str | None = None) -> QdrantView:
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
    collection_ok = _collection_answered(f"{base}/collections/{collection}") if ready_ok else False
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
    )
