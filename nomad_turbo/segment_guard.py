"""Hold knowledge-base writes a segment rewrite cannot afford, and force one when it can.

Qdrant copies a segment before it indexes or vacuums it, and it needs about
twice that segment's disk size free. max_segment_size is not a write cap, so
a plain segment can grow past the point where the copy fits. These decisions
are pure: they do not delete points, delete the collection, or set
hnsw_config.on_disk. The repair action schedules one rewrite only when the
copy fits, then puts the previous optimizer settings back. Qdrant treats
indexing_threshold 0 as disabling indexing, so the schedule is an empty
optimizer update and never sends that value.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from nomad_turbo.discover import Install
from nomad_turbo.store import MAX_SEGMENT_KB

log = logging.getLogger("nomad_turbo")

# Same cap apply_segment_cap writes, in bytes. max_segment_size is kibibytes.
SEGMENT_CAP_BYTES = MAX_SEGMENT_KB * 1024

# InRamChunkedMmap prefaults the vectors. The graph stays in RAM while
# hnsw_config.on_disk is left false. 64 bytes covers the links for m=16.
GRAPH_BYTES_PER_VECTOR = 64

# The write loop asks every batch. Walking the collection tree every time
# would stall indexing, so the sizes are reused briefly.
MEASURE_CACHE_SECONDS = 15

OPTIMIZER_KEYS = (
    "deleted_threshold",
    "vacuum_min_vector_number",
    "default_segment_number",
    "max_segment_size",
    "memmap_threshold",
    "indexing_threshold",
    "flush_interval_sec",
    "max_optimization_threads",
)

_POPCOUNT = tuple(bin(value).count("1") for value in range(256))
_cache_lock = threading.Lock()
_write_cache: dict[tuple[str, str], tuple[float, "SegmentFacts"]] = {}


def default_segment_cap_bytes() -> int:
    """4 GiB of vectors, the cap Turbo already applies."""
    return SEGMENT_CAP_BYTES


@dataclass(frozen=True)
class SegmentFacts:
    """Measured collection state. Tests build this directly and skip the live files."""

    largest_plain_vector_bytes: int = 0
    largest_plain_disk_bytes: int = 0
    largest_plain_vector_count: int = 0
    largest_deleted_disk_bytes: int = 0
    free_bytes: int | None = None
    optimization_running: bool = False
    spare_ram_bytes: int | None = None
    segment_cap_bytes: int | None = None
    unreadable: bool = False


@dataclass(frozen=True)
class HoldDecision:
    """Empty reason means new upserts may proceed. Points and the collection stay."""

    reason: str
    delete_points: bool = False
    delete_collection: bool = False


@dataclass(frozen=True)
class ForceDecision:
    """allow is the only path that may ask Qdrant to optimize."""

    allow: bool
    reason: str
    delete_points: bool = False
    delete_collection: bool = False


@dataclass(frozen=True)
class ForceResult:
    ok: bool
    requested: bool
    reason: str
    delete_points: bool = False
    delete_collection: bool = False


def index_ram_bytes(vector_bytes: int, vector_count: int) -> int:
    """RAM for one in-memory HNSW build of this segment's live vectors."""
    vectors = max(0, int(vector_bytes))
    links = max(0, int(vector_count)) * GRAPH_BYTES_PER_VECTOR
    return vectors + links


def _cap_bytes(facts: SegmentFacts) -> int:
    if facts.segment_cap_bytes is not None and facts.segment_cap_bytes > 0:
        return int(facts.segment_cap_bytes)
    return default_segment_cap_bytes()


def segment_upsert_hold(facts: SegmentFacts) -> HoldDecision:
    """Why the next upsert should wait. An optimization already running does not, by itself.

    segment: the largest plain segment is past the vector cap.
    space: free space is under twice the plain segment that would be rewritten,
    or the free space could not be read while such a segment exists.
    """
    if facts.unreadable:
        return HoldDecision("space")
    if facts.largest_plain_vector_bytes > _cap_bytes(facts):
        return HoldDecision("segment")
    disk = facts.largest_plain_disk_bytes
    if disk > 0 and (facts.free_bytes is None or facts.free_bytes < 2 * disk):
        return HoldDecision("space")
    return HoldDecision("")


def _rewrite_target(facts: SegmentFacts) -> tuple[int, int, int] | None:
    """Disk bytes, live vector bytes, and live vector count of the segment to rewrite.

    An oversized plain segment is the one the indexing optimizer copies whole.
    Otherwise a fully-deleted segment still waiting to be reclaimed is the target,
    and its live vector count is zero.
    """
    if facts.largest_plain_vector_bytes > _cap_bytes(facts):
        disk = facts.largest_plain_disk_bytes
        if disk <= 0:
            disk = facts.largest_plain_vector_bytes
        return disk, facts.largest_plain_vector_bytes, max(0, facts.largest_plain_vector_count)
    if facts.largest_deleted_disk_bytes > 0:
        return facts.largest_deleted_disk_bytes, 0, 0
    return None


def force_optimization(facts: SegmentFacts) -> ForceDecision:
    """Force a rewrite of stuck segments, or refuse and leave the collection alone."""
    if facts.unreadable:
        return ForceDecision(
            False,
            "Segment files could not be read, so optimizing was refused. "
            "The collection was left unchanged.",
        )
    target = _rewrite_target(facts)
    if target is None:
        return ForceDecision(
            False,
            "No stuck segments. Nothing is oversized, and no fully-deleted segment "
            "is waiting to be reclaimed. Nothing was changed.",
        )
    if facts.optimization_running:
        return ForceDecision(
            False,
            "Optimization is already in progress. This click did not start another one. "
            "Qdrant is still building the segment it already opened, and that work stays "
            "in memory until it finishes. The collection was left unchanged.",
        )
    disk, vector_bytes, vector_count = target
    if facts.free_bytes is None or facts.free_bytes < 2 * disk:
        return ForceDecision(
            False,
            "Not enough free space to rewrite that segment. "
            f"The copy needs {format_bytes(2 * disk)} free and {format_bytes(facts.free_bytes)} is free. "
            "The collection was left unchanged.",
        )
    needed = index_ram_bytes(vector_bytes, vector_count)
    if facts.spare_ram_bytes is None or facts.spare_ram_bytes < needed:
        return ForceDecision(
            False,
            "Not enough spare memory for an in-memory index of that segment. "
            f"The graph needs {format_bytes(needed)} and {format_bytes(facts.spare_ram_bytes)} is spare. "
            "The collection was left unchanged.",
        )
    return ForceDecision(
        True,
        "Safe to optimize stuck segments. Qdrant will copy that segment and build its index. "
        "Points already stored stay.",
    )


def format_bytes(size: int | None) -> str:
    """A size a person can read. None means the measurement was missing."""
    if size is None:
        return "an unknown amount"
    n = max(0, int(size))
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f} GiB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.1f} MiB"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n} B"


def report_optimize(stage: str, detail: str) -> str:
    """Write one Optimize line to the Turbo log and return that line."""
    line = f"Optimize stuck segments: {stage}. {detail}".strip()
    log.info(line)
    return line


@dataclass(frozen=True)
class OptimizePause:
    """One indexing pause that is waiting on a segment rewrite.

    generation increases every time a pause starts, including when the same
    reason comes back after a gap. The bar uses that to start its clock over.
    """

    active: bool = False
    generation: int = 0
    reason: str = ""
    started: float | None = None


def track_optimize_pause(
    previous: OptimizePause | None,
    reason: str,
    now: float,
) -> OptimizePause:
    """Keep the current pause, or start a new one when indexing waits again."""
    previous = previous or OptimizePause()
    if reason in ("segment", "space"):
        if previous.active and previous.reason == reason:
            return previous
        return OptimizePause(True, previous.generation + 1, reason, now)
    if previous.active:
        return OptimizePause(False, previous.generation, "", now)
    return previous


def _elapsed_text(seconds: float) -> str:
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def pause_caption(pause: OptimizePause, now: float) -> tuple[str, float | None]:
    """Bar text, and a fraction. None means Qdrant published no percent, so pulse.

    A new pause starts its own clock. The fraction is never invented.
    """
    if pause.active:
        started = now if pause.started is None else pause.started
        elapsed = _elapsed_text(now - started)
        return (
            f"Pause {pause.generation} · {elapsed} · Qdrant has not reported a percent",
            None,
        )
    if pause.generation > 0:
        return ("This optimization pause finished. Indexing can write again.", 1.0)
    return ("Not waiting on an optimization.", 0.0)


def optimize_bar_after_click(ok: bool, requested: bool, reason: str) -> tuple[str, bool]:
    """Short repair-bar text, and whether the bar should pulse."""
    low = reason.lower()
    if "already in progress" in low:
        return ("Already optimizing", True)
    if ok:
        return ("Optimizing", True)
    if requested:
        return ("Optimize failed", False)
    return ("Not started", False)


def describe_optimizer_state(
    collection_status: str,
    optimizer_error: str,
    temp_open: bool,
) -> tuple[str, str, str, bool]:
    """code, bar text, detail, pulse.

    A build folder or a yellow collection is already running. Grey means
    Qdrant is waiting for an update. No percent is invented.
    """
    status = (collection_status or "").strip().lower()
    error = (optimizer_error or "").strip()
    if error and error.lower() != "ok":
        return (
            "error",
            "Optimizer error",
            "Qdrant reported an optimizer error: "
            f"{error}. Nothing was changed by this check. "
            "Do not restart Qdrant while a segment build is open.",
            False,
        )
    if temp_open:
        return (
            "running",
            "Already optimizing",
            "Optimization is already in progress. A segment build is open, so "
            "Optimize stuck segments will not start another one. "
            "Qdrant has not reported a percent for this segment. "
            "The bar moves only to show this is still the current job. "
            "Indexed points stay where they are until the segment is committed. "
            "Do not restart Qdrant.",
            True,
        )
    if status == "yellow":
        return (
            "pending",
            "Optimization still outstanding",
            "Qdrant marks the collection yellow, so some segments are not finished. "
            "No build folder is open right now. "
            "The Optimize button checks whether a rewrite is safe before it sends one. "
            "Nothing was changed by this check.",
            True,
        )
    if status == "grey":
        return (
            "waiting",
            "Waiting for an update",
            "Qdrant has optimizations waiting and will not run them until an update is sent. "
            "Optimize stuck segments sends that update when the copy fits in free disk and spare memory. "
            "Nothing was changed by this check.",
            False,
        )
    if status == "red":
        return (
            "error",
            "Collection needs attention",
            "Qdrant reports the collection is in an error state. Nothing was changed by this check.",
            False,
        )
    if status == "green":
        return (
            "idle",
            "Not optimizing",
            "Qdrant is not optimizing. Use Optimize stuck segments to rewrite an oversized "
            "or fully-deleted segment when that is safe.",
            False,
        )
    return (
        "unknown",
        "Optimization status unknown",
        "Qdrant did not say whether an optimization is running. Nothing was changed by this check.",
        False,
    )


def optimization_open(install: Install) -> bool:
    """True when Qdrant has a segment build folder open. This does not read segment bodies."""
    root = _collection_root(install)
    if not root.is_dir():
        return False
    try:
        shards = [path for path in root.iterdir() if path.is_dir()]
    except OSError:
        return False
    for shard in shards:
        if _dir_has_entry(shard / "temp_segments"):
            return True
    return False


def optimizer_settings(raw: object) -> dict:
    """Optimizer keys only. hnsw_config and on_disk never pass through."""
    if not isinstance(raw, dict):
        return {}
    return {key: raw[key] for key in OPTIMIZER_KEYS if key in raw}


def apply_force_optimization(facts: SegmentFacts, boundary) -> ForceResult:
    """Pause embedding, schedule one rewrite only when allowed, then restore and resume.

    On a refusal the collection is not read or patched. The schedule itself
    does not change optimizer settings: an empty optimizer update is what
    Qdrant uses to start a rewrite, and indexing_threshold 0 would disable
    indexing. Previous settings are written back even if that call raises.
    If the write-back fails, the result says so and does not claim they were
    restored. This never deletes the collection.
    """
    paused = False
    before: dict | None = None
    restore_needed = False
    outcome: ForceResult | None = None
    restore_error: Exception | None = None
    restored = False
    try:
        boundary.pause_embeds()
        paused = True
        decision = force_optimization(facts)
        if not decision.allow:
            outcome = ForceResult(False, False, decision.reason)
        else:
            before = optimizer_settings(boundary.read_optimizer_config())
            if "indexing_threshold" not in before:
                outcome = ForceResult(
                    False,
                    False,
                    "Optimizer settings could not be read. The collection was left unchanged.",
                )
            else:
                # Read succeeded, so a later failure still has settings to put back.
                restore_needed = True
                try:
                    boundary.request_optimize()
                except Exception as exc:
                    outcome = ForceResult(False, True, f"Optimization failed: {exc}.")
                else:
                    outcome = ForceResult(True, True, "Optimization requested.")
    except Exception as exc:
        outcome = ForceResult(False, False, f"Optimization failed: {exc}.")
    finally:
        if restore_needed and before is not None:
            try:
                boundary.patch_optimizer_config(before)
            except Exception as exc:
                restore_error = exc
                log.warning("could not restore optimizer settings", exc_info=True)
            else:
                restored = True
        if paused:
            try:
                boundary.resume_embeds()
            except Exception:
                log.warning("could not resume file embedding", exc_info=True)
    if outcome is None:
        outcome = ForceResult(False, False, "Optimization failed.")
    if restore_needed:
        if restored:
            outcome = ForceResult(
                outcome.ok,
                outcome.requested,
                f"{outcome.reason} Optimizer settings are restored. Points and the collection stay.",
            )
        else:
            detail = f": {restore_error}" if restore_error else ""
            outcome = ForceResult(
                False,
                outcome.requested,
                f"{outcome.reason} Optimizer settings could not be put back{detail}. "
                "Points and the collection stay.",
            )
    return outcome


def cached_write_facts(install: Install, now: float | None = None) -> SegmentFacts:
    """Plain-segment sizes for the write loop. Deleted bitsets are not counted here."""
    key = (install.storage_host, install.collection)
    moment = time.monotonic() if now is None else now
    with _cache_lock:
        hit = _write_cache.get(key)
        if hit is not None and moment - hit[0] < MEASURE_CACHE_SECONDS:
            return hit[1]
    facts = measure_write_facts(install)
    with _cache_lock:
        _write_cache[key] = (moment, facts)
    return facts


def clear_write_cache() -> None:
    with _cache_lock:
        _write_cache.clear()


def measure_write_facts(install: Install) -> SegmentFacts:
    return _measure(install, count_deleted=False)


def measure_repair_facts(install: Install) -> SegmentFacts:
    """Include fully-deleted segments. Used when the repair button is pressed."""
    return _measure(install, count_deleted=True)


def _measure(install: Install, *, count_deleted: bool) -> SegmentFacts:
    root = _collection_root(install)
    free = _free_bytes(root if root.exists() else Path(install.storage_host))
    ram = _spare_ram_bytes()
    if not root.is_dir():
        return SegmentFacts(free_bytes=free, spare_ram_bytes=ram, optimization_running=False)
    try:
        shards = [path for path in root.iterdir() if path.is_dir()]
    except OSError:
        return SegmentFacts(free_bytes=free, spare_ram_bytes=ram, unreadable=True)
    running = False
    best_vectors = -1
    best_disk = 0
    best_count = 0
    best_deleted = 0
    saw_segment = False
    try:
        for shard in shards:
            if _dir_has_entry(shard / "temp_segments"):
                running = True
            segment_root = shard / "segments"
            if not segment_root.is_dir():
                continue
            for segment in segment_root.iterdir():
                if not segment.is_dir():
                    continue
                saw_segment = True
                described = _describe_segment(segment, count_deleted=count_deleted)
                if described is None:
                    # A segment we cannot classify might be the one past the cap.
                    return SegmentFacts(
                        free_bytes=free,
                        spare_ram_bytes=ram,
                        optimization_running=running,
                        unreadable=True,
                    )
                kind, vector_bytes, disk_bytes, vector_count = described
                if kind == "deleted":
                    if disk_bytes > best_deleted:
                        best_deleted = disk_bytes
                    continue
                if kind == "plain" and vector_bytes > best_vectors:
                    best_vectors = vector_bytes
                    best_disk = disk_bytes
                    best_count = vector_count
    except OSError:
        return SegmentFacts(
            free_bytes=free,
            spare_ram_bytes=ram,
            optimization_running=running,
            unreadable=True,
        )
    if not saw_segment and not running and best_vectors < 0 and best_deleted == 0:
        # An empty shard is a collection with nothing to rewrite.
        pass
    if best_vectors < 0:
        best_vectors = 0
    return SegmentFacts(
        largest_plain_vector_bytes=best_vectors,
        largest_plain_disk_bytes=best_disk,
        largest_plain_vector_count=best_count,
        largest_deleted_disk_bytes=best_deleted,
        free_bytes=free,
        optimization_running=running,
        spare_ram_bytes=ram,
    )


def _collection_root(install: Install) -> Path:
    return Path(install.storage_host) / "qdrant" / "collections" / install.collection


def _free_bytes(path: Path) -> int | None:
    target = path if path.exists() else path.parent
    if not target.exists():
        return None
    try:
        return int(shutil.disk_usage(target).free)
    except OSError:
        return None


def _spare_ram_bytes() -> int | None:
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if not line.startswith("MemAvailable:"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1]) * 1024
    return None


def _dir_has_entry(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        with os.scandir(path) as entries:
            for _entry in entries:
                return True
    except OSError:
        return False
    return False


def _describe_segment(
    segment: Path, *, count_deleted: bool
) -> tuple[str, int, int, int] | None:
    """Return kind, vector bytes, disk bytes, and vector count. None when the segment is unreadable."""
    info = _segment_index_type(segment / "segment.json")
    if info is None:
        return None
    vector_bytes = _vector_bytes(segment / "vector_storage" / "vectors")
    disk_bytes = _tree_bytes(segment)
    vector_count = _vector_count(segment / "vector_storage" / "vectors" / "status.dat")
    if vector_count is None and vector_bytes > 0:
        vector_count = vector_bytes // 3072
    if vector_count is None:
        vector_count = 0
    fully_deleted = False
    if count_deleted and vector_count > 0:
        deleted = _popcount_file(segment / "id_tracker.deleted")
        fully_deleted = deleted is not None and deleted >= vector_count
    if fully_deleted:
        return "deleted", vector_bytes, disk_bytes, vector_count
    if info == "plain":
        return "plain", vector_bytes, disk_bytes, vector_count
    return "indexed", vector_bytes, disk_bytes, vector_count


def _segment_index_type(path: Path) -> str | None:
    """plain, hnsw, or other. None when the file cannot be read."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(data, dict):
        return None
    vector_data = (data.get("config") or {}).get("vector_data") if isinstance(data.get("config"), dict) else None
    if not isinstance(vector_data, dict) or not vector_data:
        return "plain"
    first = next(iter(vector_data.values()))
    if not isinstance(first, dict):
        return "plain"
    index = first.get("index") or {}
    if not isinstance(index, dict):
        return "plain"
    kind = str(index.get("type") or "plain").strip().lower()
    return kind or "plain"


def _vector_bytes(vectors: Path) -> int:
    if not vectors.is_dir():
        return 0
    total = 0
    try:
        with os.scandir(vectors) as entries:
            for entry in entries:
                if entry.name in ("status.dat", "config.json"):
                    continue
                try:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


def _vector_count(path: Path) -> int | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if len(raw) < 8:
        return None
    return int.from_bytes(raw[:8], "little")


def _popcount_file(path: Path) -> int | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    table = _POPCOUNT
    return sum(table[byte] for byte in raw)


def _tree_bytes(root: Path) -> int:
    total = 0
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


class LiveOptimizeBoundary:
    """Qdrant and the file-embeddings queue. No NOMAD image or compose file is changed.

    The queue is paused only when this action found it running, and resumed
    only then. A queue NOMAD or Turbo already paused stays paused.
    request_optimize schedules a rewrite with an empty optimizer update.
    It does not delete the collection or change indexing_threshold.
    """

    def __init__(self, install: Install) -> None:
        self.install = install
        self._resume_after = False

    def pause_embeds(self) -> None:
        from nomad_turbo.runtime import queue_is_paused, queuectl

        try:
            already = queue_is_paused(self.install)
        except Exception:
            log.warning("could not read the embed queue; leaving it untouched")
            self._resume_after = False
            return
        self._resume_after = not already
        if self._resume_after:
            queuectl(self.install, "pause")

    def resume_embeds(self) -> None:
        if not self._resume_after:
            return
        from nomad_turbo.runtime import queuectl

        queuectl(self.install, "resume")
        self._resume_after = False

    def read_optimizer_config(self) -> dict:
        from nomad_turbo.store import _request

        info = _request(self._collection_url(), timeout=30)
        config = (info.get("result") or {}).get("config") or {}
        if not isinstance(config, dict):
            return {}
        raw = config.get("optimizer_config") or config.get("optimizers_config") or {}
        return optimizer_settings(raw)

    def patch_optimizer_config(self, settings: dict) -> None:
        from nomad_turbo.store import _request

        clean = optimizer_settings(settings)
        _request(
            self._collection_url(),
            {"optimizers_config": clean},
            timeout=60,
            method="PATCH",
        )

    def request_optimize(self) -> None:
        """Start a rewrite without changing optimizer settings.

        Qdrant waits for an update before it runs optimizations that are
        pending. An empty optimizer diff is that update. indexing_threshold
        0 disables indexing, so this body does not send one. This is not a delete.
        """
        from nomad_turbo.store import _request

        _request(
            self._collection_url(),
            {"optimizers_config": {}},
            timeout=60,
            method="PATCH",
        )

    def _collection_url(self) -> str:
        return f"{self.install.qdrant_url.rstrip('/')}/collections/{self.install.collection}"
