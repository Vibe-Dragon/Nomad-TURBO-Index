"""Repair knowledge-base rows whose files are gone.

Quick repair compares the library on disk with ingest rows and checkpoints.
Complete repair also lists the sources stored in Qdrant. Points for a file
that is still on disk are reported and left in place. Random point ids make
a second copy unsafe to delete, and repair does not embed files again.

The embed queue stays paused for the whole repair. A delete that has already
been sent runs until Qdrant answers. Stop is honored between files.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass

from nomad_turbo.decisions import drop_should_wait, nomad_embed_guard, offsets_are_safe, policy_after_stop

log = logging.getLogger("nomad_turbo")

DELETE_TIMEOUT_FLOOR = 600
DELETE_TIMEOUT_CAP = 6 * 60 * 60
SCAN_SOURCE = ""

_REASON_LABELS = (
    ("ingest", "an ingest row"),
    ("checkpoint", "a checkpoint"),
    ("points", "index points"),
)


@dataclass(frozen=True)
class RepairItem:
    source: str
    name: str
    problem: str
    action: str
    points: int | None
    chunks: int | None
    detail: str


@dataclass(frozen=True)
class RepairRow:
    source: str
    name: str
    problem: str
    points: str
    percent: int
    pulse: int
    status: str
    elapsed: str
    detail: str
    color: str
    points_value: int = -1
    elapsed_seconds: int = -1


@dataclass(frozen=True)
class RepairSnap:
    phase: str
    note: str
    mode: str
    rows: tuple[RepairRow, ...]
    fraction: float
    overall: str
    busy: bool
    stats: str = ""


def format_repair_current(snap: RepairSnap) -> tuple[str, float, str]:
    """Title, bar fraction, and bar text for the file repair is on right now."""
    active = next((row for row in snap.rows if row.pulse >= 0), None)
    if active is not None:
        bits = [f"{active.percent}%", active.status or "working"]
        if active.points and active.points != "unknown":
            bits.append(active.points)
        if active.elapsed:
            bits.append(active.elapsed)
        title = active.name or "Current file"
        return title, max(0.0, min(1.0, active.percent / 100.0)), " · ".join(bits)
    if snap.busy or snap.phase in ("scanning", "locking", "repairing"):
        return "Working through the list.", max(0.0, min(1.0, snap.fraction)), snap.overall or "Working"
    return "No file is being repaired.", max(0.0, min(1.0, snap.fraction)), snap.overall or "Idle"


def is_library_source(source: str, container_root: str) -> bool:
    """True for a ZIM or an upload. Other paths, including NOMAD's own docs, stay."""
    root = container_root.rstrip("/")
    return source.startswith(root + "/zim/") or source.startswith(root + "/kb_uploads/")


def problem_text(reasons: set[str]) -> str:
    labels = [label for key, label in _REASON_LABELS if key in reasons]
    if not labels:
        return "File is gone"
    if len(labels) == 1:
        verb = "remain" if labels[0] == "index points" else "remains"
        return f"File is gone and {labels[0]} {verb}"
    head = labels[0][0].upper() + labels[0][1:]
    if len(labels) == 2:
        return f"File is gone. {head} and {labels[1]} remain"
    return f"File is gone. {head}, {labels[1]}, and {labels[2]} remain"


def delete_detail(
    state: str,
    chunks: int | None,
    points: int | None,
    articles: int | None = None,
    done: bool | None = None,
) -> str:
    parts = ["The file is not on disk."]
    if state:
        parts.append(f"Ingest state {state}.")
    if chunks is not None:
        parts.append(f"{chunks:,} chunks recorded.")
    if points is not None:
        parts.append(f"{points:,} points in the index.")
    if articles is not None:
        where = "finished" if done else "not finished"
        parts.append(f"Checkpoint at article {articles:,}, {where}.")
    parts.append("Repair deletes those points, then removes the ingest row and checkpoint.")
    return " ".join(parts)


def library_counts(paths: set[str]) -> tuple[int, int]:
    """ZIM files and uploads among the container paths on disk."""
    zim = sum(1 for path in paths if "/zim/" in path)
    return zim, len(paths) - zim


def ingest_summary(ingest: dict[str, tuple[str, int]]) -> str:
    counts: dict[str, int] = {}
    for state, _chunks in ingest.values():
        label = state or "unknown"
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "none"
    return ", ".join(f"{counts[key]:,} {key}" for key in sorted(counts))


def checkpoint_counts(checkpoints: list[dict]) -> tuple[int, int, int]:
    """Total, still open, and finished checkpoints."""
    done = 0
    for item in checkpoints:
        if item.get("done") is True:
            done += 1
    total = len(checkpoints)
    return total, total - done, done


def _count(value: int | None, unknown: str = "…") -> str:
    if value is None:
        return unknown
    return f"{value:,}"


def format_scan_stats(elapsed: str = "", **facts: object) -> str:
    """The live numbers under the repair progress bar."""
    stage = str(facts.get("stage") or "Idle")
    lines = [f"Stage: {stage}"]
    if elapsed:
        lines.append(f"Elapsed: {elapsed}")
    library = facts.get("library_files")
    if library is not None:
        zim = facts.get("zim_files")
        uploads = facts.get("upload_files")
        line = f"Library on disk: {_count(library if isinstance(library, int) else None)}"
        if isinstance(zim, int) and isinstance(uploads, int):
            line += f" ({zim:,} ZIM, {uploads:,} uploads)"
        lines.append(line)
    else:
        lines.append("Library on disk: …")
    ingest_rows = facts.get("ingest_rows")
    states = facts.get("ingest_states")
    if isinstance(ingest_rows, int):
        line = f"Ingest rows: {ingest_rows:,}"
        if isinstance(states, str) and states:
            line += f" ({states})"
        lines.append(line)
    else:
        lines.append("Ingest rows: …")
    checkpoints = facts.get("checkpoints")
    if isinstance(checkpoints, int):
        line = f"Checkpoints: {checkpoints:,}"
        opened = facts.get("checkpoints_open")
        finished = facts.get("checkpoints_done")
        if isinstance(opened, int) and isinstance(finished, int):
            line += f" ({opened:,} still open, {finished:,} finished)"
        lines.append(line)
    else:
        lines.append("Checkpoints: …")
    sources = facts.get("index_sources")
    if sources == "skipped":
        lines.append("Index sources: not read in a quick scan")
    elif sources == "unavailable":
        lines.append("Index sources: the index did not answer")
    elif isinstance(sources, int):
        points = facts.get("index_points")
        line = f"Index sources: {sources:,}"
        if isinstance(points, int):
            line += f" ({points:,} points)"
        lines.append(line)
    else:
        lines.append("Index sources: …")
    would_delete = facts.get("would_delete")
    would_report = facts.get("would_report")
    lines.append(
        "Would delete: "
        + (_count(would_delete) if isinstance(would_delete, int) else "…")
    )
    lines.append(
        "Report only: "
        + (_count(would_report) if isinstance(would_report, int) else "…")
    )
    index_state = facts.get("index_state")
    if isinstance(index_state, str) and index_state:
        lines.append(f"Index: {index_state}")
    download = facts.get("download")
    if isinstance(download, str) and download:
        lines.append(download)
    return "\n".join(lines)


def delete_timeout(points: int | None, chunks: int | None) -> int:
    """How long to wait for one file's delete. Large files get longer, up to six hours."""
    count = points if points is not None else chunks
    if count is None or count <= 0:
        return DELETE_TIMEOUT_FLOOR
    return int(max(DELETE_TIMEOUT_FLOOR, min(DELETE_TIMEOUT_CAP, count * 0.5)))


def delete_may_still_be_running(message: str) -> bool:
    """A timed-out delete can still be running inside Qdrant. Do not start another.

    A connection that never reached Qdrant is not an in-flight delete.
    """
    text = message.lower()
    if "connection refused" in text or "failed to establish" in text or "name or service not known" in text:
        return False
    return "timed out" in text or "timeout" in text or "unreachable" in text


def restore_policy(previous: str | None, stuck: bool) -> str | None:
    """Policy to write when the repair releases the index. None means it was never taken."""
    if previous is None:
        return None
    if stuck:
        return "Manual"
    return policy_after_stop(previous, False)


def should_resume(paused: bool, stuck: bool) -> bool:
    """Resume the embed queue only after a repair that is sure its last delete finished."""
    return paused and not stuck


def format_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def points_label(points: int | None, chunks: int | None) -> str:
    if points is not None:
        return f"{points:,}"
    if chunks is not None:
        return f"{chunks:,} chunks"
    return "unknown"


def _file_name(source: str) -> str:
    name = source.rsplit("/", 1)[-1]
    return name or source


def _note(
    source: str,
    problem: str,
    detail: str,
    points: int | None = None,
    chunks: int | None = None,
) -> RepairItem:
    return RepairItem(
        source=source,
        name=_file_name(source) if source else "Index listing",
        problem=problem,
        action="note",
        points=points,
        chunks=chunks,
        detail=detail,
    )


class _Gone:
    def __init__(self, source: str) -> None:
        self.source = source
        self.reasons: set[str] = set()
        self.state = ""
        self.chunks: int | None = None
        self.points: int | None = None
        self.articles: int | None = None
        self.done: bool | None = None

    def item(self) -> RepairItem:
        return RepairItem(
            source=self.source,
            name=_file_name(self.source),
            problem=problem_text(self.reasons),
            action="delete",
            points=self.points,
            chunks=self.chunks,
            detail=delete_detail(self.state, self.chunks, self.points, self.articles, self.done),
        )


def plan_repair(
    *,
    mode: str,
    on_disk: set[str],
    ingest: dict[str, tuple[str, int]],
    checkpoints: list[dict],
    qdrant_sources: dict[str, int] | None,
    container_root: str = "/app/storage",
) -> list[RepairItem]:
    """Files to delete, then files to report. A file still on disk is never deleted."""
    if mode not in ("quick", "complete"):
        raise ValueError(f"unknown repair mode {mode}")
    gone: dict[str, _Gone] = {}

    def mark(
        source: str,
        reason: str,
        *,
        state: str = "",
        chunks: int | None = None,
        points: int | None = None,
        articles: int | None = None,
        done: bool | None = None,
    ) -> None:
        if not source or not is_library_source(source, container_root) or source in on_disk:
            return
        row = gone.get(source)
        if row is None:
            row = _Gone(source)
            gone[source] = row
        row.reasons.add(reason)
        if state and not row.state:
            row.state = state
        if chunks is not None and row.chunks is None:
            row.chunks = int(chunks)
        if points is not None:
            row.points = int(points)
        if isinstance(articles, int) and row.articles is None:
            row.articles = articles
        if done is not None and row.done is None:
            row.done = done

    for source, (state, chunks) in ingest.items():
        mark(source, "ingest", state=state, chunks=chunks)
    for checkpoint in checkpoints:
        source = str(checkpoint.get("source") or "")
        raw = checkpoint.get("chunks")
        chunks = raw if isinstance(raw, int) else None
        articles = checkpoint.get("articles") if isinstance(checkpoint.get("articles"), int) else None
        done = checkpoint.get("done") if isinstance(checkpoint.get("done"), bool) else None
        mark(source, "checkpoint", chunks=chunks, articles=articles, done=done)

    notes: dict[str, RepairItem] = {}
    if mode == "complete" and qdrant_sources is None:
        notes[SCAN_SOURCE] = _note(
            SCAN_SOURCE,
            "Qdrant did not list its sources",
            "The source list did not return. Files already known to be gone are still repaired.",
        )
    if mode == "complete" and qdrant_sources is not None:
        for source, count in qdrant_sources.items():
            count = int(count)
            if count <= 0 or not source:
                continue
            if not is_library_source(source, container_root):
                notes[source] = _note(
                    source,
                    "Points sit outside the ZIM and upload folders",
                    f"{count:,} points are stored for {source}. Repair leaves them in place.",
                    points=count,
                )
                continue
            if source not in on_disk:
                mark(source, "points", points=count)
                continue
            recorded = ingest.get(source)
            if recorded is None:
                notes[source] = _note(
                    source,
                    "File is on disk with points and no ingest row",
                    f"The file is on disk and the index has {count:,} points. "
                    "There is no ingest row. Those points stay, and the file is not embedded again.",
                    points=count,
                )
                continue
            state, chunks = recorded
            if chunks != count:
                notes[source] = _note(
                    source,
                    "Point count does not match the ingest row",
                    f"The file is on disk. Ingest state {state}, {chunks:,} chunks recorded, "
                    f"{count:,} points in the index. Those points stay, and the file is not embedded again.",
                    points=count,
                    chunks=chunks,
                )
        for source, (state, chunks) in ingest.items():
            if source in notes or source in gone or source not in on_disk:
                continue
            if not is_library_source(source, container_root):
                continue
            if state == "indexed" and chunks > 0 and source not in qdrant_sources:
                notes[source] = _note(
                    source,
                    "Ingest row says indexed but no points were found",
                    f"The file is on disk. Ingest state indexed, {chunks:,} chunks recorded. "
                    "The source list has no points for it. The row stays, and the file is not embedded again.",
                    chunks=chunks,
                )

    deletes = [row.item() for row in gone.values()]
    deletes.sort(key=lambda item: (item.points if item.points is not None else item.chunks if item.chunks is not None else 10**15, item.name))
    noted = [notes[key] for key in sorted(notes, key=lambda key: (key != SCAN_SOURCE, notes[key].name))]
    return deletes + noted


_COLORS = {
    "Waiting": "#4b5563",
    "Deleting points": "#1d4ed8",
    "Waiting for a download to finish": "#a16207",
    "Waiting for a download to settle": "#a16207",
    "Waiting for the disk": "#a16207",
    "Clearing records": "#1d4ed8",
    "Repaired": "#15803d",
    "Failed": "#b91c1c",
    "Noted": "#a16207",
    "Stopped": "#4b5563",
    "Would delete": "#b45309",
    "Report only": "#1d4ed8",
}
_PULSE = {
    "Deleting points",
    "Waiting for a download to finish",
    "Waiting for a download to settle",
    "Waiting for the disk",
    "Clearing records",
}
_FINISHED = {"Repaired", "Noted", "Failed", "Stopped", "Would delete", "Report only"}
_SCAN_DONE = {"Would delete", "Report only"}


class _Job:
    def __init__(self, item: RepairItem) -> None:
        self.source = item.source
        self.name = item.name
        self.problem = item.problem
        self.action = item.action
        self.points = item.points
        self.chunks = item.chunks
        self.base_detail = item.detail
        self.status = "Waiting"
        self.percent = 0
        self.started: float | None = None
        self.finished: float | None = None

    def row(self, now: float) -> RepairRow:
        if self.finished is not None and self.started is not None:
            elapsed = self.finished - self.started
        elif self.started is not None:
            elapsed = max(0.0, now - self.started)
        else:
            elapsed = None
        elapsed_text = format_elapsed(elapsed)
        detail = self.base_detail
        if elapsed_text and self.status != "Noted":
            detail = f"{detail} Elapsed {elapsed_text}."
        pulse = int(elapsed * 2) % 24 if elapsed is not None and self.status in _PULSE else -1
        percent = 100 if self.status in ("Repaired", "Noted", "Would delete", "Report only") else self.percent
        if self.points is not None:
            points_value = int(self.points)
        elif self.chunks is not None:
            points_value = int(self.chunks)
        else:
            points_value = -1
        return RepairRow(
            source=self.source,
            name=self.name,
            problem=self.problem,
            points=points_label(self.points, self.chunks),
            percent=percent,
            pulse=pulse,
            status=self.status,
            elapsed=elapsed_text,
            detail=detail,
            color=_COLORS.get(self.status, "#111827"),
            points_value=points_value,
            elapsed_seconds=-1 if elapsed is None else int(elapsed),
        )


class RepairBoard:
    """Worker writes this. The window reads a snapshot on the GTK timer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.phase = "idle"
        self.note = ""
        self.mode = "quick"
        self.kind = "repair"
        self.jobs: list[_Job] = []
        self.started: float | None = None
        self.stage_fraction = 0.0
        self.facts: dict[str, object] = {}

    def set_phase(self, phase: str, note: str = "") -> None:
        with self._lock:
            self.phase = phase
            if note:
                self.note = note
            if self.started is None and phase != "idle":
                self.started = time.monotonic()

    def set_facts(self, **facts: object) -> None:
        with self._lock:
            self.facts.update(facts)
            stage = facts.get("stage")
            fraction = facts.get("fraction")
            if isinstance(stage, str) and stage:
                self.note = stage
            if isinstance(fraction, (int, float)):
                self.stage_fraction = float(fraction)

    def set_jobs(self, items: list[RepairItem]) -> None:
        with self._lock:
            self.jobs = [_Job(item) for item in items]

    def update(self, source: str, **fields: object) -> None:
        with self._lock:
            for job in self.jobs:
                if job.source != source:
                    continue
                for key, value in fields.items():
                    setattr(job, key, value)
                return

    def stop_waiting(self, now: float) -> None:
        with self._lock:
            for job in self.jobs:
                if job.status == "Waiting":
                    job.status = "Stopped"
                    job.finished = now

    def snapshot(self, now: float | None = None) -> RepairSnap:
        moment = time.monotonic() if now is None else now
        with self._lock:
            rows = tuple(job.row(moment) for job in self.jobs)
            phase = self.phase
            note = self.note
            mode = self.mode
            kind = self.kind
            started = self.started
            stage_fraction = self.stage_fraction
            facts = dict(self.facts)
        total = len(rows)
        finished = sum(1 for row in rows if row.status in _FINISHED)
        working = next((row for row in rows if row.status not in _FINISHED and row.status != "Waiting"), None)
        elapsed = format_elapsed(None if started is None else moment - started)
        if kind == "scan" and phase == "scanning":
            fraction = stage_fraction
            overall = note or "Scanning"
            if elapsed:
                overall += f" · {elapsed}"
        elif kind == "scan" and phase in ("done", "stopped"):
            fraction = 1.0
            deletes = sum(1 for row in rows if row.status == "Would delete")
            reports = sum(1 for row in rows if row.status == "Report only")
            if total == 0:
                overall = "Nothing is broken" if phase == "done" else "Scan stopped"
            else:
                overall = f"{deletes} would be deleted · {reports} report only"
            if elapsed:
                overall += f" · {elapsed}"
        elif total:
            fraction = finished / total
            if working is not None:
                fraction = min(0.99, (finished + 0.15) / total)
            overall = f"{finished} of {total} done"
            if working is not None:
                overall += f" · {working.name}"
            if elapsed:
                overall += f" · {elapsed}"
        elif phase == "done":
            fraction = 1.0
            overall = "Nothing to repair"
        elif phase in ("locking", "scanning"):
            fraction = stage_fraction
            overall = note or "Working"
            if elapsed:
                overall += f" · {elapsed}"
        else:
            fraction = 0.0
            overall = "Idle"
        stats = format_scan_stats(elapsed, **facts) if facts else ""
        if rows:
            summary = f"Rows listed: {total:,}"
            working_now = sum(1 for row in rows if row.pulse >= 0)
            if working_now:
                summary += f" · in progress: {working_now}"
            known_points = sum(row.points_value for row in rows if row.points_value > 0)
            if known_points:
                summary += f" · {known_points:,} points across these rows"
            stats = summary if not stats else summary + "\n" + stats
        return RepairSnap(
            phase=phase,
            note=note,
            mode=mode,
            rows=rows,
            fraction=fraction,
            overall=overall,
            busy=phase in ("locking", "scanning", "repairing"),
            stats=stats,
        )


def _download_note(reason: str) -> str:
    if reason == "download":
        return "A download is in progress. This picture can change when it finishes."
    if reason == "settle":
        return "A download just finished. Rows may still be changing."
    if reason == "disk":
        return "The disk is busy. Reads can be slow."
    return ""


def _scan_facts(board: RepairBoard, **facts: object) -> None:
    board.set_facts(**facts)


def run_scan(stop: threading.Event, board: RepairBoard, mode: str, deps) -> None:
    """Read the library and the index. Change nothing.

    A complete scan lists Qdrant sources only when the index is answering.
    Stop is checked between steps. The source list itself runs until it returns.
    """
    if mode not in ("quick", "complete"):
        board.set_phase("error", "Unknown repair mode.")
        return
    with board._lock:
        board.kind = "scan"
        board.mode = mode
    board.set_phase("scanning", "Checking whether the index is answering.")
    facts: dict[str, object] = {
        "stage": "Checking whether the index is answering.",
        "fraction": 0.05,
        "would_delete": None,
        "would_report": None,
    }
    try:
        try:
            ready = bool(deps.qdrant_up())
        except Exception:
            log.exception("scan could not reach the index")
            ready = False
        facts["index_state"] = "answering" if ready else "not answering"
        try:
            facts["download"] = _download_note(deps.hold_reason())
        except Exception:
            log.exception("scan could not check downloads")
        if stop.is_set():
            _scan_facts(board, **facts)
            board.set_phase("stopped", "Scan stopped before it read the library.")
            return
        facts["stage"] = "Reading the library on disk."
        facts["fraction"] = 0.2
        _scan_facts(board, **facts)
        on_disk = set(deps.on_disk())
        zim, uploads = library_counts(on_disk)
        facts["library_files"] = len(on_disk)
        facts["zim_files"] = zim
        facts["upload_files"] = uploads
        if stop.is_set():
            _scan_facts(board, **facts)
            board.set_phase("stopped", "Scan stopped.")
            return
        facts["stage"] = "Reading ingest rows."
        facts["fraction"] = 0.4
        _scan_facts(board, **facts)
        ingest = deps.ingest()
        facts["ingest_rows"] = len(ingest)
        facts["ingest_states"] = ingest_summary(ingest)
        if stop.is_set():
            _scan_facts(board, **facts)
            board.set_phase("stopped", "Scan stopped.")
            return
        facts["stage"] = "Reading checkpoints."
        facts["fraction"] = 0.6
        _scan_facts(board, **facts)
        checkpoints = list(deps.checkpoints())
        total, opened, done = checkpoint_counts(checkpoints)
        facts["checkpoints"] = total
        facts["checkpoints_open"] = opened
        facts["checkpoints_done"] = done
        listed = None
        if mode == "quick":
            facts["index_sources"] = "skipped"
            facts["stage"] = "Comparing files, ingest rows, and checkpoints."
            facts["fraction"] = 0.85
            _scan_facts(board, **facts)
        elif not ready:
            facts["index_sources"] = "unavailable"
            facts["stage"] = "The index is not answering. Comparing files, rows, and checkpoints."
            facts["fraction"] = 0.85
            _scan_facts(board, **facts)
        else:
            facts["stage"] = "Listing sources in the index. This finishes when Qdrant answers."
            facts["fraction"] = 0.72
            _scan_facts(board, **facts)
            listed = deps.facet_sources()
            if listed is None:
                facts["index_sources"] = "unavailable"
                facts["index_state"] = "the source list did not return"
            else:
                facts["index_sources"] = len(listed)
                facts["index_points"] = sum(int(count) for count in listed.values())
                facts["index_state"] = "source list returned"
            facts["stage"] = "Comparing the index with the library."
            facts["fraction"] = 0.9
            _scan_facts(board, **facts)
        if stop.is_set() and listed is None and mode == "complete" and ready:
            _scan_facts(board, **facts)
            board.set_phase("stopped", "Scan stopped.")
            return
        items = plan_repair(
            mode=mode,
            on_disk=on_disk,
            ingest=ingest,
            checkpoints=checkpoints,
            qdrant_sources=listed,
            container_root=deps.container_root,
        )
        board.set_jobs(items)
        now = time.monotonic()
        for item in items:
            status = "Would delete" if item.action == "delete" else "Report only"
            board.update(item.source, status=status, percent=100, started=now, finished=now)
        deletes = sum(1 for item in items if item.action == "delete")
        reports = len(items) - deletes
        facts["would_delete"] = deletes
        facts["would_report"] = reports
        facts["fraction"] = 1.0
        if not items:
            facts["stage"] = "Nothing is broken."
            note = "Nothing is broken. NOMAD was not changed."
        else:
            facts["stage"] = "Scan finished. Nothing was changed."
            note = (
                f"Scan finished. {deletes} file(s) would be deleted. "
                f"{reports} would be reported and left in place. Nothing was changed."
            )
        _scan_facts(board, **facts)
        if stop.is_set():
            board.set_phase("stopped", "Scan stopped. " + note)
            return
        board.set_phase("done", note)
        log.info("scan found %s file(s) to delete and %s note(s)", deletes, reports)
    except Exception as exc:
        log.exception("scan failed")
        board.set_phase("error", str(exc))


def _hold_status(reason: str) -> str:
    if reason == "download":
        return "Waiting for a download to finish"
    if reason == "settle":
        return "Waiting for a download to settle"
    return "Waiting for the disk"


def wait_for_clearance(stop: threading.Event, hold_reason, sleeper, now=None, on_reason=None) -> str:
    """Block deletes while a download is settling. Return '' or 'stopped'."""
    clock = time.monotonic if now is None else now
    started = clock()
    announced = ""
    while True:
        reason = hold_reason()
        if not drop_should_wait(reason, clock() - started):
            return ""
        if on_reason is not None and reason != announced:
            announced = reason
            on_reason(reason)
        if stop.is_set():
            return "stopped"
        if sleeper(5):
            return "stopped"


def _guard(deps, stop: threading.Event) -> None:
    if stop.is_set():
        return
    try:
        policy = deps.get_policy()
        paused = deps.queue_is_paused()
    except Exception:
        log.exception("could not check the embed queue during repair")
        return
    corrected, must_pause = nomad_embed_guard(policy, paused)
    if corrected:
        log.warning("ingest policy was %s during repair. Setting it back to Manual.", policy)
        try:
            deps.set_policy(corrected)
        except Exception:
            log.exception("could not keep ingest policy on Manual during repair")
    if must_pause and not stop.is_set():
        log.warning("the embed queue was turned back on during repair. Pausing it again.")
        try:
            deps.pause_queue()
        except Exception:
            log.exception("could not keep the embed queue paused during repair")


def _release(previous: str | None, paused: bool, stuck: bool, deps) -> None:
    policy = restore_policy(previous, stuck)
    if policy is not None:
        try:
            deps.set_policy(policy)
            if stuck:
                log.warning("repair left ingest policy on Manual because a delete did not return")
            else:
                log.info("ingest policy restored to %s, which is how repair found it.", policy)
        except Exception:
            log.exception("could not restore the ingest policy after repair")
    if should_resume(paused, stuck):
        try:
            deps.resume_queue()
            log.info("file-embeddings queue resumed")
        except Exception:
            log.exception("could not resume file-embeddings after repair")
    try:
        deps.write_status(
            phase="off",
            paused_queue=bool(paused and stuck),
            repair_hold=bool(stuck),
            slots=0,
        )
    except Exception:
        log.exception("could not write repair status")


def run_repair(stop: threading.Event, board: RepairBoard, mode: str, deps, sleeper=None) -> None:
    """Lock NOMAD embedding, repair one file at a time, then unlock.

    sleeper(seconds) returns True when the repair should stop. Tests pass their own.
    """
    if mode not in ("quick", "complete"):
        board.set_phase("error", "Unknown repair mode.")
        return
    try:
        index_up = bool(deps.qdrant_up())
    except Exception as exc:
        log.exception("repair could not reach the index")
        board.set_phase("error", f"The index is not answering. {exc}")
        return
    if not index_up:
        board.set_phase("error", "The index is not answering. Start repair when Qdrant is up.")
        return
    pause_for = sleeper or stop.wait
    with board._lock:
        board.mode = mode
    previous = None
    paused = False
    stuck = False
    try:
        if deps.repair_hold():
            log.warning("starting repair while a previous delete may still be running")
        board.set_phase("locking", "Pausing NOMAD file embedding until this repair finishes.")
        previous = deps.get_policy()
        deps.remember_policy(previous)
        deps.set_policy("Manual")
        deps.pause_queue()
        paused = True
        deps.write_status(phase="repairing", paused_queue=True, repair_hold=False, slots=0)
        while not stop.is_set():
            active = deps.active_embeds()
            if offsets_are_safe(active):
                break
            board.set_phase("locking", f"Waiting for {active} embed job(s) to finish.")
            _guard(deps, stop)
            if pause_for(5):
                break
        if stop.is_set():
            board.set_phase("stopped", "Stopped before the repair changed the index.")
            return
        board.set_phase("scanning", "Waiting until downloads and the disk are clear.")
        if wait_for_clearance(
            stop,
            deps.hold_reason,
            pause_for,
            on_reason=lambda reason: board.set_phase("scanning", _hold_status(reason) + "."),
        ) == "stopped":
            board.set_phase("stopped", "Stopped while waiting to read the index.")
            return
        board.set_phase("scanning", "Reading the library, ingest rows, and checkpoints.")
        on_disk = deps.on_disk()
        ingest = deps.ingest()
        checkpoints = deps.checkpoints()
        listed = None
        if mode == "complete":
            board.set_phase("scanning", "Listing sources in the index.")
            listed = deps.facet_sources()
            if listed is None:
                board.set_phase("scanning", "The source list did not return. Repairing files already known to be gone.")
        items = plan_repair(
            mode=mode,
            on_disk=on_disk,
            ingest=ingest,
            checkpoints=checkpoints,
            qdrant_sources=listed,
            container_root=deps.container_root,
        )
        deletes = sum(1 for item in items if item.action == "delete")
        notes = len(items) - deletes
        log.info("repair plan has %s file(s) to delete and %s note(s)", deletes, notes)
        board.set_jobs(items)
        if not items:
            board.set_phase("done", "Nothing to repair.")
            return
        board.set_phase("repairing", f"{deletes} file(s) to repair. {notes} left as they are.")
        for item in items:
            if stop.is_set():
                board.stop_waiting(time.monotonic())
                board.set_phase("stopped", "Stopped between files.")
                return
            if item.action != "delete":
                now = time.monotonic()
                board.update(item.source, status="Noted", percent=100, started=now, finished=now)
                continue
            holding = {"started": False}

            def on_reason(reason: str, source=item.source) -> None:
                fields: dict = {"status": _hold_status(reason), "percent": 5}
                if not holding["started"]:
                    fields["started"] = time.monotonic()
                    holding["started"] = True
                board.update(source, **fields)
                board.set_phase("repairing", _hold_status(reason) + ".")

            if wait_for_clearance(stop, deps.hold_reason, pause_for, on_reason=on_reason) == "stopped":
                board.stop_waiting(time.monotonic())
                board.set_phase("stopped", "Stopped while waiting for the disk.")
                return
            _guard(deps, stop)
            if stop.is_set():
                board.stop_waiting(time.monotonic())
                board.set_phase("stopped", "Stopped between files.")
                return
            board.update(item.source, status="Deleting points", percent=15, started=time.monotonic())
            log.info("repair deleting points for %s", item.name)
            try:
                deps.delete_points(item.source, delete_timeout(item.points, item.chunks))
            except Exception as exc:
                message = str(exc)
                log.exception("repair could not delete points for %s", item.source)
                board.update(item.source, status="Failed", percent=15, finished=time.monotonic())
                if delete_may_still_be_running(message):
                    stuck = True
                    board.stop_waiting(time.monotonic())
                    board.set_phase(
                        "error",
                        "That delete did not return. The embed queue stays paused so another delete is not started. "
                        "When the index is idle, start repair again.",
                    )
                    return
                continue
            board.update(item.source, status="Clearing records", percent=90)
            cleared = True
            for step, label in (
                (deps.forget_ingest, "ingest row"),
                (deps.drop_checkpoint, "checkpoint"),
                (deps.forget_jobs, "embed jobs"),
            ):
                try:
                    step(item.source)
                except Exception:
                    cleared = False
                    log.exception("repair deleted points but could not clear the %s for %s", label, item.source)
            if not cleared:
                board.update(item.source, status="Failed", percent=90, finished=time.monotonic())
                continue
            board.update(item.source, status="Repaired", percent=100, finished=time.monotonic())
            log.info("repaired %s", item.name)
        board.set_phase("done", "Repair finished. NOMAD embedding is unlocked.")
    except Exception as exc:
        log.exception("repair failed")
        board.set_phase("error", str(exc))
    finally:
        if previous is not None or paused:
            _release(previous, paused, stuck, deps)


@dataclass
class RepairDeps:
    get_policy: object
    set_policy: object
    remember_policy: object
    pause_queue: object
    resume_queue: object
    active_embeds: object
    queue_is_paused: object
    on_disk: object
    ingest: object
    checkpoints: object
    facet_sources: object
    delete_points: object
    forget_ingest: object
    drop_checkpoint: object
    forget_jobs: object
    hold_reason: object
    write_status: object
    repair_hold: object
    qdrant_up: object
    container_root: str = "/app/storage"


def live_deps(install) -> RepairDeps:
    """Talk to the NOMAD containers. Used by the window, not by the tests."""
    from nomad_turbo.discover import state_dir
    from nomad_turbo.engine import _checkpoint_file, _forget_source_jobs, _library_hold, _sources
    from nomad_turbo.engine import _status
    from nomad_turbo.runtime import active_embeds, queue_is_paused, queuectl
    from nomad_turbo.store import (
        _health_ok,
        delete_source_points,
        facet_sources,
        forget_ingest,
        get_policy,
        ingest_details,
        set_policy,
    )

    def remember_policy(previous: str) -> None:
        path = state_dir() / "policy"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(previous)

    def on_disk() -> set[str]:
        return {source for _host, _kind, source in _sources(install, include_fresh=True)}

    def checkpoints() -> list[dict]:
        folder = state_dir() / "checkpoints"
        found: list[dict] = []
        if not folder.is_dir():
            return found
        for path in sorted(folder.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                found.append(data)
        return found

    def drop_checkpoint(source: str) -> None:
        path = _checkpoint_file(source)
        if path.exists():
            path.unlink()

    def held() -> bool:
        path = state_dir() / "status.json"
        if not path.exists():
            return False
        try:
            return bool(json.loads(path.read_text()).get("repair_hold"))
        except (OSError, json.JSONDecodeError):
            return False

    return RepairDeps(
        get_policy=lambda: get_policy(install),
        set_policy=lambda policy: set_policy(install, policy),
        remember_policy=remember_policy,
        pause_queue=lambda: queuectl(install, "pause"),
        resume_queue=lambda: queuectl(install, "resume"),
        active_embeds=lambda: active_embeds(install),
        queue_is_paused=lambda: queue_is_paused(install),
        on_disk=on_disk,
        ingest=lambda: ingest_details(install),
        checkpoints=checkpoints,
        facet_sources=lambda: facet_sources(install),
        delete_points=lambda source, timeout: delete_source_points(install, source, timeout=timeout),
        forget_ingest=lambda source: forget_ingest(install, source),
        drop_checkpoint=drop_checkpoint,
        forget_jobs=lambda source: _forget_source_jobs(install, source),
        hold_reason=lambda: _library_hold(install),
        write_status=_status,
        repair_hold=held,
        qdrant_up=lambda: _health_ok(install.qdrant_url.rstrip("/")),
        container_root=install.container_storage,
    )


