"""Read-only picture of the library. This does not pause NOMAD or start containers."""

from __future__ import annotations

from dataclasses import dataclass

from nomad_turbo.catalog import CatalogRow, catalog_state, origin_for
from nomad_turbo.decisions import choose_resume
from nomad_turbo.discover import Install


@dataclass
class ScanResult:
    rows: list[CatalogRow]
    active_jobs: int
    downloads: bool


def scan_library(install: Install) -> ScanResult:
    from nomad_turbo.engine import _indexable, _read_checkpoint, _walk_library
    from nomad_turbo.runtime import active_embeds, list_jobs
    from nomad_turbo.store import ingest_details

    snap = _walk_library(install)
    jobs = list_jobs(install)
    grouped: dict[str, list[dict]] = {}
    for job in jobs:
        data = job.get("data") or {}
        source = data.get("filePath")
        if source:
            grouped.setdefault(source, []).append(data)
    details = ingest_details(install)
    rows: list[CatalogRow] = []
    for walk_index, item in enumerate(_indexable(snap)):
        host, kind, source = item.host, item.kind, item.source
        entries = grouped.get(source, [])
        checkpoint = _read_checkpoint(source) or {}
        offsets = [int(entry.get("batchOffset") or 0) for entry in entries]
        resume = choose_resume(offsets, checkpoint.get("articles"))
        total = None
        for entry in entries:
            if entry.get("totalArticles"):
                total = int(entry["totalArticles"])
        if checkpoint.get("total_articles"):
            total = int(checkpoint["total_articles"])
        ingest_state, ingest_chunks = details.get(source, (None, 0))
        state = catalog_state(
            ingest_state=ingest_state,
            resume_articles=resume,
            checkpoint_done=bool(checkpoint.get("done")),
            has_continuation=bool(entries),
        )
        chunks = max(int(checkpoint.get("chunks") or 0), int(ingest_chunks or 0))
        size = item.size
        mtime = item.mtime
        rows.append(
            CatalogRow(
                source=source,
                name=host.name,
                origin=origin_for(source),
                kind=kind,
                size_bytes=size,
                state=state,
                articles=0 if state == "indexed" else resume,
                total_articles=total,
                chunks=chunks,
                mtime=mtime,
                walk_index=walk_index,
            )
        )
    order = {"running": 0, "partial": 1, "pending": 2, "failed": 3, "deferred": 4, "indexed": 5}
    rows.sort(key=lambda row: (order.get(row.state, 9), row.origin, row.name.lower()))
    try:
        active = active_embeds(install)
    except Exception:
        active = 0
    return ScanResult(rows, active, snap.downloads)
