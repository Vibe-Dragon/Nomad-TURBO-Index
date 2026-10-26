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
    from nomad_turbo.engine import _downloads_pending, _read_checkpoint, _sources
    from nomad_turbo.runtime import active_embeds, list_jobs
    from nomad_turbo.store import ingest_details

    jobs = list_jobs(install)
    grouped: dict[str, list[dict]] = {}
    for job in jobs:
        data = job.get("data") or {}
        source = data.get("filePath")
        if source:
            grouped.setdefault(source, []).append(data)
    details = ingest_details(install)
    rows: list[CatalogRow] = []
    for host, kind, source in _sources(install):
        entries = grouped.get(source, [])
        checkpoint = _read_checkpoint(source) or {}
        offsets = [int(item.get("batchOffset") or 0) for item in entries]
        resume = choose_resume(offsets, checkpoint.get("articles"))
        total = None
        for item in entries:
            if item.get("totalArticles"):
                total = int(item["totalArticles"])
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
        try:
            size = host.stat().st_size
        except OSError:
            size = 0
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
            )
        )
    order = {"running": 0, "partial": 1, "pending": 2, "failed": 3, "deferred": 4, "indexed": 5}
    rows.sort(key=lambda row: (order.get(row.state, 9), row.origin, row.name.lower()))
    try:
        active = active_embeds(install)
    except Exception:
        active = 0
    return ScanResult(rows, active, _downloads_pending(install))
