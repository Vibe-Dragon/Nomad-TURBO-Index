"""Qdrant points and the stock MySQL ingest rows. Passwords stay in the container.

Turbo uses the same collection contract as stock NOMAD: nomic-embed-text,
768 dimensions, cosine distance, and the collection name on the install.
It does not change that contract. An existing collection is left in place.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from nomad_turbo.discover import Install

log = logging.getLogger("nomad_turbo")

# Upserts and point deletes share Qdrant. A delete running at the same time
# as a write makes the write time out, and one timeout used to fail the file.
_write_lock = threading.Lock()


class StoreError(RuntimeError):
    pass


# 4 GiB of vectors. Compaction rewrites one segment, so the cap has to fit
# in free space on the collection's own disk.
MAX_SEGMENT_KB = 4 * 1024 * 1024
_POINT_NS = uuid.uuid5(uuid.NAMESPACE_URL, "nomad-turbo-point")


def point_id(source: str, article_path: str, section_key: str, chunk_index: int, text: str) -> str:
    """The same chunk keeps the same id, so a retry replaces it instead of duplicating it."""
    material = "\0".join([source, article_path, section_key, str(int(chunk_index)), text])
    return str(uuid.uuid5(_POINT_NS, material))


def segment_cap_update(current_kb: object, cap_kb: int = MAX_SEGMENT_KB) -> dict | None:
    """An update when the cap is missing or larger than the one we want."""
    if isinstance(current_kb, bool):
        current_kb = None
    if isinstance(current_kb, (int, float)) and not isinstance(current_kb, bool) and int(current_kb) <= cap_kb:
        return None
    if isinstance(current_kb, str) and current_kb.isdigit() and int(current_kb) <= cap_kb:
        return None
    return {"optimizers_config": {"max_segment_size": cap_kb}}


def store_write_again(message: str, attempt: int, limit: int = 3) -> bool:
    """A timed-out write is tried again. A write Qdrant rejected is not."""
    return attempt + 1 < limit and "unreachable" in message


def _sql(install: Install, statement: str) -> str:
    try:
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                install.mysql_container,
                "sh",
                "-c",
                'mysql -N -u"$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DATABASE"',
            ],
            input=statement,
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or "").splitlines()
        safe = [line for line in message if "Using a password" not in line]
        raise StoreError("mysql failed: " + " ".join(safe[-3:])) from exc
    return result.stdout


def quote(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "''") + "'"


def get_policy(install: Install) -> str:
    from nomad_turbo.decisions import remembered_policy

    rows = _sql(
        install,
        "SELECT value FROM kv_store WHERE `key`='rag.defaultIngestPolicy';\n",
    ).strip()
    if not rows:
        return remembered_policy(None)
    return remembered_policy(rows.splitlines()[-1])


def set_policy(install: Install, policy: str) -> None:
    if policy not in ("Manual", "Always"):
        raise StoreError(f"refusing policy {policy}")
    _sql(
        install,
        "INSERT INTO kv_store (`key`, value, created_at, updated_at) "
        f"VALUES ('rag.defaultIngestPolicy', '{policy}', NOW(), NOW()) "
        "ON DUPLICATE KEY UPDATE value=VALUES(value), updated_at=NOW();\n",
    )


def active_from_cell(cell: str | None) -> bool:
    """A missing ingest row leaves the file searchable. Stock NOMAD does the same.

    The knowledge-base page stamps ``active: true`` onto every point that has
    no ``active`` field. That rewrite holds the collection until it finishes,
    so a point Turbo stores has to carry the flag itself.
    """
    if cell is None:
        return True
    text = cell.strip()
    if not text:
        return True
    return text.splitlines()[-1].strip() != "0"


def source_active(install: Install, source: str) -> bool:
    """The file's current search flag. True when the ingest row does not say off."""
    safe = source.replace("\\", "\\\\").replace("'", "''")
    text = _sql(
        install,
        f"SELECT active FROM kb_ingest_state WHERE file_path='{safe}' LIMIT 1;\n",
    ).strip()
    return active_from_cell(text or None)


def ingest_states(install: Install) -> dict[str, str]:
    return {path: state for path, (state, _chunks) in ingest_details(install).items()}


def ingest_details(install: Install) -> dict[str, tuple[str, int]]:
    text = _sql(install, "SELECT file_path, state, chunks_embedded FROM kb_ingest_state;\n")
    out: dict[str, tuple[str, int]] = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or not parts[0]:
            continue
        chunks = int(parts[2]) if len(parts) > 2 and parts[2].lstrip("-").isdigit() else 0
        out[parts[0]] = (parts[1], chunks)
    return out


def clear_ingest_rows(install: Install) -> None:
    """Remove ingest rows only. Library files stay on disk."""
    _sql(install, "DELETE FROM kb_ingest_state;\n")


def delete_collection(install: Install) -> None:
    """Drop the vector collection. A missing collection is already empty."""
    url = f"{install.qdrant_url.rstrip('/')}/collections/{install.collection}"
    try:
        _request(url, method="DELETE", timeout=180)
    except StoreError as exc:
        if "404" not in str(exc):
            raise


def mark_indexed(install: Install, file_path: str, chunks: int) -> None:
    _sql(
        install,
        "INSERT INTO kb_ingest_state (file_path, state, chunks_embedded, last_error, created_at, updated_at) "
        f"VALUES ({quote(file_path)}, 'indexed', {int(chunks)}, NULL, NOW(), NOW()) "
        "ON DUPLICATE KEY UPDATE state='indexed', chunks_embedded=VALUES(chunks_embedded), "
        "last_error=NULL, updated_at=NOW();\n",
    )


def delete_source_points(install: Install, source: str, timeout: int = 600) -> None:
    """Remove every point NOMAD stored for one file path."""
    url = (
        f"{install.qdrant_url.rstrip('/')}/collections/{install.collection}"
        "/points/delete?wait=true"
    )
    with _write_lock:
        _request(
            url,
            {"filter": {"must": [{"key": "source", "match": {"value": source}}]}},
            timeout=timeout,
            method="POST",
        )


def facet_sources(install: Install, timeout: int = 600) -> dict[str, int] | None:
    """Distinct source values and their counts. None when Qdrant does not answer.

    An exact tally is tried first. A second try allows an approximate count.
    """
    url = f"{install.qdrant_url.rstrip('/')}/collections/{install.collection}/facet"
    last = "qdrant did not list sources"
    for exact in (True, False):
        try:
            body = _request(
                url,
                {"key": "source", "limit": 10000, "exact": exact},
                timeout=timeout,
                method="POST",
            )
        except StoreError as exc:
            last = str(exc)
            log.warning("could not list index sources (exact=%s): %s", exact, exc)
            continue
        hits = (body.get("result") or {}).get("hits") if isinstance(body.get("result"), dict) else None
        if not isinstance(hits, list):
            last = "qdrant returned no source list"
            continue
        found: dict[str, int] = {}
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            value = hit.get("value")
            if not isinstance(value, str) or not value:
                continue
            try:
                found[value] = int(hit.get("count") or 0)
            except (TypeError, ValueError):
                found[value] = 0
        return found
    log.warning("index source list failed: %s", last)
    return None


def forget_ingest(install: Install, file_path: str) -> None:
    """Drop the ingest row for a file that is no longer on disk."""
    _sql(install, f"DELETE FROM kb_ingest_state WHERE file_path={quote(file_path)};\n")


def mark_pending(install: Install, file_path: str, chunks: int) -> None:
    """Block backfill_indexed without claiming the file is finished."""
    _sql(
        install,
        "INSERT INTO kb_ingest_state (file_path, state, chunks_embedded, created_at, updated_at) "
        f"VALUES ({quote(file_path)}, 'pending_decision', {int(chunks)}, NOW(), NOW()) "
        "ON DUPLICATE KEY UPDATE chunks_embedded=GREATEST(chunks_embedded, VALUES(chunks_embedded)), "
        "updated_at=NOW();\n",
    )


def _request(url: str, payload: dict | None = None, timeout: int = 60, method: str | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    verb = method or ("POST" if data is not None else "GET")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method=verb,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise StoreError(f"qdrant {exc.code}: {detail}") from exc
    except (TimeoutError, urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise StoreError(f"qdrant unreachable: {reason}") from exc
    if not body:
        return {}
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise StoreError(f"qdrant returned non-json: {body[:80]!r}") from exc
    if not isinstance(parsed, dict):
        raise StoreError("qdrant returned a non-object")
    return parsed


def collection_action(info_error: str | None, names: list[str] | None, name: str, health_ok: bool) -> str:
    """Decide whether the collection can be used. A busy info call is not a missing collection."""
    if info_error is None:
        return "ready"
    if "404" in info_error:
        return "create"
    if names is not None:
        return "ready" if name in names else "create"
    if health_ok and ("timed out" in info_error or "unreachable" in info_error):
        return "ready"
    return "raise"


def _collection_names(base: str) -> list[str] | None:
    try:
        listing = _request(f"{base}/collections", timeout=20)
    except StoreError:
        return None
    result = listing.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("collections"), list):
        return None
    return [item.get("name") for item in result["collections"] if isinstance(item, dict)]


def _health_ok(base: str) -> bool:
    """Qdrant's health check is plain text, not JSON."""
    request = urllib.request.Request(f"{base}/healthz", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return 200 <= response.status < 300
    except (TimeoutError, urllib.error.URLError, OSError):
        return False


def collection_ready(install: Install) -> None:
    """Make sure the stock collection exists and can take writes.

    The contract is install.collection (nomad_knowledge_base by default),
    install.dimension (768), and cosine distance. Keyword indexes on source,
    content_type, and collection match what NOMAD search expects. Turbo does
    not switch the distance or the dimension. A collection that is already
    there is not recreated.
    """
    base = install.qdrant_url.rstrip("/")
    name = install.collection
    info_error = None
    try:
        _request(f"{base}/collections/{name}", timeout=20)
    except StoreError as exc:
        info_error = str(exc)
    if info_error is None:
        apply_segment_cap(install)
        return
    names = _collection_names(base)
    # The name list is enough. Health is only the fallback, and it is plain text.
    health_ok = _health_ok(base) if names is None else False
    action = collection_action(info_error, names, name, health_ok)
    if action == "ready":
        apply_segment_cap(install)
        return
    if action == "raise":
        raise StoreError(info_error or "qdrant collection check failed")
    _request(
        f"{base}/collections/{name}",
        {"vectors": {"size": install.dimension, "distance": "Cosine"}},
        timeout=60,
        method="PUT",
    )
    for field in ("source", "content_type", "collection"):
        try:
            _request(
                f"{base}/collections/{name}/index",
                {"field_name": field, "field_schema": "keyword"},
                timeout=60,
            )
        except StoreError:
            # The index already exists. Leave it.
            pass
    apply_segment_cap(install)


def apply_segment_cap(install: Install) -> None:
    """Keep segments small enough that a later compact can rewrite one of them."""
    base = install.qdrant_url.rstrip("/")
    name = install.collection
    try:
        info = _request(f"{base}/collections/{name}", timeout=20)
        config = (info.get("result") or {}).get("config") or {}
        opt = config.get("optimizer_config") or config.get("optimizers_config") or {}
        payload = segment_cap_update(opt.get("max_segment_size"))
        if payload is None:
            return
        _request(f"{base}/collections/{name}", payload, timeout=60, method="PATCH")
        log.info(
            "capped each index segment at %s KB so a later compact can fit in free space",
            payload["optimizers_config"]["max_segment_size"],
        )
    except StoreError as exc:
        log.warning("could not cap index segment size: %s", exc)


# One request per 128 points. A whole article can be larger than that, and
# Qdrant still waits for each request before the next one starts.
POINT_BATCH = 128


def upsert(install: Install, points: list[dict]) -> None:
    if not points:
        return
    url = f"{install.qdrant_url.rstrip('/')}/collections/{install.collection}/points?wait=true"
    with _write_lock:
        for start in range(0, len(points), POINT_BATCH):
            batch = points[start : start + POINT_BATCH]
            for attempt in range(3):
                try:
                    _request(url, {"points": batch}, timeout=120, method="PUT")
                    break
                except StoreError as exc:
                    if not store_write_again(str(exc), attempt):
                        raise
                    log.warning("qdrant write timed out. Trying that batch again. %s", exc)
                    time.sleep(2)


def count_source(install: Install, source: str) -> int | None:
    try:
        body = _request(
            f"{install.qdrant_url.rstrip('/')}/collections/{install.collection}/points/count",
            {"filter": {"must": [{"key": "source", "match": {"value": source}}]}, "exact": False},
            timeout=20,
        )
    except StoreError as exc:
        log.warning("Qdrant problem while indexing: could not count points for %s: %s", source, exc)
        return None
    return int(body.get("result", {}).get("count", 0))


def existing_paths(install: Install, source: str, limit_pages: int = 20) -> set[str] | None:
    """Article paths already stored for one file. None if Qdrant does not answer."""
    found: set[str] = set()
    offset = None
    base = f"{install.qdrant_url.rstrip('/')}/collections/{install.collection}/points/scroll"
    for _ in range(limit_pages):
        payload: dict = {
            "filter": {"must": [{"key": "source", "match": {"value": source}}]},
            "limit": 256,
            "with_payload": ["article_path"],
            "with_vector": False,
        }
        if offset is not None:
            payload["offset"] = offset
        try:
            body = _request(base, payload, timeout=30)
        except StoreError as exc:
            log.warning(
                "Qdrant problem while indexing: could not list stored paths for %s: %s",
                source,
                exc,
            )
            return None
        result = body.get("result", {})
        for point in result.get("points") or []:
            path = (point.get("payload") or {}).get("article_path")
            if path:
                found.add(path)
        offset = result.get("next_page_offset")
        if not offset:
            return found
    return None
