"""Qdrant points and the stock MySQL ingest rows. Passwords stay in the container.

Turbo uses the same collection contract as stock NOMAD: nomic-embed-text,
768 dimensions, cosine distance, and the collection name on the install.
It does not change that contract. An existing collection is left in place.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
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
# Stop for the upsert running on this thread. Checked between point batches.
_write_stop = threading.local()


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


# The Models page reads the cap. It writes the tasks model, and the assistant's
# last model only after the selected model was deleted.
_KV_READ = re.compile(r"^[A-Za-z0-9._]+$")
_KV_WRITE = frozenset({"ai.tasksModel", "chat.lastModel"})


def kv_read_statement(key: str) -> str:
    if not _KV_READ.fullmatch(key or ""):
        raise StoreError(f"refusing key {key}")
    return f"SELECT value FROM kv_store WHERE `key`='{key}';\n"


def kv_write_statement(key: str, value: str) -> str:
    if key not in _KV_WRITE:
        raise StoreError(f"refusing key {key}")
    if any(char in value for char in ("\n", "\r", "\x00")):
        raise StoreError("refusing a value with a newline")
    return (
        "INSERT INTO kv_store (`key`, value, created_at, updated_at) "
        f"VALUES ('{key}', {quote(value)}, NOW(), NOW()) "
        "ON DUPLICATE KEY UPDATE value=VALUES(value), updated_at=NOW();\n"
    )


def get_kv(install: Install, key: str) -> str | None:
    rows = _sql(install, kv_read_statement(key)).strip()
    if not rows:
        return None
    return rows.splitlines()[-1]


def set_kv(install: Install, key: str, value: str) -> None:
    _sql(install, kv_write_statement(key, value))


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


def installed_zim_rows(install: Install) -> list[tuple[str, str, str, str]]:
    """resource id, installed version, file path, and offered version.

    An empty offered version means no update is waiting. The row is the
    edition on disk. It does not name a path that was already replaced.
    """
    text = _sql(
        install,
        "SELECT resource_id, version, file_path, IFNULL(available_update_version, '') "
        "FROM installed_resources WHERE resource_type='zim';\n",
    )
    rows: list[tuple[str, str, str, str]] = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 4 or not parts[2]:
            continue
        rows.append((parts[0], parts[1], parts[2], parts[3]))
    return rows


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
    """Keep a partial file pending_decision, including a row that already exists.

    An indexed row is not passed here. record_file_finish leaves that state
    alone. backfill_indexed and any other unfinished state become
    pending_decision, so a restart does not treat the file as finished.
    """
    _sql(
        install,
        "INSERT INTO kb_ingest_state (file_path, state, chunks_embedded, created_at, updated_at) "
        f"VALUES ({quote(file_path)}, 'pending_decision', {int(chunks)}, NOW(), NOW()) "
        "ON DUPLICATE KEY UPDATE state='pending_decision', "
        "chunks_embedded=GREATEST(chunks_embedded, VALUES(chunks_embedded)), "
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


def collection_copy_state(install: Install) -> str:
    """present, missing, or unknown. A timeout is unknown and must not clear logs."""
    base = install.qdrant_url.rstrip("/")
    name = install.collection
    info_error = None
    try:
        _request(f"{base}/collections/{name}", timeout=3)
    except StoreError as exc:
        info_error = str(exc)
    if info_error is None:
        return "present"
    if "404" in info_error:
        return "missing"
    names = _collection_names(base, timeout=3)
    if names is not None:
        return "present" if name in names else "missing"
    if _health_ok(base) and ("timed out" in info_error or "unreachable" in info_error):
        return "unknown"
    return "unknown"


def _collection_names(base: str, timeout: int = 20) -> list[str] | None:
    try:
        listing = _request(f"{base}/collections", timeout=timeout)
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


def read_collection_view(install: Install) -> dict:
    """The settings page's picture of Qdrant. A missing collection is not an error."""
    from nomad_turbo.qdrant_settings import (
        canonical_datatype,
        parse_version,
        snapshot_from_info,
        storage_choices,
    )

    base = install.qdrant_url.rstrip("/")
    version_text = ""
    try:
        root = _request(f"{base}/", timeout=5)
        version_text = str(root.get("version") or "")
    except StoreError as exc:
        return {
            "ok": False,
            "loaded": True,
            "present": False,
            "error": str(exc),
            "version": "",
            "choices": storage_choices(None),
            "datatype": canonical_datatype(getattr(install, "qdrant_datatype", "float32")),
            "quantization": bool(getattr(install, "qdrant_quantization", False)),
            "hnsw_on_disk": bool(getattr(install, "qdrant_hnsw_on_disk", False)),
            "points": 0,
        }
    choices = storage_choices(parse_version(version_text))
    saved_type = canonical_datatype(getattr(install, "qdrant_datatype", "float32"))
    if saved_type not in choices:
        saved_type = "float32"
    try:
        info = _request(f"{base}/collections/{install.collection}", timeout=8)
    except StoreError as exc:
        if "404" not in str(exc):
            return {
                "ok": False,
                "loaded": True,
                "present": False,
                "error": str(exc),
                "version": version_text,
                "choices": choices,
                "datatype": saved_type,
                "quantization": bool(getattr(install, "qdrant_quantization", False)),
                "hnsw_on_disk": bool(getattr(install, "qdrant_hnsw_on_disk", False)),
                "points": 0,
            }
        return {
            "ok": True,
            "loaded": True,
            "present": False,
            "error": None,
            "version": version_text,
            "choices": choices,
            "datatype": saved_type,
            "quantization": bool(getattr(install, "qdrant_quantization", False)),
            "hnsw_on_disk": bool(getattr(install, "qdrant_hnsw_on_disk", False)),
            "points": 0,
            "distance": "Cosine",
            "size": int(install.dimension),
        }
    view = snapshot_from_info(info.get("result"))
    view.update(
        ok=True,
        loaded=True,
        error=None,
        version=version_text,
        choices=choices,
    )
    return view


# One collection read per minute. The graph grows much slower than that.
_graph_auto_lock = threading.Lock()
_graph_auto_checked = 0.0
GRAPH_AUTO_INTERVAL = 60.0


def consider_graph_auto(install: Install) -> bool:
    """Move the graph to disk when automatic is on and RAM cannot hold it.

    Returns True when a patch was sent. Never moves the graph back to memory.
    A read or a decision that says no waits out the interval. Automatic off
    does not. A patch also saves the on-disk preference so a recreated
    collection matches.
    """
    if not bool(getattr(install, "qdrant_graph_auto", False)):
        return False
    global _graph_auto_checked
    now = time.time()
    with _graph_auto_lock:
        if now - _graph_auto_checked < GRAPH_AUTO_INTERVAL:
            return False
        _graph_auto_checked = now
    try:
        view = read_collection_view(install)
    except Exception:
        log.exception("could not read the collection while checking the graph")
        return False
    if not view.get("ok") or not view.get("present") or view.get("hnsw_on_disk"):
        return False
    from nomad_turbo.qdrant_settings import graph_auto_wants_disk
    from nomad_turbo.segment_guard import _spare_ram_bytes

    points = int(view.get("points") or 0)
    if not graph_auto_wants_disk(
        automatic=True,
        on_disk=False,
        points=points,
        spare_ram_bytes=_spare_ram_bytes(),
    ):
        return False
    log.info(
        "Spare memory cannot hold the search graph (%s points). Moving it to disk.",
        points,
    )
    update_collection_options(install, hnsw_on_disk=True)
    install.qdrant_hnsw_on_disk = True
    try:
        from nomad_turbo.discover import save_install

        save_install(install)
    except Exception:
        log.exception("the graph was moved to disk, and the saved preference was not written")
    return True


_vectors_auto_lock = threading.Lock()
_vectors_auto_checked = 0.0


def consider_vectors_auto(install: Install) -> bool:
    """Move the vectors to disk when automatic is on and the int8 copy would not fit.

    Returns True when a patch was sent. Never moves the vectors back to the
    in-RAM storage that pins the int8 copy. A collection without quantization
    is left alone.
    """
    if not bool(getattr(install, "qdrant_vectors_auto", True)):
        return False
    global _vectors_auto_checked
    now = time.time()
    with _vectors_auto_lock:
        if now - _vectors_auto_checked < GRAPH_AUTO_INTERVAL:
            return False
        _vectors_auto_checked = now
    try:
        view = read_collection_view(install)
    except Exception:
        log.exception("could not read the collection while checking the vectors")
        return False
    if not view.get("ok") or not view.get("present"):
        return False
    from nomad_turbo.qdrant_settings import vectors_auto_wants_disk
    from nomad_turbo.segment_guard import _spare_ram_bytes

    points = int(view.get("points") or 0)
    dimension = int(view.get("size") or 768)
    if not vectors_auto_wants_disk(
        automatic=True,
        on_disk=bool(view.get("vectors_on_disk")),
        quantization=bool(view.get("quantization")),
        points=points,
        dimension=dimension,
        spare_ram_bytes=_spare_ram_bytes(),
    ):
        return False
    log.info(
        "Spare memory cannot hold the int8 vector copy (%s points). Moving the vectors to disk.",
        points,
    )
    update_collection_options(install, vectors_on_disk=True)
    install.qdrant_vectors_on_disk = True
    try:
        from nomad_turbo.discover import save_install

        save_install(install)
    except Exception:
        log.exception("the vectors were moved to disk, and the saved preference was not written")
    return True


def update_collection_options(
    install: Install,
    *,
    quantization: bool | None = None,
    hnsw_on_disk: bool | None = None,
    vectors_on_disk: bool | None = None,
) -> None:
    """Change quantization or the graph location on the collection already there.

    The storage type is not included. Qdrant keeps that for the life of the
    collection. Points stay.
    """
    from nomad_turbo.qdrant_settings import update_body

    changes = {}
    if quantization is not None:
        changes["quantization"] = bool(quantization)
    if hnsw_on_disk is not None:
        changes["hnsw_on_disk"] = bool(hnsw_on_disk)
    if vectors_on_disk is not None:
        changes["vectors_on_disk"] = bool(vectors_on_disk)
    body = update_body(changes)
    if not body:
        return
    name = install.collection
    _request(
        f"{install.qdrant_url.rstrip('/')}/collections/{name}",
        body,
        timeout=60,
        method="PATCH",
    )


def _create_collection(install: Install) -> None:
    """Create the collection with the storage choices saved on the install."""
    from nomad_turbo.qdrant_settings import canonical_datatype, create_body

    datatype = canonical_datatype(getattr(install, "qdrant_datatype", "float32"))
    body = create_body(
        int(install.dimension),
        datatype,
        bool(getattr(install, "qdrant_quantization", False)),
        bool(getattr(install, "qdrant_hnsw_on_disk", False)),
        bool(getattr(install, "qdrant_vectors_on_disk", False)),
    )
    _request(
        f"{install.qdrant_url.rstrip('/')}/collections/{install.collection}",
        body,
        timeout=60,
        method="PUT",
    )


def collection_ready(install: Install) -> None:
    """Make sure the stock collection exists and can take writes.

    The contract is install.collection (nomad_knowledge_base by default),
    install.dimension (768), and cosine distance. Keyword indexes on source,
    content_type, and collection match what NOMAD search expects. Turbo does
    not switch the distance or the dimension. A collection that is already
    there is not recreated. Missing keyword indexes are created anyway,
    because a filtered point count is wrong without them.
    """
    base = install.qdrant_url.rstrip("/")
    name = install.collection
    info_error = None
    try:
        _request(f"{base}/collections/{name}", timeout=20)
    except StoreError as exc:
        info_error = str(exc)
    if info_error is None:
        _ensure_keyword_indexes(install)
        apply_segment_cap(install)
        return
    names = _collection_names(base)
    # The name list is enough. Health is only the fallback, and it is plain text.
    health_ok = _health_ok(base) if names is None else False
    action = collection_action(info_error, names, name, health_ok)
    if action == "ready":
        _ensure_keyword_indexes(install)
        apply_segment_cap(install)
        return
    if action == "raise":
        raise StoreError(info_error or "qdrant collection check failed")
    _create_collection(install)
    _ensure_keyword_indexes(install)
    apply_segment_cap(install)


def _ensure_keyword_indexes(install: Install) -> None:
    """Keyword indexes Qdrant needs for an exact filtered count.

    Qdrant creates a payload index with PUT. POST returns 404, and an
    approximate count with no index is about half the points in the collection
    for every file.
    """
    base = install.qdrant_url.rstrip("/")
    name = install.collection
    for field in ("source", "content_type", "collection"):
        try:
            _request(
                f"{base}/collections/{name}/index",
                {"field_name": field, "field_schema": "keyword"},
                timeout=120,
                method="PUT",
            )
        except StoreError as exc:
            if index_already_exists(str(exc)):
                continue
            log.warning("could not index payload field %s: %s", field, exc)


def index_already_exists(detail: str) -> bool:
    text = detail.lower()
    return "already" in text and "index" in text


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


@contextlib.contextmanager
def stop_between_writes(stop):
    """Notice Stop between point batches of the upsert on this thread."""
    previous = getattr(_write_stop, "event", None)
    _write_stop.event = stop
    try:
        yield
    finally:
        _write_stop.event = previous


def upsert(install: Install, points: list[dict]) -> None:
    if not points:
        return
    from nomad_turbo.ocr_budget import note_qdrant_fault, note_upsert

    note_upsert(True)
    try:
        url = f"{install.qdrant_url.rstrip('/')}/collections/{install.collection}/points?wait=true"
        stop = getattr(_write_stop, "event", None)
        with _write_lock:
            for start in range(0, len(points), POINT_BATCH):
                # The request already sent finishes. Stop leaves the rest of
                # the slice uncommitted so resume can write it again.
                if stop is not None and stop.is_set():
                    raise RuntimeError("stopped")
                batch = points[start : start + POINT_BATCH]
                for attempt in range(3):
                    try:
                        _request(url, {"points": batch}, timeout=120, method="PUT")
                        break
                    except StoreError as exc:
                        note_qdrant_fault(str(exc))
                        if not store_write_again(str(exc), attempt):
                            raise
                        log.warning("qdrant write timed out. Trying that batch again. %s", exc)
                        time.sleep(2)
    finally:
        note_upsert(False)


def count_source(install: Install, source: str) -> int | None:
    """Exact point count for one file. None when Qdrant does not answer.

    An approximate count is wrong when the source index is missing: Qdrant
    then reports about half of the whole collection for every filter. A skip
    decision cannot use that number.
    """
    try:
        body = _request(
            f"{install.qdrant_url.rstrip('/')}/collections/{install.collection}/points/count",
            {"filter": {"must": [{"key": "source", "match": {"value": source}}]}, "exact": True},
            timeout=60,
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
