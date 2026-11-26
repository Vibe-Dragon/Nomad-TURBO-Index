"""Storage choices for the knowledge-base collection.

NOMAD searches one unnamed cosine vector of the install's dimension. These
choices change how that vector is stored. They do not change the distance,
the dimension, or the payload fields NOMAD filters on. uint8 is not offered:
Qdrant stores that type as values from 0 to 255, and the embedder sends
ordinary floats.
"""

from __future__ import annotations

FLOAT_TYPES = ("float32", "float16")
# Compressed storage that still accepts an ordinary float on write.
# Qdrant 1.20 is the first release that stores these as the vector itself.
TURBO_TYPES = ("turbo16", "turbo8", "turbo4")
STORAGE_TYPES = FLOAT_TYPES + TURBO_TYPES

STORAGE_LABELS = {
    "float32": "Float32",
    "float16": "Float16",
    "turbo16": "Turbo16",
    "turbo8": "Turbo8",
    "turbo4": "Turbo4",
}

# One byte per number, kept beside the original vector. always_ram stays off
# so the copy is not pinned into RAM.
SCALAR_QUANTIZATION = {"scalar": {"type": "int8", "quantile": 0.99, "always_ram": False}}

# Uncompressed HNSW links for m=16: 32 neighbor ids, 4 bytes each. The
# compressed file is smaller. The int8 vector copy is separate and much
# larger. It is pinned while the original vectors use in-RAM storage, so a
# quantized collection stores those vectors on disk and the int8 copy follows.
GRAPH_RAM_BYTES_PER_POINT = 128
GRAPH_RAM_RESERVE_BYTES = 4 * 1024 ** 3
# One int8 number per dimension. This is the copy Qdrant pins when the
# original vectors use in-RAM storage.
INT8_BYTES_PER_DIM = 1


def parse_version(text: object) -> tuple[int, int, int] | None:
    """The leading numbers of a Qdrant version. A suffix such as -dev is ignored."""
    raw = str(text or "").strip()
    if not raw:
        return None
    parts: list[int] = []
    for piece in raw.split(".", 2):
        digits = []
        for char in piece:
            if char.isdigit():
                digits.append(char)
                continue
            break
        if not digits:
            break
        parts.append(int("".join(digits)))
    if not parts:
        return None
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def storage_choices(version: tuple[int, int, int] | None) -> tuple[str, ...]:
    """Types this Qdrant can store for an ordinary float embedding.

    An unknown version still offers float16. That type has been accepted
    since 1.9, and a server that rejects it reports the error on Apply.
    Turbo types wait until the server says it is 1.20 or newer.
    """
    if version is not None and version < (1, 9, 0):
        return ("float32",)
    choices = ["float32", "float16"]
    if version is not None and version >= (1, 20, 0):
        choices.extend(TURBO_TYPES)
    return tuple(choices)


def storage_label(datatype: str) -> str:
    return STORAGE_LABELS.get(datatype, datatype)


def canonical_datatype(value: object) -> str:
    """A missing datatype is float32, which is what Qdrant stores when the field is absent."""
    text = str(value or "").strip().lower()
    if text in STORAGE_TYPES:
        return text
    return "float32" if not text else text


def accepted_datatype(value: object, choices: tuple[str, ...]) -> str:
    """The type Apply may write. Anything outside the server's list is refused."""
    text = str(value or "").strip().lower()
    if text not in choices or text not in STORAGE_TYPES:
        raise ValueError(f"refusing storage type {value}")
    return text


def bytes_per_vector(datatype: object, dimension: int) -> int:
    """Bytes of one stored vector. turbo4 keeps four bits per number."""
    dim = max(1, int(dimension))
    kind = str(datatype or "float32").strip().lower()
    if kind == "turbo4":
        return max(1, (dim + 1) // 2)
    width = {"float32": 4, "float16": 2, "turbo16": 2, "turbo8": 1}.get(kind, 4)
    return dim * width


def _vector_params(vectors: object) -> dict:
    """The unnamed vector, or the first named one. NOMAD uses the unnamed vector."""
    if not isinstance(vectors, dict):
        return {}
    if "size" in vectors or "datatype" in vectors or "distance" in vectors:
        return vectors
    for item in vectors.values():
        if isinstance(item, dict):
            return item
    return {}


def vector_bytes_from_config(config: object, fallback_dimension: int = 768) -> int:
    """Bytes of one vector from a collection config. A missing datatype is float32."""
    if not isinstance(config, dict):
        return bytes_per_vector("float32", fallback_dimension)
    params = config.get("params") if isinstance(config.get("params"), dict) else config
    if not isinstance(params, dict):
        return bytes_per_vector("float32", fallback_dimension)
    vectors = _vector_params(params.get("vectors"))
    try:
        size = int(vectors.get("size") or fallback_dimension)
    except (TypeError, ValueError):
        size = fallback_dimension
    return bytes_per_vector(vectors.get("datatype"), size)


def quantization_enabled(config: object) -> bool:
    if not isinstance(config, dict):
        return False
    quant = config.get("quantization_config")
    return isinstance(quant, dict) and bool(quant)


def snapshot_from_info(result: object) -> dict:
    """The live collection, as the settings page compares it."""
    info = result if isinstance(result, dict) else {}
    config = info.get("config") if isinstance(info.get("config"), dict) else {}
    params = config.get("params") if isinstance(config.get("params"), dict) else {}
    vectors = _vector_params(params.get("vectors"))
    hnsw = config.get("hnsw_config") if isinstance(config.get("hnsw_config"), dict) else {}
    try:
        points = int(info.get("points_count") or 0)
    except (TypeError, ValueError):
        points = 0
    try:
        size = int(vectors.get("size") or 0)
    except (TypeError, ValueError):
        size = 0
    return {
        "present": True,
        "datatype": canonical_datatype(vectors.get("datatype")),
        "quantization": quantization_enabled(config),
        "hnsw_on_disk": bool(hnsw.get("on_disk")),
        "vectors_on_disk": bool(vectors.get("on_disk")),
        "points": points,
        "distance": str(vectors.get("distance") or ""),
        "size": size,
    }


def create_body(
    dimension: int,
    datatype: str,
    quantization: bool,
    hnsw_on_disk: bool,
    vectors_on_disk: bool = False,
) -> dict:
    """A new collection NOMAD can search. Payloads stay on disk."""
    kind = accepted_datatype(datatype, STORAGE_TYPES)
    vectors: dict = {"size": int(dimension), "distance": "Cosine"}
    if kind != "float32":
        vectors["datatype"] = kind
    body: dict = {
        "vectors": vectors,
        "hnsw_config": {"on_disk": bool(hnsw_on_disk)},
        "on_disk_payload": True,
    }
    if quantization or vectors_on_disk:
        # In-RAM vector storage pins the int8 copy even when always_ram is
        # false. On-disk vectors make that copy cold.
        vectors["on_disk"] = True
    if quantization:
        body["quantization_config"] = dict(SCALAR_QUANTIZATION)
    return body


def update_body(changes: dict) -> dict:
    """A live patch. Only the keys in `changes` are sent, so an unchanged setting is left alone."""
    body: dict = {}
    if "quantization" in changes:
        if changes["quantization"]:
            body["quantization_config"] = dict(SCALAR_QUANTIZATION)
            # Turning quantization on stores the vectors on disk so the int8
            # copy is not pinned. An explicit vectors choice below wins.
            if "vectors_on_disk" not in changes:
                body["vectors"] = {"": {"on_disk": True}}
        else:
            body["quantization_config"] = None
    if "vectors_on_disk" in changes:
        # The default vector has an empty name. A patch has to use that
        # name; a bare on_disk boolean is rejected.
        body["vectors"] = {"": {"on_disk": bool(changes["vectors_on_disk"])}}
    if "hnsw_on_disk" in changes:
        body["hnsw_config"] = {"on_disk": bool(changes["hnsw_on_disk"])}
    return body


def apply_plan(
    current: dict | None,
    datatype: str,
    quantization: bool,
    hnsw_on_disk: bool,
    vectors_on_disk: bool | None = None,
) -> str:
    """What Apply has to do. create, rebuild, update, or none.

    The storage type is fixed for the life of a collection, so a different
    type means a new collection. Quantization, where the vectors live, and
    the graph location can change on the collection that is already there.
    """
    if not current or not current.get("present"):
        return "create"
    live = canonical_datatype(current.get("datatype"))
    chosen = canonical_datatype(datatype)
    if live != chosen:
        return "rebuild"
    if bool(current.get("quantization")) != bool(quantization):
        return "update"
    if vectors_on_disk is not None and bool(current.get("vectors_on_disk")) != bool(vectors_on_disk):
        return "update"
    if bool(current.get("hnsw_on_disk")) != bool(hnsw_on_disk):
        return "update"
    return "none"


def graph_auto_wants_disk(
    *,
    automatic: bool,
    on_disk: bool,
    points: int,
    spare_ram_bytes: int | None,
) -> bool:
    """Whether indexing should move the graph to disk.

    Automatic never turns the graph off. A graph that is already on disk, or
    a missing memory reading, is left alone. Spare RAM is MemAvailable. Swap
    is not counted, because a graph that only fits in swap is the failure
    this switch exists to avoid.
    """
    if not automatic or on_disk or spare_ram_bytes is None:
        return False
    try:
        count = max(0, int(points))
        spare = int(spare_ram_bytes)
    except (TypeError, ValueError):
        return False
    needed = count * GRAPH_RAM_BYTES_PER_POINT + GRAPH_RAM_RESERVE_BYTES
    return needed > spare


def vectors_auto_wants_disk(
    *,
    automatic: bool,
    on_disk: bool,
    quantization: bool,
    points: int,
    dimension: int,
    spare_ram_bytes: int | None,
) -> bool:
    """Whether indexing should move the original vectors to disk.

    That is what stops Qdrant pinning the int8 copy. Automatic never moves
    the vectors back. Without quantization there is no int8 copy to pin.
    Spare RAM is MemAvailable. Swap is not counted.
    """
    if not automatic or on_disk or not quantization or spare_ram_bytes is None:
        return False
    try:
        count = max(0, int(points))
        width = max(1, int(dimension))
        spare = int(spare_ram_bytes)
    except (TypeError, ValueError):
        return False
    needed = count * width * INT8_BYTES_PER_DIM + GRAPH_RAM_RESERVE_BYTES
    return needed > spare


def changed_options(
    current: dict | None,
    quantization: bool,
    hnsw_on_disk: bool,
    vectors_on_disk: bool | None = None,
) -> dict:
    """Quantization and graph fields that differ from the live collection."""
    current = current or {}
    changes = {}
    if bool(current.get("quantization")) != bool(quantization):
        changes["quantization"] = bool(quantization)
    if vectors_on_disk is not None and bool(current.get("vectors_on_disk")) != bool(vectors_on_disk):
        changes["vectors_on_disk"] = bool(vectors_on_disk)
    if bool(current.get("hnsw_on_disk")) != bool(hnsw_on_disk):
        changes["hnsw_on_disk"] = bool(hnsw_on_disk)
    return changes


def status_text(view: dict, collection: str) -> str:
    """The sentence at the top of the page."""
    if not view.get("loaded"):
        return "The collection is read when this page is opened."
    if not view.get("ok"):
        detail = str(view.get("error") or "Qdrant did not answer")
        return detail
    version = str(view.get("version") or "unknown")
    if not view.get("present"):
        return (
            f"Qdrant {version} has no {collection} collection yet. "
            "Apply creates an empty one. NOMAD search uses it once it exists."
        )
    points = int(view.get("points") or 0)
    stored = storage_label(str(view.get("datatype") or "float32"))
    quant = "on" if view.get("quantization") else "off"
    graph = "on disk" if view.get("hnsw_on_disk") else "in memory"
    vectors_place = "on disk" if view.get("vectors_on_disk") else "in memory"
    return (
        f"Collection {collection} on Qdrant {version}. "
        f"{points:,} points. Storage is {stored}. "
        f"Quantization is {quant}. Vectors are {vectors_place}. The graph is {graph}."
    )


def apply_note(plan: str, datatype: str, points: int, *, busy: bool) -> str:
    """What the Apply button will do with the choices currently on screen."""
    label = storage_label(canonical_datatype(datatype))
    if plan == "rebuild" and busy:
        return "Stop indexing before changing the storage type. The collection is left as it is."
    if plan == "rebuild":
        return (
            f"Apply deletes this collection ({points:,} points) and creates it again as {label}. "
            "Library files stay on disk. Indexing does not start. "
            "Type DELETE to confirm."
        )
    if plan == "update":
        return (
            "Apply changes the live collection. Points stay. "
            "Qdrant rebuilds segments in the background, so indexing may pause."
        )
    if plan == "create":
        return f"Apply creates an empty {label} collection. Nothing is deleted."
    return "The collection already uses these settings."


def rebuild_warning(points: int, datatype: str) -> tuple[str, str]:
    """Title and body for the confirmation dialog. The body asks for DELETE."""
    label = storage_label(canonical_datatype(datatype))
    title = "Replace the knowledge index?"
    body = (
        f"This deletes every stored vector ({points:,} points) and creates the collection again as {label}. "
        "ZIM files and uploaded documents stay on disk. "
        "Indexing does not start. Click Start indexing when you want to rebuild. "
        "Ingest stays on Manual, so NOMAD does not start embedding on its own.\n\n"
        "Type DELETE to confirm."
    )
    return title, body
