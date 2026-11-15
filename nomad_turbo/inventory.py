"""Facts Measure reads before it trusts article pace alone.

Page counts come from PDF metadata. Nothing here draws a page. The xref
is followed to the root page tree, so a count in the middle of a large
file is read by seeking to it. A page tree stored in a FlateDecode
object stream is inflated. A ZIM's documents come from its directory:
kind, size, and a PDF page count when that count can be read without
decompressing a whole large document.
"""

from __future__ import annotations

import ctypes
import re
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

_PAGES = re.compile(r"/Type\s*/Pages\b")
_COUNT = re.compile(r"/Count\s+(\d+)")
_LENGTH = re.compile(br"/Length\s+(\d+)")
_INDIRECT_LENGTH = re.compile(br"\s+\d+\s+R\b")
_PREDICTOR = re.compile(br"/Predictor\s+(\d+)")
_LINEAR_N = re.compile(r"/N\s+(\d+)")
# A compressed ZIM PDF above the whole-file cap is scanned from its start.
_SCAN_BYTES = 2 * 1024 * 1024
_WHOLE_BYTES = 32 * 1024 * 1024
_XREF_OUT = 64 * 1024 * 1024
_XREF_CHAIN = 32
# Object streams are inflated only up to these caps, so a hostile file cannot hang Measure.
_STREAM_CAP = 4000
_STREAM_OUT = 8 * 1024 * 1024
_DICT_WINDOW = 64 * 1024
_LOOKBACK = 8192
_OTHER_FILTERS = (b"ASCII85Decode", b"ASCIIHexDecode", b"LZWDecode", b"RunLengthDecode", b"Crypt")
_ZIM_MAGIC = 0x044D495A
_REDIRECT = 0xFFFF
_LINK = 0xFFFE
_DELETED = 0xFFFD
# Past this many directory entries, HTML is counted and not stored one by one.
# Clusters are opened for PDFs and other non-HTML documents either way.
_DETAIL_ENTRIES = 40_000
_MAX_ENTRIES = 50_000_000
_ZSTD = None


@dataclass
class Inventory:
    kind: str
    pages: int | None
    size_bytes: int
    documents: list[dict] = field(default_factory=list)
    total_units: int = 0


def pdf_page_count(path: Path) -> int | None:
    """Page count from the root page tree. None when that count cannot be read.

    The xref is followed, so a /Count past the ends of a large file still
    counts. A small file with no xref is read whole, including a tree stored
    only in a FlateDecode object stream. A head-and-tail slice is not a page
    count. Nothing is drawn.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        return None
    found = None
    try:
        with path.open("rb") as handle:
            found = _page_count_from_xref(handle, size)
    except OSError:
        found = None
    if found:
        return found
    if size > _WHOLE_BYTES:
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return _page_count_bytes(data)


def inventory_zim_file(path: Path) -> Inventory:
    """Kinds, sizes, and PDF page counts stored in one ZIM.

    The directory is read. Article bodies are not. A PDF page count comes
    from that PDF's page tree. A compressed PDF larger than 32MB is counted
    from its start only, so Measure does not decompress the rest of it.
    """
    try:
        return _inventory_zim_path(Path(path))
    except (OSError, ValueError):
        return Inventory(kind="zim", pages=None, size_bytes=0, documents=[], total_units=0)


def inventory_pdf(path: Path) -> Inventory:
    """Pages and size of one uploaded PDF. The page count is the work total."""
    path = Path(path)
    pages = pdf_page_count(path)
    try:
        size = int(path.stat().st_size)
    except OSError:
        size = 0
    units = int(pages) if pages else 0
    return Inventory(
        kind="pdf",
        pages=pages,
        size_bytes=size,
        documents=[{"kind": "pdf", "size_bytes": size, "pages": pages}],
        total_units=units,
    )


def inventory_zim(entries: list[dict] | None) -> Inventory:
    """Kinds, sizes, and known page counts for documents inside a ZIM.

    An HTML or plain-text entry is one unit. A PDF with a page count costs
    that many units. No page is drawn.
    """
    documents: list[dict] = []
    pages = 0
    size = 0
    units = 0
    saw_pages = False
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        kind = str(entry.get("kind") or "")
        nbytes = int(entry.get("size_bytes") or entry.get("size") or 0)
        raw_pages = entry.get("pages", entry.get("page_count"))
        entry_pages = int(raw_pages) if isinstance(raw_pages, int) and raw_pages > 0 else None
        documents.append({"kind": kind, "size_bytes": nbytes, "pages": entry_pages})
        size += nbytes
        if entry_pages:
            pages += entry_pages
            units += entry_pages
            saw_pages = True
        elif kind in ("html", "text", "pdf", "ocr", "epub", "docx"):
            units += 1
        elif nbytes:
            units += 1
    return Inventory(
        kind="zim",
        pages=pages if saw_pages else None,
        size_bytes=size,
        documents=documents,
        total_units=units,
    )


def inventory_eta_seconds(inventory: Inventory | None, article_rate: float | None, articles_done: int = 0) -> int | None:
    """Seconds left from the inventory total and the sampled articles per second.

    An article rate with no total has no ETA. A known page or document total does.
    """
    if inventory is None or not inventory.total_units:
        return None
    if not article_rate or article_rate <= 0:
        return None
    remaining = max(0, int(inventory.total_units) - int(articles_done or 0))
    return int(remaining / float(article_rate))


def _page_count_bytes(data: bytes) -> int | None:
    """Largest /Type /Pages /Count in clear text and in FlateDecode object streams.

    A linearized /N is used only when no page-tree count exists. No page is drawn.
    """
    if not data:
        return None
    text = data.decode("latin1", errors="ignore")
    best = _best_pages_count(text)
    inflated = 0
    streams = 0
    for payload in _objstm_payloads(data):
        streams += 1
        inflated += len(payload)
        found = _best_pages_count(payload.decode("latin1", errors="ignore"))
        if found is not None and (best is None or found > best):
            best = found
        if streams >= _STREAM_CAP or inflated >= _WHOLE_BYTES:
            break
    if best is not None:
        return best
    return _linearized_count(text)


def _best_pages_count(text: str) -> int | None:
    best: int | None = None
    if not text:
        return None
    for match in _PAGES.finditer(text):
        count = _direct_pages_count(_enclosing_dict(text, match.start(), _DICT_WINDOW))
        if count and (best is None or count > best):
            best = count
    return best


def _enclosing_dict(text: str, pos: int, limit: int) -> str:
    """The << >> dictionary that holds pos, or empty when it is wider than limit."""
    start = max(0, pos - limit)
    depth = 0
    i = pos - 1
    dict_at = None
    while i > start:
        pair = text[i - 1 : i + 1]
        if pair == ">>":
            depth += 1
            i -= 2
            continue
        if pair == "<<":
            if depth == 0:
                dict_at = i - 1
                break
            depth -= 1
            i -= 2
            continue
        i -= 1
    if dict_at is None:
        return ""
    depth = 0
    i = dict_at
    end_limit = min(len(text), dict_at + limit)
    while i < end_limit - 1:
        pair = text[i : i + 2]
        if pair == "<<":
            depth += 1
            i += 2
            continue
        if pair == ">>":
            depth -= 1
            i += 2
            if depth == 0:
                return text[dict_at:i]
            continue
        i += 1
    return ""


def _direct_pages_count(dict_text: str) -> int | None:
    """The /Count written on this dictionary, not one nested inside it."""
    if not dict_text:
        return None
    depth = 0
    best: int | None = None
    i = 0
    length = len(dict_text)
    while i < length:
        pair = dict_text[i : i + 2]
        if pair == "<<":
            depth += 1
            i += 2
            continue
        if pair == ">>":
            depth -= 1
            i += 2
            continue
        if depth == 1 and dict_text.startswith("/Count", i):
            found = _COUNT.match(dict_text, i)
            if found is not None:
                count = int(found.group(1))
                if count and (best is None or count > best):
                    best = count
                i = found.end()
                continue
        i += 1
    return best


def _linearized_count(text: str) -> int | None:
    # A linearized PDF stores the page count in the first dictionary, as /N.
    mark = text.find("/Linearized")
    if mark < 0:
        return None
    end = text.find(">>", mark)
    if end < 0 or end - mark > 2000:
        return None
    found = _LINEAR_N.search(text[mark:end])
    if found is None:
        return None
    count = int(found.group(1))
    return count or None


def _objstm_payloads(data: bytes):
    """Inflated FlateDecode /ObjStm bodies. /N in the stream dictionary is ignored."""
    pos = 0
    produced = 0
    total = 0
    length = len(data)
    while pos < length and produced < _STREAM_CAP and total < _WHOLE_BYTES:
        mark = data.find(b"stream", pos)
        if mark < 0:
            return
        if mark >= 3 and data[mark - 3 : mark] == b"end":
            pos = mark + 6
            continue
        dict_end = mark
        while dict_end > 0 and data[dict_end - 1] in b" \t\r\n":
            dict_end -= 1
        data_at = _stream_data_start(data, mark)
        dictionary = _bytes_dict_before(data, dict_end, _LOOKBACK)
        if (
            dictionary is None
            or b"/ObjStm" not in dictionary
            or b"FlateDecode" not in dictionary
            or _skipped_filter(dictionary)
        ):
            pos = max(data_at, mark + 6)
            continue
        payload, nxt = _objstm_payload(data, data_at, dictionary, _WHOLE_BYTES - total)
        pos = max(nxt, mark + 6)
        if not payload:
            continue
        total += len(payload)
        produced += 1
        yield payload


def _skipped_filter(dictionary: bytes) -> bool:
    if any(name in dictionary for name in _OTHER_FILTERS):
        return True
    predictor = _PREDICTOR.search(dictionary)
    return predictor is not None and int(predictor.group(1)) > 1


def _bytes_dict_before(data: bytes, end: int, limit: int) -> bytes | None:
    if end < 2 or data[end - 2 : end] != b">>":
        return None
    start = max(0, end - limit)
    depth = 1
    i = end - 2
    while i > start:
        pair = data[i - 1 : i + 1]
        if pair == b">>":
            depth += 1
            i -= 2
            continue
        if pair == b"<<":
            depth -= 1
            if depth == 0:
                return data[i - 1 : end]
            i -= 2
            continue
        i -= 1
    return None


def _stream_data_start(data: bytes, mark: int) -> int:
    i = mark + len(b"stream")
    if data[i : i + 2] == b"\r\n":
        return i + 2
    if i < len(data) and data[i : i + 1] in (b"\n", b"\r"):
        return i + 1
    return i


def _direct_length(dictionary: bytes) -> int | None:
    match = _LENGTH.search(dictionary)
    if match is None:
        return None
    if _INDIRECT_LENGTH.match(dictionary, match.end()):
        return None
    return int(match.group(1))


def _objstm_payload(data: bytes, data_at: int, dictionary: bytes, cap: int) -> tuple[bytes | None, int]:
    room = min(_STREAM_OUT, cap)
    length = _direct_length(dictionary)
    if length is not None and 0 <= length <= len(data) - data_at:
        payload = _inflate_capped(data[data_at : data_at + length], room)
        if payload:
            return payload, data_at + length
    end = data.find(b"endstream", data_at, data_at + _WHOLE_BYTES + 64)
    if end < 0:
        return None, data_at + 1
    raw = data[data_at:end]
    if raw.endswith(b"\r\n"):
        raw = raw[:-2]
    elif raw.endswith((b"\n", b"\r")):
        raw = raw[:-1]
    return _inflate_capped(raw, room), end + len(b"endstream")


def _inflate_capped(raw: bytes, cap: int) -> bytes | None:
    if not raw or cap <= 0 or len(raw) > _WHOLE_BYTES:
        return None
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
        try:
            out = zlib.decompressobj(wbits).decompress(raw, cap)
        except zlib.error:
            continue
        if out:
            return out
    return None


@dataclass
class _ZimHeader:
    article_count: int
    cluster_count: int
    path_ptr: int
    title_ptr: int
    cluster_ptr: int
    mime_pos: int
    checksum: int


def _inventory_zim_path(path: Path) -> Inventory:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header = _read_zim_header(handle, file_size)
        if header is None or header.article_count <= 0:
            return Inventory(kind="zim", pages=None, size_bytes=0, documents=[], total_units=0)
        mimes = _read_mimes(handle, header, file_size)
        slots = _mime_slots(mimes)
        detailed, html_count = _scan_dirents(handle, header, mimes, slots)
        if detailed:
            _fill_cluster_facts(handle, header, file_size, detailed)
    documents: list[dict] = []
    total_units = 0
    total_size = 0
    total_pages = 0
    saw_pages = False
    for item in detailed:
        pages = item.get("pages")
        size = int(item.get("size_bytes") or 0)
        documents.append({"kind": item["kind"], "size_bytes": size, "pages": pages})
        total_size += size
        if isinstance(pages, int) and pages > 0:
            total_pages += pages
            total_units += pages
            saw_pages = True
        else:
            total_units += 1
    if html_count:
        documents.append({"kind": "html", "size_bytes": 0, "pages": None, "count": html_count})
        total_units += html_count
    return Inventory(
        kind="zim",
        pages=total_pages if saw_pages else None,
        size_bytes=total_size,
        documents=documents,
        total_units=total_units,
    )


def _read_zim_header(handle, file_size: int) -> _ZimHeader | None:
    if file_size < 80:
        return None
    handle.seek(0)
    raw = handle.read(80)
    if len(raw) < 80:
        return None
    magic, major, _minor = struct.unpack_from("<IHH", raw, 0)
    if magic != _ZIM_MAGIC or major not in (5, 6):
        return None
    articles, clusters = struct.unpack_from("<II", raw, 24)
    path_ptr, title_ptr, cluster_ptr, mime_pos = struct.unpack_from("<QQQQ", raw, 32)
    checksum = struct.unpack_from("<Q", raw, 72)[0]
    if articles <= 0 or articles > _MAX_ENTRIES:
        return None
    if clusters > articles + 16:
        return None
    if not (0 <= mime_pos < file_size and 0 < path_ptr < file_size):
        return None
    return _ZimHeader(articles, clusters, path_ptr, title_ptr, cluster_ptr, mime_pos, checksum)


def _read_mimes(handle, header: _ZimHeader, file_size: int) -> list[str]:
    handle.seek(header.mime_pos)
    raw = handle.read(min(256 * 1024, file_size - header.mime_pos))
    mimes: list[str] = []
    cursor = 0
    while cursor < len(raw):
        end = raw.find(b"\0", cursor)
        if end < 0 or end == cursor:
            break
        mimes.append(raw[cursor:end].decode("latin1", errors="replace"))
        cursor = end + 1
    return mimes


def _mime_slots(mimes: list[str]) -> list[bool]:
    """True when this MIME can be a document Measure should time."""
    from nomad_turbo.passes import classify_entry

    slots: list[bool] = []
    for mime in mimes:
        keep = classify_entry(mime, "x.bin", 0) != "skip"
        if not keep:
            for name in ("x.pdf", "x.html", "x.epub", "x.docx", "x.txt"):
                if classify_entry(mime, name, 0) != "skip":
                    keep = True
                    break
        slots.append(keep)
    return slots


def _kind_of(mime: str, path: str) -> str:
    from nomad_turbo.passes import classify_entry

    classified = classify_entry(mime, path, 0)
    if classified == "html":
        return "html"
    if classified == "ocr":
        return "pdf"
    if classified != "text":
        return ""
    lowered = path.lower()
    name = mime.split(";", 1)[0].strip().lower()
    if lowered.endswith(".epub") or "epub" in name:
        return "epub"
    if lowered.endswith(".docx") or "wordprocessingml" in name:
        return "docx"
    return "text"


def _scan_dirents(handle, header: _ZimHeader, mimes: list[str], slots: list[bool]):
    """Directory entries that are documents. Tiles, pictures, and redirects are left out."""
    detailed: list[dict] = []
    html_count = 0
    keep_html = header.article_count <= _DETAIL_ENTRIES
    pointer = header.path_ptr
    left = header.article_count
    mime_count = len(mimes)
    while left > 0:
        take = min(4096, left)
        handle.seek(pointer)
        raw = handle.read(take * 8)
        if len(raw) != take * 8:
            break
        pointers = struct.unpack("<" + "Q" * take, raw)
        pointer += take * 8
        left -= take
        found = _dirents_in(handle, pointers, mimes, slots, mime_count)
        for kind, cluster, blob in found:
            if kind == "html" and not keep_html:
                html_count += 1
                continue
            detailed.append({"kind": kind, "cluster": cluster, "blob": blob, "size_bytes": 0, "pages": None})
    return detailed, html_count


def _dirents_in(handle, pointers, mimes, slots, mime_count: int) -> list[tuple[str, int, int]]:
    start = pointers[0]
    extra = 65536
    for _attempt in range(3):
        handle.seek(start)
        blob = handle.read(pointers[-1] - start + extra)
        parsed = _parse_dirent_span(blob, pointers, start, mimes, slots, mime_count)
        if parsed is not None:
            return parsed
        extra *= 4
    return []


def _parse_dirent_span(blob: bytes, pointers, start: int, mimes, slots, mime_count: int):
    found: list[tuple[str, int, int]] = []
    last = len(pointers) - 1
    blob_len = len(blob)
    for index, ptr in enumerate(pointers):
        rel = ptr - start
        if rel < 0 or rel + 8 > blob_len:
            return None
        mime_index = blob[rel] | (blob[rel + 1] << 8)
        if mime_index >= _DELETED or mime_index >= mime_count or not slots[mime_index]:
            continue
        if rel + 16 > blob_len:
            return None
        cluster, blob_index = struct.unpack_from("<II", blob, rel + 8)
        limit = blob_len if index == last else pointers[index + 1] - start
        if limit > blob_len:
            limit = blob_len
        path_at = rel + 16
        end = blob.find(b"\0", path_at, limit)
        if end < 0:
            return None
        path = blob[path_at:end].decode("utf-8", "surrogateescape")
        kind = _kind_of(mimes[mime_index], path)
        if not kind:
            continue
        found.append((kind, int(cluster), int(blob_index)))
    return found


def _fill_cluster_facts(handle, header: _ZimHeader, file_size: int, detailed: list[dict]) -> None:
    starts = _cluster_starts(handle, header, file_size)
    if not starts:
        return
    by_cluster: dict[int, list[int]] = {}
    for index, item in enumerate(detailed):
        by_cluster.setdefault(int(item["cluster"]), []).append(index)
    for cluster, indexes in by_cluster.items():
        if cluster < 0 or cluster >= len(starts):
            continue
        limit = _cluster_limit(starts, cluster, file_size, header.checksum)
        view = _open_cluster(handle, starts[cluster], limit, header.article_count)
        if view is None:
            continue
        pdfs: list[tuple[int, int, int]] = []
        try:
            for index in indexes:
                item = detailed[index]
                blob_index = int(item["blob"])
                if blob_index < 0 or blob_index + 1 >= len(view.offsets):
                    continue
                size = int(view.offsets[blob_index + 1] - view.offsets[blob_index])
                if size < 0:
                    continue
                item["size_bytes"] = size
                if item["kind"] == "pdf" and size > 0:
                    pdfs.append((int(view.offsets[blob_index]), size, index))
            pdfs.sort()
            for start, size, index in pdfs:
                pages = view.page_count(start, size)
                if pages:
                    detailed[index]["pages"] = pages
        finally:
            view.close()


def _cluster_starts(handle, header: _ZimHeader, file_size: int) -> list[int]:
    count = header.cluster_count
    if count <= 0 or header.cluster_ptr <= 0 or header.cluster_ptr >= file_size:
        return []
    handle.seek(header.cluster_ptr)
    raw = handle.read(count * 8)
    if len(raw) != count * 8:
        return []
    return list(struct.unpack("<" + "Q" * count, raw))


def _cluster_limit(starts: list[int], index: int, file_size: int, checksum: int) -> int:
    start = starts[index]
    limit = file_size
    if 0 < checksum < file_size and checksum > start:
        limit = min(limit, checksum)
    for other in starts:
        if start < other < limit:
            limit = other
    return limit


class _ClusterView:
    def __init__(self, offsets: list[int], read_range, close, compressed: bool):
        self.offsets = offsets
        self._read_range = read_range
        self._close = close
        self.compressed = compressed

    def close(self) -> None:
        closer = self._close
        self._close = None
        if closer is not None:
            closer()

    def page_count(self, start: int, size: int) -> int | None:
        if size <= 0 or start < 0:
            return None
        # An uncompressed PDF can be sought. A compressed one is read whole
        # only while it stays inside the same cap used for an uploaded file.
        if not self.compressed and size > _WHOLE_BYTES:
            return _page_count_from_xref(_BlobReader(self._read_range, start, size), size)
        if size <= _WHOLE_BYTES:
            data = self._read_range(start, size)
        elif self.compressed:
            data = self._read_range(start, min(size, _SCAN_BYTES))
        else:
            return None
        return _page_count_bytes(data)


def _open_cluster(handle, start: int, limit: int, article_count: int) -> _ClusterView | None:
    if start <= 0 or start >= limit:
        return None
    handle.seek(start)
    info = handle.read(1)
    if not info:
        return None
    flags = info[0]
    compressed_code = flags & 0x0F
    extended = bool(flags & 0x10)
    payload = start + 1
    if compressed_code in (0, 1):
        offsets = _offsets_at(handle, payload, limit - payload, extended, article_count)
        if offsets is None:
            return None

        def read_range(offset: int, length: int) -> bytes:
            if offset < 0 or length <= 0:
                return b""
            handle.seek(payload + offset)
            return handle.read(length)

        return _ClusterView(offsets, read_range, None, compressed=False)
    if compressed_code != 5:
        return None
    window = _ZstdWindow(handle, payload, limit)
    try:
        offsets = _offsets_from_window(window, extended, article_count)
    except (OSError, ValueError):
        window.close()
        return None
    if offsets is None:
        window.close()
        return None
    return _ClusterView(offsets, window.read_range, window.close, compressed=True)


def _offsets_at(handle, payload: int, available: int, extended: bool, article_count: int) -> list[int] | None:
    width = 8 if extended else 4
    if available < width * 2:
        return None
    handle.seek(payload)
    head = handle.read(width)
    if len(head) < width:
        return None
    first = struct.unpack("<Q" if extended else "<I", head)[0]
    count = _offset_count(first, width, article_count)
    if count is None or first > available:
        return None
    handle.seek(payload)
    raw = handle.read(first)
    if len(raw) < first:
        return None
    return _offsets_checked(raw, width, count)


def _offsets_from_window(window, extended: bool, article_count: int) -> list[int] | None:
    width = 8 if extended else 4
    head = window.read_next(width)
    if len(head) < width:
        return None
    first = struct.unpack("<Q" if extended else "<I", head)[0]
    count = _offset_count(first, width, article_count)
    if count is None:
        return None
    rest = window.read_next(first - width)
    raw = head + rest
    if len(raw) < first:
        return None
    return _offsets_checked(raw, width, count)


def _offset_count(first: int, width: int, article_count: int) -> int | None:
    if first < width * 2 or first % width:
        return None
    count = first // width
    cap = max(2, int(article_count) + 1)
    if count > cap:
        return None
    return count


def _offsets_checked(raw: bytes, width: int, count: int) -> list[int] | None:
    fmt = "<" + ("Q" if width == 8 else "I") * count
    try:
        offsets = list(struct.unpack_from(fmt, raw, 0))
    except struct.error:
        return None
    previous = -1
    for offset in offsets:
        if offset < previous:
            return None
        previous = offset
    return offsets


class _ZSTDIn(ctypes.Structure):
    _fields_ = [("src", ctypes.c_void_p), ("size", ctypes.c_size_t), ("pos", ctypes.c_size_t)]


class _ZSTDOut(ctypes.Structure):
    _fields_ = [("dst", ctypes.c_void_p), ("size", ctypes.c_size_t), ("pos", ctypes.c_size_t)]


def _zstd():
    global _ZSTD
    if _ZSTD is False:
        return None
    if _ZSTD is not None:
        return _ZSTD
    try:
        lib = ctypes.CDLL("libzstd.so.1")
    except OSError:
        _ZSTD = False
        return None
    lib.ZSTD_createDCtx.restype = ctypes.c_void_p
    lib.ZSTD_freeDCtx.argtypes = [ctypes.c_void_p]
    lib.ZSTD_freeDCtx.restype = ctypes.c_size_t
    lib.ZSTD_decompressStream.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ZSTDOut), ctypes.POINTER(_ZSTDIn)]
    lib.ZSTD_decompressStream.restype = ctypes.c_size_t
    lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
    lib.ZSTD_isError.restype = ctypes.c_uint
    _ZSTD = lib
    return lib


class _ZstdWindow:
    """Decompress a cluster from the start and keep only the bytes still needed."""

    def __init__(self, handle, start: int, limit: int):
        self.handle = handle
        self.file_pos = start
        self.limit = limit
        self.buf = bytearray()
        self.produced = 0
        self.done = False
        self.failed = False
        self._hold = None
        self._incoming = None
        self._cursor = 0
        lib = _zstd()
        self._lib = lib
        self.dctx = lib.ZSTD_createDCtx() if lib is not None else None
        if not self.dctx:
            self.failed = True

    def close(self) -> None:
        dctx = self.dctx
        self.dctx = None
        if dctx and self._lib is not None:
            self._lib.ZSTD_freeDCtx(dctx)

    def read_next(self, count: int) -> bytes:
        if count <= 0:
            return b""
        start = self.produced - len(self.buf)
        # read_next continues at the first byte that has not been handed out.
        # The buffer may already hold it.
        target = self._cursor + count
        self._produce_until(target)
        base = self.produced - len(self.buf)
        rel = self._cursor - base
        if rel < 0:
            return b""
        data = bytes(self.buf[rel : rel + count])
        self._cursor += len(data)
        drop = self._cursor - base
        if drop > 0:
            del self.buf[:drop]
        return data

    def read_range(self, start: int, length: int) -> bytes:
        if length <= 0 or start < 0:
            return b""
        if start < self._cursor:
            return b""
        if start > self._cursor:
            self._produce_until(start)
            base = self.produced - len(self.buf)
            drop = start - base
            if drop > 0:
                del self.buf[: min(drop, len(self.buf))]
            self._cursor = start
        return self.read_next(length)

    def _produce_until(self, target: int) -> None:
        spins = 0
        while self.produced < target and not self.done and not self.failed:
            before = (self.produced, self.file_pos, self._incoming.pos if self._incoming is not None else -1)
            self._pump()
            spins += 1
            after = (self.produced, self.file_pos, self._incoming.pos if self._incoming is not None else -1)
            if after == before or spins > 100000:
                self.failed = True
                return

    def _pump(self) -> None:
        lib = self._lib
        if lib is None or not self.dctx or self.failed:
            self.failed = True
            return
        if self._incoming is None or self._incoming.pos >= self._incoming.size:
            if self.file_pos >= self.limit:
                self.failed = True
                return
            amount = min(65536, self.limit - self.file_pos)
            self.handle.seek(self.file_pos)
            chunk = self.handle.read(amount)
            if not chunk:
                self.failed = True
                return
            self.file_pos += len(chunk)
            self._hold = ctypes.create_string_buffer(chunk)
            self._incoming = _ZSTDIn(ctypes.addressof(self._hold), len(chunk), 0)
        out_hold = ctypes.create_string_buffer(65536)
        outgoing = _ZSTDOut(ctypes.addressof(out_hold), 65536, 0)
        result = lib.ZSTD_decompressStream(self.dctx, ctypes.byref(outgoing), ctypes.byref(self._incoming))
        if lib.ZSTD_isError(result):
            self.failed = True
            return
        if outgoing.pos:
            self.buf.extend(out_hold.raw[: outgoing.pos])
            self.produced += outgoing.pos
        if result == 0:
            self.done = True


# The next unread decompressed byte. Set on the class via the instance in __init__.
_ZstdWindow._cursor = 0


class _BlobReader:
    """A seekable view of one blob inside a ZIM cluster."""

    def __init__(self, read_range, start: int, size: int):
        self._read_range = read_range
        self._start = start
        self.size = size
        self.pos = 0

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 1:
            offset += self.pos
        elif whence == 2:
            offset += self.size
        self.pos = max(0, offset)
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, count: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        if count is None or count < 0:
            count = self.size - self.pos
        count = min(count, self.size - self.pos)
        data = self._read_range(self._start + self.pos, count) or b""
        self.pos += len(data)
        return data

    def readline(self) -> bytes:
        if self.pos >= self.size:
            return b""
        start = self.pos
        data = self.read(min(256, self.size - self.pos))
        mark = data.find(b"\n")
        if mark < 0:
            return data
        self.pos = start + mark + 1
        return data[: mark + 1]


def _page_count_from_xref(handle, size: int) -> int | None:
    """Root /Pages /Count by seeking through the xref. None when it cannot be read."""
    try:
        return _xref_page_count(handle, size)
    except (OSError, ValueError, zlib.error, OverflowError, IndexError, TypeError, re.error):
        return None


def _xref_page_count(handle, size: int) -> int | None:
    if size <= 0:
        return None
    offset = _startxref(handle, size)
    if offset is None or offset < 0 or offset >= size:
        return None
    entries, trailer = _merged_xref(handle, offset, set())
    if trailer.get("encrypt"):
        return None
    root = trailer.get("root")
    if root is None:
        return None
    cache: dict = {}
    catalog = _object_body(handle, entries, root, cache)
    if not catalog:
        return None
    dictionary, _end = _first_dict(catalog)
    pages_num = _ref_number(dictionary or catalog, b"Pages")
    if pages_num is None:
        return None
    pages = _object_body(handle, entries, pages_num, cache)
    if not pages:
        return None
    dictionary, _end = _first_dict(pages)
    return _direct_int(dictionary or pages, b"Count")


def _startxref(handle, size: int) -> int | None:
    window = min(size, 1024 * 1024)
    handle.seek(size - window)
    tail = handle.read(window)
    mark = tail.rfind(b"startxref")
    if mark < 0:
        return None
    match = re.search(br"startxref\s+(\d+)", tail[mark : mark + 64])
    if match is None:
        return None
    return int(match.group(1))


def _merged_xref(handle, offset: int, seen: set, depth: int = 0):
    if offset in seen or depth > _XREF_CHAIN or offset < 0:
        return {}, {}
    seen.add(offset)
    entries, trailer = _load_xref(handle, offset)
    merged: dict = {}
    if "prev" in trailer:
        older, _trailer = _merged_xref(handle, trailer["prev"], seen, depth + 1)
        merged.update(older)
    if "xrefstm" in trailer:
        extra, _trailer = _merged_xref(handle, trailer["xrefstm"], seen, depth + 1)
        merged.update(extra)
    merged.update(entries)
    return merged, trailer


def _load_xref(handle, offset: int):
    handle.seek(offset)
    if handle.read(4) == b"xref":
        return _load_xref_table(handle, offset)
    return _load_xref_stream(handle, offset)


def _load_xref_table(handle, offset: int):
    handle.seek(offset)
    if not handle.readline().startswith(b"xref"):
        raise ValueError("xref")
    entries = {}
    line = b""
    while True:
        pos = handle.tell()
        line = handle.readline()
        if not line or line.startswith(b"trailer"):
            break
        parts = line.split()
        if len(parts) != 2 or not parts[0].isdigit():
            handle.seek(pos)
            break
        start, count = int(parts[0]), int(parts[1])
        if count < 0 or count > 2_000_000:
            raise ValueError("xref size")
        blob = handle.read(count * 20)
        if len(blob) != count * 20:
            raise ValueError("xref row")
        for index in range(count):
            row = blob[index * 20 : (index + 1) * 20]
            if row[17:18] != b"n":
                continue
            entries[start + index] = (1, int(row[:10]), int(row[11:16]))
    if line.startswith(b"trailer"):
        data = line[len(b"trailer") :] + handle.read(256 * 1024)
    else:
        data = handle.read(256 * 1024)
        mark = data.find(b"trailer")
        if mark < 0:
            raise ValueError("trailer")
        data = data[mark + len(b"trailer") :]
    dictionary, _end = _first_dict(data)
    if not dictionary:
        raise ValueError("trailer")
    return entries, _trailer_info(dictionary)


def _load_xref_stream(handle, offset: int):
    dictionary, raw = _stream_at(handle, offset)
    if any(name in dictionary for name in _OTHER_FILTERS):
        raise ValueError("filter")
    if b"FlateDecode" in dictionary:
        raw = _xref_inflate(raw)
        if raw is None:
            raise ValueError("inflate")
    elif b"/Filter" in dictionary:
        raise ValueError("filter")
    info = _trailer_info(dictionary)
    widths = re.search(br"/W\s*\[\s*(\d+)\s+(\d+)\s+(\d+)\s*\]", dictionary)
    if widths is None:
        raise ValueError("W")
    fields = tuple(int(widths.group(index)) for index in (1, 2, 3))
    index_match = re.search(br"/Index\s*\[([^\]]*)\]", dictionary, re.S)
    if index_match:
        nums = [int(number) for number in re.findall(br"\d+", index_match.group(1))]
        pairs = list(zip(nums[0::2], nums[1::2]))
    else:
        pairs = [(0, info.get("size") or 0)]
    predictor = _PREDICTOR.search(dictionary)
    columns = re.search(br"/Columns\s+(\d+)", dictionary)
    if predictor is not None:
        width = int(columns.group(1)) if columns else sum(fields)
        raw = _unpredict(raw, int(predictor.group(1)), width)
        if raw is None:
            raise ValueError("predictor")
    return _xref_entries(raw, fields, pairs), info


def _xref_entries(raw: bytes, fields: tuple, pairs: list) -> dict:
    width = sum(fields)
    if width <= 0:
        raise ValueError("W")
    entries = {}
    pos = 0
    for start, count in pairs:
        if count < 0 or count > 2_000_000:
            raise ValueError("xref size")
        for number in range(start, start + count):
            if pos + width > len(raw):
                return entries
            chunk = raw[pos : pos + width]
            pos += width
            values = []
            at = 0
            for field in fields:
                if field == 0:
                    values.append(0)
                else:
                    values.append(int.from_bytes(chunk[at : at + field], "big"))
                    at += field
            kind, second, third = values
            if fields[0] == 0:
                kind = 1
            if kind == 0:
                continue
            entries[number] = (kind, second, third)
    return entries


def _trailer_info(data: bytes) -> dict:
    info = {}
    root = _ref_number(data, b"Root")
    if root is not None:
        info["root"] = root
    prev = _direct_int(data, b"Prev")
    if prev is not None:
        info["prev"] = prev
    xrefstm = _direct_int(data, b"XRefStm")
    if xrefstm is not None:
        info["xrefstm"] = xrefstm
    size = _direct_int(data, b"Size")
    if size is not None:
        info["size"] = size
    if b"/Encrypt" in data:
        info["encrypt"] = True
    return info


def _ref_number(data: bytes, name: bytes) -> int | None:
    match = re.search(br"/" + name + br"\s+(\d+)\s+\d+\s+R\b", data)
    if match is None:
        return None
    return int(match.group(1))


def _direct_int(data: bytes, name: bytes) -> int | None:
    match = re.search(br"/" + name + br"\s+(\d+)", data)
    if match is None:
        return None
    if _INDIRECT_LENGTH.match(data, match.end()):
        return None
    value = int(match.group(1))
    return value or None


def _first_dict(data: bytes):
    start = data.find(b"<<")
    if start < 0:
        return None, 0
    depth = 0
    index = start
    limit = len(data) - 1
    while index < limit:
        pair = data[index : index + 2]
        if pair == b"<<":
            depth += 1
            index += 2
            continue
        if pair == b">>":
            depth -= 1
            index += 2
            if depth == 0:
                return data[start:index], index
            continue
        index += 1
    return None, 0


def _stream_at(handle, offset: int):
    handle.seek(offset)
    data = handle.read(4 * 1024 * 1024)
    match = re.match(br"\s*\d+\s+\d+\s+obj\b", data)
    if match is None:
        raise ValueError("obj")
    dictionary, rel = _first_dict(data[match.end() :])
    if not dictionary:
        raise ValueError("dict")
    cursor = match.end() + rel
    while cursor < len(data) and data[cursor] in b" \t\r\n":
        cursor += 1
    if data[cursor : cursor + 6] != b"stream":
        raise ValueError("stream")
    cursor += 6
    if data[cursor : cursor + 2] == b"\r\n":
        cursor += 2
    elif cursor < len(data) and data[cursor : cursor + 1] in (b"\n", b"\r"):
        cursor += 1
    length = _direct_int(dictionary, b"Length")
    if length is not None and length <= _WHOLE_BYTES:
        raw = data[cursor : cursor + length]
        if len(raw) < length:
            handle.seek(offset + cursor)
            raw = handle.read(length)
    else:
        end = data.find(b"endstream", cursor)
        if end < 0:
            raise ValueError("endstream")
        raw = data[cursor:end]
        if raw.endswith(b"\r\n"):
            raw = raw[:-2]
        elif raw.endswith((b"\n", b"\r")):
            raw = raw[:-1]
    return dictionary, raw


def _object_body(handle, entries: dict, number: int, cache: dict, depth: int = 0):
    if depth > 6 or number not in entries:
        return None
    kind, second, third = entries[number]
    if kind == 1:
        handle.seek(second)
        data = handle.read(1024 * 1024)
        match = re.match(br"\s*\d+\s+\d+\s+obj\b", data)
        if match is None:
            return None
        return data[match.end() :]
    if kind == 2:
        return _objstm_object(handle, entries, second, third, cache, depth + 1)
    return None


def _objstm_object(handle, entries: dict, stream_number: int, index: int, cache: dict, depth: int):
    if stream_number not in cache:
        kind, offset, _gen = entries.get(stream_number, (0, 0, 0))
        if kind != 1:
            return None
        dictionary, raw = _stream_at(handle, offset)
        if b"/ObjStm" not in dictionary or any(name in dictionary for name in _OTHER_FILTERS):
            return None
        if b"FlateDecode" in dictionary:
            raw = _xref_inflate(raw)
        elif b"/Filter" in dictionary:
            return None
        count = _direct_int(dictionary, b"N")
        first = _direct_int(dictionary, b"First")
        if not raw or count is None or first is None:
            return None
        cache[stream_number] = (raw, count, first)
    payload, count, first = cache[stream_number]
    nums = [int(number) for number in re.findall(br"\d+", payload[:first])]
    if index < 0 or index >= count or len(nums) < count * 2:
        return None
    offsets = [nums[item * 2 + 1] for item in range(count)]
    start = first + offsets[index]
    end = first + offsets[index + 1] if index + 1 < count else len(payload)
    if start < 0 or end > len(payload) or start > end:
        return None
    return payload[start:end]


def _xref_inflate(raw: bytes) -> bytes | None:
    if not raw or len(raw) > _WHOLE_BYTES:
        return None
    for bits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
        try:
            out = zlib.decompressobj(bits).decompress(raw, _XREF_OUT)
        except zlib.error:
            continue
        if out:
            return out
    return None


def _unpredict(data: bytes, predictor: int, columns: int) -> bytes | None:
    if predictor <= 1:
        return data
    columns = columns or 1
    if predictor == 2:
        out = bytearray()
        for row_at in range(0, len(data), columns):
            row = bytearray(data[row_at : row_at + columns])
            for index in range(len(row)):
                left = row[index - 1] if index else 0
                row[index] = (row[index] + left) & 255
            out.extend(row)
        return bytes(out)
    if predictor < 10:
        return None
    stride = columns + 1
    out = bytearray()
    prev = bytearray(columns)
    for row_at in range(0, len(data), stride):
        row = data[row_at : row_at + stride]
        if len(row) < stride:
            break
        algo = row[0]
        cur = bytearray(row[1:])
        if algo == 1:
            for index in range(columns):
                left = cur[index - 1] if index else 0
                cur[index] = (cur[index] + left) & 255
        elif algo == 2:
            for index in range(columns):
                cur[index] = (cur[index] + prev[index]) & 255
        elif algo == 3:
            for index in range(columns):
                left = cur[index - 1] if index else 0
                cur[index] = (cur[index] + ((left + prev[index]) // 2)) & 255
        elif algo == 4:
            for index in range(columns):
                left = cur[index - 1] if index else 0
                up = prev[index]
                upleft = prev[index - 1] if index else 0
                cur[index] = (cur[index] + _paeth(left, up, upleft)) & 255
        elif algo != 0:
            return None
        prev = cur
        out.extend(cur)
    return bytes(out)


def _paeth(left: int, up: int, upleft: int) -> int:
    estimate = left + up - upleft
    if abs(estimate - left) <= abs(estimate - up) and abs(estimate - left) <= abs(estimate - upleft):
        return left
    if abs(estimate - up) <= abs(estimate - upleft):
        return up
    return upleft
