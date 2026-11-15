"""Measure reads page counts and document sizes without drawing pages."""

import struct
import subprocess
import tempfile
import unittest
import zlib
from pathlib import Path

from nomad_turbo.engine import FileWork, _apply_measure_inventory
from nomad_turbo.inventory import (
    inventory_eta_seconds,
    inventory_pdf,
    inventory_zim,
    inventory_zim_file,
    pdf_page_count,
)
from nomad_turbo.schedule import file_eta_view


_PDF = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Count 3 /Kids [3 0 R 4 0 R 5 0 R] >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
4 0 obj
<< /Type /Page /Parent 2 0 R >>
endobj
5 0 obj
<< /Type /Page /Parent 2 0 R >>
endobj
trailer
<< /Root 1 0 R >>
%%EOF
"""


class MeasureInventoryTest(unittest.TestCase):
    def test_pdf_pages_and_zim_documents_change_the_eta(self):
        directory = tempfile.TemporaryDirectory(prefix="nomad-turbo-inventory-")
        self.addCleanup(directory.cleanup)
        pdf_path = Path(directory.name) / "chapter.pdf"
        pdf_path.write_bytes(_PDF)
        pdf = inventory_pdf(pdf_path)
        self.assertEqual(pdf.pages, 3)
        self.assertEqual(pdf.total_units, 3)
        self.assertGreater(pdf.size_bytes, 0)
        self.assertEqual(pdf.documents[0]["kind"], "pdf")
        self.assertEqual(pdf.documents[0]["pages"], 3)

        entries = [
            {"kind": "html", "size_bytes": 4096},
            {"kind": "pdf", "size_bytes": 80000, "pages": 10},
            {"kind": "docx", "size_bytes": 1200},
        ]
        zim = inventory_zim(entries)
        self.assertEqual([item["size_bytes"] for item in zim.documents], [4096, 80000, 1200])
        self.assertEqual(zim.documents[1]["pages"], 10)
        self.assertEqual(zim.documents[1]["kind"], "pdf")
        self.assertEqual(zim.total_units, 12)

        unknown = file_eta_view(
            state="pending",
            articles=0,
            total=None,
            live_rate=None,
            measured_rate=2.0,
        )
        self.assertIsNone(unknown[1])
        paced = file_eta_view(
            state="pending",
            articles=0,
            total=None,
            live_rate=None,
            measured_rate=2.0,
            inventory_units=zim.total_units,
        )
        self.assertEqual(paced[1], 6)
        self.assertNotEqual(paced[1], unknown[1])
        self.assertEqual(inventory_eta_seconds(pdf, 2.0), 1)
        self.assertIsNone(inventory_eta_seconds(pdf, None))

    def test_measure_reads_a_zim_directory_when_no_entry_list_was_stored(self):
        directory = tempfile.TemporaryDirectory(prefix="nomad-turbo-inventory-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        pdf_path = root / "chapter.pdf"
        pdf_path.write_bytes(_PDF)
        linear = root / "linear.pdf"
        linear.write_bytes(
            b"%PDF-1.5\n1 0 obj\n<< /Linearized 1 /L 98235 /N 24 /T 100 >>\nendobj\n%%EOF\n"
        )
        self.assertEqual(inventory_pdf(linear).pages, 24)
        self.assertEqual(inventory_pdf(linear).total_units, 24)
        zim_path = root / "shelf.zim"
        html = b"<p>Hello</p>"
        png = b"\x89PNG\r\n"
        _write_zim(zim_path, [html, _PDF, png])

        found = inventory_zim_file(zim_path)
        kinds = [item["kind"] for item in found.documents]
        self.assertEqual(kinds, ["html", "pdf"])
        self.assertEqual(found.documents[0]["size_bytes"], len(html))
        self.assertEqual(found.documents[1]["size_bytes"], len(_PDF))
        self.assertEqual(found.documents[1]["pages"], 3)
        self.assertEqual(found.pages, 3)
        self.assertEqual(found.total_units, 4)

        class _Book:
            def __init__(self):
                self.got = None

            def note_inventory(self, source, *, units, pages):
                self.got = (source, units, pages)

        class _Stats:
            def __init__(self):
                self.got = None

            def note_inventory(self, source, *, pages, units):
                self.got = (source, pages, units)

        class _Session:
            def __init__(self):
                self.book = _Book()
                self.stats = _Stats()

            def get_book(self):
                return self.book

            def file_stats(self):
                return self.stats

        session = _Session()
        zim_work = FileWork(host=zim_path, kind="zim", source="shelf.zim")
        self.assertIsNone(getattr(zim_work, "inventory_entries", None))
        _apply_measure_inventory(session, zim_work)
        self.assertEqual(session.book.got, ("shelf.zim", 4, 3))
        self.assertEqual(session.stats.got, ("shelf.zim", 3, 4))

        session = _Session()
        _apply_measure_inventory(session, FileWork(host=pdf_path, kind="pdf", source="chapter.pdf"))
        self.assertEqual(session.book.got, ("chapter.pdf", 3, 3))
        self.assertEqual(session.stats.got[1], 3)

    def test_object_stream_page_count_is_recorded(self):
        directory = tempfile.TemporaryDirectory(prefix="nomad-turbo-inventory-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        hidden = root / "questions.pdf"
        # /Count sits after the kids list, and nowhere in the clear text.
        hidden.write_bytes(_object_stream_pdf(133))
        self.assertEqual(pdf_page_count(hidden), 133)
        found = inventory_pdf(hidden)
        self.assertEqual(found.pages, 133)
        self.assertEqual(found.total_units, 133)

        class _Book:
            def __init__(self):
                self.got = None

            def note_inventory(self, source, *, units, pages):
                self.got = (source, units, pages)

        class _Stats:
            def __init__(self):
                self.got = None

            def note_inventory(self, source, *, pages, units):
                self.got = (source, pages, units)

        class _Session:
            def __init__(self):
                self.book = _Book()
                self.stats = _Stats()

            def get_book(self):
                return self.book

            def file_stats(self):
                return self.stats

        session = _Session()
        _apply_measure_inventory(session, FileWork(host=hidden, kind="pdf", source="questions.pdf"))
        self.assertEqual(session.book.got, ("questions.pdf", 133, 133))
        self.assertEqual(session.stats.got, ("questions.pdf", 133, 133))

        # A smaller clear-text /Count loses to the root count inside the stream.
        mixed = root / "code.pdf"
        mixed.write_bytes(_object_stream_pdf(352, clear_count=345, count_first=True))
        self.assertEqual(pdf_page_count(mixed), 352)
        self.assertEqual(inventory_pdf(mixed).total_units, 352)

        # /N on an object stream is the object count, not a page count.
        decoy = root / "decoy.pdf"
        decoy.write_bytes(_object_stream_without_pages(7))
        self.assertIsNone(pdf_page_count(decoy))
        self.assertEqual(inventory_pdf(decoy).total_units, 0)

    def test_page_count_past_the_head_and_tail(self):
        directory = tempfile.TemporaryDirectory(prefix="nomad-turbo-inventory-")
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "biology.pdf"
        _write_pdf_with_count_past_the_window(path)
        self.assertGreater(path.stat().st_size, 32 * 1024 * 1024)
        with path.open("rb") as handle:
            head = handle.read(2 * 1024 * 1024)
            handle.seek(path.stat().st_size - 2 * 1024 * 1024)
            tail = handle.read()
        self.assertIn(b"/Count 125", head)
        self.assertNotIn(b"/Count 1525", head)
        self.assertNotIn(b"/Count 1525", tail)
        self.assertEqual(pdf_page_count(path), 1525)
        found = inventory_pdf(path)
        self.assertEqual(found.pages, 1525)
        self.assertEqual(found.total_units, 1525)

        class _Book:
            def __init__(self):
                self.got = None

            def note_inventory(self, source, *, units, pages):
                self.got = (source, units, pages)

        class _Stats:
            def __init__(self):
                self.got = None

            def note_inventory(self, source, *, pages, units):
                self.got = (source, pages, units)

        class _Session:
            def __init__(self):
                self.book = _Book()
                self.stats = _Stats()

            def get_book(self):
                return self.book

            def file_stats(self):
                return self.stats

        session = _Session()
        _apply_measure_inventory(session, FileWork(host=path, kind="pdf", source="biology.pdf"))
        self.assertEqual(session.book.got, ("biology.pdf", 1525, 1525))
        self.assertEqual(session.stats.got, ("biology.pdf", 1525, 1525))


def _write_pdf_with_count_past_the_window(path: Path) -> None:
    """A PDF bigger than 32MB whose real /Count is outside the 2MB head and tail.

    The head carries a decoy count of 125. The xref root points at the page
    tree in the middle, whose /Count is 1525.
    """
    decoy = b"%PDF-1.4\n9 0 obj\n<< /Type /Pages /Count 125 /Kids [8 0 R] >>\nendobj\n"
    catalog = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    pages = b"2 0 obj\n<< /Type /Pages /Count 1525 /Kids [3 0 R] >>\nendobj\n"
    page = b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
    with path.open("wb") as handle:
        handle.write(decoy)
        handle.write(b"\n" * (3 * 1024 * 1024))
        offsets = {}
        for number, blob in ((1, catalog), (2, pages), (3, page)):
            offsets[number] = handle.tell()
            handle.write(blob)
        handle.seek(34 * 1024 * 1024)
        xref_at = handle.tell()
        rows = ["0000000000 65535 f \n"]
        for number in (1, 2, 3):
            rows.append(f"{offsets[number]:010d} 00000 n \n")
        xref = (
            "xref\n0 4\n"
            + "".join(rows)
            + "trailer\n<< /Size 4 /Root 1 0 R >>\n"
            + f"startxref\n{xref_at}\n%%EOF\n"
        )
        handle.write(xref.encode())


def _object_stream_pdf(count: int, *, clear_count: int | None = None, count_first: bool = False) -> bytes:
    """A PDF whose page tree is only inside a FlateDecode object stream."""
    kids = " ".join(f"{number} 0 R" for number in range(6, 22))
    if count_first:
        body = f"<< /Count {count} /Kids [{kids}] /Type /Pages >>".encode()
    else:
        body = f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode()
    header = b"6 0 "
    compressed = zlib.compress(header + body)
    stream = (
        b"4 0 obj\n"
        + (
            f"<< /Type /ObjStm /N 1 /First {len(header)} "
            f"/Filter /FlateDecode /Length {len(compressed)} >>\n"
        ).encode()
        + b"stream\n"
        + compressed
        + b"\nendstream\nendobj\n"
    )
    clear = b""
    if clear_count is not None:
        clear = f"9 0 obj\n<< /Type /Pages /Count {clear_count} /Kids [8 0 R] >>\nendobj\n".encode()
    return b"%PDF-1.5\n" + clear + stream + b"%%EOF\n"


def _object_stream_without_pages(object_count: int) -> bytes:
    header = b"2 0 "
    compressed = zlib.compress(header + b"<< /Type /Catalog /Pages 2 0 R >>")
    return (
        b"%PDF-1.5\n4 0 obj\n"
        + (
            f"<< /Type /ObjStm /N {object_count} /First {len(header)} "
            f"/Filter /FlateDecode /Length {len(compressed)} >>\n"
        ).encode()
        + b"stream\n"
        + compressed
        + b"\nendstream\nendobj\n%%EOF\n"
    )


def _dirent(mime: int, cluster: int, blob: int, path: str) -> bytes:
    encoded = path.encode()
    return struct.pack("<HB", mime, 0) + b"C" + struct.pack("<III", 0, cluster, blob) + encoded + b"\0t\0"


def _redirect(path: str) -> bytes:
    encoded = path.encode()
    return struct.pack("<HB", 0xFFFF, 0) + b"C" + struct.pack("<II", 0, 0) + encoded + b"\0\0"


def _write_zim(path: Path, blobs: list[bytes]) -> None:
    """A small Zstd ZIM: HTML, a 3-page PDF, a redirect, and a map tile."""
    width = 4
    first = (len(blobs) + 1) * width
    offsets = [first]
    for blob in blobs:
        offsets.append(offsets[-1] + len(blob))
    raw = struct.pack("<" + "I" * len(offsets), *offsets) + b"".join(blobs)
    compressed = subprocess.run(["zstd", "-c", "-q"], input=raw, capture_output=True, check=True).stdout
    cluster = bytes([5]) + compressed
    dirents = [
        _dirent(0, 0, 0, "article.html"),
        _dirent(1, 0, 1, "chapter.pdf"),
        _redirect("old-name"),
        _dirent(2, 0, 2, "tiles/0.png"),
    ]
    mime = b"text/html\0application/pdf\0image/png\0\0"
    mime_pos = 80
    cluster_off = mime_pos + len(mime)
    dirent_off = cluster_off + len(cluster)
    packed = b"".join(dirents)
    pointers = []
    cursor = dirent_off
    for entry in dirents:
        pointers.append(cursor)
        cursor += len(entry)
    path_ptr = dirent_off + len(packed)
    cluster_ptr = path_ptr + 8 * len(pointers)
    checksum = cluster_ptr + 8
    body = bytearray(checksum + 16)
    body[mime_pos : mime_pos + len(mime)] = mime
    body[cluster_off : cluster_off + len(cluster)] = cluster
    body[dirent_off : dirent_off + len(packed)] = packed
    struct.pack_into("<" + "Q" * len(pointers), body, path_ptr, *pointers)
    struct.pack_into("<Q", body, cluster_ptr, cluster_off)
    struct.pack_into("<IHH", body, 0, 0x044D495A, 6, 3)
    struct.pack_into("<II", body, 24, len(dirents), 1)
    struct.pack_into("<QQQQ", body, 32, path_ptr, 0xFFFFFFFFFFFFFFFF, cluster_ptr, mime_pos)
    struct.pack_into("<IIQ", body, 64, 0xFFFFFFFF, 0xFFFFFFFF, checksum)
    path.write_bytes(body)


if __name__ == "__main__":
    unittest.main()
