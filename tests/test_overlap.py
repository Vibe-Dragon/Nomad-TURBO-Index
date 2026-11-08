"""CPU OCR beside GPU indexing.

The lane reads one PDF or picture at a time. It does not start a GPU reader.
While a ZIM is on the GPU it also reads uploaded PDFs that come after other files.
"""

import tempfile
import threading
import time
import unittest
from pathlib import Path

from nomad_turbo.devices import Assignment, Backend, ComputeDevice
from nomad_turbo.overlap import (
    OcrJob,
    VisualLane,
    cpu_ocr_overlaps_gpu,
    parallel_ocr_enabled,
    gpu_candidates,
    ocr_ram_budget,
    preview_order,
    text_bytes,
    visual_reads,
)
from nomad_turbo.schedule import OrderFile


def _device(kind: str, ident: str) -> ComputeDevice:
    return ComputeDevice(ident, kind, kind, None if kind == "cpu" else 0, 0, 0, 0, 0, "")


def _assignment(index_kind: str, backend: str, ocr_kind: str = "cpu") -> Assignment:
    return Assignment(_device(index_kind, "index"), _device(ocr_kind, "ocr"), Backend(backend, backend), False)


def _file(source: str, kind: str, walk: int) -> OrderFile:
    return OrderFile(
        source=source,
        name=source,
        file_type=kind,
        size_bytes=100,
        mtime=1.0,
        zim_date=None,
        articles_total=None,
        completed=0.0,
        chunks=0,
        measured_eta=None,
        state="pending",
        walk_index=walk,
    )


class _Work:
    def __init__(self, source: str, kind: str = "pdf", zim_pass: str = "html") -> None:
        self.source = source
        self.kind = kind
        self.zim_pass = zim_pass


class OverlapChoiceTest(unittest.TestCase):
    def test_only_cpu_ocr_beside_a_gpu_overlaps(self):
        self.assertTrue(cpu_ocr_overlaps_gpu(_assignment("nvidia", "tesseract")))
        self.assertTrue(cpu_ocr_overlaps_gpu(_assignment("nvidia", "tesseract", ocr_kind="nvidia")))
        self.assertFalse(cpu_ocr_overlaps_gpu(_assignment("cpu", "tesseract")))
        self.assertFalse(cpu_ocr_overlaps_gpu(_assignment("nvidia", "easyocr", ocr_kind="nvidia")))
        self.assertFalse(cpu_ocr_overlaps_gpu(None))

    def test_parallel_stays_off_until_it_is_turned_on(self):
        gpu = _assignment("nvidia", "tesseract")
        self.assertFalse(parallel_ocr_enabled(False, gpu))
        self.assertTrue(parallel_ocr_enabled(True, gpu))
        self.assertFalse(parallel_ocr_enabled(True, _assignment("cpu", "tesseract")))
        self.assertFalse(parallel_ocr_enabled(True, _assignment("nvidia", "easyocr", ocr_kind="nvidia")))
        self.assertFalse(parallel_ocr_enabled(True, None))
        engine = Path(__file__).resolve().parents[1].joinpath("nomad_turbo", "engine.py").read_text(encoding="utf-8")
        self.assertEqual(engine.count("parallel_ocr_enabled("), 2)
        from nomad_turbo.discover import Install, load_install
        import inspect

        self.assertFalse(Install.__dataclass_fields__["parallel_ocr"].default)
        self.assertIn('"parallel_ocr"', inspect.getsource(load_install))


class VisualReadTest(unittest.TestCase):
    def test_a_text_file_stops_the_read_before_a_later_pdf(self):
        order = [("a.pdf", "pdf"), ("notes.txt", "text"), ("b.pdf", "pdf")]
        self.assertEqual(visual_reads(order, reserved=set(), hold=None, limit=3), ["a.pdf"])

    def test_a_zim_on_the_gpu_still_reads_the_pdfs_after_it(self):
        order = [("book.zim", "zim"), ("a.pdf", "pdf"), ("b.png", "image"), ("notes.txt", "text"), ("c.pdf", "pdf")]
        found = visual_reads(order, reserved=set(), hold="book.zim", limit=3)
        self.assertEqual(found, ["a.pdf", "b.png"])
        past = visual_reads(order, reserved=set(), hold="book.zim", past_ordinary=True)
        self.assertEqual(past, ["a.pdf", "b.png", "c.pdf"])

    def test_a_zim_reaches_an_upload_that_follows_another_zim(self):
        files = [
            _file("maps.zim", "zim", 0),
            _file("wiki.zim", "zim", 1),
            _file("notes.txt", "text", 2),
            _file("book.pdf", "pdf", 3),
        ]
        kinds = {item.source: item.file_type for item in files}
        order = preview_order(
            files,
            choice="library",
            index_now="maps.zim",
            defer=set(),
            kind_of=lambda source: kinds[source],
            limit=None,
            past_ordinary=True,
        )
        found = visual_reads(order, reserved=set(), hold="maps.zim", past_ordinary=True)
        self.assertEqual(found, ["book.pdf"])

    def test_the_cap_reserved_files_and_a_full_lookahead_still_reads_the_hold(self):
        order = [("a.pdf", "pdf"), ("b.pdf", "pdf"), ("c.pdf", "pdf")]
        self.assertEqual(visual_reads(order, reserved=set(), hold="a.pdf", limit=2), ["a.pdf", "b.pdf"])
        self.assertEqual(visual_reads(order, reserved={"a.pdf"}, hold="a.pdf", limit=2), ["b.pdf"])
        full = visual_reads(order, reserved={"b.pdf", "c.pdf"}, hold="a.pdf", limit=2)
        self.assertEqual(full, ["a.pdf"])

    def test_preview_keeps_files_in_index_order_and_stops_at_the_cap(self):
        files = [
            _file("t1.txt", "text", 0),
            _file("a.pdf", "pdf", 1),
            _file("t2.txt", "text", 2),
            _file("b.pdf", "pdf", 3),
        ]
        kinds = {item.source: "pdf" if item.source.endswith(".pdf") else "text" for item in files}
        order = preview_order(
            files, choice="library", index_now=None, defer=set(), kind_of=lambda source: kinds[source], limit=3
        )
        self.assertEqual([source for source, _kind in order], ["t1.txt", "a.pdf", "t2.txt"])
        self.assertNotIn("b.pdf", [source for source, _kind in order])
        later = [
            _file("t1.txt", "text", 0),
            _file("t2.txt", "text", 1),
            _file("a.pdf", "pdf", 2),
        ]
        later_kinds = {item.source: "pdf" if item.source.endswith(".pdf") else "text" for item in later}
        stopped = preview_order(
            later, choice="library", index_now=None, defer=set(), kind_of=lambda source: later_kinds[source], limit=3
        )
        self.assertEqual([source for source, _kind in stopped], ["t1.txt", "t2.txt"])
        pulled = preview_order(
            files, choice="library", index_now="b.pdf", defer=set(), kind_of=lambda source: kinds[source], limit=3
        )
        self.assertEqual(pulled[0][0], "b.pdf")
        self.assertNotIn("a.pdf", [source for source, _kind in pulled])
        pdfs = [_file(f"p{index}.pdf", "pdf", index) for index in range(5)]
        capped = preview_order(
            pdfs,
            choice="library",
            index_now=None,
            defer=set(),
            kind_of=lambda _source: "pdf",
            limit=3,
        )
        self.assertEqual(len(capped), 3)
        opened = preview_order(
            pdfs,
            choice="library",
            index_now=None,
            defer=set(),
            kind_of=lambda _source: "pdf",
            limit=None,
        )
        self.assertEqual(len(opened), 5)


class VisualLaneTest(unittest.TestCase):
    def test_one_read_at_a_time_in_offer_order(self):
        gate = threading.Event()
        started: list[str] = []
        stop = threading.Event()

        def reader(work):
            started.append(work.source)
            if work.source == "a.pdf":
                self.assertTrue(gate.wait(2))
            return {"text": work.source}

        lane = VisualLane(reader, stop)
        self.addCleanup(lane.close)
        self.addCleanup(gate.set)
        lane.offer([_Work("a.pdf"), _Work("b.pdf"), _Work("a.pdf")])
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and "a.pdf" not in started:
            time.sleep(0.01)
        time.sleep(0.05)
        self.assertEqual(started, ["a.pdf"])
        gate.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not lane.has_result("b.pdf"):
            time.sleep(0.01)
        visual, err = lane.take("a.pdf")
        self.assertEqual(visual, {"text": "a.pdf"})
        self.assertIsNone(err)
        visual, err = lane.take("b.pdf")
        self.assertEqual(visual, {"text": "b.pdf"})
        self.assertIsNone(err)
        self.assertEqual(started, ["a.pdf", "b.pdf"])

    def test_a_failed_read_is_kept_and_the_next_file_still_runs(self):
        stop = threading.Event()

        def reader(work):
            if work.source == "a.pdf":
                raise RuntimeError("bad read")
            return None

        lane = VisualLane(reader, stop)
        self.addCleanup(lane.close)
        lane.offer([_Work("a.pdf"), _Work("b.pdf")])
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not lane.has_result("b.pdf"):
            time.sleep(0.01)
        visual, err = lane.take("a.pdf")
        self.assertIsNone(visual)
        self.assertEqual(str(err), "bad read")
        visual, err = lane.take("b.pdf")
        self.assertIsNone(visual)
        self.assertIsNone(err)

    def test_close_stops_the_open_read(self):
        started = threading.Event()
        stop = threading.Event()
        holder: dict = {}

        def reader(work):
            started.set()
            while not holder["lane"].stop_flag().is_set():
                time.sleep(0.01)
            raise RuntimeError("stopped")

        lane = VisualLane(reader, stop)
        holder["lane"] = lane
        self.addCleanup(lane.close)
        lane.offer([_Work("a.pdf"), _Work("b.pdf")])
        self.assertTrue(started.wait(2))
        lane.close()
        self.assertFalse(lane._thread.is_alive())
        _visual, err = lane.take("a.pdf")
        self.assertEqual(str(err), "stopped")
        _visual, err = lane.take("b.pdf")
        self.assertEqual(str(err), "ocr did not start")

    def test_text_past_the_memory_budget_waits_on_disk(self):
        self.assertEqual(ocr_ram_budget(1000), 250)
        self.assertEqual(ocr_ram_budget(0), 0)
        page = {"page": 1, "layer": "hello", "ocr": ""}
        self.assertEqual(text_bytes({"pages": [page], "drew": True}), len(b"hello"))
        stop = threading.Event()
        body = "x" * 40

        def reader(work):
            return {"pages": [{"text": body if work.source == "big.pdf" else "ok"}]}

        folder = Path(tempfile.mkdtemp(prefix="nomad-turbo-ocr-"))
        lane = VisualLane(reader, stop, spill_dir=folder, budget=len(b"ok"))
        self.addCleanup(lane.close)
        lane.offer([_Work("small.pdf"), _Work("big.pdf")])
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not lane.has_result("big.pdf"):
            time.sleep(0.01)
        self.assertIn("small.pdf", lane._ready)
        self.assertIn("big.pdf", lane._spilled)
        self.assertTrue(any(folder.glob("*.json")))
        small, err = lane.take("small.pdf")
        self.assertEqual(small["pages"][0]["text"], "ok")
        self.assertIsNone(err)
        big, err = lane.take("big.pdf")
        self.assertEqual(big["pages"][0]["text"], body)
        self.assertIsNone(err)
        self.assertEqual(list(folder.glob("*.json")), [])
        lane.close()
        self.assertEqual(list(folder.glob("*.json")), [])

    def test_a_zim_pdf_jumps_ahead_of_uploads_that_have_not_started(self):
        gate = threading.Event()
        started: list[str] = []
        stop = threading.Event()

        def reader(work):
            started.append(work.source)
            if work.source == "upload.pdf":
                self.assertTrue(gate.wait(2))
            return {"text": work.source}

        lane = VisualLane(reader, stop)
        self.addCleanup(lane.close)
        self.addCleanup(gate.set)
        lane.offer([_Work("upload.pdf"), _Work("other.pdf")])
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and started != ["upload.pdf"]:
            time.sleep(0.01)
        lane.offer([_Work("zim.pdf")], front=True)
        gate.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and set(started) != {"upload.pdf", "zim.pdf", "other.pdf"}:
            time.sleep(0.01)
        self.assertEqual(started, ["upload.pdf", "zim.pdf", "other.pdf"])

    def test_arrange_moves_waiting_reads_and_leaves_the_open_one(self):
        gate = threading.Event()
        started: list[str] = []
        stop = threading.Event()

        def reader(work):
            started.append(work.source)
            if work.source == "a.pdf":
                self.assertTrue(gate.wait(2))
            return {"text": work.source}

        lane = VisualLane(reader, stop)
        self.addCleanup(lane.close)
        self.addCleanup(gate.set)
        lane.offer([_Work("a.pdf"), _Work("b.pdf"), _Work("c.pdf")])
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and started != ["a.pdf"]:
            time.sleep(0.01)
        lane.arrange(["c.pdf", "missing.pdf"])
        view = lane.view()
        self.assertEqual(view["reading"]["source"], "a.pdf")
        self.assertEqual([item["source"] for item in view["queued"]], ["c.pdf", "b.pdf"])
        gate.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and set(started) != {"a.pdf", "c.pdf", "b.pdf"}:
            time.sleep(0.01)
        self.assertEqual(started, ["a.pdf", "c.pdf", "b.pdf"])
        self.assertEqual(lane.ready_sources(), ["a.pdf", "c.pdf", "b.pdf"])
        lane.arrange(["b.pdf", "a.pdf"])
        self.assertEqual(lane.ready_sources(), ["b.pdf", "a.pdf", "c.pdf"])


class OcrJobReportTest(unittest.TestCase):
    def test_a_queued_pdf_can_be_reported_before_it_has_articles(self):
        from nomad_turbo.engine import _report

        class Book:
            def __init__(self):
                self.kwargs = None

            def update(self, source, **kwargs):
                self.kwargs = kwargs

        class Session:
            def __init__(self):
                self.book = Book()

            def get_book(self):
                return self.book

        session = Session()
        job = OcrJob(source="/app/storage/kb_uploads/a.pdf", host="/tmp/a.pdf", kind="pdf", title="a.pdf")
        _report(session, job, "running")
        self.assertEqual(session.book.kwargs["articles"], 0)
        self.assertEqual(session.book.kwargs["chunks"], 0)
        self.assertEqual(session.book.kwargs["state"], "running")


class GpuCandidateTest(unittest.TestCase):
    def test_markdown_stays_embeddable_while_a_pdf_waits_for_ocr(self):
        files = [
            _Work("maps.zim", "zim"),
            _Work("notes.md", "text"),
            _Work("book.pdf", "pdf"),
            _Work("done.zim", "zim", "queued"),
        ]
        found = [item.source for item in gpu_candidates(files, overlap=True)]
        self.assertEqual(found, ["maps.zim", "notes.md"])
        now = [item.source for item in gpu_candidates(files, overlap=True, index_now="book.pdf")]
        self.assertIn("book.pdf", now)
        self.assertIn("notes.md", now)
        plain = [item.source for item in gpu_candidates(files, overlap=False)]
        self.assertEqual(plain, ["maps.zim", "notes.md", "book.pdf", "done.zim"])


class QueuedZimRowTest(unittest.TestCase):
    def test_one_ocr_article_leaves_the_zim_partial(self):
        import os
        from unittest import mock

        from nomad_turbo import engine
        from nomad_turbo.progress import ProgressBook
        from nomad_turbo.catalog import CatalogRow

        source = "/app/storage/zim/queued.zim"
        book = ProgressBook(
            [
                CatalogRow(
                    source=source,
                    name="queued.zim",
                    origin="zim",
                    kind="zim",
                    size_bytes=10,
                    state="partial",
                )
            ]
        )

        class Session:
            def get_book(self):
                return book

            def set_note(self, _note):
                return None

        work = engine.FileWork(
            host=Path("/tmp/queued.zim"),
            kind="zim",
            source=source,
            articles=4,
            chunks=10,
            zim_pass="queued",
            ocr_pending=["a.pdf", "b.pdf"],
            label="OCR text",
            phase="ocr-text",
        )
        job = OcrJob(
            source=source + "\na.pdf",
            host=Path("/tmp/a.pdf"),
            kind="pdf",
            title="a.pdf",
            owner=source,
            article_path="a.pdf",
        )

        class Lane:
            def job(self, _source):
                return job

            def take(self, _source):
                return {"drew": True, "pages": [{"text": "page"}]}, None

        session = Session()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"NOMAD_TURBO_STATE": str(tmp)}):
                with (
                    mock.patch.object(engine, "_embed_one_article", return_value=True),
                    mock.patch.object(engine, "_finish_queued_zim", lambda *_args: None),
                ):
                    stored = engine._embed_ready_source(
                        Lane(),
                        job.source,
                        None,
                        None,
                        threading.Event(),
                        session,
                        [work],
                        set(),
                        [],
                        None,
                        None,
                        set(),
                    )
        self.assertTrue(stored)
        shown = book.files_for([source])[source]
        self.assertEqual(shown.state, "partial")
        self.assertEqual(shown.status, "partial")
        self.assertEqual(work.ocr_pending, ["b.pdf"])

    def test_a_missing_zim_pdf_is_skipped_once(self):
        import json
        import os
        from unittest import mock

        from nomad_turbo import engine

        source = "/app/storage/zim/library.zim"
        path = "www.example.com/missing.pdf"
        warnings = []

        class Lane:
            def pending(self, _source):
                return False

            def has_result(self, _source):
                return False

            def offer(self, _jobs, front=False):
                raise AssertionError("a missing PDF was queued")

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"NOMAD_TURBO_STATE": str(tmp)}):
                work = engine.FileWork(host=Path(tmp) / "library.zim", kind="zim", source=source)
                spill = engine._spill_paths(source)
                work.spill = spill
                spill["ocr"].write_text(json.dumps(path) + "\n", encoding="utf-8")
                with mock.patch.object(engine.log, "warning", lambda *args: warnings.append(args)):
                    engine._catch_up_exports(Lane(), work)
                    engine._catch_up_exports(Lane(), work)
                self.assertEqual(len(warnings), 1)
                self.assertEqual(work.ocr_pending, [])
                self.assertEqual(work.ocr_skipped, [path])
                saved = json.loads(engine._checkpoint_file(source).read_text())
                self.assertEqual(saved["ocr_skipped"], [path])
                again = engine.FileWork(
                    host=work.host,
                    kind="zim",
                    source=source,
                    ocr_skipped=saved["ocr_skipped"],
                )
                again.spill = spill
                with mock.patch.object(engine.log, "warning", lambda *args: warnings.append(args)):
                    engine._catch_up_exports(Lane(), again)
                self.assertEqual(len(warnings), 1)
