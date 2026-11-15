"""Queue window order. Drags change what comes next. The running file stays."""

import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomad_turbo.engine import FileWork, _next_work
from nomad_turbo.queue_view import (
    QUEUE_SHOW,
    apply_manual,
    build_queue,
    choose_embed_source,
    display_title,
    load_queues,
    locked_moved,
    pinned_files,
    queue_caption,
    rank_embed_waiting,
    rewrite_side,
    save_queues,
    shown_slots,
    waiting_order,
)
from nomad_turbo.schedule import SWITCH_REASONS, OrderFile, pick_next
from nomad_turbo.session import Session


def _file(source: str, name: str, *, state: str = "pending", status: str = "Waiting", walk: int = 0, priority: bool = False, eta_seconds=None) -> SimpleNamespace:
    return SimpleNamespace(
        source=source,
        name=name,
        state=state,
        status=status,
        file_type="",
        size_bytes=10,
        mtime=None,
        total_articles=None,
        articles=0,
        chunks=0,
        eta_seconds=eta_seconds,
        walk_index=walk,
        priority=priority,
        percent=0,
        count="—",
        size_text="10 B",
        chunks_text="—",
        eta="",
        color="",
    )


def _order(source: str, *, priority: bool = False, state: str = "pending") -> OrderFile:
    return OrderFile(
        source=source,
        name=source,
        file_type="zim",
        size_bytes=10,
        mtime=None,
        zim_date=None,
        articles_total=None,
        completed=0.0,
        chunks=0,
        measured_eta=None,
        state=state,
        walk_index=0,
        priority=priority,
    )


class QueueOrderTest(unittest.TestCase):
    def test_one_waiting_list_splits_embed_and_ocr(self):
        files = [
            _file("zimA", "zimA.zim", walk=0),
            _file("pdf1", "pdf1.pdf", walk=1),
            _file("zimB", "zimB.zim", walk=2),
            _file("pdf2", "pdf2.pdf", walk=3),
            _file("md", "notes.md", walk=4),
        ]
        boards = build_queue(files, choice="library")
        self.assertEqual([slot.source for slot in boards.embed], ["zimA", "zimB", "md"])
        self.assertEqual([slot.source for slot in boards.ocr], ["pdf1", "pdf2"])
        self.assertFalse(any(slot.locked for slot in boards.embed))
        self.assertEqual(list(boards.sequential), ["zimA", "pdf1", "zimB", "pdf2", "md"])
        moved = rewrite_side(list(boards.sequential), boards.ocr_waiting, ["pdf2", "pdf1"])
        self.assertEqual(moved, ["zimA", "pdf2", "zimB", "pdf1", "md"])

    def test_the_file_being_read_stays_put_and_ocr_text_is_embedding(self):
        files = [
            _file("scan", "scan.pdf", state="running", status="OCR", walk=0),
            _file("zimA", "zimA.zim", walk=1),
            _file("later", "later.pdf", walk=2),
        ]
        boards = build_queue(files, choice="library")
        self.assertEqual([(slot.source, slot.locked) for slot in boards.ocr], [("scan", True), ("later", False)])
        self.assertEqual([slot.source for slot in boards.embed], ["zimA"])
        self.assertNotIn("scan", boards.sequential)
        text = [
            _file("scan", "scan.pdf", state="running", status="text", walk=0),
            _file("zimA", "zimA.zim", walk=1),
        ]
        stored = build_queue(text, choice="library")
        self.assertEqual([(slot.source, slot.locked) for slot in stored.embed], [("scan", True), ("zimA", False)])
        self.assertEqual(list(stored.ocr), [])
        stale = [
            _file("scan", "scan.pdf", state="running", status="OCR text", walk=0),
            _file("zimA", "zimA.zim", walk=1),
        ]
        leftover = build_queue(stale, choice="library")
        self.assertEqual([(slot.source, slot.locked) for slot in leftover.embed], [("scan", True), ("zimA", False)])
        self.assertEqual(list(leftover.ocr), [])
        self.assertTrue(locked_moved(["scan", "later"], ["later", "scan"], {"scan"}))
        self.assertFalse(locked_moved(["scan", "later"], ["scan", "other"], {"scan"}))

    def test_a_saved_queue_beats_priority_and_index_now_beats_the_queue(self):
        files = [
            _order("small", priority=True),
            _order("big"),
        ]
        picked = pick_next(files, choice="size_desc", current=None, index_now=None, manual=["big", "small"])
        self.assertEqual((picked.target, picked.reason), ("big", "queue"))
        stayed = pick_next(files, choice="size_desc", current="small", index_now=None, manual=["big"])
        self.assertEqual((stayed.target, stayed.switch, stayed.reason), ("small", False, "stay"))
        now = pick_next(files, choice="library", current=None, index_now="small", manual=["big"])
        self.assertEqual((now.target, now.reason), ("small", "now"))
        empty = pick_next(files, choice="library", current=None, index_now=None, manual=[])
        self.assertEqual(empty.target, "small")
        self.assertNotIn("queue", SWITCH_REASONS)
        self.assertEqual(waiting_order(["small", "big"], {"small"}, ["big", "small"]), ["big"])
        self.assertEqual(apply_manual(["a", "b", "c"], None), ["a", "b", "c"])

    def test_parallel_ocr_uses_the_lane_and_finished_text_waits_to_embed(self):
        files = [
            _file("zim", "book.zim", state="running", status="HTML", walk=0),
            _file("notes", "notes.md", walk=1),
            _file("upload", "upload.pdf", walk=2),
            _file("done", "done.pdf", walk=3),
        ]
        lane = {
            "reading": {
                "source": "zim\nchapter.pdf",
                "title": "chapter.pdf",
                "kind": "pdf",
                "articles": 2,
                "total_articles": 8,
                "chunks": 0,
            },
            "queued": [{"source": "upload", "title": "upload.pdf", "kind": "pdf"}],
            "ready": [{"source": "done", "title": "done.pdf", "kind": "pdf"}],
        }
        boards = build_queue(files, choice="library", lane=lane, embed_order=["notes", "done"])
        self.assertTrue(boards.parallel)
        self.assertEqual([(slot.source, slot.locked) for slot in boards.embed], [("zim", True), ("notes", False), ("done", False)])
        self.assertEqual([(slot.source, slot.locked) for slot in boards.ocr], [("zim\nchapter.pdf", True), ("upload", False)])
        self.assertEqual(boards.ocr[0].item.name, "chapter.pdf")
        self.assertEqual(boards.ocr[0].item.status, "OCR")
        self.assertNotIn("upload", [slot.source for slot in boards.embed])

    def test_each_column_shows_the_next_200_and_a_drag_keeps_the_rest(self):
        self.assertEqual(QUEUE_SHOW, 200)
        files = [_file(f"zim{i:04d}", f"zim{i:04d}.zim", walk=i) for i in range(250)]
        files[0] = _file("zim0000", "zim0000.zim", state="running", status="HTML", walk=0)
        boards = build_queue(files, choice="library")
        self.assertEqual(len(boards.embed), 250)
        shown = shown_slots(boards.embed)
        self.assertEqual(len(shown), 200)
        self.assertTrue(shown[0].locked)
        self.assertEqual(shown[0].source, "zim0000")
        self.assertEqual(queue_caption("Embedding", len(shown), len(boards.embed)), "Embedding · 200 of 250")
        self.assertEqual(queue_caption("OCR", 2, 2), "OCR")
        visible = [slot.source for slot in shown if not slot.locked]
        dragged = [visible[1], visible[0], *visible[2:]]
        full = list(boards.embed_waiting)
        merged = apply_manual(full, dragged)
        self.assertEqual(merged[0], visible[1])
        self.assertEqual(merged[1], visible[0])
        self.assertEqual(merged[199:], full[199:])

        queued = [
            {
                "source": f"zim\nfiles/{i}.pdf",
                "title": "null",
                "article_path": f"files/{i}.pdf",
                "kind": "pdf",
            }
            for i in range(250)
        ]
        lane = {
            "reading": {
                "source": "zim\nfiles/current.pdf",
                "title": "null",
                "article_path": "files/current.pdf",
                "kind": "pdf",
            },
            "queued": queued,
            "ready": [],
        }
        ocr = build_queue([_file("zim", "book.zim", state="running", status="HTML")], choice="library", lane=lane)
        self.assertEqual(len(ocr.ocr), 251)
        self.assertEqual(ocr.ocr[0].item.name, "current.pdf")
        self.assertEqual(ocr.ocr[1].item.name, "0.pdf")
        shown_ocr = shown_slots(ocr.ocr)
        self.assertEqual(len(shown_ocr), 200)
        self.assertTrue(shown_ocr[0].locked)
        self.assertEqual(shown_ocr[-1].source, "zim\nfiles/198.pdf")
        waiting = [slot.source for slot in shown_ocr if not slot.locked]
        moved = [waiting[1], waiting[0], *waiting[2:]]
        kept = apply_manual(list(ocr.ocr_waiting), moved)
        self.assertEqual(kept[0], waiting[1])
        self.assertEqual(kept[199:], list(ocr.ocr_waiting)[199:])
        self.assertEqual(len(kept), len(ocr.ocr_waiting))

    def test_shortest_eta_lists_the_next_embed_files_after_the_running_one(self):
        running = _file("running-zim", "running.zim", state="running", status="HTML", eta_seconds=1)
        notes = [
            _file(f"note-{index:04d}.md", f"note-{index:04d}.md", eta_seconds=5000 - index)
            for index in range(210)
        ]
        visual = _file("soon.pdf", "soon.pdf", eta_seconds=1)
        lane = {
            "reading": None,
            "queued": [],
            "ready": [{"source": "ready-ocr", "title": "Scanned page", "kind": "pdf"}],
        }
        boards = build_queue([running, *notes, visual], choice="eta_asc", lane=lane)
        sources = [slot.source for slot in boards.embed]
        self.assertEqual(sources[0], "running-zim")
        self.assertTrue(boards.embed[0].locked)
        self.assertEqual(sources[1], "ready-ocr")
        self.assertFalse(boards.embed[1].locked)
        waiting = sources[2:]
        self.assertEqual(waiting[0], "note-0209.md")
        self.assertEqual(waiting[-1], "note-0000.md")
        etas = [slot.item.eta_seconds for slot in boards.embed[2:]]
        self.assertEqual(etas, sorted(etas))
        self.assertNotIn("soon.pdf", sources)
        self.assertGreater(len(boards.embed), QUEUE_SHOW)
        shown = shown_slots(boards.embed)
        self.assertEqual(len(shown), QUEUE_SHOW)
        self.assertEqual([slot.source for slot in shown], sources[:QUEUE_SHOW])
        self.assertGreater(len(boards.embed_waiting), QUEUE_SHOW)
        self.assertEqual(list(boards.embed_waiting), sources[1:])

    def test_priority_stays_ahead_of_finished_ocr(self):
        running = _file("running-zim", "running.zim", state="running", status="HTML", eta_seconds=1)
        laws = _file("laws", "laws.zim", eta_seconds=9000, walk=1)
        scan = _file("scan.pdf", "scan.pdf", eta_seconds=50, walk=2)
        short = _file("short.md", "short.md", eta_seconds=1, walk=3)
        lane = {
            "reading": None,
            "queued": [],
            "ready": [
                {"source": "other.pdf", "title": "Other", "kind": "pdf"},
                {"source": "scan.pdf", "title": "scan.pdf", "kind": "pdf"},
            ],
        }
        boards = build_queue(
            [running, laws, scan, short],
            choice="eta_asc",
            priorities={"laws", "scan.pdf"},
            lane=lane,
        )
        sources = [slot.source for slot in boards.embed]
        self.assertEqual(sources[:5], ["running-zim", "scan.pdf", "laws", "other.pdf", "short.md"])
        self.assertTrue(boards.embed[0].locked)
        self.assertFalse(boards.embed[1].locked)

        program = ["short.md", "scan.pdf", "laws"]
        self.assertEqual(
            rank_embed_waiting(program, ["other.pdf", "scan.pdf"], {"laws", "scan.pdf"}),
            ["scan.pdf", "laws", "other.pdf", "short.md"],
        )
        rows = [
            _order("laws", priority=True),
            _order("scan.pdf", priority=True),
            _order("short.md"),
        ]
        rows[0] = OrderFile(**{**rows[0].__dict__, "measured_eta": 9000})
        rows[1] = OrderFile(**{**rows[1].__dict__, "measured_eta": 50, "file_type": "pdf", "name": "scan.pdf"})
        rows[2] = OrderFile(**{**rows[2].__dict__, "measured_eta": 1, "file_type": "md", "name": "short.md"})
        self.assertEqual(
            choose_embed_source(rows, ["other.pdf", "scan.pdf"], choice="eta_asc", index_now=None, manual=None),
            ("scan.pdf", "priority"),
        )
        self.assertEqual(
            choose_embed_source(rows, ["other.pdf", "scan.pdf"], choice="eta_asc", index_now="short.md", manual=None),
            ("short.md", "now"),
        )
        self.assertEqual(
            choose_embed_source(rows, ["other.pdf", "scan.pdf"], choice="eta_asc", index_now=None, manual=["short.md"]),
            ("short.md", "queue"),
        )
        plain = [row for row in rows if row.source != "scan.pdf"]
        self.assertEqual(
            choose_embed_source(plain, ["other.pdf"], choice="eta_asc", index_now=None, manual=None),
            ("laws", "priority"),
        )
        unmarked = [OrderFile(**{**row.__dict__, "priority": False}) for row in plain]
        self.assertEqual(
            choose_embed_source(unmarked, ["other.pdf"], choice="eta_asc", index_now=None, manual=None),
            ("other.pdf", "ocr"),
        )

    def test_a_null_title_uses_the_file_name_and_the_read_is_pinned(self):
        self.assertEqual(display_title("Chapter", "files/a.pdf"), "Chapter")
        self.assertEqual(display_title("null", "files/Traditional Ways (2).pdf"), "Traditional Ways (2).pdf")
        self.assertEqual(display_title(" None ", "files/a.pdf"), "a.pdf")
        self.assertEqual(display_title("", "zim\nfiles/a.pdf"), "a.pdf")
        source = "zim\nfiles/Traditional Ways (2).pdf"
        lane = {
            "reading": {
                "source": source,
                "title": "null",
                "article_path": "files/Traditional Ways (2).pdf",
                "kind": "pdf",
                "articles": 42,
                "total_articles": 383,
            },
            "queued": [],
            "ready": [],
        }
        running = [_file("zim", "book.zim", state="running", status="HTML")]
        pinned = pinned_files(running, lane)
        self.assertEqual([item.source for item in pinned], ["zim", source])
        self.assertEqual(pinned[1].name, "Traditional Ways (2).pdf")
        self.assertEqual(pinned[1].state, "running")
        upload = [
            _file("now.pdf", "now.pdf", state="running", status="HTML"),
            _file("mid.pdf", "mid.pdf", state="running", status="Waiting"),
        ]
        same = pinned_files(upload, {"reading": {"source": "mid.pdf", "title": "mid.pdf", "kind": "pdf"}})
        self.assertEqual([item.source for item in same], ["now.pdf", "mid.pdf"])
        self.assertEqual([item.source for item in pinned_files(upload, None)], ["now.pdf", "mid.pdf"])
        # The read sorts first. The row above the list is still the file being stored.
        ordered = [
            _file("scan.pdf", "scan.pdf", state="running", status="OCR"),
            _file("laws.zim", "laws.zim", state="running", status="HTML"),
        ]
        lane_read = {"reading": {"source": "scan.pdf", "title": "scan.pdf", "kind": "pdf"}}
        self.assertEqual([item.source for item in pinned_files(ordered, lane_read)], ["laws.zim", "scan.pdf"])
        self.assertEqual([item.source for item in pinned_files(ordered, None)], ["laws.zim", "scan.pdf"])
        alone = pinned_files([], lane)
        self.assertEqual([item.source for item in alone], [source])
        self.assertEqual(alone[0].name, "Traditional Ways (2).pdf")

    def test_the_pdf_being_stored_stays_above_the_one_being_read(self):
        from nomad_turbo.engine import _note_embedding_chunks, _publish_embedding
        from nomad_turbo.overlap import OcrJob

        session = Session(1)
        job = OcrJob(
            source="zim\nfiles/Stored.pdf",
            host=Path("/tmp/stored.pdf"),
            kind="pdf",
            title="null",
            owner="zim",
            article_path="files/Stored.pdf",
        )
        upload = OcrJob(source="/tmp/plain.pdf", host=Path("/tmp/plain.pdf"), kind="pdf", title="plain.pdf")
        self.assertIsNone(_publish_embedding(session, upload, "text"))
        self.assertIsNone(session.embedding_file())
        _publish_embedding(session, job, "text")
        face = session.embedding_file()
        self.assertEqual(face["title"], "Stored.pdf")
        self.assertEqual(face["status"], "text")
        _note_embedding_chunks(session, 40)
        _note_embedding_chunks(session, 40)
        self.assertEqual(session.embedding_file()["chunks"], 40)
        running = [_file("zim", "book.zim", state="running", status="HTML")]
        lane = {
            "reading": {
                "source": "zim\nfiles/Reading.pdf",
                "title": "Reading.pdf",
                "article_path": "files/Reading.pdf",
                "kind": "pdf",
            },
            "queued": [],
            "ready": [{"source": job.source, "title": "Stored.pdf", "kind": "pdf"}],
        }
        pinned = pinned_files(running, lane, session.embedding_file())
        self.assertEqual([item.source for item in pinned], [job.source, "zim\nfiles/Reading.pdf"])
        self.assertEqual(pinned[0].name, "Stored.pdf")
        self.assertEqual(pinned[0].status, "text")
        self.assertEqual(pinned[0].state, "running")
        self.assertEqual(pinned[0].chunks_text, "40")
        self.assertEqual(pinned[1].name, "Reading.pdf")
        self.assertEqual(pinned[1].status, "OCR")
        boards = build_queue(running, choice="library", lane=lane, embedding=session.embedding_file())
        self.assertEqual(boards.embed[0].source, job.source)
        self.assertTrue(boards.embed[0].locked)
        self.assertEqual(boards.embed[0].item.name, "Stored.pdf")
        self.assertEqual(boards.embed[0].item.status, "text")
        self.assertNotIn(job.source, boards.embed_waiting)
        self.assertEqual(boards.embed[1].source, "zim")
        quiet = pinned_files([], lane, session.embedding_file())
        self.assertEqual([item.name for item in quiet], ["Stored.pdf", "Reading.pdf"])
        _publish_embedding(session, None)
        self.assertIsNone(session.embedding_file())
        restored = pinned_files(running, lane, session.embedding_file())
        self.assertEqual(restored[0].source, "zim")
        self.assertEqual(restored[1].name, "Reading.pdf")

    def test_the_saved_lists_round_trip_and_the_next_file_follows_them(self):
        state = tempfile.TemporaryDirectory(prefix="nomad-turbo-queue-")
        self.addCleanup(state.cleanup)
        previous = os.environ.get("NOMAD_TURBO_STATE")
        os.environ["NOMAD_TURBO_STATE"] = state.name
        try:
            self.assertEqual(load_queues(), (None, None))
            save_queues(["/b.md", "/a.md"], ["/b.pdf"])
            self.assertEqual(load_queues(), (["/b.md", "/a.md"], ["/b.pdf"]))
            save_queues(None, None)
            self.assertFalse((Path(state.name) / "queue.json").exists())
            session = Session(8)
            session.set_embed_queue(["/b.md", "/a.md"])
            works = [
                FileWork(host=Path("/tmp/a.md"), kind="text", source="/a.md"),
                FileWork(host=Path("/tmp/b.md"), kind="text", source="/b.md"),
            ]
            work, *_rest = _next_work(None, None, session, works, threading.Event())
            self.assertEqual(work.source, "/b.md")
            bare = SimpleNamespace(get_order=lambda: "library", get_book=lambda: None)
            again, *_rest = _next_work(None, None, bare, works, threading.Event())
            self.assertEqual(again.source, "/a.md")
        finally:
            if previous is None:
                os.environ.pop("NOMAD_TURBO_STATE", None)
            else:
                os.environ["NOMAD_TURBO_STATE"] = previous
