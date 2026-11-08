"""Queue window order. Drags change what comes next. The running file stays."""

import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomad_turbo.engine import FileWork, _next_work
from nomad_turbo.queue_view import (
    apply_manual,
    build_queue,
    load_queues,
    locked_moved,
    rewrite_side,
    save_queues,
    waiting_order,
)
from nomad_turbo.schedule import SWITCH_REASONS, OrderFile, pick_next
from nomad_turbo.session import Session


def _file(source: str, name: str, *, state: str = "pending", status: str = "Waiting", walk: int = 0, priority: bool = False) -> SimpleNamespace:
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
        eta_seconds=None,
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
            _file("scan", "scan.pdf", state="running", status="OCR text", walk=0),
            _file("zimA", "zimA.zim", walk=1),
        ]
        stored = build_queue(text, choice="library")
        self.assertEqual([(slot.source, slot.locked) for slot in stored.embed], [("scan", True), ("zimA", False)])
        self.assertEqual(list(stored.ocr), [])
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
