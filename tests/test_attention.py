"""Color for the Qdrant line and the pause line under it."""

import unittest
from pathlib import Path

from nomad_turbo.attention import (
    ALERT,
    READY,
    STOP_HANDOFF_NOTE,
    WATCH,
    WORKING,
    activity_status,
    markup,
    note_level,
    qdrant_level,
)
from nomad_turbo.engine import QDRANT_HOLD_NOTE

ROOT = Path(__file__).resolve().parents[1]


class AttentionTest(unittest.TestCase):
    def test_qdrant_color_follows_severity(self):
        ready = qdrant_level("Qdrant is ready for changes.", ready=True)
        self.assertEqual(ready, READY)
        self.assertIn("#1f7a3a", markup("Qdrant is ready for changes.", ready))
        self.assertIn("A &amp; B &lt;file&gt;", markup("A & B <file>", ready))
        writing = qdrant_level(
            "Qdrant is still writing. Do not restart it or delete the index.",
            ready=False,
        )
        self.assertEqual(writing, WATCH)
        self.assertEqual(
            qdrant_level("Qdrant is running. The ready check has not passed.", ready=False),
            WATCH,
        )
        self.assertEqual(
            qdrant_level("The knowledge index is not there.", ready=False, collection_missing=True),
            WATCH,
        )
        self.assertEqual(
            qdrant_level("Qdrant is ready for changes.", ready=True, collection_status="yellow"),
            WATCH,
        )
        self.assertEqual(
            qdrant_level("Qdrant is ready for changes.", ready=True, collection_status="red"),
            ALERT,
        )
        self.assertEqual(
            qdrant_level(
                "Qdrant is ready for changes.",
                ready=True,
                optimizer_error="segment build failed",
            ),
            ALERT,
        )
        self.assertEqual(
            qdrant_level("Qdrant is running and has not passed the health check.", ready=False),
            ALERT,
        )
        self.assertEqual(qdrant_level("Qdrant is exited. Not ready for changes.", ready=False), ALERT)
        self.assertEqual(qdrant_level("Checking Qdrant…", ready=False, unreadable=True), ALERT)

    def test_a_pause_is_named_when_the_file_list_is_still(self):
        self.assertEqual(activity_status(phase="running", work_alive=True), ("Indexing.", READY))
        self.assertEqual(note_level("Indexing."), None)
        text, level = activity_status(phase="running", note=STOP_HANDOFF_NOTE, work_alive=True)
        self.assertEqual(text, STOP_HANDOFF_NOTE)
        self.assertEqual(level, WORKING)
        text, level = activity_status(phase="running", note=QDRANT_HOLD_NOTE, work_alive=True)
        self.assertIn("Do not restart", text)
        self.assertEqual(level, WATCH)
        text, level = activity_status(
            phase="running",
            note="Indexing is paused so a segment can be optimized.",
            optimize_hold="segment",
            work_alive=True,
        )
        self.assertIn("paused so a segment", text)
        self.assertEqual(level, WATCH)
        text, level = activity_status(
            phase="waiting",
            note="Waiting for NOMAD to finish its current embed batches.",
        )
        self.assertEqual(level, WORKING)
        self.assertIn("Waiting for NOMAD", text)
        text, level = activity_status(
            phase="running",
            note="The disk is busy. Slowing knowledge-base writes.",
            work_alive=True,
        )
        self.assertEqual(level, WATCH)
        text, level = activity_status(
            phase="running",
            note="A ZIM download is in progress. Knowledge-base writes are waiting.",
            work_alive=True,
        )
        self.assertEqual(level, WORKING)
        self.assertEqual(activity_status(scanning=True)[0], "Scanning the library.")
        self.assertEqual(activity_status(repairing=True, repair_note="3 of 10 done")[1], WORKING)
        self.assertEqual(activity_status(phase="error", error="qdrant unreachable")[1], ALERT)
        self.assertEqual(activity_status()[0], "Idle.")
        waiting, level = activity_status(qdrant_ready=False, qdrant_attention=WATCH)
        self.assertEqual(level, WATCH)
        self.assertIn("Waiting until Qdrant", waiting)

    def test_stop_tells_the_window_before_the_records_finish(self):
        engine = (ROOT / "nomad_turbo" / "engine.py").read_text(encoding="utf-8")
        window = (ROOT / "nomad_turbo" / "window.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(engine.count("_note_stop_handoff("), 5)
        self.assertIn("STOP_HANDOFF_NOTE", engine)
        self.assertLess(
            window.index("root.pack_start(self.qdrant_label"),
            window.index("root.pack_start(self.activity_label"),
        )
        self.assertIn('TIPS["activity"]', window)
        self.assertIn("qdrant_level(", window)
        self.assertIn("activity_status(", window)
