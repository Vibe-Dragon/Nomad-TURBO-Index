import threading
import unittest

from nomad_turbo.repair import (
    format_repair_current,
    delete_may_still_be_running,
    delete_timeout,
    format_scan_stats,
    plan_repair,
    problem_text,
    restore_policy,
    run_repair,
    run_scan,
    should_resume,
    wait_for_clearance,
    RepairBoard,
    RepairDeps,
    RepairRow,
    RepairSnap,
)

ROOT = "/app/storage"
GONE = "/app/storage/zim/old.zim"
LIVE = "/app/storage/zim/live.zim"
UPLOAD = "/app/storage/kb_uploads/notes.txt"
DOCS = "/app/storage/docs/guide.pdf"


def _plan(**overrides):
    args = {
        "mode": "quick",
        "on_disk": {LIVE},
        "ingest": {},
        "checkpoints": [],
        "qdrant_sources": None,
        "container_root": ROOT,
    }
    args.update(overrides)
    return plan_repair(**args)


class PlanTest(unittest.TestCase):
    def test_a_missing_library_file_with_an_ingest_row_is_deleted(self):
        items = _plan(ingest={GONE: ("indexed", 40), LIVE: ("indexed", 5)})
        self.assertEqual([item.source for item in items], [GONE])
        self.assertEqual(items[0].action, "delete")
        self.assertEqual(items[0].problem, "File is gone and an ingest row remains")
        self.assertEqual(items[0].chunks, 40)
        self.assertIsNone(items[0].points)

    def test_a_checkpoint_for_a_missing_file_is_deleted(self):
        items = _plan(checkpoints=[{"source": GONE, "chunks": 2}, {"source": LIVE, "chunks": 9}])
        self.assertEqual([item.source for item in items], [GONE])
        self.assertEqual(items[0].problem, "File is gone and a checkpoint remains")

    def test_ingest_and_checkpoint_for_the_same_missing_file_are_one_row(self):
        items = _plan(
            ingest={GONE: ("indexed", 8)},
            checkpoints=[{"source": GONE, "chunks": 3}],
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].problem, "File is gone. An ingest row and a checkpoint remain")
        self.assertEqual(items[0].chunks, 8)

    def test_files_outside_the_library_are_left_alone_in_a_quick_repair(self):
        items = _plan(ingest={DOCS: ("indexed", 12)})
        self.assertEqual(items, [])

    def test_smaller_files_are_repaired_first(self):
        big = "/app/storage/zim/big.zim"
        small = "/app/storage/zim/small.zim"
        items = _plan(ingest={big: ("indexed", 5000), small: ("indexed", 3), GONE: ("indexed", 80)})
        self.assertEqual([item.name for item in items], ["small.zim", "old.zim", "big.zim"])

    def test_a_complete_repair_deletes_orphan_points_for_a_missing_file(self):
        items = _plan(
            mode="complete",
            ingest={LIVE: ("indexed", 5)},
            qdrant_sources={GONE: 26, LIVE: 5},
        )
        self.assertEqual([item.source for item in items], [GONE])
        self.assertEqual([item.action for item in items], ["delete"])
        self.assertEqual(items[0].points, 26)
        self.assertIn("index points", items[0].problem)

    def test_a_file_still_on_disk_is_reported_and_not_deleted(self):
        items = _plan(
            mode="complete",
            ingest={LIVE: ("indexed", 5)},
            qdrant_sources={LIVE: 9},
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].action, "note")
        self.assertEqual(items[0].problem, "Point count does not match the ingest row")

    def test_a_matching_file_is_not_listed(self):
        items = _plan(
            mode="complete",
            ingest={LIVE: ("indexed", 5)},
            qdrant_sources={LIVE: 5},
        )
        self.assertEqual(items, [])

    def test_points_with_no_ingest_row_are_reported_when_the_file_remains(self):
        items = _plan(mode="complete", qdrant_sources={LIVE: 4})
        self.assertEqual(items[0].action, "note")
        self.assertEqual(items[0].problem, "File is on disk with points and no ingest row")

    def test_an_indexed_file_with_no_points_is_reported(self):
        items = _plan(mode="complete", ingest={LIVE: ("indexed", 15)}, qdrant_sources={})
        self.assertEqual(items[0].problem, "Ingest row says indexed but no points were found")
        self.assertEqual(items[0].action, "note")

    def test_points_outside_the_library_are_not_deleted(self):
        items = _plan(mode="complete", qdrant_sources={DOCS: 70})
        self.assertEqual(items[0].action, "note")
        self.assertEqual(items[0].problem, "Points sit outside the ZIM and upload folders")

    def test_a_failed_source_list_still_repairs_known_missing_files(self):
        items = _plan(mode="complete", ingest={GONE: ("indexed", 1)}, qdrant_sources=None)
        self.assertEqual([item.action for item in items], ["delete", "note"])
        self.assertEqual(items[1].problem, "Qdrant did not list its sources")

    def test_problem_text_lists_every_leftover(self):
        self.assertEqual(
            problem_text({"ingest", "checkpoint", "points"}),
            "File is gone. An ingest row, a checkpoint, and index points remain",
        )

    def test_upload_paths_count_as_library_files(self):
        items = _plan(ingest={UPLOAD: ("indexed", 1)})
        self.assertEqual(items[0].source, UPLOAD)


class LockTest(unittest.TestCase):
    def test_a_finished_repair_puts_back_the_policy_it_found(self):
        self.assertEqual(restore_policy("Always", False), "Always")
        self.assertEqual(restore_policy("Manual", False), "Manual")
        self.assertEqual(restore_policy(None, False), None)
        self.assertTrue(should_resume(True, False))

    def test_a_delete_that_did_not_return_keeps_the_queue_paused(self):
        self.assertEqual(restore_policy("Always", True), "Manual")
        self.assertFalse(should_resume(True, True))
        self.assertTrue(delete_may_still_be_running("qdrant unreachable: timed out"))
        self.assertFalse(delete_may_still_be_running("qdrant 400: bad request"))
        self.assertFalse(delete_may_still_be_running("qdrant unreachable: [Errno 111] Connection refused"))

    def test_delete_timeout_grows_with_the_file_and_stays_bounded(self):
        self.assertEqual(delete_timeout(None, None), 600)
        self.assertEqual(delete_timeout(100, None), 600)
        self.assertEqual(delete_timeout(24000, None), 12000)
        self.assertEqual(delete_timeout(2_000_000, None), 6 * 60 * 60)

    def test_a_download_holds_the_repair_until_it_clears_or_stop_is_pressed(self):
        stop = threading.Event()
        reasons = iter(["download", "download", ""])

        def hold():
            return next(reasons)

        self.assertEqual(wait_for_clearance(stop, hold, lambda _seconds: False), "")

        stop.set()
        self.assertEqual(
            wait_for_clearance(stop, lambda: "download", lambda _seconds: False),
            "stopped",
        )


class _Fake:
    def __init__(self) -> None:
        self.policy = "Always"
        self.paused = False
        self.resumed = False
        self.deleted: list[str] = []
        self.forgot: list[str] = []
        self.hold = ""
        self.status: list[dict] = []
        self.facet = {GONE: 10}
        self.fail_delete = False

    def deps(self) -> RepairDeps:
        return RepairDeps(
            get_policy=lambda: self.policy,
            set_policy=self._set_policy,
            remember_policy=lambda _previous: None,
            pause_queue=self._pause,
            resume_queue=self._resume,
            active_embeds=lambda: 0,
            queue_is_paused=lambda: self.paused,
            on_disk=lambda: {LIVE},
            ingest=lambda: {GONE: ("indexed", 10)},
            checkpoints=lambda: [],
            facet_sources=lambda: self.facet,
            delete_points=self._delete,
            forget_ingest=self.forgot.append,
            drop_checkpoint=lambda _source: None,
            forget_jobs=lambda _source: None,
            hold_reason=lambda: self.hold,
            write_status=lambda **fields: self.status.append(fields),
            repair_hold=lambda: False,
            qdrant_up=lambda: True,
        )

    def _set_policy(self, policy: str) -> None:
        self.policy = policy

    def _pause(self) -> None:
        self.paused = True

    def _resume(self) -> None:
        self.resumed = True
        self.paused = False

    def _delete(self, source: str, _timeout: int) -> None:
        if self.fail_delete:
            raise RuntimeError("qdrant unreachable: timed out")
        self.deleted.append(source)


class RunTest(unittest.TestCase):
    def test_quick_repair_deletes_the_missing_file_and_restores_always(self):
        fake = _Fake()
        board = RepairBoard()
        run_repair(threading.Event(), board, "quick", fake.deps(), sleeper=lambda _seconds: False)
        self.assertEqual(fake.deleted, [GONE])
        self.assertEqual(fake.forgot, [GONE])
        self.assertEqual(fake.policy, "Always")
        self.assertTrue(fake.resumed)
        snap = board.snapshot()
        self.assertEqual(snap.phase, "done")
        self.assertEqual(snap.rows[0].status, "Repaired")
        self.assertEqual(snap.rows[0].percent, 100)
        self.assertFalse(any(item.get("repair_hold") for item in fake.status))

    def test_stop_leaves_the_next_file_untouched_and_unlocks(self):
        other = "/app/storage/zim/other.zim"
        fake = _Fake()
        original = fake.deps()

        def ingest():
            return {GONE: ("indexed", 1), other: ("indexed", 2)}

        original.ingest = ingest
        stop = threading.Event()

        def delete(source: str, _timeout: int) -> None:
            fake.deleted.append(source)
            stop.set()

        original.delete_points = delete
        board = RepairBoard()
        run_repair(stop, board, "quick", original, sleeper=lambda _seconds: stop.is_set())
        self.assertEqual(fake.deleted, [GONE])
        self.assertTrue(fake.resumed)
        self.assertEqual(fake.policy, "Always")
        snap = board.snapshot()
        self.assertEqual(snap.phase, "stopped")
        self.assertEqual(snap.rows[1].status, "Stopped")

    def test_a_timed_out_delete_does_not_resume_the_queue(self):
        fake = _Fake()
        fake.fail_delete = True
        board = RepairBoard()
        run_repair(threading.Event(), board, "quick", fake.deps(), sleeper=lambda _seconds: False)
        self.assertFalse(fake.resumed)
        self.assertEqual(fake.policy, "Manual")
        self.assertTrue(fake.status[-1]["repair_hold"])
        self.assertTrue(fake.status[-1]["paused_queue"])
        snap = board.snapshot()
        self.assertEqual(snap.phase, "error")
        self.assertEqual(snap.rows[0].status, "Failed")

    def test_a_down_index_does_not_pause_embedding(self):
        fake = _Fake()
        deps = fake.deps()
        deps.qdrant_up = lambda: False
        board = RepairBoard()
        run_repair(threading.Event(), board, "quick", deps, sleeper=lambda _seconds: False)
        self.assertFalse(fake.paused)
        self.assertEqual(fake.deleted, [])
        self.assertEqual(fake.policy, "Always")
        self.assertEqual(board.snapshot().phase, "error")

    def test_the_row_spec_changes_as_time_passes(self):
        board = RepairBoard()
        items = _plan(ingest={GONE: ("indexed", 4)})
        board.set_jobs(items)
        board.update(GONE, status="Deleting points", percent=15, started=10.0)
        first = board.snapshot(now=12.0)
        second = board.snapshot(now=75.0)
        self.assertEqual(first.rows[0].elapsed, "2s")
        self.assertEqual(second.rows[0].elapsed, "1m 5s")
        self.assertIn("Elapsed 1m 5s", second.rows[0].detail)
        self.assertGreaterEqual(second.rows[0].pulse, 0)
        self.assertNotEqual(first.rows[0].detail, second.rows[0].detail)


class ScanTest(unittest.TestCase):
    def test_the_current_repair_bar_names_the_file_in_progress(self):
        active = RepairRow(
            source="/zim/a.zim",
            name="a.zim",
            problem="index points",
            points="1,200",
            percent=40,
            pulse=3,
            status="Deleting points",
            elapsed="4s",
            detail="spec",
            color="#111",
            points_value=1200,
            elapsed_seconds=4,
        )
        snap = RepairSnap(
            phase="repairing",
            note="",
            mode="quick",
            rows=(active,),
            fraction=0.2,
            overall="0 of 1 done",
            busy=True,
        )
        title, fraction, text = format_repair_current(snap)
        self.assertEqual(title, "a.zim")
        self.assertAlmostEqual(fraction, 0.4)
        self.assertIn("Deleting points", text)
        self.assertIn("1,200", text)
        self.assertIn("4s", text)
        idle = RepairSnap("idle", "", "quick", (), 0.0, "Idle", False)
        self.assertEqual(format_repair_current(idle)[0], "No file is being repaired.")

    def test_scan_stats_list_the_stage_and_leave_unknowns_blank(self):
        text = format_scan_stats(
            "3s",
            stage="Reading ingest rows.",
            library_files=2,
            zim_files=1,
            upload_files=1,
            ingest_rows=3,
            ingest_states="2 indexed, 1 pending_decision",
            checkpoints=4,
            checkpoints_open=1,
            checkpoints_done=3,
            index_sources="skipped",
            would_delete=None,
            would_report=0,
            index_state="answering",
        )
        self.assertIn("Stage: Reading ingest rows.", text)
        self.assertIn("Elapsed: 3s", text)
        self.assertIn("Library on disk: 2 (1 ZIM, 1 uploads)", text)
        self.assertIn("Ingest rows: 3 (2 indexed, 1 pending_decision)", text)
        self.assertIn("Checkpoints: 4 (1 still open, 3 finished)", text)
        self.assertIn("Index sources: not read in a quick scan", text)
        self.assertIn("Would delete: …", text)
        self.assertIn("Report only: 0", text)
        self.assertIn("Index: answering", text)

    def test_a_scan_bar_follows_the_stage(self):
        board = RepairBoard()
        with board._lock:
            board.kind = "scan"
        board.set_phase("scanning", "Reading the library on disk.")
        board.set_facts(
            stage="Reading the library on disk.",
            fraction=0.2,
            library_files=3,
            zim_files=2,
            upload_files=1,
        )
        snap = board.snapshot(now=board.started)
        self.assertAlmostEqual(snap.fraction, 0.2)
        self.assertIn("Reading the library on disk.", snap.overall)
        self.assertIn("Library on disk: 3 (2 ZIM, 1 uploads)", snap.stats)

    def test_a_scan_does_not_pause_or_delete(self):
        fake = _Fake()
        board = RepairBoard()
        run_scan(threading.Event(), board, "quick", fake.deps())
        self.assertFalse(fake.paused)
        self.assertEqual(fake.deleted, [])
        self.assertEqual(fake.forgot, [])
        self.assertEqual(fake.policy, "Always")
        self.assertFalse(fake.resumed)
        self.assertEqual(fake.status, [])
        snap = board.snapshot()
        self.assertEqual(snap.phase, "done")
        self.assertEqual(snap.rows[0].status, "Would delete")
        self.assertEqual(snap.rows[0].source, GONE)
        self.assertEqual(snap.rows[0].percent, 100)
        self.assertIn("Would delete: 1", snap.stats)
        self.assertIn("not read in a quick scan", snap.stats)
        self.assertIn("Nothing was changed", snap.note)
        self.assertIn("would be deleted", snap.overall)

    def test_a_complete_scan_skips_the_source_list_when_qdrant_is_down(self):
        fake = _Fake()
        deps = fake.deps()
        deps.qdrant_up = lambda: False
        called = []
        deps.facet_sources = lambda: called.append("facet")
        board = RepairBoard()
        run_scan(threading.Event(), board, "complete", deps)
        self.assertEqual(called, [])
        self.assertFalse(fake.paused)
        self.assertEqual(fake.deleted, [])
        snap = board.snapshot()
        self.assertEqual(snap.rows[0].status, "Would delete")
        self.assertIn("the index did not answer", snap.stats)
        self.assertIn("Nothing was changed", snap.note)

    def test_a_complete_scan_counts_sources_when_the_index_answers(self):
        fake = _Fake()
        deps = fake.deps()
        calls = {"n": 0}

        def facet():
            calls["n"] += 1
            return {GONE: 10}

        deps.facet_sources = facet
        board = RepairBoard()
        run_scan(threading.Event(), board, "complete", deps)
        self.assertEqual(calls["n"], 1)
        self.assertFalse(fake.paused)
        snap = board.snapshot()
        self.assertEqual(snap.rows[0].status, "Would delete")
        self.assertIn("Index sources: 1 (10 points)", snap.stats)


if __name__ == "__main__":
    unittest.main()
