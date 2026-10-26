import logging
import unittest

from nomad_turbo.decisions import article_outcome, hand_back_when_idle, nomad_embed_guard, repeat_failure
from nomad_turbo.catalog import (
    catalog_state,
    origin_for,
    row_color,
    wipe_allowed,
    wipe_phrase_ok,
    CatalogRow,
)
from nomad_turbo.pool import should_change_slots, target_slots
from nomad_turbo.progress import (
    format_bytes,
    format_current_file,
    preferred_column_width,
    ProgressBook,
    format_eta,
    format_run_stats,
    follow_target,
    log_line_visible,
    log_lines_to_drop,
    log_should_stick,
    next_log_height,
    saved_log_height,
    usage_text,
)
from nomad_turbo.window import _MemoryLog, index_store_types, repair_store_types, window_close_action


class IdleTest(unittest.TestCase):
    def test_an_empty_queue_hands_indexing_back(self):
        self.assertTrue(hand_back_when_idle(keep_control=False, downloads_pending=False))

    def test_stay_on_keeps_turbo_in_control(self):
        self.assertFalse(hand_back_when_idle(keep_control=True, downloads_pending=False))

    def test_a_download_holds_the_queue_open(self):
        self.assertFalse(hand_back_when_idle(keep_control=False, downloads_pending=True))

    def test_a_policy_change_is_put_back_to_manual(self):
        self.assertEqual(nomad_embed_guard("Manual", True), ("", False))
        self.assertEqual(nomad_embed_guard("Always", True), ("Manual", False))
        self.assertEqual(nomad_embed_guard("Always", False), ("Manual", True))
        self.assertEqual(nomad_embed_guard("Manual", False), ("", True))

    def test_a_bad_page_is_counted_apart_from_an_empty_one(self):
        self.assertEqual(article_outcome("parse", already_stored=False, stored_chunks=0), "bad")
        self.assertEqual(article_outcome("read", already_stored=False, stored_chunks=0), "bad")
        self.assertEqual(article_outcome("empty", already_stored=False, stored_chunks=0), "skipped")
        self.assertEqual(article_outcome(None, already_stored=False, stored_chunks=0), "skipped")
        self.assertEqual(article_outcome(None, already_stored=False, stored_chunks=4), "indexed")
        self.assertEqual(article_outcome("parse", already_stored=True, stored_chunks=0), "indexed")
        self.assertEqual(article_outcome("embed", already_stored=False, stored_chunks=0), "bad")
        self.assertEqual(article_outcome("embed", already_stored=True, stored_chunks=0), "indexed")

    def test_a_file_is_failed_only_after_the_same_article_breaks_twice(self):
        seen: set[tuple[str, int]] = set()
        self.assertFalse(repeat_failure(seen, "a.zim", 10))
        self.assertTrue(repeat_failure(seen, "a.zim", 10))
        self.assertFalse(repeat_failure(seen, "a.zim", 11))


class CatalogTest(unittest.TestCase):
    def test_uploads_and_zims_are_different_colors(self):
        self.assertEqual(origin_for("/app/storage/kb_uploads/notes.md"), "upload")
        self.assertEqual(origin_for("/app/storage/zim/wiki.zim"), "zim")
        self.assertNotEqual(row_color("zim", "pending"), row_color("upload", "pending"))
        self.assertNotEqual(row_color("zim", "pending"), row_color("zim", "indexed"))

    def test_a_continuation_is_not_finished(self):
        self.assertEqual(
            catalog_state(
                ingest_state="indexed",
                resume_articles=3243900,
                checkpoint_done=False,
                has_continuation=True,
            ),
            "partial",
        )
        self.assertEqual(
            catalog_state(
                ingest_state="indexed",
                resume_articles=0,
                checkpoint_done=False,
                has_continuation=False,
            ),
            "indexed",
        )
        self.assertEqual(
            catalog_state(
                ingest_state=None,
                resume_articles=0,
                checkpoint_done=False,
                has_continuation=False,
            ),
            "pending",
        )

    def test_wipe_needs_the_button_and_the_typed_phrase(self):
        self.assertTrue(wipe_phrase_ok(" DELETE "))
        self.assertFalse(wipe_phrase_ok("delete"))
        self.assertFalse(wipe_allowed(True, False))
        self.assertFalse(wipe_allowed(False, True))
        self.assertTrue(wipe_allowed(True, True))


class ProgressTest(unittest.TestCase):
    def test_eta_text(self):
        self.assertEqual(format_eta(None), "measuring")
        self.assertEqual(format_eta(0), "done")
        self.assertEqual(format_eta(45), "45s")
        self.assertEqual(format_eta(90 * 60), "1h 30m")
        self.assertEqual(format_eta(3 * 86400 + 4 * 3600), "3d 4h")

    def test_overall_bar_counts_finished_files_and_file_eta(self):
        book = ProgressBook(
            [
                CatalogRow("a.zim", "a.zim", "zim", "zim", 1000, "indexed", 10, 10),
                CatalogRow("b.zim", "b.zim", "zim", "zim", 3000, "partial", 1000, 4000),
            ]
        )
        early = book.snapshot(now=0)
        self.assertEqual(early.done_files, 1)
        self.assertAlmostEqual(early.overall_fraction, 0.4375, places=2)
        self.assertEqual(early.files[1].eta, "")
        book.update("b.zim", articles=1000, total=4000, chunks=1, state="running", now=0)
        book.update("b.zim", articles=2000, total=4000, chunks=2, state="running", now=10)
        later = book.snapshot(now=10)
        self.assertEqual(later.files[1].eta, "20s")
        self.assertIn("about", later.overall_text)
        self.assertAlmostEqual(later.articles_per_second, 100.0)

    def test_a_flat_library_rate_stays_on_measuring(self):
        book = ProgressBook(
            [CatalogRow("b.zim", "b.zim", "zim", "zim", 3000, "partial", 1000, 4000)]
        )
        book.update("b.zim", articles=1000, total=4000, chunks=1, state="running", now=0)
        book.update("b.zim", articles=1000, total=4000, chunks=1, state="running", now=5)
        snap = book.snapshot(now=5)
        self.assertIn("measuring", snap.overall_text)

    def test_a_new_file_appears_without_resetting_the_one_in_progress(self):
        book = ProgressBook(
            [CatalogRow("old.zim", "old.zim", "zim", "zim", 1000, "running", 50, 100)]
        )
        added = book.ensure_present(
            "/app/storage/zim/new.zim",
            name="new.zim",
            origin="zim",
            kind="zim",
            size_bytes=20,
        )
        self.assertTrue(added)
        self.assertFalse(
            book.ensure_present(
                "old.zim",
                name="old.zim",
                origin="zim",
                kind="zim",
                size_bytes=5,
            )
        )
        snap = book.snapshot(now=0)
        by_name = {item.name: item for item in snap.files}
        self.assertEqual(by_name["old.zim"].state, "running")
        self.assertEqual(by_name["old.zim"].count, "50 / 100")
        self.assertEqual(by_name["new.zim"].status, "waiting")
        book.drop("/app/storage/zim/new.zim")
        self.assertNotIn("new.zim", {item.name for item in book.snapshot(now=0).files})

    def test_chunk_rate_and_the_current_file_bar(self):
        book = ProgressBook(
            [CatalogRow("b.zim", "b.zim", "zim", "zim", 3_000_000, "running", 1000, 4000, chunks=100)]
        )
        book.update("b.zim", articles=1000, total=4000, chunks=100, state="running", now=0)
        book.update("b.zim", articles=1600, total=4000, chunks=700, state="running", now=10)
        snap = book.snapshot(now=10)
        self.assertAlmostEqual(snap.chunks_per_second, 60.0)
        self.assertEqual(snap.chunks_stored, 700)
        self.assertEqual(snap.files[0].chunks_text, "700")
        self.assertEqual(snap.files[0].size_text, "2.86 MB")
        self.assertIn("chk/s", snap.files[0].rate_text)
        line = format_run_stats("running", 10, snap)
        self.assertIn("3,600 chunks/min", line)
        self.assertIn("700 chunks stored", line)
        title, fraction, text = format_current_file(snap)
        self.assertEqual(title, "b.zim")
        self.assertAlmostEqual(fraction, 0.4)
        self.assertIn("700 chunks", text)
        self.assertIn("ETA", text)
        self.assertEqual(format_current_file(None)[0], "No file is indexing.")

    def test_column_width_uses_the_longest_text_and_stays_inside_the_limits(self):
        self.assertEqual(format_bytes(0), "—")
        self.assertEqual(format_bytes(1536), "1.50 KB")
        narrow = preferred_column_width(["ab"], header="Hi", minimum=40, maximum=80, px_per_char=10, padding=0)
        wide = preferred_column_width(["abcdefghijklmnop"], header="Hi", minimum=40, maximum=80, px_per_char=10, padding=0)
        self.assertEqual(narrow, 40)
        self.assertEqual(wide, 80)

    def test_skipped_and_bad_show_on_the_run_line(self):
        book = ProgressBook(
            [CatalogRow("b.zim", "b.zim", "zim", "zim", 3000, "running", 10, 100)]
        )
        book.note_article("skipped")
        book.note_article("bad")
        snap = book.snapshot(now=0)
        self.assertEqual((snap.skipped, snap.bad), (2, 1))
        line = format_run_stats("running", 12, snap)
        self.assertIn("Skipped 2", line)
        self.assertIn("Bad 1", line)
        self.assertNotIn("Skipped", format_run_stats("idle", None, snap))
        book.mark_all_unindexed()
        self.assertEqual((book.snapshot(now=0).skipped, book.snapshot(now=0).bad), (0, 0))

    def test_closing_while_indexing_hides_the_window(self):
        self.assertEqual(window_close_action(True), "hide")
        self.assertEqual(window_close_action(False), "quit")

    def test_scale_down_waits_for_the_open_batch(self):
        self.assertFalse(should_change_slots(True, 14, 6))
        self.assertTrue(should_change_slots(False, 14, 6))
        self.assertFalse(should_change_slots(False, 6, 6))

    def test_hundred_percent_asks_for_more_processes_than_forty(self):
        common = dict(
            total_mib=16384,
            used_before=400,
            per_slot_mib=1000,
            cpu_count=4,
            max_slots=64,
            forced=None,
            use_gpu=True,
            gpu_index=0,
            already_running=False,
        )
        self.assertGreater(target_slots(percent=100, **common), target_slots(percent=40, **common))

    def test_usage_line_shows_the_batch_and_the_server(self):
        text = usage_text(set_percent=80, chunks=51, server_up=True)
        self.assertIn("Set 80%", text)
        self.assertIn("51 chunks per call", text)
        self.assertIn("One embed server is running.", text)
        idle = usage_text(set_percent=100, chunks=64, server_up=False, slots_ignored=True)
        self.assertIn("Embed server: idle.", idle)
        self.assertIn("slots setting is ignored", idle)

    def test_log_filters(self):
        self.assertTrue(log_line_visible(logging.INFO, True, False))
        self.assertFalse(log_line_visible(logging.INFO, False, True, True))
        self.assertFalse(log_line_visible(logging.DEBUG, True, False))
        self.assertTrue(log_line_visible(logging.DEBUG, False, True))
        self.assertTrue(log_line_visible(logging.ERROR, False, False, True))
        self.assertTrue(log_line_visible(logging.WARNING, False, False, True))
        self.assertFalse(log_line_visible(logging.ERROR, True, True, False))
        self.assertFalse(log_line_visible(logging.WARNING, True, True, False))
        self.assertFalse(log_should_stick(False, 0, 100, 100))
        self.assertTrue(log_should_stick(True, 400, 100, 500))
        self.assertFalse(log_should_stick(True, 100, 100, 500))
        self.assertTrue(log_should_stick(True, 0, 0, 0))
        self.assertEqual(log_lines_to_drop(100), 0)
        self.assertEqual(log_lines_to_drop(4000), 0)
        self.assertEqual(log_lines_to_drop(4001), 1)
        self.assertEqual(log_lines_to_drop(8_000_000), 7_996_000)

    def test_a_file_over_two_gigabytes_fits_in_the_list(self):
        import gi

        gi.require_version("Gtk", "3.0")
        from gi.repository import Gtk

        if not Gtk.init_check():
            self.skipTest("no display")
        askubuntu = 2767228112
        index = Gtk.ListStore(*index_store_types())
        index.append(
            (
                "askubuntu.com_en_all_2026-10.zim",
                "askubuntu",
                "ZIM",
                "waiting",
                0,
                "0",
                "0",
                "2.6 GB",
                "",
                "",
                "#4488ff",
                9_000_000,
                50_000_000,
                askubuntu,
                0,
                10**9,
            )
        )
        self.assertEqual(index[0][13], askubuntu)
        repair = Gtk.ListStore(*repair_store_types())
        repair.append(("src", "name", "problem", "3e9", 0, -1, "waiting", "0s", "tip", "#4488ff", 3_000_000_000, 0))
        self.assertEqual(repair[0][10], 3_000_000_000)

    def test_the_file_list_keeps_a_few_files_ahead_in_view(self):
        self.assertEqual(follow_target(0, 20), 3)
        self.assertEqual(follow_target(10, 20), 13)
        self.assertEqual(follow_target(18, 20), 19)
        self.assertEqual(follow_target(19, 20), 19)
        self.assertIsNone(follow_target(None, 20))
        self.assertIsNone(follow_target(0, 0))
        self.assertIsNone(follow_target(-1, 5))
        self.assertIsNone(follow_target(5, 5))

    def test_the_log_height_ignores_an_old_window_split(self):
        self.assertEqual(saved_log_height(0), 220)
        self.assertEqual(saved_log_height(822), 220)
        self.assertEqual(saved_log_height(160), 160)
        self.assertEqual(next_log_height(220, 40, room=600), 260)
        self.assertEqual(next_log_height(220, -500, room=600), 120)
        self.assertEqual(next_log_height(220, 900, room=2000), 720)
        self.assertEqual(next_log_height(220, 900, room=400), 400)

    def test_the_log_bar_is_under_the_log_row(self):
        from pathlib import Path

        text = Path("nomad_turbo/window.py").read_text(encoding="utf-8")
        self.assertNotIn("gtk.Paned", text)
        self.assertIn("log_header.pack_start(log_head", text)
        handle = text.index("log_body.pack_start(self._build_log_handle()")
        scroll = text.index("log_body.pack_start(log_scroll")
        self.assertLess(handle, scroll)

    def test_the_pane_reads_only_lines_it_has_not_shown(self):
        memory = _MemoryLog()
        memory.setFormatter(logging.Formatter("%(message)s"))
        for text in ("one", "two"):
            memory.emit(logging.LogRecord("nomad_turbo", logging.INFO, __file__, 1, text, (), None))
        self.assertEqual([(seq, msg) for seq, _level, msg in memory.since(0)], [(1, "one"), (2, "two")])
        self.assertEqual([(seq, msg) for seq, _level, msg in memory.since(1)], [(2, "two")])
        self.assertEqual(memory.since(2), [])


class WindowScanNamesTest(unittest.TestCase):
    def test_repair_scan_does_not_replace_the_library_scan(self):
        import ast
        from pathlib import Path

        tree = ast.parse(Path("nomad_turbo/window.py").read_text())
        found = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name != "_Window":
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in {"_scan", "_repair_scan"}:
                    found.append((item.name, [arg.arg for arg in item.args.args]))
        self.assertEqual(found, [("_scan", ["self"]), ("_repair_scan", ["self", "mode"])])


if __name__ == "__main__":
    unittest.main()
