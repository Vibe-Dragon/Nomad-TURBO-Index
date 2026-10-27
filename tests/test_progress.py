import logging
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

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
    initial_log_split,
    fitted_list_position,
    list_pane_position,
    restored_log_top,
    usage_text,
)
from nomad_turbo import __version__
from nomad_turbo.window import (
    NOMAD_REPO_URL,
    REPO_URL,
    TESTED_NOMAD_VERSION,
    _MemoryLog,
    about_facts,
    index_store_types,
    repair_store_types,
    window_close_action,
)

ROOT = Path(__file__).resolve().parents[1]
_ABOUT_PHRASES = (
    "Vibe-Dragon",
    "MIT",
    "Copyright 2026",
    "when I get to it",
    "I am not responsible",
)


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

    def test_the_log_split_keeps_a_saved_height_and_an_old_position(self):
        self.assertEqual(initial_log_split(0, 900), 664)
        self.assertEqual(initial_log_split(822, 900), 820)
        self.assertEqual(initial_log_split(160, 900), 724)
        self.assertEqual(initial_log_split(0, 100), 50)
        self.assertEqual(restored_log_top(720, 0, 900), 164)
        self.assertEqual(restored_log_top(720, 500, 900), 500)
        self.assertEqual(restored_log_top(160, 0, 100), 50)

    def test_an_undragged_list_keeps_the_spare_space(self):
        self.assertEqual(list_pane_position(0, 600, 200), 384)
        self.assertEqual(list_pane_position(250, 600, 200), 250)
        self.assertEqual(list_pane_position(0, 100, 400), 48)
        self.assertEqual(list_pane_position(0, 80, 400), 40)
        self.assertEqual(list_pane_position(0, 428, 105), 307)
        # A tall pane is unchanged. A short pane keeps a list and a control strip.
        self.assertEqual(fitted_list_position(0, 428, 105), 307)
        self.assertEqual(fitted_list_position(250, 600, 200), 250)
        short = fitted_list_position(0, 81, 400)
        self.assertEqual(short, 25)
        self.assertGreater(short, 0)
        self.assertLessEqual(short + 48 + 8, 81)
        self.assertGreater(fitted_list_position(300, 81, 105), 0)
        self.assertLessEqual(fitted_list_position(300, 81, 105) + 48 + 8, 81)
        self.assertEqual(fitted_list_position(0, 62, 400), 6)
        # The list widget's own minimum is kept when the pane can hold it.
        self.assertEqual(fitted_list_position(0, 86, 105, 30), 30)
        self.assertEqual(fitted_list_position(0, 428, 105, 30), 307)

    def test_the_window_panes_can_shrink(self):
        from pathlib import Path

        text = Path("nomad_turbo/window.py").read_text(encoding="utf-8")
        self.assertIn("set_resizable(True)", text)
        self.assertIn("gtk.Paned", text)
        self.assertNotIn("set_size_request(-1, self._log_height)", text)
        self.assertLess(text.index("top.pack_start(log_header"), text.index("pack2(log_scroll"))
        self.assertLess(text.index("root.pack_start(self.qdrant_label"), text.index("root.pack_start(self.content_pane"))
        self.assertNotIn("top.pack_start(self.qdrant_label", text)
        self.assertIn("min-height: 6px", text)
        self.assertIn("self.index_pane", text)
        self.assertIn("self.repair_pane", text)
        self.assertIn("set_min_content_height(48)", text)
        self.assertNotIn("pack1(scrolled, True, False)", text)
        self.assertEqual(text.count("pack1(scrolled, True, True)"), 2)
        self.assertEqual(text.count("pack2(self._portion(lower), False, False)"), 2)
        self.assertEqual(text.count("set_min_content_height(1)"), 2)
        self.assertIn("set_min_content_height(48)", text)
        self.assertIn("pack1(top, True, False)", text)
        self.assertIn("pack2(log_scroll, False, True)", text)
        self.assertIn("https://github.com/Vibe-Dragon/Nomad-TURBO-Index", text)
        self.assertIn("Nomad TURBO Index", text)
        self.assertIn("bars.pack_start(self.current_bar", text)
        self.assertIn("bars.pack_start(self.overall", text)
        self.assertIn("bars.pack_start(self.scale", text)
        self.assertLess(text.index("bars.pack_start(self.current_bar"), text.index("bars.pack_start(self.overall"))
        self.assertLess(text.index("bars.pack_start(self.overall"), text.index("bars.pack_start(self.scale"))
        self.assertIn("repair_bars.pack_start(self.repair_current_bar", text)
        self.assertIn("repair_bars.pack_start(self.repair_overall", text)
        self.assertIn('add_titled(page, "about", "About")', text)
        self.assertIn("from nomad_turbo import __version__", text)
        self.assertNotIn(f'"{__version__}"', text)
        self.assertIn("set_size_request(480, 280)", text)

    def test_the_pane_reads_only_lines_it_has_not_shown(self):
        memory = _MemoryLog()
        memory.setFormatter(logging.Formatter("%(message)s"))
        for text in ("one", "two"):
            memory.emit(logging.LogRecord("nomad_turbo", logging.INFO, __file__, 1, text, (), None))
        self.assertEqual([(seq, msg) for seq, _level, msg in memory.since(0)], [(1, "one"), (2, "two")])
        self.assertEqual([(seq, msg) for seq, _level, msg in memory.since(1)], [(2, "two")])
        self.assertEqual(memory.since(2), [])


def _stub_install():
    from nomad_turbo.discover import Install

    install = Install(
        admin_container="nomad_admin",
        admin_image="project-nomad:test",
        network="nomad",
        redis_host="redis",
        redis_port=6379,
        redis_container="nomad_redis",
        mysql_container="nomad_mysql",
        qdrant_url="http://127.0.0.1:9",
        ollama_image="ollama/ollama:test",
        ollama_container="nomad_ollama",
        models_dir="/tmp/nomad-turbo-test-models",
        storage_host="/tmp/nomad-turbo-test-storage",
    )
    install.log_open = True
    install.log_split = 0
    install.index_pane = 0
    install.repair_pane = 0
    install.log_pane = 0
    return install


def _ui_log_path():
    override = os.environ.get("NOMAD_TURBO_UI_LOG")
    if override:
        return Path(override)
    scratch = Path("/tmp/grok-goal-d75e593b4696/implementer/ui-layout.log")
    if scratch.parent.is_dir():
        return scratch
    return None


def _dump(fields: dict) -> str:
    lines = [f"{key}={fields[key]}" for key in sorted(fields)]
    return "\n".join(lines) + "\n"


def _parse_dump(text: str) -> dict:
    fields = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key] = value
    return fields


def _source_layout_ok() -> str:
    """Structural checks used when a window cannot be mapped at the default size."""
    text = (ROOT / "nomad_turbo" / "window.py").read_text(encoding="utf-8")
    facts = about_facts()
    checks = {
        "version import": "from nomad_turbo import __version__" in text,
        "no second version literal": f'"{__version__}"' not in text,
        "about builder": "about_facts()" in text and 'add_titled(page, "about", "About")' in text,
        "shared index row": (
            "bars.pack_start(self.current_bar" in text
            and "bars.pack_start(self.overall" in text
            and "bars.pack_start(self.scale" in text
            and text.index("bars.pack_start(self.current_bar") < text.index("bars.pack_start(self.overall")
            and text.index("bars.pack_start(self.overall") < text.index("bars.pack_start(self.scale")
        ),
        "shared repair row": (
            "repair_bars.pack_start(self.repair_current_bar" in text
            and "repair_bars.pack_start(self.repair_overall" in text
        ),
        "floor request": "set_size_request(480, 280)" in text,
        "facts version": facts["version"] == __version__,
        "facts tested": f"Project NOMAD {TESTED_NOMAD_VERSION}" in facts["tested"],
        "facts repos": facts["turbo_repo"] == REPO_URL and facts["nomad_repo"] == NOMAD_REPO_URL,
        "facts phrases": all(phrase in facts["body"] for phrase in _ABOUT_PHRASES),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return "ok" if not failed else "failed: " + ", ".join(failed)


def _pump(gtk, seconds: float = 0.05) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        while gtk.events_pending():
            gtk.main_iteration_do(False)
        time.sleep(0.01)


def _widget_in(widget, root) -> bool:
    while widget is not None:
        if widget is root:
            return True
        widget = widget.get_parent()
    return False


def _span(widget, origin):
    point = widget.translate_coordinates(origin, 0, 0)
    if point is None:
        return None
    _x, y = point
    height = widget.get_allocated_height()
    if height <= 0:
        return None
    return (y, y + height)


def _control_scroller(pane):
    child = pane.get_child2()
    return child


def _content_origin(scroller):
    """Widget whose coordinates match the scroller adjustments."""
    origin = scroller.get_child() if scroller is not None else None
    if origin is not None and type(origin).__name__ == "Viewport":
        inner = origin.get_child()
        if inner is not None:
            return inner
    return origin


def _nudge(adjustment, start: int, size: int) -> None:
    if adjustment is None:
        return
    page = adjustment.get_page_size()
    if page <= 0:
        return
    lower = adjustment.get_lower()
    upper = adjustment.get_upper()
    value = adjustment.get_value()
    if start < value:
        value = start
    elif start + size > value + page:
        value = start + size - page
    adjustment.set_value(max(lower, min(value, max(lower, upper - page))))


def _scroll_into(button, scroller) -> None:
    """Scroll the real adjustments the way a short window expects a reader to."""
    if scroller is None or not scroller.get_mapped():
        return
    origin = _content_origin(scroller)
    if origin is None:
        return
    point = button.translate_coordinates(origin, 0, 0)
    if point is None:
        return
    x, y = point
    _nudge(scroller.get_hadjustment(), x, max(1, button.get_allocated_width()))
    _nudge(scroller.get_vadjustment(), y, max(1, button.get_allocated_height()))


def _rects_intersect(button, target) -> bool:
    if button is None or target is None:
        return False
    if not button.get_mapped() or not target.get_mapped():
        return False
    if button.get_allocated_width() <= 0 or button.get_allocated_height() <= 0:
        return False
    if target.get_allocated_width() <= 0 or target.get_allocated_height() <= 0:
        return False
    point = button.translate_coordinates(target, 0, 0)
    if point is None:
        return False
    x, y = point
    return (
        x < target.get_allocated_width()
        and x + button.get_allocated_width() > 0
        and y < target.get_allocated_height()
        and y + button.get_allocated_height() > 0
    )


def _placed_in(widget, parent) -> tuple[int, int]:
    """Origin y and the height that lies inside parent. A negative y is clipped off the top."""
    if widget is None or parent is None or not widget.get_mapped() or not parent.get_mapped():
        return -999, 0
    point = widget.translate_coordinates(parent, 0, 0)
    if point is None:
        return -999, 0
    _x, y = point
    height = widget.get_allocated_height()
    visible = min(parent.get_allocated_height(), y + height) - max(0, y)
    return int(y), max(0, int(visible))


def _window_span(widget, window) -> int:
    """How many pixels of widget's height lie inside the window."""
    if widget is None or not widget.get_mapped() or not window.get_mapped():
        return 0
    point = widget.translate_coordinates(window, 0, 0)
    if point is None:
        return 0
    _x, y = point
    height = widget.get_allocated_height()
    window_height = window.get_allocated_height()
    return max(0, min(window_height, y + height) - max(0, y))


def _floor_reach(app, window, gtk, page_name: str, button) -> dict:
    """Map one page at the current size and scroll a control into the scroller."""
    app.stack.set_visible_child_name(page_name)
    pane = app.index_pane if page_name == "index" else app.repair_pane
    listing = app.index_list if page_name == "index" else app.repair_list
    scroller = _control_scroller(pane)
    window_height = window.get_allocated_height()
    for _ in range(50):
        _pump(gtk, 0.05)
        window_height = window.get_allocated_height()
        list_y, list_visible = _placed_in(listing, pane)
        pane_height = pane.get_allocated_height()
        on_screen = _window_span(scroller, window)
        pane_on_screen = _window_span(pane, window)
        if (
            scroller is not None
            and scroller.get_mapped()
            and on_screen >= 34
            and list_y >= -1
            and list_visible >= 48
            and 48 < pane_height <= window_height + 4
            and pane_on_screen >= pane_height - 4
        ):
            break
    _scroll_into(button, scroller)
    _pump(gtk, 0.15)
    height = 0 if scroller is None else int(scroller.get_allocated_height())
    on_screen = _window_span(scroller, window)
    mapped = bool(scroller is not None and scroller.get_mapped() and on_screen >= 34)
    # The button has to sit in the scroller's on-screen viewport. Touching the
    # window from a stale off-screen pane does not count.
    visible = _rects_intersect(button, scroller) and on_screen >= 34 and _rects_intersect(button, window)
    list_y, list_visible = _placed_in(listing, pane)
    scroller_y, scroller_visible = _placed_in(scroller, pane)
    return {
        "list": int(listing.get_allocated_height()),
        "list_mapped": "1" if listing.get_mapped() and list_visible >= 48 and list_y >= -1 else "0",
        "list_y": list_y,
        "list_visible": list_visible,
        "pane": int(pane.get_allocated_height()),
        "mapped": "1" if mapped else "0",
        "height": on_screen if on_screen else scroller_visible,
        "scroller_y": scroller_y,
        "visible": "1" if visible else "0",
    }


def _overlaps(spans: dict) -> str:
    names = [name for name in ("current_bar", "overall", "scale", "buttons") if spans.get(name)]
    hits = []
    for index, left in enumerate(names):
        a = spans[left]
        for right in names[index + 1 :]:
            b = spans[right]
            if a[0] < b[1] and b[0] < a[1]:
                hits.append(f"{left}+{right}")
    return ",".join(hits)


def layout_report(floor: bool = True) -> str:
    """Build the real window and read its allocation. Does not call open_window or run."""
    facts = about_facts()
    base = {
        "version": facts["version"],
        "turbo_uri": facts["turbo_repo"],
        "nomad_uri": facts["nomad_repo"],
        "tested": facts["tested"],
        "body": facts["body"].replace("\n", " | "),
        "source": _source_layout_ok(),
    }
    try:
        import gi

        gi.require_version("Gtk", "3.0")
        from gi.repository import GLib, Gtk
    except Exception as exc:
        base.update(mode="structural", reason=f"gtk import failed: {exc}")
        return _dump(base)
    ready = Gtk.init_check()
    if isinstance(ready, tuple):
        ready = ready[0]
    if not ready:
        base.update(mode="structural", reason="Gtk.init_check failed")
        return _dump(base)

    from nomad_turbo.session import Session
    from nomad_turbo.window import _Window

    previous_state = os.environ.get("NOMAD_TURBO_STATE")
    previous_config = os.environ.get("NOMAD_TURBO_CONFIG")
    state = tempfile.TemporaryDirectory(prefix="nomad-turbo-layout-")
    os.environ["NOMAD_TURBO_STATE"] = str(Path(state.name) / "state")
    os.environ["NOMAD_TURBO_CONFIG"] = str(Path(state.name) / "config.json")
    app = None
    try:
        app = _Window(_stub_install(), Session(55), _MemoryLog(), Path(state.name) / "session.log", Gtk, GLib)
        window = app.window
        window.set_opacity(0)
        window.show_all()
        window.unmaximize()
        window.resize(1180, 820)
        size = (0, 0)
        for _ in range(40):
            _pump(Gtk, 0.05)
            alloc = window.get_allocation()
            size = (alloc.width, alloc.height)
            if abs(size[0] - 1180) <= 80 and abs(size[1] - 820) <= 80 and app.index_list.get_allocated_height() > 0:
                _pump(Gtk, 0.1)
                alloc = window.get_allocation()
                size = (alloc.width, alloc.height)
                break
        near = abs(size[0] - 1180) <= 80 and abs(size[1] - 820) <= 80
        if not near:
            base.update(
                mode="structural",
                reason=f"mapped window is {size[0]}x{size[1]}, not near 1180x820",
                window_size=f"{size[0]}x{size[1]}",
            )
            return _dump(base)

        def page_heights(page, listing, pane):
            last = None
            best = (0, 0)
            for _ in range(40):
                _pump(Gtk, 0.05)
                height = listing.get_allocated_height()
                page_height = page.get_allocated_height()
                position = pane.get_position()
                if page_height > 100 and height > 48:
                    best = (height, page_height)
                # Hidden pages report 1×1. An early allocate can also be taller
                # than the split the idle place is about to set. Wait for both.
                if (
                    height == last
                    and height > 48
                    and page_height > 100
                    and abs(height - position) <= 4
                ):
                    return height, page_height
                last = height
            return best

        index_list, index_page = page_heights(app.index_page, app.index_list, app.index_pane)
        spans = {
            "current_bar": _span(app.current_bar, window),
            "overall": _span(app.overall, window),
            "scale": _span(app.scale, window),
            "buttons": _span(app.index_buttons, window),
        }
        app.stack.set_visible_child_name("repair")
        repair_list, repair_page = page_heights(app.repair_page, app.repair_list, app.repair_pane)
        app.stack.set_visible_child_name("about")
        _pump(Gtk, 0.1)
        request = window.get_size_request()
        fields = {
            "mode": "measured",
            "reason": "",
            "version": app.version_label.get_text(),
            "title": window.get_title() or "",
            "turbo_uri": app.about_turbo_link.get_uri(),
            "nomad_uri": app.about_nomad_link.get_uri(),
            "tested": app.about_tested.get_text(),
            "about_heading": app.about_heading.get_text(),
            "body": app.about_body.get_text().replace("\n", " | "),
            "index_list": str(index_list),
            "index_page": str(index_page),
            "repair_list": str(repair_list),
            "repair_page": str(repair_page),
            "overlap": _overlaps(spans),
            "window_size": f"{size[0]}x{size[1]}",
            "request": f"{request.width}x{request.height}",
            "resizable": "1" if window.get_resizable() else "0",
            "source": base["source"],
            "about_parent": type(app.about_page.get_parent()).__name__ if app.about_page.get_parent() else "",
        }
        if floor:
            window.resize(480, 280)
            app.stack.set_visible_child_name("index")
            floor_size = (0, 0)
            for _ in range(30):
                _pump(Gtk, 0.05)
                alloc = window.get_allocation()
                floor_size = (alloc.width, alloc.height)
                if abs(floor_size[0] - 480) <= 40 and abs(floor_size[1] - 280) <= 40:
                    _pump(Gtk, 0.1)
                    alloc = window.get_allocation()
                    floor_size = (alloc.width, alloc.height)
                    break
            fields["floor_size"] = f"{floor_size[0]}x{floor_size[1]}"
            index_floor = _floor_reach(app, window, Gtk, "index", app.start_button)
            fields["floor_list"] = str(index_floor["list"])
            fields["floor_list_mapped"] = index_floor["list_mapped"]
            fields["floor_list_y"] = str(index_floor["list_y"])
            fields["floor_list_visible"] = str(index_floor["list_visible"])
            fields["floor_pane"] = str(index_floor["pane"])
            fields["floor_scroller_y"] = str(index_floor["scroller_y"])
            fields["floor_scroller_mapped"] = index_floor["mapped"]
            fields["floor_scroller_height"] = str(index_floor["height"])
            fields["floor_start_visible"] = index_floor["visible"]
            fields["start_in_tree"] = "1" if _widget_in(app.start_button, window) else "0"
            fields["start_label"] = app.start_button.get_label()
            repair_floor = _floor_reach(app, window, Gtk, "repair", app.repair_start)
            fields["floor_repair_list"] = str(repair_floor["list"])
            fields["floor_repair_list_mapped"] = repair_floor["list_mapped"]
            fields["floor_repair_list_y"] = str(repair_floor["list_y"])
            fields["floor_repair_list_visible"] = str(repair_floor["list_visible"])
            fields["floor_repair_pane"] = str(repair_floor["pane"])
            fields["floor_repair_scroller_y"] = str(repair_floor["scroller_y"])
            fields["floor_repair_scroller_mapped"] = repair_floor["mapped"]
            fields["floor_repair_scroller_height"] = str(repair_floor["height"])
            fields["floor_repair_visible"] = repair_floor["visible"]
            fields["repair_start_label"] = app.repair_start.get_label()
            final = window.get_allocation()
            fields["floor_size"] = f"{final.width}x{final.height}"
        return _dump(fields)
    except Exception as exc:
        base.update(mode="error", reason=f"{exc.__class__.__name__}: {exc}")
        return _dump(base)
    finally:
        if app is not None:
            try:
                app.drop_tray()
            except Exception:
                pass
            try:
                app.window.destroy()
            except Exception:
                pass
            _pump(Gtk, 0.05)
        if previous_state is None:
            os.environ.pop("NOMAD_TURBO_STATE", None)
        else:
            os.environ["NOMAD_TURBO_STATE"] = previous_state
        if previous_config is None:
            os.environ.pop("NOMAD_TURBO_CONFIG", None)
        else:
            os.environ["NOMAD_TURBO_CONFIG"] = previous_config
        state.cleanup()


class AboutAndLayoutTest(unittest.TestCase):
    def test_the_readme_and_changelog_match_the_about_page(self):
        facts = about_facts()
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        window = (ROOT / "nomad_turbo" / "window.py").read_text(encoding="utf-8")
        self.assertEqual(facts["version"], __version__)
        self.assertEqual(TESTED_NOMAD_VERSION, "1.35.1")
        self.assertIn(f"Project NOMAD {TESTED_NOMAD_VERSION}", facts["tested"])
        self.assertIn(f"tested with Project NOMAD {TESTED_NOMAD_VERSION}", readme)
        self.assertIn(facts["turbo_repo"], readme)
        self.assertIn(facts["nomad_repo"], readme)
        self.assertIn("About", readme)
        self.assertIn("package version", readme)
        self.assertIn("at least half", readme)
        self.assertIn("share one row", readme)
        for phrase in _ABOUT_PHRASES:
            self.assertIn(phrase, facts["body"])
        notes = changelog.split(f"## {__version__}", 1)[1].split("\n## ", 1)[0]
        self.assertIn("About", notes)
        self.assertIn("half the page", notes)
        self.assertIn("Project NOMAD 1.35.1", notes)
        self.assertIn("shared rows", notes)
        self.assertIn("about_facts()", window)
        self.assertEqual(_source_layout_ok(), "ok")

    def test_the_default_window_gives_the_lists_half_the_page(self):
        own = _parse_dump(layout_report(floor=True))
        env = os.environ.copy()
        child_state = tempfile.mkdtemp(prefix="nomad-turbo-layout-child-state-")
        env["NOMAD_TURBO_STATE"] = child_state
        env["NOMAD_TURBO_CONFIG"] = str(Path(child_state) / "config.json")
        script = (
            "import importlib.util\n"
            "from pathlib import Path\n"
            f"path = Path({str(Path(__file__))!r})\n"
            "spec = importlib.util.spec_from_file_location('turbo_layout_probe', path)\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "print(module.layout_report(floor=False), end='')\n"
        )
        try:
            child = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(ROOT),
                env=env,
                text=True,
                capture_output=True,
                timeout=120,
            )
        finally:
            import shutil

            shutil.rmtree(child_state, ignore_errors=True)
        child_text = child.stdout
        other = _parse_dump(child_text)
        log_path = _ui_log_path()
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("== parent ==\n")
                handle.write(_dump(own))
                handle.write("== child ==\n")
                handle.write(child_text if child_text.endswith("\n") or not child_text else child_text + "\n")
                if child.returncode != 0:
                    handle.write(f"== child stderr ({child.returncode}) ==\n")
                    handle.write(child.stderr)
                    handle.write("\n")
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(own.get("source"), "ok", own)
        self.assertEqual(other.get("source"), "ok", other)
        self.assertEqual(own.get("version"), other.get("version"))
        self.assertEqual(own.get("turbo_uri"), other.get("turbo_uri"))
        self.assertEqual(own.get("nomad_uri"), other.get("nomad_uri"))
        self.assertEqual(own.get("version"), __version__)
        self.assertEqual(own.get("turbo_uri"), REPO_URL)
        self.assertEqual(own.get("nomad_uri"), NOMAD_REPO_URL)
        self.assertIn(f"Project NOMAD {TESTED_NOMAD_VERSION}", own.get("tested", ""))
        for phrase in _ABOUT_PHRASES:
            self.assertIn(phrase, own.get("body", ""))
        for report in (own, other):
            self.assertNotEqual(report.get("mode"), "error", report)
            if report.get("mode") == "measured":
                index_list = int(report["index_list"])
                index_page = int(report["index_page"])
                repair_list = int(report["repair_list"])
                repair_page = int(report["repair_page"])
                self.assertGreater(index_page, 100, report)
                self.assertGreater(repair_page, 100, report)
                self.assertLess(abs(index_page - repair_page), 80, report)
                self.assertGreaterEqual(index_list * 2, index_page, report)
                self.assertGreaterEqual(repair_list * 2, repair_page, report)
                self.assertIn("+", report.get("overlap", ""), report)
                self.assertEqual(report.get("about_heading"), "About")
                self.assertIn(__version__, report.get("title", ""))
                self.assertLessEqual(int(report["request"].split("x")[0]), 480)
                self.assertLessEqual(int(report["request"].split("x")[1]), 280)
                self.assertEqual(report.get("resizable"), "1")
            else:
                self.assertEqual(report.get("mode"), "structural", report)
                self.assertTrue(report.get("reason"), report)
        if own.get("mode") == "measured":
            self.assertGreaterEqual(int(own.get("floor_list_visible", "0")), 48, own)
            self.assertGreaterEqual(int(own.get("floor_list_y", "-999")), -1, own)
            self.assertLessEqual(int(own.get("floor_pane", "9999")), int(own["floor_size"].split("x")[1]) + 4, own)
            self.assertEqual(own.get("floor_list_mapped"), "1", own)
            self.assertEqual(own.get("floor_scroller_mapped"), "1", own)
            self.assertGreaterEqual(int(own.get("floor_scroller_y", "-999")), -1, own)
            self.assertGreaterEqual(int(own.get("floor_scroller_height", "0")), 34, own)
            self.assertLessEqual(
                int(own["floor_list_visible"]) + int(own["floor_scroller_height"]),
                int(own["floor_pane"]) + 8,
                own,
            )
            self.assertEqual(own.get("floor_start_visible"), "1", own)
            self.assertEqual(own.get("start_in_tree"), "1", own)
            self.assertEqual(own.get("start_label"), "Start indexing")
            self.assertGreaterEqual(int(own.get("floor_repair_list_visible", "0")), 48, own)
            self.assertGreaterEqual(int(own.get("floor_repair_list_y", "-999")), -1, own)
            self.assertLessEqual(
                int(own.get("floor_repair_pane", "9999")),
                int(own["floor_size"].split("x")[1]) + 4,
                own,
            )
            self.assertEqual(own.get("floor_repair_list_mapped"), "1", own)
            self.assertEqual(own.get("floor_repair_scroller_mapped"), "1", own)
            self.assertGreaterEqual(int(own.get("floor_repair_scroller_y", "-999")), -1, own)
            self.assertGreaterEqual(int(own.get("floor_repair_scroller_height", "0")), 34, own)
            self.assertLessEqual(
                int(own["floor_repair_list_visible"]) + int(own["floor_repair_scroller_height"]),
                int(own["floor_repair_pane"]) + 8,
                own,
            )
            self.assertEqual(own.get("floor_repair_visible"), "1", own)
            self.assertEqual(own.get("repair_start_label"), "Start repair")
            floor_w, floor_h = own.get("floor_size", "0x0").split("x")
            self.assertLessEqual(abs(int(floor_w) - 480), 80, own)
            self.assertLessEqual(abs(int(floor_h) - 280), 80, own)
        self.assertEqual(own.get("mode"), other.get("mode"), (own, other))


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
