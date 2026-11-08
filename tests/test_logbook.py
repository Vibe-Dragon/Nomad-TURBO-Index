"""Collection logs stay for the life of the current index. No window is opened."""

import logging
import tempfile
import unittest
import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nomad_turbo.logbook import (
    DailyLogHandler,
    align_running_logs,
    clear_collection_logs,
    collection_directory,
    daily_log_path,
    export_logs,
    log_files,
    log_help_text,
    parse_stamp,
    prepare_collection_logs,
)
from nomad_turbo import logbook
from nomad_turbo.store import StoreError, collection_copy_state


def _clock(moment: datetime):
    return ("present", int(moment.timestamp()))


class CollectionLogTest(unittest.TestCase):
    def setUp(self):
        logbook._seen.clear()
        self._handlers = logging.getLogger().handlers[:]
        self._level = logging.getLogger().level

    def tearDown(self):
        root = logging.getLogger()
        saved = {id(handler) for handler in self._handlers}
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            if id(handler) not in saved:
                handler.close()
        for handler in self._handlers:
            if handler not in root.handlers:
                root.addHandler(handler)
        root.setLevel(self._level)
        logbook._seen.clear()

    def test_logs_stay_for_the_collection_and_roll_by_day(self):
        born = datetime(2026, 11, 7, 17, 23, 47)
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            (directory / "session-previous.log").write_text(
                "2026-11-07 17:09:18,532 INFO from the previous collection\n"
            )
            (directory / "session.log").write_text(
                "2026-11-07 17:26:37,826 INFO this collection\n"
                "2026-11-08 00:00:01,100 INFO after midnight\n"
            )
            (directory / "session-old.log").write_text("2026-11-01 01:00:00,000 INFO ancient\n")
            (directory / "turbo.log").write_text("console stays\n")
            with patch("nomad_turbo.logbook._read_clock", return_value=_clock(born)):
                prepare_collection_logs(directory, object())
            self.assertFalse((directory / "session-previous.log").exists())
            self.assertFalse((directory / "session.log").exists())
            self.assertFalse((directory / "session-old.log").exists())
            self.assertEqual((directory / "turbo.log").read_text(), "console stays\n")
            kept = (directory / "logs" / "2026-11-07.log").read_text()
            self.assertIn("this collection", kept)
            self.assertNotIn("previous collection", kept)
            self.assertIn("after midnight", (directory / "logs" / "2026-11-08.log").read_text())
            with patch("nomad_turbo.logbook._read_clock", return_value=_clock(born)):
                prepare_collection_logs(directory, object())
            self.assertEqual((directory / "logs" / "2026-11-07.log").read_text(), kept)

    def test_a_file_that_starts_before_the_collection_is_removed_whole(self):
        born = datetime(2026, 11, 7, 17, 23, 47)
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            (directory / "session.log").write_text(
                "2026-11-07 17:00:00,000 INFO before\n"
                "2026-11-07 18:00:00,000 INFO after the collection existed\n"
            )
            with patch("nomad_turbo.logbook._read_clock", return_value=_clock(born)):
                prepare_collection_logs(directory, object())
            self.assertEqual(log_files(directory), [])

    def test_a_missing_collection_clears_logs_and_a_timeout_does_not(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            day = directory / "logs"
            day.mkdir()
            (day / "2026-11-07.log").write_text("2026-11-07 18:00:00,000 INFO kept\n")
            (directory / "turbo.log").write_text("console\n")
            with patch("nomad_turbo.logbook._read_clock", return_value=("unknown", 0)):
                prepare_collection_logs(directory, object())
            self.assertIn("kept", (day / "2026-11-07.log").read_text())
            with patch("nomad_turbo.logbook._read_clock", return_value=("missing", 0)):
                prepare_collection_logs(directory, object())
            self.assertFalse((day / "2026-11-07.log").exists())
            self.assertEqual((directory / "turbo.log").read_text(), "console\n")

    def test_an_unreadable_first_line_is_not_guessed_away(self):
        born = datetime(2026, 11, 7, 17, 23, 47)
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            (directory / "session.log").write_text("this line has no timestamp\n")
            with patch("nomad_turbo.logbook._read_clock", return_value=_clock(born)):
                prepare_collection_logs(directory, object())
            self.assertEqual((directory / "session.log").read_text(), "this line has no timestamp\n")

    def test_midnight_opens_the_next_day_and_export_zips_both(self):
        when = {"now": datetime(2026, 11, 7, 23, 59, 0)}
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            handler = DailyLogHandler(directory, clock=lambda: when["now"])
            logger = logging.getLogger("nomad_turbo_logbook_midnight")
            logger.handlers = [handler]
            logger.propagate = False
            logger.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.info("before midnight")
            when["now"] = datetime(2026, 11, 8, 0, 0, 1)
            logger.info("after midnight")
            handler.close()
            self.assertIn("before midnight", (directory / "logs" / "2026-11-07.log").read_text())
            self.assertIn("after midnight", (directory / "logs" / "2026-11-08.log").read_text())
            again = DailyLogHandler(directory, clock=lambda: when["now"])
            logger.handlers = [again]
            logger.info("still the eighth")
            again.close()
            eighth = (directory / "logs" / "2026-11-08.log").read_text()
            self.assertIn("after midnight", eighth)
            self.assertIn("still the eighth", eighth)
            target = directory / "out" / "nomad-turbo-logs.zip"
            count = export_logs(directory, target)
            self.assertEqual(count, 2)
            with zipfile.ZipFile(target) as archive:
                self.assertEqual(archive.namelist(), ["2026-11-07.log", "2026-11-08.log"])
                self.assertIn("before midnight", archive.read("2026-11-07.log").decode())

    def test_clearing_starts_a_new_file_for_the_next_line(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            folder = directory / "logs"
            folder.mkdir()
            (folder / "2026-11-06.log").write_text("2026-11-06 12:00:00,000 INFO yesterday\n")
            (folder / "2026-11-07.log").write_text("2026-11-07 12:00:00,000 INFO today\n")
            clear_collection_logs(directory)
            logging.getLogger("nomad_turbo_after_clear").warning("knowledge index deleted")
            self.assertFalse((folder / "2026-11-06.log").exists())
            today = daily_log_path(directory).read_text(encoding="utf-8")
            self.assertIn("knowledge index deleted", today)
            self.assertNotIn("today", today)
            self.assertNotIn("yesterday", today)

    def test_the_collection_directory_follows_nomad_storage(self):
        path = collection_directory("/mnt/nomad/storage", "nomad_knowledge_base")
        self.assertEqual(
            path,
            Path("/mnt/nomad/storage/qdrant/collections/nomad_knowledge_base"),
        )

    def test_a_timeout_is_not_a_missing_collection(self):
        install = SimpleNamespace(qdrant_url="http://127.0.0.1:9", collection="nomad_knowledge_base")

        def unreachable(url, payload=None, timeout=60, method=None):
            raise StoreError("qdrant unreachable: timed out")

        def missing(url, payload=None, timeout=60, method=None):
            raise StoreError("qdrant 404: not found")

        def present(url, payload=None, timeout=60, method=None):
            return {"result": {"status": "green"}}

        with (
            patch("nomad_turbo.store._request", side_effect=unreachable),
            patch("nomad_turbo.store._health_ok", return_value=False),
        ):
            self.assertEqual(collection_copy_state(install), "unknown")
        with patch("nomad_turbo.store._request", side_effect=missing):
            self.assertEqual(collection_copy_state(install), "missing")
        with patch("nomad_turbo.store._request", side_effect=present):
            self.assertEqual(collection_copy_state(install), "present")

    def test_the_open_handler_follows_a_new_collection(self):
        born = datetime(2026, 11, 7, 17, 23, 47)
        later = datetime(2026, 11, 7, 19, 0, 0)
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            (directory / "session.log").write_text("2026-11-07 17:26:37,826 INFO this collection\n")
            with patch("nomad_turbo.logbook._read_clock", return_value=_clock(born)):
                align_running_logs(directory, object())
            # The open handler follows the machine clock. Pin it to the
            # collection's day so the next line stays in that day's file.
            for handler in logging.getLogger().handlers:
                if isinstance(handler, DailyLogHandler):
                    handler.clock = lambda: datetime(2026, 11, 7, 18, 0, 0)
            follow = logging.getLogger("nomad_turbo_align")
            follow.setLevel(logging.DEBUG)
            follow.info("still going")
            text = (directory / "logs" / "2026-11-07.log").read_text(encoding="utf-8")
            self.assertIn("this collection", text)
            self.assertIn("still going", text)
            with patch("nomad_turbo.logbook._read_clock", return_value=_clock(later)):
                align_running_logs(directory, object())
            self.assertFalse((directory / "logs" / "2026-11-07.log").exists())

    def test_the_help_and_the_window_describe_the_zip(self):
        text = log_help_text()
        self.assertIn("only change which lines the pane draws", text)
        self.assertIn("Debug starts off", text)
        self.assertIn("midnight", text)
        self.assertIn("one zip", text)
        self.assertIn("current copy of the knowledge index", text)
        self.assertNotIn("last session", text.lower())
        self.assertNotIn("Older session logs are deleted", text)
        self.assertIn("Optimize: not running.", text)
        self.assertIn("keeps Qdrant's full message", text)
        self.assertIn("cannot explain those in advance", text)
        window = Path("nomad_turbo/window.py").read_text(encoding="utf-8")
        engine = Path("nomad_turbo/engine.py").read_text(encoding="utf-8")
        self.assertIn("self.debug_check.set_active(False)", window)
        self.assertIn("file_handler.setLevel(logging.DEBUG)", window)
        switcher = window.index("header.pack_start(switcher")
        host = window.index("header.pack_start(self.host_button")
        export_button = window.index("self.log_export_group.pack_start(self.export_button")
        help_button = window.index("self.log_export_group.pack_start(self.log_help_button")
        self.assertLess(switcher, host)
        self.assertNotIn("header.pack_start(self.log_help_button", window)
        self.assertNotIn("log_header.pack_end(self.log_help_button", window)
        self.assertLess(export_button, help_button)
        self.assertIn("nomad-turbo-logs.zip", window)
        self.assertIn("prepare_collection_logs", window)
        self.assertIn("export_logs", window)
        self.assertNotIn("Current session", window)
        self.assertNotIn("Last session", window)
        self.assertIn("clear_collection_logs", engine)
        self.assertIn("align_running_logs", engine)
        self.assertIn("watch_collection_logs", engine)
        stamp = parse_stamp("2026-11-07 17:26:37,826 INFO this collection")
        self.assertEqual(stamp, datetime(2026, 11, 7, 17, 26, 37, 826000))


if __name__ == "__main__":
    unittest.main()
