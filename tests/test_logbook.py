"""Session logs keep the previous run and drop anything older. No window is opened."""

import unittest
from pathlib import Path

from nomad_turbo.logbook import CURRENT_NAME, PREVIOUS_NAME, log_help_text, prepare_session_logs


class SessionLogTest(unittest.TestCase):
    def test_a_new_session_keeps_only_the_previous_log(self):
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            (directory / CURRENT_NAME).write_text("first session\n")
            (directory / "session-old.log").write_text("older\n")
            (directory / "turbo.log").write_text("console grew\n")
            current, previous = prepare_session_logs(directory)
            self.assertEqual(current.name, CURRENT_NAME)
            self.assertEqual(previous.name, PREVIOUS_NAME)
            self.assertEqual(previous.read_text(), "first session\n")
            self.assertFalse(current.exists())
            self.assertFalse((directory / "session-old.log").exists())
            self.assertEqual((directory / "turbo.log").stat().st_size, 0)
            current.write_text("second\n")
            (directory / "session-extra.log").write_text("extra\n")
            prepare_session_logs(directory)
            self.assertEqual(previous.read_text(), "second\n")
            self.assertFalse((directory / "session-extra.log").exists())
            self.assertFalse(current.exists())

    def test_an_empty_session_does_not_replace_the_previous_log(self):
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            (directory / PREVIOUS_NAME).write_text("kept\n")
            (directory / CURRENT_NAME).write_text("")
            _current, previous = prepare_session_logs(directory)
            self.assertEqual(previous.read_text(), "kept\n")

    def test_the_help_explains_known_lines_and_not_unforeseen_errors(self):
        text = log_help_text()
        self.assertIn("only change which lines the pane draws", text)
        self.assertIn("last session", text.lower())
        self.assertIn("Older session logs are deleted", text)
        self.assertIn("Optimize: not running.", text)
        self.assertIn("keeps Qdrant's full message", text)
        self.assertIn("cannot explain those in advance", text)
        window = Path("nomad_turbo/window.py").read_text(encoding="utf-8")
        switcher = window.index("header.pack_start(switcher")
        help_button = window.index("header.pack_start(self.log_help_button")
        host = window.index("header.pack_start(self.host_button")
        self.assertLess(switcher, help_button)
        self.assertLess(help_button, host)
        self.assertIn("Current session", window)
        self.assertIn("Last session", window)
        self.assertIn("prepare_session_logs", window)
