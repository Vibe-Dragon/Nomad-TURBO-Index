"""Skipped and bad counts follow the same collection clock as the log files."""

import tempfile
import time
import unittest
from pathlib import Path

from nomad_turbo.run_counts import RunCounts


class RunCountsTest(unittest.TestCase):
    def test_counts_reload_and_follow_the_collection_clock(self):
        directory = tempfile.TemporaryDirectory(prefix="nomad-turbo-counts-")
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "run-counts.json"
        counts = RunCounts(path)
        counts.note("skipped")
        counts.note("bad")
        counts.save()

        loaded = RunCounts(path)
        loaded.load()
        self.assertEqual((loaded.skipped, loaded.bad), (2, 1))

        kept = RunCounts(path)
        kept.load_and_apply("unknown", 0)
        self.assertEqual((kept.skipped, kept.bad), (2, 1))
        again = RunCounts(path)
        again.load()
        self.assertEqual((again.skipped, again.bad), (2, 1))

        older = RunCounts(path)
        older.load_and_apply("present", 1)
        self.assertEqual((older.skipped, older.bad), (2, 1))

        dropped = RunCounts(path)
        dropped.load_and_apply("present", int(time.time()) + 3600)
        self.assertEqual((dropped.skipped, dropped.bad), (0, 0))
        self.assertFalse(path.exists())

        fresh = RunCounts(path)
        fresh.note("bad")
        fresh.save()
        self.assertEqual((fresh.skipped, fresh.bad), (1, 1))
        cleared = RunCounts(path)
        cleared.load_and_apply("missing", 0)
        self.assertEqual((cleared.skipped, cleared.bad), (0, 0))
        self.assertFalse(path.exists())
        empty = RunCounts(path)
        empty.load()
        self.assertEqual((empty.skipped, empty.bad), (0, 0))


if __name__ == "__main__":
    unittest.main()
