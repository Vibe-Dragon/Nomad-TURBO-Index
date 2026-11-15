"""Per-ZIM skipped, bad, and OCR'd counts on the details lines."""

import tempfile
import time
import unittest
from pathlib import Path

from nomad_turbo.file_stats import FileStats, detail_lines


class ZimDetailCountsTest(unittest.TestCase):
    def test_details_show_skipped_bad_and_ocr_counts_after_reload(self):
        directory = tempfile.TemporaryDirectory(prefix="nomad-turbo-details-")
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "file-stats.json"
        source = "/app/storage/zim/guides.zim"
        stats = FileStats(path)
        stats.note_library(source, name="guides.zim", file_type="zim", size_bytes=1000, state="partial")
        stats.note_zim_counts(source, skipped=12, bad=3, ocr_documents=5)
        stats.set_priority(source, True)
        stats.save()

        loaded = FileStats(path)
        loaded.load()
        saved = loaded.get(source)
        lines = dict(detail_lines(saved))
        self.assertEqual(lines["Status"], "partial")
        self.assertEqual(dict(detail_lines(saved, status="Waiting OCR"))["Status"], "Waiting OCR")
        self.assertEqual(lines["Skipped articles"], "12")
        self.assertEqual(lines["Bad articles"], "3")
        self.assertEqual(lines["OCR'd documents"], "5")

        self.assertFalse(loaded.apply_collection_clock("unknown", 0))
        self.assertEqual(loaded.get(source).skipped_articles, 12)
        self.assertEqual(loaded.get(source).bad_articles, 3)
        self.assertEqual(loaded.get(source).ocr_documents, 5)

        newer = FileStats(path)
        newer.load()
        self.assertTrue(newer.apply_collection_clock("present", int(time.time()) + 3600))
        self.assertEqual(newer.get(source).skipped_articles, 0)
        self.assertEqual(newer.get(source).bad_articles, 0)
        self.assertEqual(newer.get(source).ocr_documents, 0)
        self.assertTrue(newer.get(source).priority)

        cleared = FileStats(path)
        cleared.load()
        cleared.note_zim_counts(source, skipped=4, bad=1, ocr_documents=2)
        cleared.save()
        self.assertTrue(cleared.apply_collection_clock("missing", 0))
        self.assertEqual(cleared.get(source).skipped_articles, 0)
        self.assertEqual(cleared.get(source).bad_articles, 0)
        self.assertEqual(cleared.get(source).ocr_documents, 0)
        self.assertTrue(cleared.get(source).priority)


if __name__ == "__main__":
    unittest.main()
