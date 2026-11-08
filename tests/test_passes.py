"""OCR quality, ZIM passes, spelling, and duplicate text."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nomad_turbo.catalog import CatalogRow
from nomad_turbo.discover import Install, state_dir
from nomad_turbo.engine import FileWork, _write_checkpoint
from nomad_turbo.packing import prepare_article
from nomad_turbo.passes import (
    ArticlePrints,
    ChunkLedger,
    bar_color,
    boilerplate,
    classify_entry,
    correct_scan,
    fingerprint,
    join_pages,
    merge_page,
    normalize_ocr_quality,
    same_work,
)
from nomad_turbo.progress import ProgressBook
from nomad_turbo.runtime import start_extract, visual_pages


def _install(storage: str) -> Install:
    return Install(
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
        models_dir=str(Path(storage) / "models"),
        storage_host=storage,
    )


class QualityTest(unittest.TestCase):
    def test_unknown_quality_is_medium(self):
        self.assertEqual(normalize_ocr_quality("high"), "high")
        self.assertEqual(normalize_ocr_quality("LOW"), "low")
        self.assertEqual(normalize_ocr_quality("nope"), "medium")
        self.assertEqual(normalize_ocr_quality(None), "medium")

    def test_bar_colors(self):
        self.assertEqual(bar_color("HTML"), "#2f9e44")
        self.assertEqual(bar_color("text"), "#1c7ed6")
        self.assertEqual(bar_color("OCR"), "#e8590c")
        self.assertEqual(bar_color("OCR text"), "#7048e8")
        self.assertEqual(bar_color("indexing"), "")


class ClassifyTest(unittest.TestCase):
    def test_text_pdf_and_pictures_are_kept_and_tiles_are_not(self):
        self.assertEqual(classify_entry("text/html", "A/article.html", 10), "html")
        self.assertEqual(classify_entry("application/pdf", "shelf/book.pdf", 100_000), "ocr")
        self.assertEqual(classify_entry("application/epub+zip", "books/tale.epub", 1000), "text")
        self.assertEqual(
            classify_entry(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "books/tale.docx",
                1000,
            ),
            "text",
        )
        self.assertEqual(classify_entry("image/svg+xml", "fig.svg", 500), "skip")
        self.assertEqual(classify_entry("image/png", "maps/tiles/0/0/0.png", 100_000), "skip")
        self.assertEqual(classify_entry("image/webp", "natural_earth/ne2sr/6/16/29.webp", 100_000), "skip")
        self.assertEqual(classify_entry("image/jpeg", "photos/2020/01/15.jpg", 100_000), "skip")
        self.assertEqual(classify_entry("image/png", "icons/dot.png", 100), "skip")
        self.assertEqual(classify_entry("image/png", "diagram.png", 20_000), "skip")
        self.assertEqual(classify_entry("text/plain", "notes/readme.txt", 50), "skip")

    def test_a_short_readme_is_boilerplate_and_a_long_one_is_not(self):
        self.assertTrue(boilerplate("docs/README.md", "hello there"))
        self.assertFalse(boilerplate("docs/README.md", "word " * 250))
        self.assertFalse(boilerplate("docs/chapter.txt", "hello"))


class PageJoinTest(unittest.TestCase):
    def test_pages_are_grouped_near_three_thousand_characters(self):
        sections = join_pages(
            [(4, "a" * 3000), (5, "b" * 1000), (6, "c" * 1000), (7, "d" * 500)],
            limit=3000,
        )
        self.assertEqual([item["heading"] for item in sections], ["page 4", "pages 5-7"])
        prepared = prepare_article(
            {
                "title": "bandaging.pdf",
                "path": "shelf/bandaging.pdf",
                "sections": sections,
            },
            "medicine.zim",
            {},
            "doc",
        )
        self.assertEqual(prepared.sections[0].meta["section_title"], "page 4")
        self.assertEqual(prepared.sections[1].meta["section_title"], "pages 5-7")
        self.assertTrue(prepared.inputs[0].startswith("search_document: bandaging.pdf > page 4\n"))
        self.assertTrue(prepared.inputs[1].startswith("search_document: bandaging.pdf > pages 5-7\n"))
        self.assertFalse(prepared.sections[0].payloads[0].startswith("bandaging"))

    def test_a_chunk_already_stored_is_left_out(self):
        body = "catheter " * 40

        def keep(payload: str) -> bool:
            return "catheter" not in payload

        prepared = prepare_article(
            {"title": "note", "path": "note.txt", "sections": [{"heading": "", "text": body, "level": 2}]},
            "note.txt",
            {},
            "doc",
            keep_chunk=keep,
        )
        self.assertEqual(prepared.inputs, [])
        self.assertEqual(prepared.sections, [])


class SpellingTest(unittest.TestCase):
    def test_a_one_edit_miss_loses_and_an_unknown_word_stays(self):
        words = {"catheter", "label"}
        merged = merge_page("The catheter is placed.", "The catheer is placed. label", words)
        self.assertIn("catheter", merged)
        self.assertNotIn("catheer", merged)
        self.assertIn("label", merged)
        unknown = merge_page("The catheter is placed.", "qirath", words)
        self.assertIn("qirath", unknown)
        self.assertNotIn("catheter", unknown.split("qirath")[-1])

    def test_a_scan_changes_only_a_single_lowercase_neighbor(self):
        self.assertEqual(correct_scan("catheer", {"catheter"}), "catheter")
        self.assertEqual(correct_scan("abcde", {"abxde", "axcde"}), "abcde")
        self.assertEqual(correct_scan("Catheer", {"catheter"}), "Catheer")
        self.assertEqual(correct_scan("model12", {"model"}), "model12")
        self.assertEqual(correct_scan("form", {"from"}), "form")


class DedupeTest(unittest.TestCase):
    def test_the_same_long_article_matches_and_a_short_one_does_not(self):
        long = " ".join(f"word{index}" for index in range(220))
        other = " ".join(f"other{index}" for index in range(220))
        self.assertTrue(same_work(fingerprint(long), fingerprint(long)))
        self.assertFalse(same_work(fingerprint(long), fingerprint(other)))
        self.assertFalse(same_work(fingerprint("too short"), fingerprint("too short")))

    def test_prints_round_trip_and_chunk_hashes_ignore_short_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prints"
            long = " ".join(f"word{index}" for index in range(220))
            prints = ArticlePrints()
            prints.bind(path)
            prints.keep(long)
            prints.close()
            loaded = ArticlePrints()
            loaded.merge_file(path)
            self.assertTrue(loaded.duplicate(long))
            ledger = ChunkLedger(Path(tmp) / "chunks.sqlite")
            short = "note"
            ledger.remember(short)
            self.assertFalse(ledger.seen(short))
            body = "catheter " * 80
            ledger.remember(body)
            self.assertTrue(ledger.seen("  Catheter  " * 80))
            ledger.close()


class PassProgressTest(unittest.TestCase):
    def test_a_later_pass_restarts_the_file_bar_without_resetting_articles(self):
        book = ProgressBook(
            [
                CatalogRow(
                    source="medicine.zim",
                    name="medicine.zim",
                    origin="zim",
                    kind="zim",
                    size_bytes=1000,
                    state="pending",
                )
            ]
        )
        book.update(
            "medicine.zim",
            articles=10,
            total=10,
            chunks=4,
            state="running",
            now=1,
            label="HTML",
        )
        first = book.snapshot(now=1).files[0]
        self.assertEqual(first.status, "HTML")
        self.assertEqual(first.percent, 100)
        book.update(
            "medicine.zim",
            articles=10,
            total=10,
            chunks=4,
            state="running",
            now=2,
            label="text",
            pass_done=0,
            pass_total=5,
        )
        second = book.snapshot(now=2).files[0]
        self.assertEqual(second.status, "text")
        self.assertEqual(second.percent, 0)
        self.assertEqual(second.count, "0 / 5")
        self.assertEqual(second.articles, 10)
        book.update(
            "medicine.zim",
            articles=12,
            total=10,
            chunks=6,
            state="indexed",
            now=3,
            label="text",
            pass_done=1,
            pass_total=5,
        )
        done = book.snapshot(now=3).files[0]
        self.assertEqual(done.status, "indexed")
        self.assertEqual(done.percent, 100)


class ExtractLaunchTest(unittest.TestCase):
    def test_later_passes_keep_the_gc_flag_next_to_the_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            zim = storage / "zim" / "medicine.zim"
            zim.parent.mkdir()
            zim.write_bytes(b"zim")
            install = _install(str(storage))
            with mock.patch("nomad_turbo.runtime._remove_ours"), mock.patch(
                "nomad_turbo.runtime.subprocess.Popen",
                return_value=mock.Mock(),
            ) as popen:
                start_extract(
                    install,
                    str(zim),
                    4,
                    None,
                    9,
                    quality="high",
                    spill_id="abc123",
                    zim_pass="ocr",
                    resume_path="shelf/book.pdf",
                )
            argv = popen.call_args.args[0]
            script = argv.index("/app/nomad-turbo-extract.mjs")
            self.assertEqual(argv[script - 1], "--expose-gc")
            self.assertEqual(argv[script + 1], "--zim")
            self.assertIn("--quality", argv)
            self.assertEqual(argv[argv.index("--quality") + 1], "high")
            self.assertIn("--spill-id", argv)
            self.assertIn("--pass", argv)
            self.assertEqual(argv[argv.index("--pass") + 1], "ocr")
            self.assertIn("--resume-path", argv)
            self.assertNotIn("--after-path", argv)
            mounted = " ".join(argv)
            self.assertIn("/app/zim-extra.mjs", mounted)
            self.assertIn("/app/nomad-turbo-ocr.mjs", mounted)
            self.assertIn("/tessdata", mounted)
            self.assertIn("/turbo-state", mounted)
            self.assertNotIn("--ocr-ahead", argv)

    def test_html_overlap_asks_the_extractor_to_read_zim_files_ahead(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            zim = storage / "zim" / "medicine.zim"
            zim.parent.mkdir()
            zim.write_bytes(b"zim")
            install = _install(str(storage))
            with mock.patch("nomad_turbo.runtime._remove_ours"), mock.patch(
                "nomad_turbo.runtime.subprocess.Popen",
                return_value=mock.Mock(),
            ) as popen:
                start_extract(
                    install,
                    str(zim),
                    0,
                    None,
                    0,
                    spill_id="abc123",
                    zim_pass="html",
                    ocr_ahead=True,
                )
            argv = popen.call_args.args[0]
            self.assertIn("--ocr-ahead", argv)
            self.assertNotIn("--ocr-queue", argv)
            self.assertEqual(argv[argv.index("--pass") + 1], "html")
            with mock.patch("nomad_turbo.runtime._remove_ours"), mock.patch(
                "nomad_turbo.runtime.subprocess.Popen",
                return_value=mock.Mock(),
            ) as queued:
                start_extract(
                    install,
                    str(zim),
                    0,
                    None,
                    0,
                    spill_id="abc123",
                    zim_pass="html",
                    ocr_queue=True,
                )
            queued_argv = queued.call_args.args[0]
            self.assertIn("--ocr-queue", queued_argv)
            self.assertNotIn("--ocr-ahead", queued_argv)

    def test_zim_ocr_tally_names_one_file_and_many(self):
        from nomad_turbo.engine import _ocr_tally_note

        self.assertEqual(
            _ocr_tally_note(1),
            "Indexing. 1 file in this ZIM is waiting for OCR.",
        )
        self.assertEqual(
            _ocr_tally_note(4),
            "Indexing. 4 files in this ZIM are waiting for OCR.",
        )
        self.assertEqual(
            _ocr_tally_note(4).removeprefix("Indexing. "),
            "4 files in this ZIM are waiting for OCR.",
        )

    def test_pages_json_and_a_checkpoint_remember_the_pass(self):
        self.assertIsNone(visual_pages(1, '{"drew": false}'))
        self.assertFalse(visual_pages(0, '{"pages": [], "drew": false}')["drew"])
        self.assertEqual(visual_pages(0, "plain text")["text"], "plain text")
        previous = os.environ.get("NOMAD_TURBO_STATE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["NOMAD_TURBO_STATE"] = tmp
            try:
                work = FileWork(host=Path(tmp) / "a.zim", kind="zim", source="/app/storage/zim/a.zim")
                work.zim_pass = "text"
                work.path = "book.epub"
                _write_checkpoint(work)
                saved = json.loads(next(state_dir().joinpath("checkpoints").glob("*.json")).read_text())
            finally:
                if previous is None:
                    os.environ.pop("NOMAD_TURBO_STATE", None)
                else:
                    os.environ["NOMAD_TURBO_STATE"] = previous
        self.assertEqual(saved["pass"], "text")
        self.assertEqual(saved["path"], "book.epub")


if __name__ == "__main__":
    unittest.main()
