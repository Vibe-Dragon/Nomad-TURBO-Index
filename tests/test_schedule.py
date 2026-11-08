"""Index columns, order, Index now, priority, and Measure.

These call the functions the Index page and the indexing loop use.
"""

import json
import os
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomad_turbo.catalog import CatalogRow
from nomad_turbo.discover import Install
from nomad_turbo.file_stats import FileStats, detail_text, format_downloaded, library_location
from nomad_turbo.progress import ProgressBook, format_bytes, format_eta
from nomad_turbo.runtime import ZIM_DATE_NAME, read_zim_date, start_extract, zim_date_command
from nomad_turbo.schedule import (
    COL_ART_MIN,
    COL_ARTICLES,
    COL_CHK_S,
    COL_CHUNKS,
    COL_ETA,
    COL_MARK,
    COL_NAME,
    COL_SIZE,
    COL_STATUS,
    COL_TYPE,
    EM_DASH,
    FOLDER,
    FOLDER_HOLD,
    INDEX_HEADERS,
    MEASURE_SECONDS,
    PRIORITY_COLOR,
    PRIORITY_MARK,
    OrderFile,
    INDEXING_MARK,
    apply_sample_batch,
    engine_switch_reason,
    file_eta_view,
    file_needs_measure,
    file_type_label,
    files_for_list_order,
    folder_store_row,
    hold_store_row,
    index_store_row,
    library_place,
    library_tree,
    list_display_order,
    measure_ending,
    measure_leave_reason,
    measure_start_chunks,
    release_measure,
    ordered_sources,
    order_choice_labels,
    pick_next,
    row_menu_labels,
    sample_expired,
    sum_known_etas,
    then_index_action,
)
from nomad_turbo.session import Session
from nomad_turbo.slots import safe_call_chunks
from nomad_turbo import engine


ROOT = Path(__file__).resolve().parents[1]


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


def _file(**kwargs) -> OrderFile:
    base = dict(
        source="a",
        name="a.zim",
        file_type="zim",
        size_bytes=100,
        mtime=1_000.0,
        zim_date=None,
        articles_total=10,
        completed=0.0,
        chunks=0,
        measured_eta=None,
        state="pending",
        walk_index=0,
        priority=False,
    )
    base.update(kwargs)
    return OrderFile(**base)


def _stamp(year: int, month: int = 1, day: int = 1) -> float:
    return datetime(year, month, day, tzinfo=timezone.utc).timestamp()


class ColumnCellsTest(unittest.TestCase):
    def test_headers_and_file_type_are_the_extension(self):
        self.assertEqual(
            INDEX_HEADERS,
            (
                "File",
                "File type",
                "Size",
                "Status",
                "Progress",
                "Articles",
                "art/min",
                "Chunks",
                "chk/s",
                "ETA",
            ),
        )
        self.assertNotIn("Kind", INDEX_HEADERS)
        self.assertNotIn("Rate", INDEX_HEADERS)
        names = {
            "wiki.zim": "zim",
            "notes.pdf": "pdf",
            "chapter.docx": "docx",
            "book.epub": "epub",
            "readme.txt": "txt",
            "notes.md": "md",
            "letter.rtf": "rtf",
            "photo.JPEG": "jpeg",
            "pic.PNG": "png",
            "scan.webp": "webp",
            "partial.zim.part": "",
        }
        for name, expected in names.items():
            self.assertEqual(file_type_label(name), expected, name)
            self.assertNotIn(file_type_label(name), ("ZIM", "Upload"))

    def test_row_cells_follow_the_header_order(self):
        book = ProgressBook(
            [
                CatalogRow("/app/storage/zim/wiki.zim", "wiki.zim", "zim", "zim", 5_000_000, "pending"),
                CatalogRow("/app/storage/kb_uploads/notes.pdf", "notes.pdf", "upload", "pdf", 2000, "pending"),
                CatalogRow("/app/storage/kb_uploads/photo.JPEG", "photo.JPEG", "upload", "image", 300, "pending"),
            ]
        )
        book.update("/app/storage/zim/wiki.zim", articles=0, total=1000, chunks=0, state="running", now=0)
        book.update("/app/storage/zim/wiki.zim", articles=120, total=1000, chunks=120, state="running", now=60)
        snap = book.snapshot(now=60)
        by_name = {item.name: item for item in snap.files}
        running = index_store_row(by_name["wiki.zim"], priority=True)
        waiting = index_store_row(by_name["notes.pdf"], priority=False)
        photo = index_store_row(by_name["photo.JPEG"], priority=False)
        self.assertEqual(len(running), 19)
        self.assertEqual(running[COL_NAME], "wiki.zim")
        self.assertEqual(running[COL_MARK], PRIORITY_MARK)
        self.assertNotIn("*", running[COL_NAME])
        self.assertEqual(running[COL_TYPE], "zim")
        self.assertEqual(waiting[COL_TYPE], "pdf")
        self.assertEqual(photo[COL_TYPE], "jpeg")
        self.assertNotEqual(waiting[COL_TYPE], "Upload")
        self.assertEqual(running[COL_SIZE], by_name["wiki.zim"].size_text)
        self.assertIn("MB", running[COL_SIZE])
        self.assertEqual(running[COL_ART_MIN], "120")
        self.assertEqual(running[COL_CHK_S], "2")
        self.assertNotIn("art/min", running[COL_ART_MIN])
        self.assertNotIn("chk/s", running[COL_CHK_S])
        self.assertEqual(waiting[COL_ART_MIN], EM_DASH)
        self.assertEqual(waiting[COL_CHK_S], EM_DASH)
        self.assertEqual(waiting[COL_MARK], "")
        self.assertIn("chk/s", by_name["wiki.zim"].rate_text)
        self.assertEqual(INDEX_HEADERS[1], "File type")
        self.assertEqual(INDEX_HEADERS[2], "Size")
        self.assertEqual(INDEX_HEADERS[6], "art/min")
        self.assertEqual(INDEX_HEADERS[8], "chk/s")
        self.assertLess(COL_TYPE, COL_SIZE)
        self.assertLess(COL_ARTICLES, COL_ART_MIN)
        self.assertLess(COL_CHUNKS, COL_CHK_S)
        self.assertLess(COL_CHK_S, COL_ETA)

    def test_a_measured_file_keeps_its_eta_after_another_file_is_current(self):
        book = ProgressBook(
            [
                CatalogRow("a.pdf", "a.pdf", "upload", "pdf", 1000, "pending", 0, 200),
                CatalogRow("b.pdf", "b.pdf", "upload", "pdf", 1000, "pending", 0, 200),
            ]
        )
        book.update("a.pdf", articles=0, total=200, chunks=0, state="running", now=0)
        book.update("a.pdf", articles=100, total=200, chunks=10, state="running", now=10)
        book.keep_measured_rate("a.pdf")
        book.update("a.pdf", articles=100, total=200, chunks=10, state="partial", now=11)
        book.update("b.pdf", articles=0, total=200, chunks=0, state="running", now=11)
        book.update("b.pdf", articles=50, total=200, chunks=5, state="running", now=21)
        book.keep_measured_rate("b.pdf")
        book.update("b.pdf", articles=50, total=200, chunks=5, state="partial", now=22)
        snap = book.snapshot(now=22)
        by_name = {item.name: item for item in snap.files}
        self.assertEqual(by_name["a.pdf"].state, "partial")
        self.assertEqual(by_name["a.pdf"].art_text, EM_DASH)
        self.assertEqual(by_name["a.pdf"].chk_text, EM_DASH)
        self.assertEqual(by_name["a.pdf"].eta_seconds, 10)
        self.assertEqual(by_name["b.pdf"].eta_seconds, 30)
        self.assertEqual(by_name["a.pdf"].eta, format_eta(10))
        row = index_store_row(by_name["a.pdf"], priority=False)
        self.assertEqual(row[COL_ETA], format_eta(10))
        self.assertEqual(row[COL_ART_MIN], EM_DASH)
        self.assertIn("about 40s", snap.overall_text)
        self.assertIsNone(sum_known_etas([("partial", 10), ("pending", None)]))
        self.assertEqual(sum_known_etas([("partial", 10), ("indexed", 99), ("deferred", None)]), 10)


class OrderTest(unittest.TestCase):
    def _library(self) -> list[OrderFile]:
        return [
            _file(
                source="newmeta",
                name="newmeta.zim",
                file_type="zim",
                size_bytes=50,
                mtime=_stamp(2010),
                zim_date="2024-06",
                articles_total=5,
                completed=0.2,
                chunks=1,
                measured_eta=100.0,
                walk_index=2,
            ),
            _file(
                source="oldmeta",
                name="oldmeta.zim",
                file_type="zim",
                size_bytes=500,
                mtime=_stamp(2023),
                zim_date="2015-01-01",
                articles_total=100,
                completed=0.9,
                chunks=40,
                measured_eta=10.0,
                walk_index=0,
            ),
            _file(
                source="nodate",
                name="nodate.zim",
                file_type="zim",
                size_bytes=20,
                mtime=_stamp(2018),
                zim_date="",
                articles_total=None,
                completed=0.0,
                chunks=0,
                measured_eta=None,
                walk_index=1,
            ),
            _file(
                source="baddate",
                name="baddate.zim",
                file_type="zim",
                size_bytes=20,
                mtime=_stamp(2019),
                zim_date="not-a-date",
                articles_total=8,
                completed=0.5,
                chunks=3,
                measured_eta=50.0,
                walk_index=3,
            ),
            _file(
                source="notes",
                name="notes.pdf",
                file_type="pdf",
                size_bytes=80,
                mtime=_stamp(2012),
                zim_date="2099-01-01",
                articles_total=2,
                completed=0.1,
                chunks=2,
                measured_eta=5.0,
                walk_index=4,
            ),
            _file(
                source="mystery",
                name="mystery.pdf",
                file_type="pdf",
                size_bytes=80,
                mtime=None,
                zim_date=None,
                articles_total=None,
                completed=0.4,
                chunks=7,
                measured_eta=None,
                walk_index=5,
            ),
        ]

    def test_the_list_puts_the_current_file_first_and_finished_files_last(self):
        current = _file(
            source="now.pdf",
            name="now.pdf",
            file_type="pdf",
            size_bytes=10,
            state="running",
            walk_index=4,
        )
        big = _file(source="big.zim", name="big.zim", size_bytes=500, state="pending", walk_index=1, priority=True)
        small = _file(source="small.zim", name="small.zim", size_bytes=20, state="pending", walk_index=0, priority=True)
        mid = _file(
            source="mid.pdf",
            name="mid.pdf",
            file_type="pdf",
            size_bytes=50,
            state="pending",
            walk_index=2,
        )
        bad = _file(
            source="bad.pdf",
            name="bad.pdf",
            file_type="pdf",
            size_bytes=5,
            state="failed",
            walk_index=3,
        )
        done = _file(
            source="done.zim",
            name="done.zim",
            size_bytes=900,
            state="indexed",
            walk_index=5,
            priority=True,
        )
        files = [done, mid, big, current, small, bad]
        self.assertEqual(
            list_display_order(files, "size_desc", current="now.pdf"),
            ["now.pdf", "big.zim", "small.zim", "mid.pdf", "bad.pdf", "done.zim"],
        )
        self.assertEqual(
            list_display_order(files, "size_desc", current="now.pdf", hide_finished=True),
            ["now.pdf", "big.zim", "small.zim", "mid.pdf", "bad.pdf"],
        )
        self.assertEqual(
            list_display_order(files, "library", current="now.pdf"),
            ["now.pdf", "small.zim", "big.zim", "mid.pdf", "bad.pdf", "done.zim"],
        )
        views = [
            SimpleNamespace(
                source="/old",
                name="old.zim",
                state="pending",
                size_bytes=1,
                articles=0,
                total_articles=None,
                chunks=0,
                eta_seconds=None,
                file_type="zim",
                mtime=_stamp(2024),
                walk_index=0,
            ),
            SimpleNamespace(
                source="/new",
                name="new.zim",
                state="pending",
                size_bytes=1,
                articles=0,
                total_articles=None,
                chunks=0,
                eta_seconds=None,
                file_type="zim",
                mtime=_stamp(2010),
                walk_index=1,
            ),
        ]
        dated = files_for_list_order(views, priorities=set(), dates={"/new": "2024", "/old": "2020"})
        self.assertEqual(list_display_order(dated, "date_desc"), ["/new", "/old"])

    def test_both_directions_and_unknowns_sort_last(self):
        files = self._library()
        expect = {
            "library": ["oldmeta", "nodate", "newmeta", "baddate", "notes", "mystery"],
            "size_desc": ["oldmeta", "mystery", "notes", "newmeta", "baddate", "nodate"],
            "size_asc": ["baddate", "nodate", "newmeta", "mystery", "notes", "oldmeta"],
            "date_desc": ["newmeta", "baddate", "nodate", "oldmeta", "notes", "mystery"],
            "date_asc": ["notes", "oldmeta", "nodate", "baddate", "newmeta", "mystery"],
            "articles_desc": ["oldmeta", "baddate", "newmeta", "notes", "mystery", "nodate"],
            "articles_asc": ["notes", "newmeta", "baddate", "oldmeta", "mystery", "nodate"],
            "completed_desc": ["oldmeta", "baddate", "mystery", "newmeta", "notes", "nodate"],
            "completed_asc": ["nodate", "notes", "newmeta", "mystery", "baddate", "oldmeta"],
            "name_asc": ["baddate", "mystery", "newmeta", "nodate", "notes", "oldmeta"],
            "name_desc": ["oldmeta", "notes", "nodate", "newmeta", "mystery", "baddate"],
            "eta_asc": ["notes", "oldmeta", "baddate", "newmeta", "mystery", "nodate"],
            "eta_desc": ["newmeta", "baddate", "oldmeta", "notes", "mystery", "nodate"],
            "chunks_desc": ["oldmeta", "mystery", "baddate", "notes", "newmeta", "nodate"],
            "chunks_asc": ["nodate", "newmeta", "notes", "baddate", "mystery", "oldmeta"],
            "type_asc": ["mystery", "notes", "baddate", "newmeta", "nodate", "oldmeta"],
            "type_desc": ["baddate", "newmeta", "nodate", "oldmeta", "mystery", "notes"],
        }
        self.assertEqual(set(expect), {choice for choice, _label in __import__("nomad_turbo.schedule", fromlist=["ORDER_CHOICES"]).ORDER_CHOICES})
        for choice, sources in expect.items():
            self.assertEqual(ordered_sources(files, choice), sources, choice)
        tied = [
            _file(source="b", name="Same.ZIM", size_bytes=10, walk_index=1),
            _file(source="a", name="same.zim", size_bytes=10, walk_index=0),
        ]
        self.assertEqual(ordered_sources(tied, "size_desc"), ["a", "b"])
        self.assertEqual(ordered_sources(tied, "name_desc"), ["a", "b"])
        self.assertEqual(order_choice_labels()[0], "Library order")
        self.assertIn("Newest date first", order_choice_labels())
        self.assertIn("Shortest ETA first", order_choice_labels())

    def test_a_date_read_does_not_use_the_extract_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            zim = storage / "zim" / "wiki.zim"
            zim.parent.mkdir()
            zim.write_bytes(b"not a real zim")
            install = _install(str(storage))
            command = zim_date_command(install, str(zim))
            name = command[command.index("--name") + 1]
            self.assertEqual(name, ZIM_DATE_NAME)
            self.assertEqual(name, "nomad-turbo-zimdate")
            self.assertNotEqual(name, "nomad-turbo-extract")
            self.assertIn("--meta-only", command)
            self.assertNotIn("--skip", command)
            seen = {}

            def run(argv, **kwargs):
                seen["argv"] = list(argv)
                seen["timeout"] = kwargs.get("timeout")

                class Done:
                    returncode = 0
                    stdout = '{"type": "meta", "date": "2024-06-01"}\n'
                    stderr = ""

                return Done()

            with mock.patch("nomad_turbo.runtime._remove_ours", side_effect=AssertionError("removed a container")):
                found = read_zim_date(install, str(zim), run=run)
            self.assertEqual(found, "2024-06-01")
            self.assertEqual(seen["argv"][seen["argv"].index("--name") + 1], "nomad-turbo-zimdate")
            self.assertIn("--meta-only", seen["argv"])

            def failed(argv, **kwargs):
                del argv, kwargs

                class Done:
                    returncode = 1
                    stdout = ""
                    stderr = "no"

                return Done()

            self.assertIsNone(read_zim_date(install, str(zim), run=failed))
        text = (ROOT / "worker" / "extract.mjs").read_text(encoding="utf-8")
        meta_at = text.find('if (process.argv.includes("--meta-only"))')
        workers_at = text.find("const workers = startWorkers();", meta_at)
        self.assertGreater(meta_at, 0)
        self.assertGreater(workers_at, meta_at)
        self.assertIn("return", text[meta_at:workers_at])

    def test_extract_releases_passed_entries_and_asks_for_gc(self):
        text = (ROOT / "worker" / "extract.mjs").read_text(encoding="utf-8")
        walk = text.split("function* entries(", 1)[1].split("function* selectedArticles(", 1)[0]
        self.assertIn("collectEntries(visited)", walk)
        self.assertNotIn("articleCount", walk)
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            zim = storage / "zim" / "maps.zim"
            zim.parent.mkdir()
            zim.write_bytes(b"not a real zim")
            install = _install(str(storage))
            with mock.patch("nomad_turbo.runtime._remove_ours") as removed, mock.patch(
                "nomad_turbo.runtime.subprocess.Popen",
                return_value=mock.Mock(),
            ) as popen:
                start_extract(install, str(zim), 0, "tiles/0/0/0.pbf", 3)
            removed.assert_called_once_with("nomad-turbo-extract")
            argv = popen.call_args.args[0]
            script = argv.index("/app/nomad-turbo-extract.mjs")
            self.assertEqual(argv[script - 1], "--expose-gc")
            self.assertEqual(argv[script + 1], "--zim")
            self.assertNotIn("--max-old-space-size", argv)
            self.assertIn("--after-path", argv)

    def test_the_engine_learns_one_zim_date_without_starting_extract(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            first = storage / "zim" / "first.zim"
            second = storage / "zim" / "second.zim"
            first.parent.mkdir()
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            install = _install(str(storage))
            works = [
                engine.FileWork(host=first, kind="zim", source="/app/storage/zim/first.zim", mtime=_stamp(2011)),
                engine.FileWork(host=second, kind="zim", source="/app/storage/zim/second.zim", mtime=_stamp(2012)),
            ]
            session = Session(16)
            calls = []

            def fake_read(_install, host, run=None):
                del run
                calls.append(Path(host).name)
                if Path(host).name == "first.zim":
                    return "2021-05-01"
                return None

            with (
                mock.patch.object(engine, "read_zim_date", fake_read),
                mock.patch.object(engine, "start_extract", side_effect=AssertionError("extract started")),
            ):
                session.set_order("library")
                engine._learn_zim_dates(install, session, works)
                self.assertEqual(calls, [])
                session.set_order("date_desc")
                engine._learn_zim_dates(install, session, works)
                self.assertEqual(calls, ["first.zim", "second.zim"])
                self.assertEqual(session.zim_date(works[0].source), "2021-05-01")
                self.assertIsNone(session.zim_date(works[1].source))
                self.assertTrue(session.zim_date_known(works[1].source))
                engine._learn_zim_dates(install, session, works)
                self.assertEqual(calls, ["first.zim", "second.zim"])
            rows = engine._order_rows(works, session)
            by_source = {item.source: item for item in rows}
            self.assertEqual(by_source[works[0].source].zim_date, "2021-05-01")
            self.assertIsNone(by_source[works[1].source].zim_date)
            self.assertEqual(
                ordered_sources(rows, "date_desc")[0],
                works[0].source,
            )

    def test_newest_zim_date_is_read_before_the_first_pick(self):
        """Newest-first reads every missing Date, then picks that archive.

        old.zim is first in the walk and has the newest file time. Its Date
        is the oldest. Reading only that one Date used to pick mid.zim.
        """
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            specs = (
                ("old.zim", 2024, "2020"),
                ("mid.zim", 2023, "2022"),
                ("new.zim", 2010, "2024"),
            )
            works = []
            dates = {}
            for name, mtime_year, published in specs:
                path = storage / "zim" / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(name.encode())
                source = f"/app/storage/zim/{name}"
                dates[name] = published
                works.append(
                    engine.FileWork(host=path, kind="zim", source=source, mtime=_stamp(mtime_year))
                )
            session = Session(16)
            session.set_order("date_desc")
            calls = []

            def fake_read(_install, host, run=None):
                del run
                name = Path(host).name
                calls.append(name)
                return dates[name]

            install = _install(str(storage))
            stop = threading.Event()
            with (
                mock.patch.object(engine, "read_zim_date", fake_read),
                mock.patch.object(engine, "start_extract", side_effect=AssertionError("extract started")),
            ):
                work, _started, _epoch, action = engine._next_work(install, None, session, works, stop)
                self.assertEqual(calls, ["old.zim", "mid.zim", "new.zim"])
                self.assertEqual(work.host.name, "new.zim")
                self.assertEqual(action, "")
                ranked = ordered_sources(engine._order_rows(works, session), "date_desc")
                self.assertEqual(ranked, [item.source for item in (works[2], works[1], works[0])])
                again, *_rest = engine._next_work(install, None, session, works, stop)
                self.assertEqual(again.host.name, "new.zim")
                self.assertEqual(calls, ["old.zim", "mid.zim", "new.zim"])


class NextFileTest(unittest.TestCase):
    def _files(self) -> list[OrderFile]:
        return [
            _file(source="small", name="small.zim", size_bytes=1, walk_index=0, priority=True),
            _file(source="big", name="big.zim", size_bytes=100, walk_index=1, priority=True),
            _file(source="mid", name="mid.zim", size_bytes=50, walk_index=2, priority=False),
            _file(source="done", name="done.zim", size_bytes=1000, walk_index=3, state="indexed", priority=True),
        ]

    def test_index_now_and_priority_follow_the_selected_order(self):
        files = self._files()
        self.assertEqual(row_menu_labels(priority=False), ("Index now", "Set priority", "Details"))
        self.assertEqual(row_menu_labels(priority=True), ("Index now", "Clear priority", "Details"))
        largest = pick_next(files, choice="size_desc", current=None, index_now=None)
        self.assertEqual((largest.target, largest.switch, largest.reason), ("big", False, "priority"))
        smallest = pick_next(files, choice="size_asc", current=None, index_now=None)
        self.assertEqual(smallest.target, "small")
        switched = pick_next(files, choice="size_desc", current="mid", index_now=None)
        self.assertEqual((switched.target, switched.switch, switched.reason), ("big", True, "priority"))
        stay = pick_next(files, choice="size_desc", current="small", index_now=None)
        self.assertEqual((stay.target, stay.switch, stay.reason), ("small", False, "stay"))
        now = pick_next(files, choice="library", current="big", index_now="mid")
        self.assertEqual((now.target, now.switch, now.reason), ("mid", True, "now"))
        same = pick_next(files, choice="library", current="mid", index_now="mid")
        self.assertEqual((same.target, same.switch), ("mid", False))
        ordered = pick_next(files, choice="size_desc", current="mid", index_now=None)
        plain = [
            _file(source="small", name="small.zim", size_bytes=1, walk_index=0),
            _file(source="big", name="big.zim", size_bytes=100, walk_index=1),
        ]
        held = pick_next(plain, choice="size_desc", current="small", index_now=None)
        self.assertEqual((held.target, held.switch, held.reason), ("small", False, "stay"))
        nxt = pick_next(plain, choice="size_desc", current=None, index_now=None)
        self.assertEqual(nxt.target, "big")
        sampled = pick_next(
            files,
            choice="size_desc",
            current=None,
            index_now="mid",
            defer={"mid", "big", "small"},
        )
        self.assertEqual((sampled.target, sampled.reason), ("mid", "now"))
        self.assertEqual(ordered.reason, "priority")
        session = Session(8)
        session.request_index_now("small")
        session.request_index_now("mid")
        self.assertEqual(session.index_now(), "mid")
        session.clear_index_now("mid")
        self.assertIsNone(session.index_now())

    def test_order_changes_do_not_switch_and_a_new_priority_does(self):
        session = Session(8)
        armed = session.arm_priority()
        session.set_order("size_asc")
        self.assertIsNone(engine_switch_reason(session, "mid", None, 0, armed))
        session.set_priority("big", True)
        self.assertEqual(engine_switch_reason(session, "mid", None, 0, armed), "priority")
        session.set_priority("mid", True)
        self.assertIsNone(engine_switch_reason(session, "mid", None, 0, session.arm_priority()))
        session.set_priority("big", True)
        self.assertIsNone(engine_switch_reason(session, "mid", None, 0, session.arm_priority()))
        session.request_index_now("mid")
        self.assertIsNone(engine_switch_reason(session, "mid", 0.0, 10_000.0, 0))
        session.request_index_now("small")
        self.assertEqual(engine_switch_reason(session, "mid", 0.0, 1.0, 0), "now")


class SafeSwitchTest(unittest.TestCase):
    def test_a_switch_checkpoints_and_a_missing_file_drops(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state"
            state.mkdir()
            install = _install(tmp)
            with mock.patch.dict(os.environ, {"NOMAD_TURBO_STATE": str(state)}):
                kept = engine.FileWork(
                    host=Path(tmp) / "kept.zim",
                    kind="zim",
                    source="/app/storage/zim/kept.zim",
                    articles=4,
                    chunks=2,
                    total_articles=20,
                )
                with (
                    mock.patch.object(engine, "_request_drop", side_effect=AssertionError("dropped")),
                    mock.patch.object(engine, "delete_source_points", side_effect=AssertionError("deleted points")),
                    mock.patch.object(engine, "set_policy", side_effect=AssertionError("policy")),
                ):
                    outcome = engine.settle_file_stop(install, None, kept, "measure")
                self.assertEqual(outcome, "partial")
                self.assertFalse(kept.removed)
                empty = engine.FileWork(
                    host=Path(tmp) / "scan.pdf",
                    kind="pdf",
                    source="/app/storage/kb_uploads/scan.pdf",
                )
                stored = engine.FileWork(
                    host=Path(tmp) / "stored.pdf",
                    kind="pdf",
                    source="/app/storage/kb_uploads/stored.pdf",
                    articles=1,
                    chunks=3,
                )
                book = ProgressBook(
                    [
                        CatalogRow(empty.source, "scan.pdf", "upload", "pdf", 100, "running"),
                        CatalogRow(stored.source, "stored.pdf", "upload", "pdf", 100, "running"),
                    ]
                )
                session = SimpleNamespace(get_book=lambda: book)
                with mock.patch.object(engine, "_request_drop", side_effect=AssertionError("dropped")):
                    engine.settle_file_stop(install, session, empty, "measure")
                    engine.settle_file_stop(install, session, stored, "measure")
                shown = {item.name: item for item in book.snapshot(now=0).files}
                self.assertEqual(shown["scan.pdf"].state, "pending")
                self.assertEqual(shown["scan.pdf"].status, "waiting")
                self.assertEqual(shown["stored.pdf"].state, "partial")
                self.assertEqual(shown["stored.pdf"].status, "partial")
                saved = json.loads(engine._checkpoint_file(kept.source).read_text())
                self.assertEqual(saved["source"], kept.source)
                self.assertEqual(saved["articles"], 4)
                self.assertEqual(saved["chunks"], 2)
                self.assertFalse(saved["done"])
                finished = engine.FileWork(
                    host=Path(tmp) / "done.zim",
                    kind="zim",
                    source="/app/storage/zim/done.zim",
                    articles=9,
                    chunks=9,
                    done=True,
                )
                with mock.patch.object(engine, "_request_drop", side_effect=AssertionError("dropped")):
                    indexed = engine.settle_file_stop(install, None, finished, "now")
                self.assertEqual(indexed, "indexed")
                self.assertTrue(json.loads(engine._checkpoint_file(finished.source).read_text())["done"])
                gone = engine.FileWork(
                    host=Path(tmp) / "gone.zim",
                    kind="zim",
                    source="/app/storage/zim/gone.zim",
                    articles=3,
                    chunks=1,
                    done=True,
                )
                dropped = []

                def record(_install, _session, source):
                    dropped.append(source)

                with (
                    mock.patch.object(engine, "_request_drop", record),
                    mock.patch.object(engine, "delete_source_points", side_effect=AssertionError("deleted points")),
                    mock.patch.object(engine, "set_policy", side_effect=AssertionError("policy")),
                ):
                    outcome = engine.settle_file_stop(install, None, gone, "replaced")
                self.assertEqual(outcome, "drop")
                self.assertEqual(dropped, [gone.source])
                self.assertTrue(gone.removed)
                self.assertFalse(gone.done)
                self.assertFalse(engine._checkpoint_file(gone.source).exists())

    def test_halt_checks_between_embed_calls(self):
        stop = threading.Event()
        retire = threading.Event()
        self.assertIsNone(engine._halt_reason(stop, None, lambda: None))
        self.assertEqual(engine._halt_reason(stop, None, lambda: "measure"), "measure")
        retire.set()
        self.assertEqual(engine._halt_reason(stop, retire, lambda: "now"), "replaced")
        stop.set()
        self.assertEqual(engine._halt_reason(stop, None, None), "stopped")

    def test_the_in_flight_embed_finishes_and_another_does_not_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            install = _install(tmp)
            work = engine.FileWork(
                host=Path(tmp) / "hold.pdf",
                kind="pdf",
                source="/app/storage/kb_uploads/hold.pdf",
            )
            session = Session(8)
            session.start_measure(from_idle=False)
            clock = {"now": 0.0}
            calls = {"embed": 0}

            def switch_check():
                return engine_switch_reason(session, work.source, 0.0, clock["now"], 0)

            articles = [
                {
                    "index": 1,
                    "path": "/A/One",
                    "title": "One",
                    "text": "hello world",
                    "sections": [{"heading": "", "text": "hello world", "level": 2}],
                },
                {
                    "index": 2,
                    "path": "/A/Two",
                    "title": "Two",
                    "text": "more text",
                    "sections": [{"heading": "", "text": "more text", "level": 2}],
                },
            ]

            def pull(_timeout):
                if not articles:
                    return None
                return articles.pop(0)

            def fake_embed(_url, _model, texts):
                calls["embed"] += 1
                clock["now"] = 300.0
                self.assertEqual(len(texts), 1)
                return [[0.0] * install.dimension for _text in texts]

            class _Pool:
                urls = ["http://127.0.0.1:9/embed"]

                def batch_width(self, _chunks: int) -> int:
                    return 1

            (Path(tmp) / "state").mkdir()
            with mock.patch.dict(os.environ, {"NOMAD_TURBO_STATE": str(Path(tmp) / "state")}):
                with (
                    mock.patch.object(engine, "embed", fake_embed),
                    mock.patch.object(engine, "upsert", lambda *_args, **_kwargs: None),
                    mock.patch.object(engine, "_wait_to_write", lambda *_args, **_kwargs: True),
                    mock.patch.object(engine, "delete_source_points", side_effect=AssertionError("deleted points")),
                    mock.patch.object(engine, "set_policy", side_effect=AssertionError("policy")),
                ):
                    with self.assertRaises(RuntimeError) as caught:
                        engine._consume_articles(
                            install,
                            _Pool(),
                            work,
                            threading.Event(),
                            session,
                            set(),
                            pull,
                            None,
                            switch_check,
                        )
            self.assertEqual(str(caught.exception), "measure")
            self.assertEqual(calls["embed"], 1)
            self.assertTrue(sample_expired(0.0, 300.0))
            self.assertFalse(sample_expired(0.0, 299.0))
            self.assertEqual(MEASURE_SECONDS, 300)

    def test_a_file_that_finishes_during_the_call_is_indexed(self):
        with tempfile.TemporaryDirectory() as tmp:
            install = _install(tmp)
            work = engine.FileWork(
                host=Path(tmp) / "last.pdf",
                kind="pdf",
                source="/app/storage/kb_uploads/last.pdf",
            )
            session = Session(8)
            session.start_measure(from_idle=True)
            clock = {"now": 0.0}
            calls = {"embed": 0}
            items = [
                {
                    "index": 1,
                    "path": "/A/One",
                    "title": "One",
                    "text": "hello world",
                    "sections": [{"heading": "", "text": "hello world", "level": 2}],
                },
                engine._EXTRACT_DONE,
            ]

            def pull(_timeout):
                if not items:
                    return None
                return items.pop(0)

            def fake_embed(_url, _model, texts):
                calls["embed"] += 1
                clock["now"] = 300.0
                return [[0.0] * install.dimension for _text in texts]

            class _Pool:
                urls = ["http://127.0.0.1:9/embed"]

                def batch_width(self, _chunks: int) -> int:
                    return 8

            def switch_check():
                return engine_switch_reason(session, work.source, 0.0, clock["now"], 0)

            (Path(tmp) / "state").mkdir()
            with mock.patch.dict(os.environ, {"NOMAD_TURBO_STATE": str(Path(tmp) / "state")}):
                with (
                    mock.patch.object(engine, "embed", fake_embed),
                    mock.patch.object(engine, "upsert", lambda *_args, **_kwargs: None),
                    mock.patch.object(engine, "_wait_to_write", lambda *_args, **_kwargs: True),
                ):
                    outcome = engine._consume_articles(
                        install,
                        _Pool(),
                        work,
                        threading.Event(),
                        session,
                        set(),
                        pull,
                        None,
                        switch_check,
                    )
            self.assertEqual(outcome, "done")
            self.assertEqual(calls["embed"], 1)

    def test_index_now_on_a_finished_or_missing_file_lets_indexing_continue(self):
        done = "/app/storage/zim/done.zim"
        later = "/app/storage/zim/later.zim"
        failed = "/app/storage/zim/bad.zim"
        missing = "/app/storage/zim/gone.zim"
        current = "/app/storage/kb_uploads/current.pdf"
        nxt = "/app/storage/kb_uploads/next.pdf"
        listed = [
            _file(source=done, name="done.zim", state="indexed"),
            _file(source=later, name="later.zim", state="deferred"),
            _file(source=failed, name="bad.zim", state="failed"),
            _file(source=current, name="current.pdf", file_type="pdf", state="pending"),
            _file(source=nxt, name="next.pdf", file_type="pdf", state="pending", walk_index=1),
        ]
        session = Session(8)
        for source in (done, later, missing):
            session.request_index_now(source)
            self.assertIsNone(engine_switch_reason(session, current, None, 0, 0, listed))
            self.assertIsNone(session.index_now())
        session.request_index_now(failed)
        self.assertEqual(engine_switch_reason(session, current, None, 0, 0, listed), "now")
        self.assertEqual(session.index_now(), failed)
        session.clear_index_now()

        kept = Session(8)
        kept.request_index_now("small")
        self.assertEqual(engine_switch_reason(kept, "mid", None, 0), "now")
        self.assertEqual(kept.index_now(), "small")

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state"
            state.mkdir()
            install = _install(tmp)
            work = engine.FileWork(host=Path(tmp) / "current.pdf", kind="pdf", source=current)
            other = engine.FileWork(host=Path(tmp) / "next.pdf", kind="pdf", source=nxt)
            batch = [work, other]
            book = ProgressBook(
                [
                    CatalogRow(done, "done.zim", "zim", "zim", 100, "indexed"),
                    CatalogRow(later, "later.zim", "zim", "zim", 100, "deferred"),
                    CatalogRow(failed, "bad.zim", "zim", "zim", 80, "failed"),
                    CatalogRow(current, "current.pdf", "upload", "pdf", 50, "pending"),
                    CatalogRow(nxt, "next.pdf", "upload", "pdf", 40, "pending"),
                ]
            )
            session = Session(8)
            session.bind_book(book)
            stop = threading.Event()
            with (
                mock.patch.dict(os.environ, {"NOMAD_TURBO_STATE": str(state)}),
                mock.patch.object(engine, "read_zim_date", side_effect=AssertionError("date read")),
                mock.patch.object(engine, "start_extract", side_effect=AssertionError("extract started")),
                mock.patch.object(engine, "set_policy", side_effect=AssertionError("policy")),
            ):
                for source in (done, later, missing):
                    session.request_index_now(source)
                    chosen, *_rest = engine._next_work(install, None, session, batch, stop)
                    self.assertEqual(chosen.source, current)
                    self.assertIsNone(session.index_now())
                session.request_index_now(failed)
                self.assertEqual(
                    engine_switch_reason(session, current, None, 0, 0, engine._order_rows(batch, session)),
                    "now",
                )
                self.assertEqual(session.index_now(), failed)
                session.clear_index_now()
                session.request_index_now(done)
                articles = [
                    {
                        "index": 1,
                        "path": "/A/One",
                        "title": "One",
                        "text": "hello world",
                        "sections": [{"heading": "", "text": "hello world", "level": 2}],
                    },
                    {
                        "index": 2,
                        "path": "/A/Two",
                        "title": "Two",
                        "text": "more text",
                        "sections": [{"heading": "", "text": "more text", "level": 2}],
                    },
                    engine._EXTRACT_DONE,
                ]
                calls = {"embed": 0}

                def pull(_timeout):
                    if not articles:
                        return None
                    return articles.pop(0)

                def fake_embed(_url, _model, texts):
                    calls["embed"] += 1
                    self.assertEqual(len(texts), 1)
                    return [[0.0] * install.dimension for _text in texts]

                class _Pool:
                    urls = ["http://127.0.0.1:9/embed"]

                    def batch_width(self, _chunks: int) -> int:
                        return 1

                def switch_check(started=None, source=work.source, epoch=0):
                    rows = engine._order_rows(batch, session)
                    return engine_switch_reason(session, source, started, 0.0, epoch, rows)

                with (
                    mock.patch.object(engine, "embed", fake_embed),
                    mock.patch.object(engine, "upsert", lambda *_args, **_kwargs: None),
                    mock.patch.object(engine, "_wait_to_write", lambda *_args, **_kwargs: True),
                    mock.patch.object(engine, "delete_source_points", side_effect=AssertionError("deleted points")),
                ):
                    outcome = engine._consume_articles(
                        install,
                        _Pool(),
                        work,
                        threading.Event(),
                        session,
                        set(),
                        pull,
                        None,
                        switch_check,
                    )
            self.assertEqual(outcome, "done")
            self.assertEqual(calls["embed"], 2)
            self.assertIsNone(session.index_now())

    def test_index_now_on_a_file_the_queue_will_not_open_stays_put(self):
        """A stale partial row must not abort the file that is already running.

        The window can still say partial after the knowledge base finished the
        file. That file is not in the work list. Index now is cleared and the
        current file keeps going. A file touched in the last 10 seconds stays
        requested so the next walk can open it.
        """
        done_source = "/app/storage/zim/done.zim"
        current_source = "/app/storage/zim/current.zim"
        fresh_source = "/app/storage/zim/fresh.zim"
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            zim = storage / "zim"
            zim.mkdir()
            done = zim / "done.zim"
            done.write_bytes(b"zim")
            os.utime(done, (1_000_000_000, 1_000_000_000))
            current_path = zim / "current.zim"
            current_path.write_bytes(b"zim")
            fresh = zim / "fresh.zim"
            fresh.write_bytes(b"zim")
            install = _install(str(storage))
            book = ProgressBook(
                [
                    CatalogRow(done_source, "done.zim", "zim", "zim", 10, "partial", articles=300, total_articles=810),
                    CatalogRow(current_source, "current.zim", "zim", "zim", 10, "running", articles=12, total_articles=100),
                    CatalogRow(fresh_source, "fresh.zim", "zim", "zim", 10, "pending"),
                ]
            )
            session = Session(8)
            session.bind_book(book)
            current = engine.FileWork(host=current_path, kind="zim", source=current_source, articles=12, chunks=4)
            batch = [current]
            session.request_index_now(done_source)
            check = engine._switch_check(install, session, current, None, 0, batch)
            self.assertIsNone(check())
            self.assertIsNone(session.index_now())
            chosen, *_rest = engine._next_work(install, None, session, batch, threading.Event())
            self.assertEqual(chosen.source, current_source)

            session.request_index_now(fresh_source)
            self.assertEqual(session.index_now(), fresh_source)
            engine._release_finished_index_now(install, session, batch)
            self.assertEqual(session.index_now(), fresh_source)
            self.assertEqual(engine.article_stay_reason(session, current_source, None, 0, 0), "now")

    def test_a_failed_file_named_by_index_now_or_priority_is_still_work(self):
        done_src = "/app/storage/zim/done.zim"
        bad_src = "/app/storage/zim/bad.zim"
        session = Session(8)
        session.bind_book(
            ProgressBook(
                [
                    CatalogRow(done_src, "done.zim", "zim", "zim", 10, "indexed"),
                    CatalogRow(bad_src, "bad.zim", "zim", "zim", 10, "failed"),
                ]
            )
        )
        self.assertIs(engine._queue_has_work(session, set(), {bad_src}), False)
        session.request_index_now(bad_src)
        self.assertIs(engine._queue_has_work(session, set(), {bad_src}), True)
        self.assertEqual(session.index_now(), bad_src)
        session.clear_index_now()
        self.assertIs(engine._queue_has_work(session, set(), {bad_src}), False)
        session.set_priority(bad_src, True)
        self.assertIs(engine._queue_has_work(session, set(), {bad_src}), True)

        sibling = Session(8)
        sibling.bind_book(
            ProgressBook(
                [
                    CatalogRow(done_src, "done.zim", "zim", "zim", 10, "indexed"),
                    CatalogRow(bad_src, "bad.zim", "zim", "zim", 10, "failed"),
                    CatalogRow("/app/storage/kb_uploads/notes.pdf", "notes.pdf", "upload", "pdf", 10, "partial"),
                ]
            )
        )
        self.assertIs(engine._queue_has_work(sibling, set(), {bad_src}), True)

        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            done_path = storage / "zim" / "done.zim"
            bad_path = storage / "zim" / "bad.zim"
            done_path.parent.mkdir()
            done_path.write_bytes(b"done")
            bad_path.write_bytes(b"bad")
            old = _stamp(2010)
            os.utime(done_path, (old, old))
            os.utime(bad_path, (old, old))
            install = _install(str(storage))
            done_real = engine.host_to_container(install, str(done_path))
            bad_real = engine.host_to_container(install, str(bad_path))
            book = ProgressBook(
                [
                    CatalogRow(done_real, "done.zim", "zim", "zim", 10, "indexed"),
                    CatalogRow(bad_real, "bad.zim", "zim", "zim", 10, "failed"),
                ]
            )
            state = Path(tmp) / "state"
            state.mkdir()
            stop = threading.Event()
            with (
                mock.patch.dict(os.environ, {"NOMAD_TURBO_STATE": str(state)}),
                mock.patch.object(engine, "ingest_states", return_value={}),
                mock.patch.object(engine, "read_zim_date", side_effect=AssertionError("date read")),
                mock.patch.object(engine, "start_extract", side_effect=AssertionError("extract started")),
                mock.patch.object(engine, "set_policy", side_effect=AssertionError("policy")),
            ):
                held = {bad_real}
                quiet = Session(8)
                quiet.bind_book(book)
                skipped = engine._queue_batch(install, quiet, [], set(), held)
                self.assertNotIn(bad_real, [item.source for item in skipped])
                self.assertIn(bad_real, held)
                self.assertIs(engine._queue_has_work(quiet, set(), held), False)

                for mark in ("now", "priority"):
                    runner = Session(8)
                    runner.bind_book(book)
                    failed = {bad_real}
                    if mark == "now":
                        runner.request_index_now(bad_real)
                    else:
                        runner.set_priority(bad_real, True)
                    self.assertIs(engine._queue_has_work(runner, set(), failed), True)
                    batch = engine._queue_batch(install, runner, [], set(), failed)
                    self.assertIn(bad_real, [item.source for item in batch])
                    self.assertNotIn(bad_real, failed)
                    work, _started, _epoch, action = engine._next_work(install, None, runner, batch, stop)
                    self.assertEqual(work.source, bad_real)
                    self.assertEqual(action, "")


class MeasureTest(unittest.TestCase):
    def test_sample_timing_batch_and_how_the_pass_ends(self):
        self.assertFalse(sample_expired(10.0, 309.0))
        self.assertTrue(sample_expired(10.0, 310.0))
        session = Session(64)
        session.start_measure(from_idle=False)
        self.assertIsNone(engine_switch_reason(session, "a.pdf", 0.0, 299.0, 0))
        self.assertEqual(engine_switch_reason(session, "a.pdf", 0.0, 300.0, 0), "measure")
        session.request_index_now("a.pdf")
        self.assertIsNone(engine_switch_reason(session, "a.pdf", 0.0, 900.0, 0))
        recommended = safe_call_chunks(gpu_total_mib=16384, host_total_mib=32000, cpu_count=8)
        user = 64 if recommended != 64 else 32
        session.set_chunks(user)
        session.set_tune_auto(False)
        self.assertEqual(measure_start_chunks(auto=False, user_chunks=user, recommended=recommended), user)
        self.assertEqual(apply_sample_batch(session, auto=False, recommended=recommended), user)
        self.assertEqual(session.get_chunks(), user)
        session.set_tune_auto(True)
        self.assertEqual(apply_sample_batch(session, auto=True, recommended=recommended), recommended)
        self.assertEqual(session.get_chunks(), recommended)
        self.assertEqual(measure_ending(from_idle=True, pass_complete=False), "")
        self.assertEqual(measure_ending(from_idle=True, pass_complete=True), "stop")
        self.assertEqual(measure_ending(from_idle=False, pass_complete=True), "resume")
        self.assertEqual(measure_ending(from_idle=True, pass_complete=True, then_index=True), "resume")
        self.assertEqual(measure_ending(from_idle=True, pass_complete=False, then_index=True), "")
        self.assertEqual(then_index_action(measuring=False), "ignore")
        idle = Session(8)
        self.assertFalse(idle.arm_then_index())
        self.assertFalse(idle.measuring())
        self.assertFalse(idle.measure_then_index())
        idle.start_measure(from_idle=True)
        clock = idle.measure_clock()
        idle.note_sampled("kept.pdf")
        sampled = idle.sampled()
        self.assertEqual(then_index_action(measuring=idle.measuring()), "arm")
        self.assertTrue(idle.arm_then_index())
        self.assertEqual(idle.measure_clock(), clock)
        self.assertEqual(idle.sampled(), sampled)
        self.assertTrue(idle.measure_from_idle())
        self.assertEqual(
            measure_ending(
                from_idle=idle.measure_from_idle(),
                pass_complete=True,
                then_index=idle.measure_then_index(),
            ),
            "resume",
        )
        plain = Session(8)
        plain.start_measure(from_idle=True)
        self.assertFalse(plain.measure_then_index())
        self.assertEqual(
            measure_ending(
                from_idle=plain.measure_from_idle(),
                pass_complete=True,
                then_index=plain.measure_then_index(),
            ),
            "stop",
        )
        stop = threading.Event()
        with mock.patch.object(engine, "set_policy", side_effect=AssertionError("policy")):
            resumed = engine.finish_measure_pass(session, stop)
        self.assertEqual(resumed, "resume")
        self.assertFalse(stop.is_set())
        self.assertFalse(session.measuring())
        self.assertEqual(session.view()["note"], "Indexing.")
        session.start_measure(from_idle=True)
        idle_stop = threading.Event()
        with mock.patch.object(engine, "set_policy", side_effect=AssertionError("policy")):
            ended = engine.finish_measure_pass(session, idle_stop)
        self.assertEqual(ended, "stop")
        self.assertTrue(idle_stop.is_set())
        self.assertFalse(session.measuring())
        session.start_measure(from_idle=True, then_index=True)
        self.assertTrue(session.measure_then_index())
        kept = threading.Event()
        with mock.patch.object(engine, "set_policy", side_effect=AssertionError("policy")):
            continued = engine.finish_measure_pass(session, kept)
        self.assertEqual(continued, "resume")
        self.assertFalse(kept.is_set())
        self.assertFalse(session.measuring())
        self.assertFalse(session.measure_then_index())
        self.assertEqual(session.view()["note"], "Indexing.")
        session.start_measure(from_idle=True)
        self.assertTrue(session.measuring())
        release_measure(session)
        self.assertFalse(session.measuring())
        release_measure(None)
        engine_source = Path("nomad_turbo/engine.py").read_text(encoding="utf-8")
        window_source = Path("nomad_turbo/window.py").read_text(encoding="utf-8")
        self.assertIn("release_measure(session)", engine_source)
        self.assertIn("schedule.release_measure(self.session)", window_source)

    def test_measure_times_the_file_already_indexing(self):
        source = "/app/storage/kb_uploads/current.pdf"
        session = Session(8)
        self.assertIsNone(engine_switch_reason(session, source, None, 10_000.0))
        captured = None

        def switch_check(started=captured, epoch=0):
            return engine_switch_reason(session, source, started, clock["now"], epoch)

        clock = {"now": 10_000.0}
        self.assertIsNone(switch_check())
        session.start_measure(from_idle=False)
        armed = session.measure_clock()
        self.assertIsNotNone(armed)
        self.assertIsNone(engine_switch_reason(session, source, None, armed + 299))
        self.assertEqual(engine_switch_reason(session, source, None, armed + 300), "measure")
        session.request_index_now(source)
        self.assertIsNone(engine_switch_reason(session, source, None, armed + 10_000))
        self.assertEqual(session.index_now(), source)
        session.clear_index_now()
        clock["now"] = armed
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state"
            state.mkdir()
            install = _install(tmp)
            work = engine.FileWork(host=Path(tmp) / "current.pdf", kind="pdf", source=source)
            articles = [
                {
                    "index": 1,
                    "path": "/A/One",
                    "title": "One",
                    "text": "hello world",
                    "sections": [{"heading": "", "text": "hello world", "level": 2}],
                },
                {
                    "index": 2,
                    "path": "/A/Two",
                    "title": "Two",
                    "text": "more text",
                    "sections": [{"heading": "", "text": "more text", "level": 2}],
                },
                engine._EXTRACT_DONE,
            ]
            calls = {"embed": 0}

            def pull(_timeout):
                if not articles:
                    return None
                return articles.pop(0)

            def fake_embed(_url, _model, texts):
                calls["embed"] += 1
                clock["now"] = armed + 300
                self.assertEqual(len(texts), 1)
                return [[0.0] * install.dimension for _text in texts]

            class _Pool:
                urls = ["http://127.0.0.1:9/embed"]

                def batch_width(self, _chunks: int) -> int:
                    return 1

            with mock.patch.dict(os.environ, {"NOMAD_TURBO_STATE": str(state)}):
                with (
                    mock.patch.object(engine, "embed", fake_embed),
                    mock.patch.object(engine, "upsert", lambda *_args, **_kwargs: None),
                    mock.patch.object(engine, "_wait_to_write", lambda *_args, **_kwargs: True),
                    mock.patch.object(engine, "delete_source_points", side_effect=AssertionError("deleted points")),
                    mock.patch.object(engine, "set_policy", side_effect=AssertionError("policy")),
                ):
                    with self.assertRaises(RuntimeError) as caught:
                        engine._consume_articles(
                            install,
                            _Pool(),
                            work,
                            threading.Event(),
                            session,
                            set(),
                            pull,
                            None,
                            switch_check,
                        )
        self.assertEqual(str(caught.exception), "measure")
        self.assertEqual(calls["embed"], 1)

    def test_the_engine_samples_in_order_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            big = storage / "kb_uploads" / "big.pdf"
            small = storage / "kb_uploads" / "small.pdf"
            big.parent.mkdir()
            big.write_bytes(b"x" * 100)
            small.write_bytes(b"y" * 10)
            install = _install(str(storage))
            works = [
                engine.FileWork(
                    host=small,
                    kind="pdf",
                    source="/app/storage/kb_uploads/small.pdf",
                    mtime=small.stat().st_mtime,
                ),
                engine.FileWork(
                    host=big,
                    kind="pdf",
                    source="/app/storage/kb_uploads/big.pdf",
                    mtime=big.stat().st_mtime,
                ),
            ]
            session = Session(64)
            session.set_order("size_desc")
            session.set_tune_auto(False)
            session.set_chunks(64)
            book = ProgressBook(
                [
                    CatalogRow(works[0].source, "small.pdf", "upload", "pdf", 10, "pending", 0, 20),
                    CatalogRow(works[1].source, "big.pdf", "upload", "pdf", 100, "pending", 0, 20),
                ]
            )
            session.bind_book(book)
            pool = SimpleNamespace(
                use_gpu=True,
                total_mib=16384,
                host_total_mib=32000,
                cpu_count=8,
                urls=["http://127.0.0.1:9/embed"],
            )
            recommended = safe_call_chunks(gpu_total_mib=16384, host_total_mib=32000, cpu_count=8)
            stop = threading.Event()
            with (
                mock.patch.dict(os.environ, {"NOMAD_TURBO_STATE": str(Path(tmp) / "state")}),
                mock.patch.object(engine, "read_zim_date", side_effect=AssertionError("date read")),
                mock.patch.object(engine, "start_extract", side_effect=AssertionError("extract started")),
                mock.patch.object(engine, "set_policy", side_effect=AssertionError("policy")),
            ):
                work, started, _epoch, action = engine._next_work(install, None, session, works, stop)
                self.assertEqual(work.source, works[1].source)
                self.assertIsNone(started)
                self.assertEqual(action, "")
                self.assertEqual(session.get_chunks(), 64)
                session.start_measure(from_idle=False)
                session.set_tune_auto(True)
                session.set_chunks(64)
                work, started, _epoch, action = engine._next_work(install, pool, session, works, stop)
                self.assertEqual(work.source, works[1].source)
                self.assertIsNotNone(started)
                self.assertEqual(session.get_chunks(), recommended)
                self.assertEqual(
                    engine_switch_reason(session, work.source, started, started + 299, 0),
                    None,
                )
                self.assertEqual(
                    engine_switch_reason(session, work.source, started, started + 300, 0),
                    "measure",
                )
                session.set_tune_auto(False)
                session.set_chunks(64)
                session.request_index_now(works[0].source)
                chosen, started, _epoch, action = engine._next_work(install, pool, session, works, stop)
                self.assertEqual(chosen.source, works[0].source)
                self.assertIsNone(started)
                self.assertEqual(session.get_chunks(), 64)
                session.clear_index_now(works[0].source)
                session.note_sampled(works[0].source)
                session.note_sampled(works[1].source)
                session.set_tune_auto(False)
                empty, started, _epoch, action = engine._next_work(install, pool, session, works, stop)
                self.assertIsNone(empty)
                self.assertEqual(action, "resume")
                self.assertFalse(stop.is_set())
                self.assertFalse(session.measuring())
                session.start_measure(from_idle=True)
                session.note_sampled(works[0].source)
                session.note_sampled(works[1].source)
                idle_stop = threading.Event()
                empty, _started, _epoch, action = engine._next_work(install, pool, session, works, idle_stop)
                self.assertIsNone(empty)
                self.assertEqual(action, "stop")
                self.assertTrue(idle_stop.is_set())

    def test_measure_skips_files_that_already_have_progress(self):
        self.assertFalse(file_needs_measure(10, 0, state="partial"))
        self.assertFalse(file_needs_measure(0, 4, state="pending"))
        self.assertFalse(file_needs_measure(0, 0, state="partial"))
        self.assertFalse(file_needs_measure(0, 0, state="indexed"))
        self.assertFalse(file_needs_measure(0, 0, state="deferred"))
        self.assertFalse(file_needs_measure(0, 0, state="pending", eta_seconds=120))
        self.assertFalse(file_needs_measure(0, 0, state="failed", eta_seconds=15))
        self.assertTrue(file_needs_measure(0, 0, state="pending"))
        self.assertTrue(file_needs_measure(0, 0, state="failed"))
        self.assertTrue(file_needs_measure(0, 0, state="running", eta_seconds=40))
        self.assertFalse(file_needs_measure(3, 0, state="running", eta_seconds=40))
        self.assertTrue(file_needs_measure(0, 0))

        reason, held = measure_leave_reason(
            sample_started=None,
            measuring=False,
            articles=0,
            chunks=0,
            index_now=None,
            had_progress=None,
        )
        self.assertIsNone(reason)
        self.assertFalse(held)
        reason, held = measure_leave_reason(
            sample_started=None,
            measuring=True,
            articles=1,
            chunks=0,
            index_now=None,
            had_progress=held,
        )
        self.assertIsNone(reason)
        self.assertFalse(held)
        reason, held = measure_leave_reason(
            sample_started=None,
            measuring=False,
            articles=12,
            chunks=0,
            index_now=None,
            had_progress=None,
        )
        self.assertIsNone(reason)
        self.assertTrue(held)
        reason, held = measure_leave_reason(
            sample_started=None,
            measuring=True,
            articles=13,
            chunks=1,
            index_now=None,
            had_progress=held,
        )
        self.assertEqual(reason, "measure-skip")
        self.assertTrue(held)
        reason, held = measure_leave_reason(
            sample_started=None,
            measuring=True,
            articles=4,
            chunks=0,
            index_now=None,
            had_progress=None,
        )
        self.assertEqual(reason, "measure-skip")
        reason, held = measure_leave_reason(
            sample_started=10.0,
            measuring=True,
            articles=8,
            chunks=2,
            index_now=None,
            had_progress=None,
        )
        self.assertIsNone(reason)
        self.assertIsNone(held)
        reason, held = measure_leave_reason(
            sample_started=None,
            measuring=True,
            articles=8,
            chunks=0,
            index_now="a.pdf",
            had_progress=True,
        )
        self.assertIsNone(reason)
        self.assertTrue(held)
        reason, held = measure_leave_reason(
            sample_started=None,
            measuring=False,
            articles=0,
            chunks=2,
            index_now=None,
            had_progress=None,
        )
        self.assertTrue(held)

        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            names = {
                "partial.pdf": 1000,
                "saved.pdf": 800,
                "chunks.pdf": 400,
                "failed.pdf": 100,
                "fresh.pdf": 10,
            }
            folder = storage / "kb_uploads"
            folder.mkdir()
            hosts = {}
            for name, size in names.items():
                path = folder / name
                path.write_bytes(b"x" * size)
                hosts[name] = path
            sources = {name: f"/app/storage/kb_uploads/{name}" for name in names}
            works = [
                engine.FileWork(
                    host=hosts["fresh.pdf"],
                    kind="pdf",
                    source=sources["fresh.pdf"],
                    mtime=hosts["fresh.pdf"].stat().st_mtime,
                ),
                engine.FileWork(
                    host=hosts["failed.pdf"],
                    kind="pdf",
                    source=sources["failed.pdf"],
                    mtime=hosts["failed.pdf"].stat().st_mtime,
                ),
                engine.FileWork(
                    host=hosts["chunks.pdf"],
                    kind="pdf",
                    source=sources["chunks.pdf"],
                    chunks=5,
                    mtime=hosts["chunks.pdf"].stat().st_mtime,
                ),
                engine.FileWork(
                    host=hosts["saved.pdf"],
                    kind="pdf",
                    source=sources["saved.pdf"],
                    mtime=hosts["saved.pdf"].stat().st_mtime,
                ),
                engine.FileWork(
                    host=hosts["partial.pdf"],
                    kind="pdf",
                    source=sources["partial.pdf"],
                    articles=40,
                    chunks=8,
                    total_articles=80,
                    mtime=hosts["partial.pdf"].stat().st_mtime,
                ),
            ]
            session = Session(64)
            session.set_order("size_desc")
            session.set_tune_auto(False)
            session.set_chunks(64)
            book = ProgressBook(
                [
                    CatalogRow(sources["fresh.pdf"], "fresh.pdf", "upload", "pdf", 10, "pending"),
                    CatalogRow(sources["failed.pdf"], "failed.pdf", "upload", "pdf", 100, "failed"),
                    CatalogRow(sources["chunks.pdf"], "chunks.pdf", "upload", "pdf", 400, "pending"),
                    CatalogRow(sources["saved.pdf"], "saved.pdf", "upload", "pdf", 800, "pending", 0, 50),
                    CatalogRow(
                        sources["partial.pdf"],
                        "partial.pdf",
                        "upload",
                        "pdf",
                        1000,
                        "partial",
                        40,
                        80,
                        8,
                    ),
                ]
            )
            saved = book._rows[sources["saved.pdf"]]
            saved.measured_rate = 1.0
            session.bind_book(book)
            session.set_priority(sources["partial.pdf"], True)
            pool = SimpleNamespace(
                use_gpu=True,
                total_mib=16384,
                host_total_mib=32000,
                cpu_count=8,
                urls=["http://127.0.0.1:9/embed"],
            )
            stop = threading.Event()
            noted = []

            def keep(source):
                noted.append(source)

            book.keep_measured_rate = keep
            with (
                mock.patch.dict(os.environ, {"NOMAD_TURBO_STATE": str(Path(tmp) / "state")}),
                mock.patch.object(engine, "read_zim_date", side_effect=AssertionError("date read")),
                mock.patch.object(engine, "set_policy", side_effect=AssertionError("policy")),
            ):
                session.start_measure(from_idle=False)
                work, started, _epoch, action = engine._next_work(install := _install(str(storage)), pool, session, works, stop)
                self.assertEqual(work.source, sources["failed.pdf"])
                self.assertIsNotNone(started)
                self.assertEqual(action, "")
                self.assertNotIn(sources["partial.pdf"], session.sampled())
                session.request_index_now(sources["partial.pdf"])
                chosen, started, _epoch, action = engine._next_work(install, pool, session, works, stop)
                self.assertEqual(chosen.source, sources["partial.pdf"])
                self.assertIsNone(started)
                session.clear_index_now()
                session.note_sampled(sources["failed.pdf"])
                work, started, _epoch, action = engine._next_work(install, pool, session, works, stop)
                self.assertEqual(work.source, sources["fresh.pdf"])
                self.assertIsNotNone(started)
                session.note_sampled(sources["fresh.pdf"])
                empty, _started, _epoch, action = engine._next_work(install, pool, session, works, stop)
                self.assertIsNone(empty)
                self.assertEqual(action, "resume")
                self.assertFalse(stop.is_set())
                self.assertNotIn(sources["partial.pdf"], session.sampled())
                self.assertNotIn(sources["saved.pdf"], session.sampled())
                self.assertNotIn(sources["chunks.pdf"], session.sampled())
                self.assertEqual(noted, [])

        source = "/app/storage/kb_uploads/current.pdf"
        session = Session(8)
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state"
            state.mkdir()
            install = _install(tmp)
            started_clean = engine.FileWork(host=Path(tmp) / "current.pdf", kind="pdf", source=source)
            progressed = engine.FileWork(
                host=Path(tmp) / "current.pdf",
                kind="pdf",
                source=source,
                articles=6,
                chunks=2,
            )
            clock = {"now": 0.0}

            def pull_factory(articles):
                def pull(_timeout):
                    if not articles:
                        return None
                    return articles.pop(0)

                return pull

            def fake_embed(_url, _model, texts):
                clock["now"] = session.measure_clock() + 10
                return [[0.0] * install.dimension for _text in texts]

            class _Pool:
                urls = ["http://127.0.0.1:9/embed"]

                def batch_width(self, _chunks: int) -> int:
                    return 1

            pages = [
                {
                    "index": 1,
                    "path": "/A/One",
                    "title": "One",
                    "text": "hello world",
                    "sections": [{"heading": "", "text": "hello world", "level": 2}],
                },
                engine._EXTRACT_DONE,
            ]
            with mock.patch.dict(os.environ, {"NOMAD_TURBO_STATE": str(state)}):
                with (
                    mock.patch.object(engine, "embed", fake_embed),
                    mock.patch.object(engine, "upsert", lambda *_args, **_kwargs: None),
                    mock.patch.object(engine, "_wait_to_write", lambda *_args, **_kwargs: True),
                    mock.patch.object(engine, "delete_source_points", side_effect=AssertionError("deleted points")),
                    mock.patch.object(engine, "set_policy", side_effect=AssertionError("policy")),
                    mock.patch.object(engine.time, "monotonic", lambda: clock["now"]),
                ):
                    session.start_measure(from_idle=False)
                    clock["now"] = session.measure_clock()
                    check = engine._switch_check(install, session, progressed, None, 0, [progressed])
                    with self.assertRaises(RuntimeError) as caught:
                        engine._consume_articles(
                            install,
                            _Pool(),
                            progressed,
                            threading.Event(),
                            session,
                            set(),
                            pull_factory(pages),
                            None,
                            check,
                        )
                    self.assertEqual(str(caught.exception), "measure-skip")
                    self.assertEqual(progressed.articles, 6)
                    fresh_pages = [
                        {
                            "index": 1,
                            "path": "/A/One",
                            "title": "One",
                            "text": "hello world",
                            "sections": [{"heading": "", "text": "hello world", "level": 2}],
                        },
                        engine._EXTRACT_DONE,
                    ]
                    session.start_measure(from_idle=False)
                    clock["now"] = session.measure_clock()

                    def embed_until_done(_url, _model, texts):
                        clock["now"] = session.measure_clock() + 300
                        return [[0.0] * install.dimension for _text in texts]

                    with mock.patch.object(engine, "embed", embed_until_done):
                        check = engine._switch_check(install, session, started_clean, None, 0, [started_clean])
                        with self.assertRaises(RuntimeError) as caught:
                            engine._consume_articles(
                                install,
                                _Pool(),
                                started_clean,
                                threading.Event(),
                                session,
                                set(),
                                pull_factory(fresh_pages),
                                None,
                                check,
                            )
                    self.assertEqual(str(caught.exception), "measure")

    def test_eta_view_uses_a_measured_rate_when_the_file_is_not_running(self):
        text, seconds = file_eta_view(
            state="partial",
            articles=25,
            total=100,
            live_rate=99.0,
            measured_rate=0.5,
        )
        self.assertEqual(seconds, 150)
        self.assertEqual(text, format_eta(150))
        running, live = file_eta_view(
            state="running",
            articles=25,
            total=100,
            live_rate=1.0,
            measured_rate=0.5,
        )
        self.assertEqual(live, 75)
        self.assertEqual(running, format_eta(75))
        blank, missing = file_eta_view(
            state="pending",
            articles=0,
            total=None,
            live_rate=None,
            measured_rate=1.0,
        )
        self.assertEqual((blank, missing), ("", None))
        self.assertIsNone(sum_known_etas([("pending", 150), ("partial", None)]))
        self.assertEqual(sum_known_etas([("pending", 150), ("indexed", 0)]), 150)


def _tree_keys(store, parent=None):
    rows = []
    iterator = store.iter_children(parent)
    while iterator is not None:
        rows.append((store.get_value(iterator, 0), _tree_keys(store, iterator)))
        iterator = store.iter_next(iterator)
    return rows


class LibraryTreeTest(unittest.TestCase):
    def test_uploads_and_zims_keep_their_directories(self):
        items = [
            SimpleNamespace(
                source="/app/storage/kb_uploads/textbooks/biology/Biology-2e_-_WEB.pdf",
                name="Biology-2e_-_WEB.pdf",
                origin="upload",
                state="indexed",
                size_bytes=400,
            ),
            SimpleNamespace(
                source="/app/storage/kb_uploads/ca-federal-law/acts/C-46.md",
                name="C-46.md",
                origin="upload",
                state="pending",
                size_bytes=20,
            ),
            SimpleNamespace(
                source="/app/storage/kb_uploads/loose.pdf",
                name="loose.pdf",
                origin="upload",
                state="pending",
                size_bytes=5,
            ),
            SimpleNamespace(
                source="/app/storage/zim/wikipedia_en.zim",
                name="wikipedia_en.zim",
                origin="zim",
                state="running",
                size_bytes=800,
            ),
            SimpleNamespace(
                source="notes.zim",
                name="notes.zim",
                origin="zim",
                state="pending",
                size_bytes=3,
            ),
        ]
        self.assertEqual(
            library_place(items[0].source, items[0].name, "upload"),
            ("kb_uploads", ("textbooks", "biology")),
        )
        roots = library_tree(items)
        self.assertEqual([folder.name for folder in roots], ["kb_uploads", "zim"])
        uploads, zims = roots
        self.assertEqual(uploads.file_count, 3)
        self.assertEqual(uploads.size_bytes, 425)
        self.assertEqual([folder.name for folder in uploads.folders], ["ca-federal-law", "textbooks"])
        self.assertEqual(uploads.files, ("/app/storage/kb_uploads/loose.pdf",))
        biology = uploads.folders[1].folders[0]
        self.assertEqual(biology.key, "kb_uploads/textbooks/biology")
        self.assertEqual(biology.files, (items[0].source,))
        self.assertEqual(zims.files, ("notes.zim", "/app/storage/zim/wikipedia_en.zim"))
        row = folder_store_row(uploads)
        self.assertEqual(len(row), 19)
        self.assertEqual(row[COL_TYPE], FOLDER)
        self.assertEqual(row[COL_NAME], "kb_uploads")
        self.assertEqual(row[COL_STATUS], "3 files")
        self.assertEqual(row[COL_ARTICLES], EM_DASH)
        hidden = library_tree(items, hide_finished=True)
        self.assertEqual(hidden[0].file_count, 2)
        self.assertEqual([folder.name for folder in hidden[0].folders], ["ca-federal-law"])
        self.assertNotIn("textbooks", [folder.name for folder in hidden[0].folders])
        held = hold_store_row()
        self.assertEqual(len(held), 19)
        self.assertEqual(held[0], FOLDER_HOLD)
        self.assertEqual(held[COL_TYPE], FOLDER)


class IndexPageControlsTest(unittest.TestCase):
    def test_the_window_uses_the_shipped_labels(self):
        text = (ROOT / "nomad_turbo" / "window.py").read_text(encoding="utf-8")
        self.assertIn("INDEX_HEADERS", text)
        self.assertIn("row_menu_labels", text)
        self.assertIn("ORDER_CHOICES", text)
        self.assertIn('label="Measure"', text)
        self.assertIn('TIPS["index_now"]', text)
        self.assertIn('TIPS["priority"]', text)
        self.assertIn('TIPS["order"]', text)
        self.assertIn('TIPS["measure"]', text)
        self.assertEqual(row_menu_labels(priority=False)[0], "Index now")
        self.assertIn("Set priority", row_menu_labels(priority=False))
        self.assertIn("Clear priority", row_menu_labels(priority=True))
        self.assertEqual(len(order_choice_labels()), 17)
        try:
            import gi

            gi.require_version("Gtk", "3.0")
            from gi.repository import GLib, Gtk
        except Exception as exc:
            self.skipTest(f"gtk import failed: {exc}")
        ready = Gtk.init_check()
        if isinstance(ready, tuple):
            ready = ready[0]
        if not ready:
            self.skipTest("Gtk.init_check failed")
        from nomad_turbo.window import _Window, describe_index_controls

        previous_state = os.environ.get("NOMAD_TURBO_STATE")
        previous_config = os.environ.get("NOMAD_TURBO_CONFIG")
        state = tempfile.TemporaryDirectory(prefix="nomad-turbo-index-")
        os.environ["NOMAD_TURBO_STATE"] = str(Path(state.name) / "state")
        os.environ["NOMAD_TURBO_CONFIG"] = str(Path(state.name) / "config.json")
        app = None
        try:
            from nomad_turbo.window import _MemoryLog

            app = _Window(_install(state.name), Session(55), _MemoryLog(), Path(state.name) / "session.log", Gtk, GLib)
            titles = [column.get_title() for column in app.file_tree.get_columns()]
            self.assertEqual(titles, list(INDEX_HEADERS))
            dump = describe_index_controls(app)
            self.assertIn("\n".join(["headers:", *INDEX_HEADERS]), dump)
            self.assertIn("\n".join(["order:", *order_choice_labels()]), dump)
            self.assertIn("measure:\nMeasure\n", dump)
            self.assertIn("then-index:\nThen index\n", dump)
            self.assertIn("hide:\nHide finished\n", dump)
            self.assertFalse(app.hide_finished.get_active())
            self.assertIn("menu:\nIndex now\nSet priority\nDetails\n", dump)
            self.assertIn("menu-priority:\nIndex now\nClear priority\nDetails\n", dump)
            self.assertEqual(app.measure_button.get_label(), "Measure")
            self.assertEqual(app.then_index_button.get_label(), "Then index")
            app.rows_ready = True
            app.qdrant_ready = True
            app._buttons("stopped")
            self.assertTrue(app.measure_button.get_sensitive())
            self.assertFalse(app.then_index_button.get_sensitive())
            self.assertIs(app.measure_button.get_parent(), app.then_index_button.get_parent())
            segment = app.measure_button.get_parent()
            self.assertTrue(segment.get_style_context().has_class("linked"))
            self.assertEqual(segment.get_spacing(), 0)
            self.assertEqual(segment.get_children(), [app.measure_button, app.then_index_button])
            self.assertEqual(
                [child.get_label() for child in app.run_group.get_children()],
                ["Start indexing", "Stop", "Rescan", "Stay on"],
            )
            self.assertEqual(app.keep_check.get_label(), "Stay on")
            self.assertIn(
                "Stay on after the last file, and index new files as they arrive",
                app.keep_check.get_tooltip_text(),
            )
            for widget in (
                app.host_button,
                app.hide_finished,
                app.keep_check,
                app.tune_auto,
                app.error_check,
                app.info_check,
                app.debug_check,
                app.follow_check,
                app.repair_quick,
                app.repair_complete,
            ):
                self.assertEqual(type(widget).__name__, "ToggleButton", type(widget).__name__)
            self.assertTrue(app.repair_quick.get_active())
            self.assertFalse(app.repair_complete.get_active())
            self.assertIs(app.repair_quick.get_parent(), app.repair_complete.get_parent())
            self.assertEqual(
                [child.get_label() for child in app.repair_actions.get_children()],
                ["Scan", "Start repair", "Stop"],
            )
            threads = threading.active_count()
            app.on_then_index(app.then_index_button)
            self.assertFalse(app.session.measuring())
            self.assertFalse(app.then_index_button.get_active())
            self.assertIsNone(app.thread)
            self.assertEqual(threading.active_count(), threads)
            app.session.start_measure(from_idle=True)
            clock = app.session.measure_clock()
            app.session.note_sampled("kept.pdf")
            sampled = set(app.session.sampled())
            app._buttons("running")
            self.assertFalse(app.measure_button.get_sensitive())
            self.assertTrue(app.then_index_button.get_sensitive())
            app.on_then_index(app.then_index_button)
            self.assertTrue(app.session.measure_then_index())
            self.assertEqual(app.session.measure_clock(), clock)
            self.assertEqual(app.session.sampled(), sampled)
            self.assertTrue(app.then_index_button.get_active())
            self.assertIsNone(app.thread)
            self.assertEqual(threading.active_count(), threads)
            self.assertEqual(
                measure_ending(
                    from_idle=app.session.measure_from_idle(),
                    pass_complete=True,
                    then_index=app.session.measure_then_index(),
                ),
                "resume",
            )
            release_measure(app.session)
            app._buttons("stopped")
            self.assertTrue(app.measure_button.get_sensitive())
            self.assertFalse(app.then_index_button.get_sensitive())
            self.assertFalse(app.then_index_button.get_active())
            self.assertIn("keep indexing", app.then_index_button.get_tooltip_text().lower())
            self.assertEqual(app.order_combo.get_active_id(), "library")
            app.session.set_order("name_asc")
            app.order_combo.set_active_id("size_desc")
            self.assertEqual(app.session.get_order(), "size_desc")
            self.assertEqual(app.install.index_order, "size_desc")
            from nomad_turbo.schedule import normalize_order

            self.assertEqual(normalize_order("not-an-order"), "library")
            saved = Install.from_json(Path(os.environ["NOMAD_TURBO_CONFIG"]).read_text())
            self.assertEqual(saved.index_order, "size_desc")
            restored = _Window(saved, Session(55), _MemoryLog(), Path(state.name) / "order.log", Gtk, GLib)
            try:
                self.assertEqual(restored.order_combo.get_active_id(), "size_desc")
                self.assertEqual(restored.session.get_order(), "size_desc")
            finally:
                restored.window.destroy()
            self.assertIsNone(engine_switch_reason(app.session, "example.zim", None, 0, app.session.arm_priority()))
            app.file_tree.get_columns()[2].clicked()
            self.assertEqual(app.session.get_order(), "size_desc")
            sorts = [column.get_sort_column_id() for column in app.file_tree.get_columns()]
            self.assertEqual(sorts, [-1] * len(INDEX_HEADERS))
            cells = app.file_tree.get_columns()[0].get_cells()
            self.assertEqual(len(cells), 2)
            from gi.repository import Pango

            self.assertIn('mark_cell.set_property("foreground", PRIORITY_COLOR)', text)
            self.assertIn("file_column.pack_start(mark_cell, False)", text)
            self.assertEqual(
                app.file_tree.get_selection().get_mode(),
                Gtk.SelectionMode.NONE,
            )
            self.assertEqual(
                app.now_tree.get_selection().get_mode(),
                Gtk.SelectionMode.NONE,
            )
            self.assertNotIn("tree.set_cursor(path, None, False)", text)
            self.assertIn('connect("test-expand-row"', text)
            self.assertEqual(cells[0].get_property("ellipsize"), Pango.EllipsizeMode.NONE)
            self.assertEqual(cells[1].get_property("ellipsize"), Pango.EllipsizeMode.MIDDLE)
            self.assertEqual(PRIORITY_COLOR, "#c0392b")
            book = ProgressBook(
                [CatalogRow("example.zim", "example.zim", "zim", "zim", 10, "pending")]
            )
            app.store.clear()
            app.store.append(None, index_store_row(book.snapshot(now=0).files[0], priority=True))
            self.assertEqual(app.store[0][COL_MARK], "*")
            self.assertEqual(app.store[0][COL_NAME], "example.zim")
            app.session.set_order("size_desc")
            app.session.bind_book(
                ProgressBook(
                    [
                        CatalogRow("done.zim", "done.zim", "zim", "zim", 900, "indexed", walk_index=5),
                        CatalogRow("now.pdf", "now.pdf", "upload", "pdf", 10, "running", walk_index=4),
                        CatalogRow("big.zim", "big.zim", "zim", "zim", 500, "pending", walk_index=1),
                        CatalogRow("small.zim", "small.zim", "zim", "zim", 20, "pending", walk_index=0),
                        CatalogRow("mid.pdf", "mid.pdf", "upload", "pdf", 50, "pending", walk_index=2),
                    ]
                )
            )
            app.session.set_priority("big.zim", True)
            app.session.set_priority("small.zim", True)
            app.session.set_priority("done.zim", True)
            app.refresh_rows()
            self.assertEqual([app.now_store[i][0] for i in range(len(app.now_store))], ["now.pdf"])
            both = app.session.get_book()
            both.update("mid.pdf", articles=0, total=None, chunks=0, state="running", now=1)
            app.session.set_ocr_activity("Reading mid.pdf with OCR. Page 4 of 10.")
            app.refresh_rows()
            self.assertEqual([app.now_store[i][0] for i in range(len(app.now_store))], ["now.pdf", "mid.pdf"])
            self.assertEqual(app.now_store[1][COL_STATUS], "OCR")
            self.assertEqual(app.now_store[1][COL_ARTICLES], "4 / 10")
            self.assertNotEqual(app.now_store[0][COL_STATUS], "OCR")
            both.update("mid.pdf", articles=0, total=None, chunks=0, state="pending", now=2)
            app.session.clear_ocr_activity("mid.pdf")
            app.refresh_rows()
            self.assertEqual([app.now_store[i][0] for i in range(len(app.now_store))], ["now.pdf"])
            self.assertEqual(_tree_keys(app.store), [
                ("kb_uploads", [(FOLDER_HOLD, [])]),
                ("zim", [(FOLDER_HOLD, [])]),
            ])
            self.assertEqual(app.store[0][COL_TYPE], FOLDER)
            self.assertEqual(app.store[0][COL_STATUS], "2 files")
            app.file_tree.expand_row(Gtk.TreePath.new_from_indices([0]), False)
            self.assertIn("kb_uploads", app._open_folders)
            self.assertEqual(_tree_keys(app.store), [
                ("kb_uploads", [("mid.pdf", []), ("now.pdf", [])]),
                ("zim", [(FOLDER_HOLD, [])]),
            ])
            app.refresh_rows()
            self.assertTrue(app.file_tree.row_expanded(Gtk.TreePath.new_from_indices([0])))
            self.assertEqual(_tree_keys(app.store), [
                ("kb_uploads", [("mid.pdf", []), ("now.pdf", [])]),
                ("zim", [(FOLDER_HOLD, [])]),
            ])
            book = app.session.get_book()
            epoch = book.structure_epoch
            book.update("now.pdf", articles=4, total=10, chunks=2, state="running", now=8)
            self.assertEqual(book.structure_epoch, epoch)
            app.refresh_rows()
            self.assertEqual(_tree_keys(app.store), [
                ("kb_uploads", [("mid.pdf", []), ("now.pdf", [])]),
                ("zim", [(FOLDER_HOLD, [])]),
            ])
            now_row = app.store.iter_children(app.store.get_iter_first())
            now_row = app.store.iter_next(now_row)
            self.assertEqual(app.store.get_value(now_row, 0), "now.pdf")
            self.assertEqual(app.store.get_value(now_row, COL_ARTICLES), "4 / 10")
            self.assertEqual(app.now_store[0][COL_ARTICLES], "4 / 10")
            self.assertIn(INDEXING_MARK, app.store.get_value(now_row, COL_MARK))
            self.assertNotIn(INDEXING_MARK, str(app.now_store[0][COL_MARK] or ""))
            pending = app.store.iter_children(app.store.get_iter_first())
            self.assertEqual(app.store.get_value(pending, 0), "mid.pdf")
            self.assertNotIn(INDEXING_MARK, app.store.get_value(pending, COL_MARK) or "")
            mark_cell = app.file_tree.get_columns()[0].get_cells()[0]
            app._paint_tree_mark(None, mark_cell, app.store, now_row, None)
            self.assertIn(INDEXING_MARK, mark_cell.get_property("text"))
            app._paint_tree_mark(None, mark_cell, app.store, pending, None)
            self.assertNotIn(INDEXING_MARK, mark_cell.get_property("text") or "")
            book.update("now.pdf", articles=4, total=10, chunks=2, state="pending", now=9)
            app.refresh_rows()
            cleared = None
            walker = app.store.iter_children(app.store.get_iter_first())
            while walker is not None:
                if app.store.get_value(walker, 0) == "now.pdf":
                    cleared = walker
                    break
                walker = app.store.iter_next(walker)
            self.assertIsNotNone(cleared)
            self.assertNotIn(INDEXING_MARK, app.store.get_value(cleared, COL_MARK) or "")
            # The rest of this window still expects the file that is indexing.
            book.update("now.pdf", articles=4, total=10, chunks=2, state="running", now=9)
            app.refresh_rows()
            app.file_tree.expand_row(Gtk.TreePath.new_from_indices([1]), False)
            self.assertEqual(_tree_keys(app.store), [
                ("kb_uploads", [("mid.pdf", []), ("now.pdf", [])]),
                ("zim", [("big.zim", []), ("done.zim", []), ("small.zim", [])]),
            ])
            zim = app.store.iter_next(app.store.get_iter_first())
            big = app.store.iter_children(zim)
            self.assertEqual(app.store.get_value(big, COL_MARK), PRIORITY_MARK)
            app.hide_finished.set_active(True)
            self.assertEqual([app.now_store[i][0] for i in range(len(app.now_store))], ["now.pdf"])
            self.assertEqual(_tree_keys(app.store), [
                ("kb_uploads", [("mid.pdf", []), ("now.pdf", [])]),
                ("zim", [("big.zim", []), ("small.zim", [])]),
            ])
            self.assertEqual(app.store[1][COL_STATUS], "2 files")
            self.assertTrue(app.file_tree.row_expanded(Gtk.TreePath.new_from_indices([0])))
            app.file_tree.collapse_row(Gtk.TreePath.new_from_indices([0]))
            self.assertNotIn("kb_uploads", app._open_folders)
            self.assertEqual(_tree_keys(app.store), [
                ("kb_uploads", [(FOLDER_HOLD, [])]),
                ("zim", [("big.zim", []), ("small.zim", [])]),
            ])
            self.assertEqual([app.now_store[i][0] for i in range(len(app.now_store))], ["now.pdf"])
            app.file_stats.note_measurement(
                "big.zim",
                article_rate=0.5,
                chunk_rate=2,
                eta_seconds=120,
                articles=1,
                total_articles=10,
                chunks=4,
                state="pending",
            )
            shown = dict(app.file_detail_rows("big.zim"))
            self.assertEqual(shown["Priority"], "prioritized")
            self.assertEqual(shown["Last measured ETA"], "2m")
            self.assertEqual(shown["File type"], "zim")
            self.assertEqual(shown["Minimum art/min"], "30")
            app.file_stats.note_library("gone.zim", name="gone.zim", file_type="zim", size_bytes=20, state="pending")
            app.file_stats.set_priority("gone.zim", True)
            app._remember_library(
                app.session.get_book(),
                [CatalogRow("big.zim", "big.zim", "zim", "zim", 500, "pending")],
            )
            self.assertIsNone(app.file_stats.get("gone.zim"))
            self.assertIsNotNone(app.file_stats.get("big.zim"))
            self.assertFalse(app.now_tree.get_columns()[0].get_expand())
            self.assertFalse(app.file_tree.get_columns()[0].get_expand())
            self.assertTrue(app.now_tree.get_columns()[-1].get_expand())
            self.assertTrue(app.file_tree.get_columns()[-1].get_expand())
            app.window.resize(1280, 640)
            app.window.show_all()
            for _ in range(40):
                while Gtk.events_pending():
                    Gtk.main_iteration_do(False)

            def starts(tree):
                x = 0
                found = []
                for column in tree.get_columns():
                    found.append(x)
                    x += column.get_width()
                return found

            self.assertGreater(app.now_tree.get_allocated_width(), 400)
            self.assertEqual(starts(app.now_tree), starts(app.file_tree))
            size = app.now_tree.get_columns()[2]
            size.set_fixed_width(130)
            app._on_column_width(size, None, "index:Size")
            for _ in range(10):
                while Gtk.events_pending():
                    Gtk.main_iteration_do(False)
            self.assertEqual(app.file_tree.get_columns()[2].get_fixed_width(), 130)
            self.assertEqual(starts(app.now_tree), starts(app.file_tree))
        finally:
            if app is not None:
                app.window.destroy()
            if previous_state is None:
                os.environ.pop("NOMAD_TURBO_STATE", None)
            else:
                os.environ["NOMAD_TURBO_STATE"] = previous_state
            if previous_config is None:
                os.environ.pop("NOMAD_TURBO_CONFIG", None)
            else:
                os.environ["NOMAD_TURBO_CONFIG"] = previous_config
            state.cleanup()


class LibraryWatchTest(unittest.TestCase):
    def test_a_running_file_is_not_a_reason_to_walk_the_library(self):
        directory = tempfile.TemporaryDirectory(prefix="nomad-turbo-watch-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "zim").mkdir()
        (root / "kb_uploads").mkdir()
        current_host = root / "zim" / "current.zim"
        old_host = root / "zim" / "old.zim"
        current_host.write_bytes(b"zim")
        old_host.write_bytes(b"zim")
        current = "/app/storage/zim/current.zim"
        old = "/app/storage/zim/old.zim"
        new = "/app/storage/kb_uploads/new.md"
        book = ProgressBook(
            [
                CatalogRow(current, "current.zim", "zim", "zim", 10, "running"),
                CatalogRow(old, "old.zim", "zim", "zim", 10, "pending"),
            ]
        )
        session = Session(8)
        session.bind_book(book)
        install = _install(str(root))
        live = engine._LiveFile()
        state = root / "state"
        state.mkdir()
        previous_live = engine._library_live
        previous_seen = engine._downloads_seen
        walks = {"n": 0}
        real_rglob = Path.rglob

        def counting(self, pattern):
            walks["n"] += 1
            return real_rglob(self, pattern)

        try:
            with mock.patch.dict(os.environ, {"NOMAD_TURBO_STATE": str(state)}):
                live.begin(current)
                engine._library_live = live
                (root / "kb_uploads" / "new.md").write_text("new", encoding="utf-8")
                old_host.unlink()
                with mock.patch.object(Path, "rglob", counting):
                    with mock.patch.object(engine, "_drop_source", return_value=True):
                        engine._sync_library(install, session, live)
                        self.assertFalse(engine._downloads_pending(install))
                self.assertEqual(walks["n"], 0)
                listed = dict(book.listed())
                self.assertIn(old, listed)
                self.assertNotIn(new, listed)
                self.assertFalse(live.retire.is_set())
                current_host.unlink()
                walks["n"] = 0
                with mock.patch.object(Path, "rglob", counting):
                    engine._sync_library(install, session, live)
                self.assertEqual(walks["n"], 0)
                self.assertTrue(live.retire.is_set())

                live.end()
                engine._library_live = None
                with mock.patch.object(engine, "_drop_source", return_value=True):
                    engine._sync_library(install, session, live)
                listed = dict(book.listed())
                self.assertNotIn(old, listed)
                self.assertIn(new, listed)
                self.assertEqual(listed[new], "pending")
        finally:
            engine._library_live = previous_live
            engine._downloads_seen = previous_seen

    def test_the_pause_between_files_walks_the_library_once(self):
        directory = tempfile.TemporaryDirectory(prefix="nomad-turbo-once-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "zim").mkdir()
        (root / "kb_uploads").mkdir()
        old_host = root / "zim" / "old.zim"
        act_host = root / "kb_uploads" / "act.md"
        fresh_host = root / "kb_uploads" / "fresh.md"
        old_host.write_bytes(b"zim")
        act_host.write_text("act", encoding="utf-8")
        fresh_host.write_text("fresh", encoding="utf-8")
        old = _stamp(2010)
        os.utime(old_host, (old, old))
        os.utime(act_host, (old, old))
        (root / "zim" / "incoming.zim.part").write_bytes(b"partial")
        install = _install(str(root))
        old_source = engine.host_to_container(install, str(old_host))
        act_source = engine.host_to_container(install, str(act_host))
        fresh_source = engine.host_to_container(install, str(fresh_host))
        session = Session(8)
        session.bind_book(ProgressBook([]))
        live = engine._LiveFile()
        state = root / "state"
        state.mkdir()
        previous_live = engine._library_live
        previous_seen = engine._downloads_seen
        previous_at = engine._last_download_at
        walks = {"n": 0}
        real_rglob = Path.rglob

        def counting(self, pattern):
            walks["n"] += 1
            return real_rglob(self, pattern)

        try:
            with (
                mock.patch.dict(os.environ, {"NOMAD_TURBO_STATE": str(state)}),
                mock.patch.object(Path, "rglob", counting),
                mock.patch.object(engine, "ingest_states", return_value={}),
            ):
                engine._library_live = None
                engine._last_download_at = None
                snap = engine._look_between_files(install, session, live)
                self.assertEqual(walks["n"], 2)
                self.assertIsNotNone(snap)
                self.assertTrue(snap.downloads)
                walks["n"] = 0
                batch = engine._queue_batch(install, session, [], set(), set(), snap)
                hold = engine._library_hold(install, snap.downloads)
                self.assertEqual(walks["n"], 0)
                self.assertEqual(hold, "download")
                sources = [item.source for item in batch]
                self.assertIn(old_source, sources)
                self.assertIn(act_source, sources)
                self.assertNotIn(fresh_source, sources)
                self.assertEqual(dict(session.get_book().listed())[fresh_source], "pending")

                live.begin(old_source)
                engine._library_live = live
                walks["n"] = 0
                self.assertIsNone(engine._look_between_files(install, session, live))
                self.assertEqual(walks["n"], 0)
        finally:
            engine._library_live = previous_live
            engine._downloads_seen = previous_seen
            engine._last_download_at = previous_at


class StayDecisionTest(unittest.TestCase):
    def test_between_articles_does_not_snapshot_the_library(self):
        count = 3000
        rows = [
            CatalogRow(
                f"/app/storage/kb_uploads/f{i}.md",
                f"f{i}.md",
                "upload",
                "text",
                1000 + i,
                "pending",
            )
            for i in range(count)
        ]
        current = "/app/storage/zim/current.zim"
        other = "/app/storage/kb_uploads/f7.md"
        rows.append(
            CatalogRow(current, "current.zim", "zim", "zim", 5000, "running", articles=10, total_articles=100, chunks=20)
        )
        book = ProgressBook(rows)
        session = Session(8)
        session.bind_book(book)
        book.update(current, articles=12, total=100, chunks=24, state="running", now=5)
        snaps = {"n": 0}
        real_snapshot = book.snapshot

        def counting(now=None):
            snaps["n"] += 1
            return real_snapshot(now)

        book.snapshot = counting
        session.request_index_now(other)
        armed = session.arm_priority()
        now_reason = engine.article_stay_reason(session, current, None, 0, armed)
        self.assertEqual(now_reason, "now")
        self.assertEqual(now_reason, engine_switch_reason(session, current, None, 0, armed))

        session.clear_index_now()
        armed = session.arm_priority()
        session.set_order("size_desc")
        order_reason = engine.article_stay_reason(session, current, None, 0, armed)
        self.assertIsNone(order_reason)
        self.assertEqual(order_reason, engine_switch_reason(session, current, None, 0, armed))

        session.set_priority(other, True)
        priority_reason = engine.article_stay_reason(session, current, None, 0, armed)
        self.assertEqual(priority_reason, "priority")
        self.assertEqual(priority_reason, engine_switch_reason(session, current, None, 0, armed))
        self.assertEqual(snaps["n"], 0)

        pulse = book.pulse(now=5)
        full = book.snapshot(now=5)
        self.assertEqual(pulse.overall_text, full.overall_text)
        self.assertEqual(pulse.done_files, full.done_files)
        self.assertEqual(pulse.chunks_stored, full.chunks_stored)
        self.assertEqual(pulse.overall_fraction, full.overall_fraction)
        self.assertEqual(len(full.files), count + 1)
        self.assertEqual(len(pulse.files), 1)
        self.assertEqual(pulse.files[0].source, current)


class FileStatsTest(unittest.TestCase):
    def test_measurements_priority_and_dates_survive_a_restart(self):
        directory = tempfile.TemporaryDirectory(prefix="nomad-turbo-stats-")
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "file-stats.json"
        source = "/app/storage/zim/wiki.zim"
        gone = "/app/storage/zim/old.zim"
        stats = FileStats(path)
        stats.note_library(
            source,
            name="wiki.zim",
            file_type="zim",
            size_bytes=10_485_760,
            location=library_location("/mnt/nomad/storage", "/app/storage", source),
            mtime=1_700_000_000,
            articles=25,
            total_articles=100,
            chunks=40,
            state="partial",
        )
        stats.set_priority(source, True)
        stats.note_zim_date(source, "2024-06")
        stats.note_measurement(
            source,
            article_rate=0.5,
            chunk_rate=2,
            eta_seconds=150,
            articles=25,
            total_articles=100,
            chunks=40,
            state="partial",
        )
        stats.note_rates(source, 0.2, 1, measured=False)
        stats.note_rates(source, 0.8, 4, measured=False)
        stats.note_library(gone, name="old.zim", file_type="zim", size_bytes=20, state="pending")
        stats.set_priority(gone, True)
        stats.save()

        loaded = FileStats(path)
        loaded.load()
        kept = loaded.get(source)
        self.assertIsNotNone(kept)
        self.assertTrue(kept.priority)
        self.assertEqual(kept.zim_date, "2024-06")
        self.assertEqual(kept.location, "/mnt/nomad/storage/zim/wiki.zim")
        self.assertAlmostEqual(kept.measured_article_rate, 0.5)
        self.assertAlmostEqual(kept.min_article_rate, 0.2)
        self.assertAlmostEqual(kept.max_article_rate, 0.8)
        self.assertAlmostEqual(kept.min_chunk_rate, 1)
        self.assertAlmostEqual(kept.max_chunk_rate, 4)
        self.assertEqual(kept.measured_eta_seconds, 150)
        self.assertEqual(loaded.priorities(), {source, gone})
        text = detail_text(kept)
        self.assertIn("File type: zim", text)
        self.assertIn("Location: /mnt/nomad/storage/zim/wiki.zim", text)
        self.assertIn("Date downloaded: " + format_downloaded(1_700_000_000), text)
        self.assertIn("ZIM date: 2024-06", text)
        self.assertIn("Priority: prioritized", text)
        self.assertIn("Articles: 25 / 100", text)
        self.assertIn("Chunks: 40", text)
        self.assertIn("Minimum art/min: 12", text)
        self.assertIn("Maximum art/min: 48", text)
        self.assertIn("Last measured ETA: 2m", text)
        self.assertIn(format_bytes(10_485_760), text)
        self.assertEqual(loaded.prune({source}), [gone])
        loaded.save()
        again = FileStats(path)
        again.load()
        self.assertIsNone(again.get(gone))
        self.assertNotIn(gone, again.priorities())
        self.assertTrue(again.get(source).priority)

        book = ProgressBook(
            [CatalogRow(source, "wiki.zim", "zim", "zim", 10_485_760, "partial", 25, 100, chunks=40)]
        )
        book.restore_saved(again)
        snap = book.snapshot(now=0)
        self.assertEqual(snap.files[0].eta_seconds, 150)
        self.assertEqual(snap.files[0].eta, format_eta(150))

        session = Session(8)
        session.bind_file_stats(again)
        session.note_zim_date(source, None)
        self.assertTrue(again.get(source).zim_known)
        self.assertEqual(again.get(source).zim_date, "")
        self.assertIsNone(session.zim_date(source))

    def test_a_sample_records_min_and_max_and_a_removed_file_is_forgotten(self):
        directory = tempfile.TemporaryDirectory(prefix="nomad-turbo-pace-")
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "file-stats.json"
        stats = FileStats(path)
        book = ProgressBook(
            [CatalogRow("a.pdf", "a.pdf", "upload", "pdf", 4096, "partial", 0, 100, mtime=1_700_000_000)]
        )
        book.restore_saved(stats)
        book.update("a.pdf", articles=0, total=100, chunks=0, state="running", now=0)
        book.update("a.pdf", articles=10, total=100, chunks=20, state="running", now=10)
        book.update("a.pdf", articles=40, total=100, chunks=80, state="running", now=20)
        book.keep_measured_rate("a.pdf")
        saved = stats.get("a.pdf")
        self.assertAlmostEqual(saved.min_article_rate, 1)
        self.assertAlmostEqual(saved.max_article_rate, 2)
        self.assertAlmostEqual(saved.measured_article_rate, 2)
        self.assertAlmostEqual(saved.min_chunk_rate, 2)
        self.assertAlmostEqual(saved.max_chunk_rate, 4)
        self.assertEqual(saved.measured_eta_seconds, 30)
        self.assertEqual(saved.file_type, "pdf")
        self.assertFalse(saved.priority)
        stats.save()
        book.drop("a.pdf")
        self.assertIsNone(stats.get("a.pdf"))
        reloaded = FileStats(path)
        reloaded.load()
        self.assertIsNone(reloaded.get("a.pdf"))

    def test_progress_rewrites_one_file_and_an_old_document_still_loads(self):
        directory = tempfile.TemporaryDirectory(prefix="nomad-turbo-one-file-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        path = root / "file-stats.json"
        stats = FileStats(path)
        count = 3000
        sources = [f"/app/storage/kb_uploads/law/f{i}.md" for i in range(count)]
        for index, source in enumerate(sources):
            stats.note_library(
                source,
                name=f"f{index}.md",
                file_type="text",
                size_bytes=2000 + index,
                mtime=1_700_000_000.0,
                articles=1,
                chunks=2,
                state="pending",
            )
            stats.set_priority(source, index % 2 == 0)
            if index % 5 == 0:
                stats.note_zim_date(source, "2024-06")
            stats.note_rates(source, 0.5, 3.0, measured=True, eta_seconds=12)
        stats.save()

        def facts(row):
            return (
                row.priority,
                row.zim_date,
                row.zim_known,
                row.articles,
                row.total_articles,
                row.chunks,
                row.state,
                row.measured_article_rate,
                row.measured_chunk_rate,
                row.last_article_rate,
                row.last_chunk_rate,
                row.measured_eta_seconds,
                row.mtime,
                row.size_bytes,
                row.name,
            )

        first = FileStats(path)
        first.load()
        remembered = {source: facts(first.get(source)) for source in sources}
        target = sources[10]
        before = {}
        for item in root.rglob("*"):
            if item.is_file():
                before[str(item.relative_to(root))] = item.read_bytes()
        self.assertIn("file-stats.json", before)
        self.assertGreater(len(before["file-stats.json"]), 100_000)

        stats.note_progress(target, articles=40, total_articles=90, chunks=15, state="running")
        stats.save()
        after = {}
        for item in root.rglob("*"):
            if item.is_file():
                after[str(item.relative_to(root))] = item.read_bytes()
        self.assertEqual(before["file-stats.json"], after["file-stats.json"])
        written = 0
        for name, data in after.items():
            if before.get(name) != data:
                written += len(data)
        self.assertLess(written, 16_384)
        self.assertGreater(written, 0)

        again = FileStats(path)
        again.load()
        for source, previous in remembered.items():
            row = again.get(source)
            self.assertIsNotNone(row)
            if source == target:
                self.assertEqual(row.articles, 40)
                self.assertEqual(row.total_articles, 90)
                self.assertEqual(row.chunks, 15)
                self.assertEqual(row.state, "running")
                self.assertEqual(row.priority, previous[0])
                self.assertEqual(row.zim_date, previous[1])
                self.assertEqual(row.measured_article_rate, previous[7])
                self.assertEqual(row.measured_chunk_rate, previous[8])
                self.assertEqual(row.measured_eta_seconds, previous[11])
                continue
            self.assertEqual(facts(row), previous)

        legacy_path = root / "legacy.json"
        legacy_path.write_text(
            json.dumps(
                {
                    "files": {
                        "/app/storage/zim/legacy.zim": {
                            "name": "legacy.zim",
                            "priority": True,
                            "zim_date": "2020-01",
                            "articles": 3,
                            "chunks": 9,
                            "state": "partial",
                            "measured_article_rate": 1.5,
                        }
                    }
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        legacy = FileStats(legacy_path)
        legacy.load()
        kept = legacy.get("/app/storage/zim/legacy.zim")
        self.assertIsNotNone(kept)
        self.assertTrue(kept.priority)
        self.assertEqual(kept.zim_date, "2020-01")
        self.assertEqual(kept.articles, 3)
        self.assertEqual(kept.chunks, 9)
        self.assertEqual(kept.state, "partial")
        self.assertAlmostEqual(kept.measured_article_rate, 1.5)

    def test_a_broken_stats_file_does_not_raise(self):
        directory = tempfile.TemporaryDirectory(prefix="nomad-turbo-bad-stats-")
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "file-stats.json"
        path.write_text("{", encoding="utf-8")
        stats = FileStats(path)
        stats.load()
        self.assertIsNone(stats.get("missing"))
        self.assertEqual(library_location("/mnt/nomad/storage", "/app/storage", "notes.pdf"), "notes.pdf")

    def test_priority_goes_away_when_a_file_finishes_or_the_index_is_deleted(self):
        directory = tempfile.TemporaryDirectory(prefix="nomad-turbo-priority-")
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "file-stats.json"
        done = "/app/storage/kb_uploads/done.md"
        kept = "/app/storage/zim/later.zim"
        stats = FileStats(path)
        stats.note_library(done, name="done.md", file_type="text", state="running")
        stats.note_library(kept, name="later.zim", file_type="zim", state="pending")
        stats.set_priority(done, True)
        stats.set_priority(kept, True)
        session = Session(4)
        session.bind_file_stats(stats)
        session.set_priority(done, True)
        session.set_priority(kept, True)

        engine._drop_priority(session, done)
        self.assertNotIn(done, session.priorities())
        self.assertFalse(stats.get(done).priority)
        self.assertIn(kept, session.priorities())
        self.assertTrue(stats.get(kept).priority)

        engine._clear_saved_priorities(session)
        self.assertEqual(session.priorities(), set())
        self.assertEqual(stats.priorities(), set())
        loaded = FileStats(path)
        loaded.load()
        self.assertEqual(loaded.priorities(), set())
