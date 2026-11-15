"""One stored slice commits its chunk hashes once, after the upsert.

The embed of the next slice overlaps that store. A further embed waits.
A crash before the commit leaves the hashes unseen.
"""

import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LONG_COUNT = 200
SHORT_COUNT = 5


def long_text(index: int) -> str:
    """Long enough to hash, short enough to stay one chunk."""
    return f"procedure {index:04d} " + ("catheter detail " * 30)


def short_text(index: int) -> str:
    return f"short {index}"


class _Install:
    dimension = 1
    embed_model = "nomic-embed-text:v1.5"
    document_prefix = "search_document: "


class _Book:
    def __init__(self) -> None:
        self.updates: list[tuple[int, int]] = []

    def update(self, _source, **fields) -> None:
        self.updates.append((int(fields.get("articles") or 0), int(fields.get("chunks") or 0)))


class _Session:
    def __init__(self, book: _Book) -> None:
        self._book = book

    def get_book(self) -> _Book:
        return self._book

    def view(self) -> dict:
        return {}


def _article(source: str, index: int, payloads: list[str]):
    from nomad_turbo.packing import PreparedArticle, SectionDraft

    return PreparedArticle(
        article={"index": index, "path": f"A/{index}", "title": f"title {index}"},
        sections=[
            SectionDraft(
                meta={
                    "source": source,
                    "article_path": f"A/{index}",
                    "section_title": "body",
                    "full_title": f"title {index}",
                    "article_title": f"title {index}",
                    "active": True,
                },
                payloads=list(payloads),
            )
        ],
        inputs=["search_document: " + payload for payload in payloads],
        section_for_input=[0] * len(payloads),
    )


def _slice_pairs(source: str):
    long_texts = [long_text(index) for index in range(LONG_COUNT)]
    short_texts = [short_text(index) for index in range(SHORT_COUNT)]
    first = long_texts[:80]
    second = long_texts[80:160] + short_texts
    third = long_texts[160:]
    pairs = [
        (_article(source, 1, first), [[0.1] for _ in first]),
        (_article(source, 2, second), [[0.1] for _ in second]),
        (_article(source, 3, third), [[0.1] for _ in third]),
    ]
    return pairs, long_texts, short_texts


def run_shipped_store(ledger_path: str) -> None:
    """Call the real post-embed store. Used under strace and in-process."""
    from nomad_turbo.engine import FileWork, _read_checkpoint, _store_ready
    from nomad_turbo.passes import ChunkLedger
    import nomad_turbo.engine as engine

    source = "/app/storage/zim/ledger-slice-probe.zim"
    pairs, long_texts, short_texts = _slice_pairs(source)
    upserted: list[str] = []

    def fake_upsert(_install, points):
        upserted.extend(str((point.get("payload") or {}).get("text") or "") for point in points)

    original = engine.upsert
    engine.upsert = fake_upsert
    ledger = ChunkLedger(Path(ledger_path))
    work = FileWork(host=Path("ledger-slice-probe.zim"), kind="zim", source=source)
    work.ledger = ledger
    try:
        _store_ready(_Install(), work, set(), None, pairs, None)
    finally:
        engine.upsert = original
        ledger.close()
    reopened = ChunkLedger(Path(ledger_path))
    try:
        missing = [text for text in long_texts if not reopened.seen(text)]
        present_short = [text for text in short_texts if reopened.seen(text)]
    finally:
        reopened.close()
    saved = _read_checkpoint(source)
    if missing or present_short or len(upserted) != LONG_COUNT + SHORT_COUNT:
        raise SystemExit(
            f"store probe failed missing={len(missing)} short={len(present_short)} "
            f"upserted={len(upserted)} checkpoint={saved}"
        )
    if not saved or int(saved.get("articles") or 0) != 3:
        raise SystemExit(f"checkpoint was not the last article: {saved}")
    print(f"LONG {LONG_COUNT}", flush=True)


class LedgerStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._old_state = os.environ.get("NOMAD_TURBO_STATE")
        os.environ["NOMAD_TURBO_STATE"] = self._tmp.name

        def restore() -> None:
            if self._old_state is None:
                os.environ.pop("NOMAD_TURBO_STATE", None)
            else:
                os.environ["NOMAD_TURBO_STATE"] = self._old_state

        self.addCleanup(restore)

    def _use_upsert(self, fn):
        import nomad_turbo.engine as engine

        original = engine.upsert
        engine.upsert = fn

        def restore_upsert() -> None:
            engine.upsert = original

        self.addCleanup(restore_upsert)

    def _use_checkpoint_spy(self, checkpoints: list[int]):
        import nomad_turbo.engine as engine

        original = engine._write_checkpoint

        def spy(work):
            checkpoints.append(int(work.articles))
            original(work)

        engine._write_checkpoint = spy

        def restore_checkpoint() -> None:
            engine._write_checkpoint = original

        self.addCleanup(restore_checkpoint)

    def _source(self, name: str) -> str:
        return f"/app/storage/zim/{name}.zim"

    def test_one_slice_commits_hashes_once_after_the_upsert_and_a_later_prepare_sees_them(self):
        from nomad_turbo.engine import FileWork, _prepare_article, _read_checkpoint, _store_ready
        from nomad_turbo.passes import ChunkLedger

        source = self._source("one-commit")
        pairs, long_texts, short_texts = _slice_pairs(source)
        path = Path(self._tmp.name) / "one-commit.sqlite"
        upserted: list[str] = []
        checkpoints: list[int] = []
        gate = {"commits": 0, "checked": False, "last": False, "first": False, "other": None, "inputs": None, "error": None}

        def fake_upsert(_install, points):
            upserted.extend(str((point.get("payload") or {}).get("text") or "") for point in points)

        self._use_upsert(fake_upsert)
        self._use_checkpoint_spy(checkpoints)
        ledger = ChunkLedger(path)
        work = FileWork(host=Path("one-commit.zim"), kind="zim", source=source)
        work.ledger = ledger
        seen: set[str] = set()
        status_writes: list[int] = []
        book = _Book()

        def trace(statement: str) -> None:
            if str(statement).strip().upper() != "COMMIT":
                return
            gate["commits"] += 1
            if gate["checked"]:
                return
            gate["checked"] = True
            try:
                gate["upserted"] = len(upserted)
                gate["first"] = ledger.seen(long_texts[0])
                gate["last"] = ledger.seen(long_texts[-1])
                other = sqlite3.connect(path, timeout=1)
                try:
                    gate["other"] = other.execute("select count(*) from chunks").fetchone()[0]
                finally:
                    other.close()
                later = FileWork(host=Path("one-commit.zim"), kind="zim", source=source)
                later.ledger = ledger
                duplicate = {
                    "index": 50,
                    "path": "later-copy",
                    "title": "later",
                    "text": long_texts[0],
                    "sections": [{"heading": "", "text": long_texts[0], "level": 2}],
                }
                prepared = _prepare_article(_Install(), later, duplicate, set())
                gate["inputs"] = list(prepared.inputs)
            except Exception as exc:
                gate["error"] = exc

        import nomad_turbo.engine as engine

        original_status = engine._status

        def spy_status(**fields) -> None:
            status_writes.append(int(fields.get("chunks") or 0))
            original_status(**fields)

        engine._status = spy_status
        self.addCleanup(lambda: setattr(engine, "_status", original_status))
        ledger._db.set_trace_callback(trace)
        try:
            _store_ready(_Install(), work, seen, _Session(book), pairs, None)
        finally:
            ledger._db.set_trace_callback(None)
            ledger.close()
        self.assertIsNone(gate["error"])
        self.assertEqual(gate["commits"], 1)
        self.assertEqual(gate["upserted"], LONG_COUNT + SHORT_COUNT)
        self.assertTrue(gate["first"])
        self.assertTrue(gate["last"])
        self.assertEqual(gate["other"], 0)
        self.assertEqual(gate["inputs"], [])
        self.assertEqual(checkpoints, [3])
        self.assertEqual(sorted(seen), ["A/1", "A/2", "A/3"])
        self.assertEqual(work.articles, 3)
        self.assertEqual(work.path, "A/3")
        self.assertEqual(status_writes, [LONG_COUNT + SHORT_COUNT])
        self.assertEqual(book.updates, [(3, LONG_COUNT + SHORT_COUNT)])
        self.assertCountEqual(upserted, long_texts + short_texts)
        reopened = ChunkLedger(path)
        try:
            self.assertTrue(all(reopened.seen(text) for text in long_texts))
            self.assertFalse(any(reopened.seen(text) for text in short_texts))
        finally:
            reopened.close()
        saved = _read_checkpoint(source)
        self.assertIsNotNone(saved)
        self.assertEqual(saved["articles"], 3)
        self.assertEqual(saved["chunks"], LONG_COUNT + SHORT_COUNT)

    def test_a_failed_upsert_leaves_those_hashes_unseen(self):
        from nomad_turbo.engine import FileWork, _store_ready
        from nomad_turbo.passes import ChunkLedger

        source = self._source("failed-upsert")
        pairs, long_texts, short_texts = _slice_pairs(source)
        path = Path(self._tmp.name) / "failed.sqlite"
        asked: list[str] = []
        checkpoints: list[int] = []

        def fake_upsert(_install, points):
            asked.extend(str((point.get("payload") or {}).get("text") or "") for point in points)
            raise RuntimeError("upsert failed")

        self._use_upsert(fake_upsert)
        self._use_checkpoint_spy(checkpoints)
        ledger = ChunkLedger(path)
        work = FileWork(host=Path("failed-upsert.zim"), kind="zim", source=source)
        work.ledger = ledger
        try:
            with self.assertRaises(RuntimeError):
                _store_ready(_Install(), work, set(), None, pairs, None)
        finally:
            ledger.close()
        self.assertCountEqual(asked, long_texts + short_texts)
        self.assertEqual(checkpoints, [])
        reopened = ChunkLedger(path)
        try:
            self.assertFalse(any(reopened.seen(text) for text in long_texts))
            self.assertFalse(any(reopened.seen(text) for text in short_texts))
        finally:
            reopened.close()

    def test_the_next_upsert_waits_until_the_first_hashes_survive_a_reopen(self):
        from nomad_turbo.engine import FileWork, _UpsertPipe, _store_ready
        from nomad_turbo.passes import ChunkLedger

        source = self._source("one-flight")
        path = Path(self._tmp.name) / "flight.sqlite"
        first_long = [long_text(index) for index in range(4)]
        first_short = [short_text(0)]
        second_long = [long_text(index) for index in range(100, 103)]
        first_pairs = [
            (_article(source, 1, first_long[:2]), [[0.1], [0.1]]),
            (_article(source, 2, first_long[2:] + first_short), [[0.1], [0.1], [0.1]]),
        ]
        second_pairs = [
            (_article(source, 3, second_long), [[0.1], [0.1], [0.1]]),
        ]
        checkpoints: list[int] = []
        flight = {"open": 0, "max": 0}
        seen_on_second = {}

        def fake_upsert(_install, points):
            flight["open"] += 1
            flight["max"] = max(flight["max"], flight["open"])
            try:
                texts = [str((point.get("payload") or {}).get("text") or "") for point in points]
                if len(texts) == len(second_long):
                    other = ChunkLedger(path)
                    try:
                        seen_on_second["first"] = [other.seen(text) for text in first_long]
                        seen_on_second["short"] = other.seen(first_short[0])
                        seen_on_second["second"] = [other.seen(text) for text in second_long]
                    finally:
                        other.close()
                    seen_on_second["checkpoints"] = list(checkpoints)
            finally:
                flight["open"] -= 1

        self._use_upsert(fake_upsert)
        self._use_checkpoint_spy(checkpoints)
        ledger = ChunkLedger(path)
        work = FileWork(host=Path("one-flight.zim"), kind="zim", source=source)
        work.ledger = ledger
        pipe = _UpsertPipe()
        try:
            pipe.submit(lambda batch=first_pairs: _store_ready(_Install(), work, set(), None, batch, None))
            pipe.submit(lambda batch=second_pairs: _store_ready(_Install(), work, set(), None, batch, None))
            pipe.retire()
        finally:
            pipe.close()
            ledger.close()
        self.assertEqual(flight["max"], 1)
        self.assertEqual(seen_on_second["first"], [True, True, True, True])
        self.assertFalse(seen_on_second["short"])
        self.assertEqual(seen_on_second["second"], [False, False, False])
        self.assertEqual(seen_on_second["checkpoints"], [2])
        self.assertEqual(checkpoints, [2, 3])

    def test_the_file_row_advances_before_the_store_returns_and_one_embed_overlaps_it(self):
        from nomad_turbo.engine import EMBED_SLICE, FileWork, _UpsertPipe, _embed_width
        from nomad_turbo.packing import ArticleFeed

        source = self._source("live-row")
        width = EMBED_SLICE * 2 + 1
        feed = ArticleFeed()
        for index in range(width):
            feed.add(_article(source, index + 1, [f"chunk {index}"]))
        release = threading.Event()
        entered = threading.Event()
        second_published = threading.Event()
        calls: list[int] = []
        published: list[tuple] = []
        upserts: list[int] = []

        def fake_embed(_url, _model, texts):
            calls.append(len(texts))
            if len(calls) == 2:
                self.assertTrue(entered.wait(30))
                self.assertFalse(release.is_set())
            if len(calls) > 2:
                self.assertTrue(release.is_set())
            return [[0.1] for _text in texts]

        def fake_upsert(_install, points):
            upserts.append(len(points))
            if len(upserts) == 1:
                entered.set()
                self.assertTrue(release.wait(30))

        class Book:
            def note_live(self, source, chunks, articles, now=None):
                published.append((chunks, articles, release.is_set()))
                if chunks >= EMBED_SLICE * 2 and articles >= EMBED_SLICE * 2:
                    second_published.set()

            def update(self, *args, **kwargs):
                return None

        class Session:
            def __init__(self, book):
                self._book = book

            def get_book(self):
                return self._book

            def view(self):
                return {}

        self._use_upsert(fake_upsert)
        work = FileWork(host=Path("live-row.zim"), kind="zim", source=source)
        pipe = _UpsertPipe()
        error = []

        def run():
            try:
                with unittest.mock.patch("nomad_turbo.engine.embed", fake_embed), unittest.mock.patch(
                    "nomad_turbo.engine._wait_to_write", return_value=True
                ), unittest.mock.patch("nomad_turbo.ocr_budget.note_embed"), unittest.mock.patch(
                    "nomad_turbo.ocr_budget.note_zim_embed_call"
                ):
                    try:
                        _embed_width(
                            _Install(),
                            type("Pool", (), {"urls": ["http://embed"]})(),
                            work,
                            Session(Book()),
                            feed,
                            pipe,
                            set(),
                            threading.Event(),
                            None,
                            None,
                            width,
                        )
                    finally:
                        release.set()
                        pipe.close()
            except Exception as exc:
                error.append(exc)
                release.set()

        worker = threading.Thread(target=run)
        worker.start()
        try:
            self.assertTrue(second_published.wait(30))
            self.assertFalse(release.is_set())
            self.assertEqual(calls, [EMBED_SLICE, EMBED_SLICE])
            self.assertEqual(upserts, [EMBED_SLICE])
            self.assertEqual(published[0], (EMBED_SLICE, EMBED_SLICE, False))
            self.assertEqual(published[1], (EMBED_SLICE * 2, EMBED_SLICE * 2, False))
            self.assertEqual(work.live_chunks, EMBED_SLICE * 2)
            self.assertEqual(work.live_articles, EMBED_SLICE * 2)
            release.set()
            worker.join(60)
        finally:
            release.set()
            worker.join(60)
        self.assertFalse(worker.is_alive())
        self.assertEqual(error, [])
        self.assertEqual(calls, [EMBED_SLICE, EMBED_SLICE, 1])
        self.assertEqual(upserts, [EMBED_SLICE, EMBED_SLICE, 1])
        self.assertEqual(work.chunks, width)
        self.assertEqual(work.articles, width)

    def test_the_shipped_store_syncs_the_ledger_under_one_tenth_of_the_long_chunks(self):
        if shutil.which("strace") is None:
            self._sync_by_time()
            return
        state = Path(self._tmp.name) / "probe-state"
        ledger = Path(self._tmp.name) / "probe.sqlite"
        trace = Path(self._tmp.name) / "strace.txt"
        state.mkdir()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["NOMAD_TURBO_STATE"] = str(state)
        code = (
            "import os, sys\n"
            "os.environ['NOMAD_TURBO_STATE'] = sys.argv[1]\n"
            "from tests.test_ledger_store import run_shipped_store\n"
            "run_shipped_store(sys.argv[2])\n"
        )
        proc = subprocess.run(
            [
                "strace",
                "-f",
                "-e",
                "trace=fsync,fdatasync",
                "-o",
                str(trace),
                sys.executable,
                "-c",
                code,
                str(state),
                str(ledger),
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:] + proc.stdout[-2000:])
        found = re.search(r"LONG (\d+)", proc.stdout)
        self.assertIsNotNone(found)
        long_count = int(found.group(1))
        self.assertGreaterEqual(long_count, 100)
        traced = trace.read_text(errors="replace")
        syncs = len(re.findall(r"(?:fdatasync|fsync)\(", traced))
        self.assertGreater(syncs, 0, traced[-500:])
        self.assertLess(syncs, long_count / 10)
        self._write_sync_evidence(
            f"method strace\nsyncs {syncs}\nlong_chunks {long_count}\n"
            f"tenth {long_count / 10}\nstdout {proc.stdout.strip()}\n"
        )

    def _sync_by_time(self):
        from nomad_turbo.passes import chunk_key

        state = Path(self._tmp.name) / "time-state"
        state.mkdir()
        os.environ["NOMAD_TURBO_STATE"] = str(state)
        texts = [long_text(index) for index in range(LONG_COUNT)]
        keys = [chunk_key(text) for text in texts]
        baseline_path = Path(self._tmp.name) / "baseline.sqlite"
        started = time.perf_counter()
        base = sqlite3.connect(baseline_path, isolation_level=None)
        base.execute("create table if not exists chunks (hash text primary key)")
        for key in keys:
            base.execute("insert or ignore into chunks (hash) values (?)", (key,))
            base.commit()
        base.close()
        baseline = time.perf_counter() - started
        shipped_path = Path(self._tmp.name) / "shipped.sqlite"
        started = time.perf_counter()
        run_shipped_store(str(shipped_path))
        shipped = time.perf_counter() - started
        self.assertGreater(baseline, 0)
        self.assertLess(shipped, baseline / 10)
        self._write_sync_evidence(
            f"method time\nshipped_seconds {shipped}\nbaseline_seconds {baseline}\n"
            f"long_chunks {LONG_COUNT}\n"
        )

    def _write_sync_evidence(self, text: str) -> None:
        raw = os.environ.get("NOMAD_GOAL_SCRATCH")
        if not raw:
            return
        folder = Path(raw)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "ledger-sync.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
