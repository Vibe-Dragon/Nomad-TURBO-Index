"""Deferred documents inside a ZIM: the list, the bar, and which one is next."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nomad_turbo.discover import state_dir
from nomad_turbo.engine import FileWork, _seal_zim_documents, _write_checkpoint
from nomad_turbo.schedule import (
    COL_ARTICLES,
    COL_PROGRESS,
    COL_SOURCE,
    COL_STATUS,
    COL_TYPE,
    FOLDER,
    ZIM_ROOT,
    folder_store_row,
    library_tree,
)
from nomad_turbo.session import Session
from nomad_turbo.zimdocs import (
    INTERNAL_KEY,
    INTERNAL_NAME,
    catalog_entries,
    doc_zim_key,
    document_file_row,
    document_percent,
    load_document_groups,
    parent_source,
    pick_document,
    source_has_pending_documents,
    virtual_source,
    write_running_document,
)


class _Item:
    def __init__(self, source, name, state="pending"):
        self.source = source
        self.name = name
        self.state = state
        self.origin = "zim"
        self.size_bytes = 10


def _group(root: Path, source: str, *, closed: bool, text=(), ocr=(), done=(), skipped=()):
    import hashlib

    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    folder = root / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{digest}.json").write_text(
        json.dumps(
            {
                "source": source,
                "done": closed,
                "pass": "done" if closed else "html",
                "documents_closed": closed,
                "pages_done": closed,
                "doc_total": len(tuple(text) + tuple(ocr)),
                "doc_done": list(done),
                "doc_skipped": list(skipped),
            }
        ),
        encoding="utf-8",
    )
    spill = root / "spill"
    spill.mkdir(parents=True, exist_ok=True)
    (spill / f"{digest}-text.jsonl").write_text(
        "".join(json.dumps(path) + "\n" for path in text),
        encoding="utf-8",
    )
    (spill / f"{digest}-ocr.jsonl").write_text(
        "".join(json.dumps(path) + "\n" for path in ocr),
        encoding="utf-8",
    )


class DocumentListTests(unittest.TestCase):
    def test_paths_stay_unique_and_a_pdf_wins(self):
        entries = catalog_entries(
            ["shelf/b.epub", "shelf/a.docx", "scan.pdf", ""],
            ["scan.pdf", "notes.pdf"],
        )
        self.assertEqual(
            entries,
            (("notes.pdf", "ocr"), ("scan.pdf", "ocr"), ("shelf/a.docx", "text"), ("shelf/b.epub", "text")),
        )

    def test_the_bar_waits_until_the_list_is_closed(self):
        self.assertIsNone(document_percent(False, 4, 1, 0))
        self.assertEqual(document_percent(True, 4, 0, 0), 0)
        self.assertEqual(document_percent(True, 4, 1, 1), 50)
        self.assertEqual(document_percent(True, 0, 0, 0), None)

    def test_priority_and_index_now_pick_a_closed_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = "/app/storage/zim/wiki.zim"
            law = "/app/storage/zim/law.zim"
            fresh = "/app/storage/zim/fresh.zim"
            _group(root, wiki, closed=True, text=["b.epub"], ocr=["a.pdf", "c.pdf"])
            _group(root, law, closed=True, ocr=["z.pdf"])
            _group(root, fresh, closed=False, ocr=["early.pdf"])
            groups = load_document_groups(root)
            first = pick_document(groups, (), None)
            self.assertEqual(first, (law, "z.pdf", "ocr"))
            marked = pick_document(groups, {wiki}, None)
            self.assertEqual(marked[0], wiki)
            self.assertEqual(marked[1], "a.pdf")
            now = pick_document(groups, {wiki}, law)
            self.assertEqual(now[:2], (law, "z.pdf"))
            self.assertIsNone(pick_document([group for group in groups if not group.closed], (), None))
            self.assertFalse(source_has_pending_documents(root, fresh))
            self.assertTrue(source_has_pending_documents(root, wiki))

    def test_the_branch_is_last_and_a_finished_zim_drops_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = "/app/storage/zim/wiki.zim"
            done = "/app/storage/zim/done.zim"
            _group(root, wiki, closed=True, ocr=["shelf/scan.pdf"], text=["shelf/book.epub"])
            _group(root, done, closed=True, ocr=["old.pdf"], done=["old.pdf"])
            groups = load_document_groups(root)
            self.assertEqual([group.source for group in groups if group.source == done], [done])
            items = [_Item("/app/storage/zim/wiki.zim", "wiki.zim"), _Item("/app/storage/kb_uploads/note.md", "note.md", "pending")]
            # kb file is an upload by name. Put a real zim and an upload path.
            items = [
                _Item("/app/storage/zim/wiki.zim", "wiki.zim"),
                _Item("/app/storage/kb_uploads/note.md", "note.md"),
            ]
            tree = library_tree(items, documents=groups)
            self.assertEqual(tree[0].key, "kb_uploads")
            zim = tree[1]
            self.assertEqual(zim.key, ZIM_ROOT)
            self.assertEqual(zim.folders[-1].key, INTERNAL_KEY)
            self.assertEqual(zim.folders[-1].name, INTERNAL_NAME)
            branch = zim.folders[-1]
            self.assertEqual([folder.name for folder in branch.folders], ["wiki.zim"])
            wiki_row = branch.folders[0]
            self.assertEqual(wiki_row.status_text, "documents")
            self.assertEqual(wiki_row.articles_text, "0 of 2")
            self.assertEqual(wiki_row.progress, 0)
            self.assertTrue(wiki_row.show_bar)
            self.assertEqual(wiki_row.doc_source, wiki)
            shelf = wiki_row.folders[0]
            self.assertEqual(shelf.name, "shelf")
            self.assertEqual(shelf.files[0], virtual_source(wiki, "shelf/book.epub"))
            stored = folder_store_row(wiki_row)
            self.assertEqual(len(stored), 20)
            self.assertEqual(stored[COL_SOURCE], doc_zim_key(wiki))
            self.assertEqual(stored[COL_TYPE], "zim")
            self.assertEqual(stored[COL_STATUS], "documents")
            self.assertEqual(stored[COL_ARTICLES], "0 of 2")
            self.assertEqual(stored[COL_PROGRESS], 0)
            plain = folder_store_row(branch)
            self.assertEqual(plain[COL_TYPE], FOLDER)
            row = document_file_row(wiki, "shelf/scan.pdf", "ocr", "waiting")
            self.assertEqual(len(row), 20)
            self.assertEqual(parent_source(row[COL_SOURCE]), wiki)
            hidden = library_tree(items, hide_finished=True, documents=groups)
            self.assertEqual(hidden[1].folders[-1].name, INTERNAL_NAME)

    def test_an_open_list_shows_a_count_and_no_shrinking_percent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = "/app/storage/zim/open.zim"
            _group(root, source, closed=False, ocr=["a.pdf", "b.pdf"])
            groups = load_document_groups(root)
            self.assertEqual(len(groups), 1)
            self.assertFalse(groups[0].closed)
            self.assertIsNone(pick_document(groups, (), None))
            tree = library_tree([], documents=groups)
            self.assertEqual(tree[0].key, ZIM_ROOT)
            row = tree[0].folders[0].folders[0]
            self.assertEqual(row.status_text, "finding")
            self.assertEqual(row.articles_text, "2 found")
            self.assertFalse(row.show_bar)
            self.assertEqual(row.progress, 0)
            write_running_document(root, source, "a.pdf")
            running = load_document_groups(root)
            self.assertEqual(running[0].running_path, "a.pdf")
            self.assertEqual(running[0].entries[0][2], "running")

    def test_no_ledger_means_no_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_document_groups(tmp), ())
            self.assertFalse(source_has_pending_documents(tmp, "/app/storage/zim/done.zim"))

    def test_sealing_pages_keeps_priority_while_documents_remain(self):
        previous = os.environ.get("NOMAD_TURBO_STATE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["NOMAD_TURBO_STATE"] = tmp
            try:
                source = "/app/storage/zim/medicine.zim"
                work = FileWork(host=Path(tmp) / "medicine.zim", kind="zim", source=source)
                work.zim_pass = "html"
                _write_checkpoint(work)
                spill = state_dir() / "spill"
                spill.mkdir(parents=True, exist_ok=True)
                digest = next(state_dir().joinpath("checkpoints").glob("*.json")).stem
                (spill / f"{digest}-text.jsonl").write_text(json.dumps("book.epub") + "\n", encoding="utf-8")
                (spill / f"{digest}-ocr.jsonl").write_text(json.dumps("scan.pdf") + "\n", encoding="utf-8")
                session = Session(1)
                session.set_priority(source, True)
                session.request_index_now(source)
                with mock.patch("nomad_turbo.engine._record_closed_file"):
                    _seal_zim_documents(None, work, session)
                self.assertTrue(work.pages_done)
                self.assertTrue(work.documents_closed)
                self.assertTrue(work.done)
                self.assertEqual(work.zim_pass, "done")
                self.assertEqual(work.doc_total, 2)
                self.assertEqual(work.ocr_pending, [])
                self.assertIn(source, session.priorities())
                saved = json.loads(next(state_dir().joinpath("checkpoints").glob("*.json")).read_text())
                self.assertTrue(saved["pages_done"])
                self.assertTrue(saved["documents_closed"])
                self.assertTrue(saved["done"])
                self.assertEqual(saved["doc_total"], 2)
                groups = load_document_groups(tmp)
                self.assertEqual(groups[0].pending[0][0], "book.epub")
            finally:
                if previous is None:
                    os.environ.pop("NOMAD_TURBO_STATE", None)
                else:
                    os.environ["NOMAD_TURBO_STATE"] = previous

    def test_an_indexed_zim_with_documents_keeps_index_now(self):
        from nomad_turbo.schedule import engine_switch_reason

        previous = os.environ.get("NOMAD_TURBO_STATE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["NOMAD_TURBO_STATE"] = tmp
            try:
                source = "/app/storage/zim/done.zim"
                _group(Path(tmp), source, closed=True, ocr=["scan.pdf"])
                session = Session(1)
                session.request_index_now(source)
                self.assertIsNone(engine_switch_reason(session, "/app/storage/zim/other.zim", None, 0, 0, []))
                self.assertEqual(session.index_now(), source)
                empty = Session(1)
                empty.request_index_now("/app/storage/zim/missing.zim")
                self.assertIsNone(engine_switch_reason(empty, "/app/storage/zim/other.zim", None, 0, 0, []))
                self.assertIsNone(empty.index_now())
            finally:
                if previous is None:
                    os.environ.pop("NOMAD_TURBO_STATE", None)
                else:
                    os.environ["NOMAD_TURBO_STATE"] = previous

    def test_html_lists_documents_instead_of_queueing_them(self):
        from nomad_turbo.engine import _index_zim

        previous = os.environ.get("NOMAD_TURBO_STATE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["NOMAD_TURBO_STATE"] = tmp
            try:
                source = "/app/storage/zim/medicine.zim"
                host = Path(tmp) / "medicine.zim"
                host.write_bytes(b"zim")
                work = FileWork(host=host, kind="zim", source=source)
                session = Session(1)
                session.set_zim_documents(True)
                proc = mock.Mock()
                proc.stdout = ['{"type": "done"}\n']
                proc.stderr = []
                proc.wait.return_value = 0
                proc.returncode = 0
                proc.poll.return_value = 0
                with (
                    mock.patch("nomad_turbo.engine.source_active", return_value=True),
                    mock.patch("nomad_turbo.engine.count_source", return_value=0),
                    mock.patch("nomad_turbo.engine.existing_paths", return_value=set()),
                    mock.patch("nomad_turbo.engine._ocr_container", return_value={}),
                    mock.patch("nomad_turbo.engine._record_closed_file"),
                    mock.patch("nomad_turbo.engine.release_extract"),
                    mock.patch("nomad_turbo.engine.start_extract", return_value=proc) as started,
                ):
                    _index_zim(mock.Mock(), None, work, mock.Mock(is_set=lambda: False), session)
                kwargs = started.call_args.kwargs
                self.assertTrue(kwargs["defer_documents"])
                self.assertTrue(kwargs["zim_documents"])
                self.assertFalse(kwargs["ocr_queue"])
                self.assertFalse(kwargs["ocr_ahead"])
                self.assertTrue(work.documents_closed)
                self.assertTrue(work.done)
            finally:
                if previous is None:
                    os.environ.pop("NOMAD_TURBO_STATE", None)
                else:
                    os.environ["NOMAD_TURBO_STATE"] = previous


if __name__ == "__main__":
    unittest.main()
