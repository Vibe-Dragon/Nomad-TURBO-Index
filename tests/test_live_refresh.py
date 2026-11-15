"""The window timer paints a published reading and does not wait on the workers."""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from nomad_turbo.attention import activity_status, qdrant_level
from nomad_turbo.catalog import CatalogRow
from nomad_turbo.discover import Install
from nomad_turbo.host import (
    DiskMeter,
    HostPoint,
    cpu_caption,
    disk_line,
    disk_series,
    gpu_caption,
    ram_caption,
    vram_caption,
)
from nomad_turbo.live import ButtonFacts, LiveBoard, Reading, Workload, control_senses, hold_interpreter
from nomad_turbo.progress import (
    OcrPace,
    ProgressBook,
    format_run_stats,
    log_line_visible,
    ocr_note_matches,
    overlay_ocr_columns,
    server_status_text,
)
from nomad_turbo.queue_view import build_queue, queue_caption, shown_slots
from nomad_turbo.repair import RepairBoard, RepairRow, RepairSnap, format_repair_current, repair_store_values
from nomad_turbo.schedule import folder_store_row, index_store_row, library_tree
from nomad_turbo.segment_guard import pause_caption, track_optimize_pause
from nomad_turbo.session import Session
from nomad_turbo.tune import content_kind, tune_advice

def _use_local_display() -> None:
    """GTK reads DISPLAY when it is imported. A local X socket is enough."""
    if os.environ.get("DISPLAY"):
        return
    for number in ("0", "1"):
        if Path(f"/tmp/.X11-unix/X{number}").exists():
            os.environ["DISPLAY"] = f":{number}"
            return


_use_local_display()

NOTES = "/opt/storage/kb_uploads/notes.pdf"
WAIT = "/opt/storage/kb_uploads/wait.pdf"
WIKI = "/opt/storage/zim/wiki.zim"


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


def _book() -> ProgressBook:
    return ProgressBook(
        [
            CatalogRow(NOTES, "notes.pdf", "upload", "pdf", 1200, "pending"),
            CatalogRow(WAIT, "wait.pdf", "upload", "pdf", 800, "pending"),
            CatalogRow(WIKI, "wiki.zim", "zim", "zim", 5_000_000, "pending", walk_index=2),
        ]
    )


def _repair(percent: int, overall: str, stats: str, note: str) -> RepairSnap:
    row = RepairRow(
        source=WIKI,
        name="wiki.zim",
        problem="Count mismatch",
        points="12 points",
        percent=percent,
        pulse=1,
        status="Deleting",
        elapsed="3s",
        detail="Removing orphan points",
        color="#111827",
        points_value=12,
        elapsed_seconds=3,
    )
    return RepairSnap(
        phase="repairing",
        note=note,
        mode="quick",
        rows=(row,),
        fraction=percent / 100.0,
        overall=overall,
        busy=True,
        stats=stats,
    )


def _point(cpu: float, read: float) -> HostPoint:
    return HostPoint(
        cpu=cpu,
        cpu_temp=47,
        ram_percent=30,
        ram_used=2 * 1024**3,
        ram_total=8 * 1024**3,
        gpu_present=True,
        gpu=22,
        gpu_temp=61,
        vram_percent=8,
        vram_used=800,
        vram_total=16311,
        disks=(
            DiskMeter("Library", used=1_000_000_000, total=4_000_000_000_000, read_mbs=read, write_mbs=4.5),
            DiskMeter("qdrant", used=2_000_000_000, total=4_000_000_000_000, read_mbs=1.5, write_mbs=9.0),
        ),
    )


def _reading(book: ProgressBook, *, articles: int, chunks: int, moment: float, **fields) -> Reading:
    book.update(NOTES, articles=articles, total=20, chunks=chunks, state="running", now=moment)
    epoch, items = book.library_items()
    snap = book.pulse(now=moment)
    return Reading(
        view=fields["view"],
        snap=snap,
        priorities=frozenset({WIKI}),
        activity=fields["view"]["note"],
        ocr_activity=fields["ocr"],
        lane=None,
        embedding=None,
        embed_order=None,
        ocr_order=None,
        order="library",
        zim_dates={},
        open_files=tuple(book.open_snaps(now=moment)),
        queue_open=True,
        library_items=tuple(items),
        structure_epoch=epoch,
        doc_sig=(False,),
        doc_groups=(),
        hide_finished=False,
        row_snaps=book.files_for([NOTES, WAIT, WIKI], now=moment),
        repair=fields["repair"],
        qdrant_text=fields["qdrant"],
        qdrant_ready=True,
        collection_status="green",
        optimizer_error="",
        collection_missing=False,
        qdrant_unreadable=False,
        gpu=(0, 8000, 1000),
        host_points=fields["host"],
        host_on=True,
        log_lines=fields["log"],
        log_reset=False,
        scanning=False,
        elapsed=fields["elapsed"],
        embed_calls=((256, 8, 0.4, True, "html", 3000),),
        embed_call_count=4,
        ocr_workers=(2, 4),
        running_source=NOTES,
        tune_auto=False,
        optimize_live=fields["optimize"],
        facts=fields["facts"],
        painted_at=moment,
    )


def _file_row(item, reading, rate):
    values = index_store_row(item, priority=item.source in set(reading.priorities or ()))
    if getattr(item, "state", "") == "running" and ocr_note_matches(reading.ocr_activity or "", getattr(item, "name", "")):
        values = overlay_ocr_columns(values, reading.ocr_activity, rate)
    return values


def _store_rows(store) -> list[tuple]:
    rows = []
    for index in range(len(store)):
        rows.append(tuple(store[index]))
    return rows


def _root_rows(store) -> list[tuple]:
    rows = []
    iterator = store.get_iter_first()
    while iterator is not None:
        rows.append(tuple(store[iterator]))
        iterator = store.iter_next(iterator)
    return rows


def _command_text(command) -> str:
    if isinstance(command, (list, tuple)):
        return " ".join(str(part) for part in command)
    return str(command or "")


class StallRefreshTest(unittest.TestCase):
    def test_refresh_paints_a_reading_while_embed_ocr_and_the_interpreter_are_busy(self):
        _use_local_display()
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

        previous_state = os.environ.get("NOMAD_TURBO_STATE")
        previous_config = os.environ.get("NOMAD_TURBO_CONFIG")
        state = tempfile.TemporaryDirectory(prefix="nomad-turbo-live-")
        os.environ["NOMAD_TURBO_STATE"] = str(Path(state.name) / "state")
        os.environ["NOMAD_TURBO_CONFIG"] = str(Path(state.name) / "config.json")
        app = None
        host = None
        listener = None
        stop_net = threading.Event()
        release = threading.Event()
        holder = None
        ocr_mark = Path(state.name) / "ocr-mark"
        records: list[str] = []
        handler = logging.Handler()
        handler.emit = lambda record: records.append(logging.Formatter().format(record))  # type: ignore[method-assign]
        logging.getLogger("nomad_turbo").addHandler(handler)
        self._log_records = records
        try:
            from nomad_turbo.window import _MemoryLog, _Window

            book = _book()
            session = Session(55)
            session.bind_book(book)
            app = _Window(_install(state.name), session, _MemoryLog(), Path(state.name) / "session.log", Gtk, GLib)
            self.assertIsInstance(app.live, LiveBoard)
            self.assertIsNone(app._publisher)
            app.qdrant_ready = True
            app.rows_ready = True
            app.repair_board = RepairBoard()
            facts_running = ButtonFacts(
                thread_alive=True,
                rows_ready=True,
                qdrant_ready=True,
                scanning=False,
                measuring=False,
            )
            facts_idle = ButtonFacts(
                thread_alive=False,
                rows_ready=True,
                qdrant_ready=True,
                scanning=False,
                measuring=False,
            )
            first = _reading(
                book,
                articles=4,
                chunks=10,
                moment=1000.0,
                view={
                    "phase": "running",
                    "note": "Indexing the library. OCR is on notes.pdf.",
                    "error": "",
                    "optimize_hold": "segment",
                    "chunks": 32,
                    "active_slots": 1,
                },
                ocr="Reading notes.pdf with OCR. Page 3 of 12.",
                repair=_repair(40, "1 of 4 done · wiki.zim", "Rows listed: 1", "Repair is deleting orphan points."),
                qdrant="Qdrant is ready.",
                host=(_point(12, 10.0), _point(18, 20.0)),
                log=((1, logging.INFO, "library walk started"),),
                elapsed=125,
                optimize=("Segment check", "Points stay.", False),
                facts=facts_running,
            )
            second = _reading(
                book,
                articles=9,
                chunks=24,
                moment=1060.0,
                view={
                    "phase": "running",
                    "note": "Still indexing. The next batch is in.",
                    "error": "",
                    "optimize_hold": "segment",
                    "chunks": 48,
                    "active_slots": 1,
                },
                ocr="Reading notes.pdf with OCR. Page 5 of 12.",
                repair=_repair(70, "2 of 4 done · wiki.zim", "Rows listed: 1 · in progress: 1", "Repair moved to the next file."),
                qdrant="Qdrant is ready. Collection answered with 12000 points.",
                host=(_point(40, 33.0), _point(55, 44.0)),
                log=(
                    (1, logging.INFO, "library walk started"),
                    (2, logging.INFO, "embed call returned"),
                ),
                elapsed=200,
                optimize=("Segment check finished", "Points stay put.", False),
                facts=facts_idle,
            )
            app.on_queue()
            self.assertIsNotNone(app.queue_window)
            if app.host_panel.widget.get_parent() is None:
                app.host_pane.pack2(app.host_panel.widget, False, True)
            app.host_panel.widget.show_all()
            self.assertIsNone(app.host_panel._timer)

            bin_dir = Path(state.name) / "bin"
            bin_dir.mkdir()
            docker = bin_dir / "docker"
            docker.write_text(
                "#!/bin/sh\n"
                'mark="$NOMAD_TURBO_OCR_MARK"\n'
                'if [ -n "$mark" ]; then echo $$ > "$mark"; fi\n'
                "exec /bin/sleep 120\n",
                encoding="utf-8",
            )
            docker.chmod(0o755)
            host = Workload.install(
                {
                    "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
                    "NOMAD_TURBO_OCR_MARK": str(ocr_mark),
                    "NOMAD_TURBO_STATE": os.environ["NOMAD_TURBO_STATE"],
                    "NOMAD_TURBO_CONFIG": os.environ["NOMAD_TURBO_CONFIG"],
                    "NO_PROXY": "127.0.0.1,localhost",
                    "no_proxy": "127.0.0.1,localhost",
                    "http_proxy": "",
                    "https_proxy": "",
                    "HTTP_PROXY": "",
                    "HTTPS_PROXY": "",
                }
            )
            self.assertIsNotNone(host)
            self.assertTrue(host.proc.is_alive())
            listener = socket.socket()
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(20)
            port = listener.getsockname()[1]
            accepted = threading.Event()
            peer_gone = threading.Event()

            def serve() -> None:
                conn = None
                try:
                    conn, _addr = listener.accept()
                    accepted.set()
                    conn.settimeout(0.5)
                    while not stop_net.is_set():
                        try:
                            chunk = conn.recv(65536)
                        except socket.timeout:
                            continue
                        if not chunk:
                            peer_gone.set()
                            return
                except Exception:
                    return
                finally:
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass

            threading.Thread(target=serve, name="nomad-turbo-test-embed", daemon=True).start()
            host.submit("nomad_turbo.runtime:embed", (f"http://127.0.0.1:{port}", "nomic", ["search_document: hello"]))
            host.submit("nomad_turbo.runtime:read_visual", (app.install, "/tmp/not-a-real.pdf", "pdf"))
            gil_mark = Path(state.name) / "gil-started"
            host.submit("nomad_turbo.live:hold_interpreter", (45.0, str(gil_mark)))
            self.assertTrue(accepted.wait(30), "embed call never reached the hung server")
            gil_deadline = time.monotonic() + 30
            while time.monotonic() < gil_deadline and not gil_mark.is_file():
                time.sleep(0.05)
            self.assertTrue(gil_mark.is_file(), "interpreter stretch never started")
            deadline = time.monotonic() + 30
            ocr_pid = 0
            while time.monotonic() < deadline:
                if ocr_mark.is_file():
                    text = ocr_mark.read_text(encoding="utf-8").strip()
                    if text.isdigit():
                        ocr_pid = int(text)
                        try:
                            os.kill(ocr_pid, 0)
                            break
                        except OSError:
                            ocr_pid = 0
                time.sleep(0.05)
            self.assertGreater(ocr_pid, 0, "ocr docker never started")
            self.assertTrue(host.proc.is_alive())

            held = threading.Event()

            def hold_locks() -> None:
                book._lock.acquire()
                session._lock.acquire()
                app.repair_board._lock.acquire()
                held.set()
                release.wait(2.0)
                app.repair_board._lock.release()
                session._lock.release()
                book._lock.release()

            holder = threading.Thread(target=hold_locks, name="nomad-turbo-test-locks", daemon=True)
            holder.start()
            self.assertTrue(held.wait(2.0), "could not take the book, session, and repair locks")

            counts = {"run": 0, "check_output": 0, "popen": 0, "embed": 0, "ocr": 0, "gpu": 0, "scan": 0, "probe": 0}
            real_run = subprocess.run
            real_check = subprocess.check_output
            real_popen = subprocess.Popen

            def forbidden(command) -> bool:
                text = _command_text(command).lower()
                return "nvidia-smi" in text or "docker" in text

            def spy_run(command, *args, **kwargs):
                if forbidden(command):
                    counts["run"] += 1
                    return subprocess.CompletedProcess(command, 1, "", "")
                return real_run(command, *args, **kwargs)

            def spy_check(command, *args, **kwargs):
                if forbidden(command):
                    counts["check_output"] += 1
                    return ""
                return real_check(command, *args, **kwargs)

            def spy_popen(command, *args, **kwargs):
                if forbidden(command):
                    counts["popen"] += 1

                    class _Stopped:
                        returncode = 1
                        stdout = None
                        stderr = None
                        pid = 0

                        def wait(self, timeout=None):
                            return 1

                        def poll(self):
                            return 1

                        def kill(self):
                            return None

                        def terminate(self):
                            return None

                    return _Stopped()
                return real_popen(command, *args, **kwargs)

            def spy_embed(*_args, **_kwargs):
                counts["embed"] += 1
                return []

            def spy_ocr(*_args, **_kwargs):
                counts["ocr"] += 1
                return None

            def spy_gpu(*_args, **_kwargs):
                counts["gpu"] += 1
                return []

            def spy_scan(*_args, **_kwargs):
                counts["scan"] += 1
                return []

            def spy_probe(*_args, **_kwargs):
                counts["probe"] += 1
                raise AssertionError("probe")

            import nomad_turbo.qdrant_status as qdrant_mod
            import nomad_turbo.runtime as runtime_mod
            import nomad_turbo.window as window_mod

            from unittest.mock import patch

            pace = OcrPace()
            with (
                patch.object(subprocess, "run", spy_run),
                patch.object(subprocess, "check_output", spy_check),
                patch.object(subprocess, "Popen", spy_popen),
                patch.object(runtime_mod, "embed", spy_embed),
                patch.object(runtime_mod, "read_visual", spy_ocr),
                patch.object(runtime_mod, "gpu_rows", spy_gpu),
                patch.object(window_mod, "gpu_rows", spy_gpu),
                patch.object(window_mod, "scan_library", spy_scan),
                patch.object(qdrant_mod, "probe_qdrant", spy_probe),
            ):
                baseline = dict(counts)
                stored = app.live.publish(first)
                durations = [self._tick(app)]
                self._assert_reading(app, stored, pace)
                kept = self._figures(app)
                durations.append(self._tick(app))
                self.assertEqual(self._figures(app), kept)
                durations.append(self._tick(app))
                self.assertEqual(self._figures(app), kept)
                self.assertEqual(app._applied_seq, stored.seq)
                stored_next = app.live.publish(second)
                durations.append(self._tick(app))
                self._assert_reading(app, stored_next, pace)
                updated = self._figures(app)
                self.assertNotEqual(updated["status"], kept["status"])
                self.assertNotEqual(updated["stats"], kept["stats"])
                self.assertNotEqual(updated["qdrant"], kept["qdrant"])
                self.assertNotEqual(updated["tray"], kept["tray"])
                self.assertNotEqual(updated["repair_overall"], kept["repair_overall"])
                self.assertNotEqual(updated["cpu"], kept["cpu"])
                self.assertNotEqual(updated["log"], kept["log"])
                self.assertNotEqual(updated["optimize"], kept["optimize"])
                self.assertNotEqual(updated["buttons"], kept["buttons"])
                durations.append(self._tick(app))
                self.assertEqual(self._figures(app), updated)
                self.assertEqual(app._applied_seq, stored_next.seq)
                self.assertFalse(peer_gone.is_set(), "embed call returned during the refresh")
                self.assertTrue(host.proc.is_alive())
                os.kill(ocr_pid, 0)
                self.assertEqual(gil_mark.read_text(encoding="utf-8"), "started")
                self.assertEqual(counts, baseline)
                for elapsed in durations:
                    print(f"on_tick {elapsed * 1000:.1f} ms", flush=True)
                self.assertGreaterEqual(len(durations), 5)
        finally:
            release.set()
            stop_net.set()
            if holder is not None:
                holder.join(3)
            current = Workload.current()
            if current is not None:
                current.close()
            Workload._current = None
            if ocr_mark.is_file():
                text = ocr_mark.read_text(encoding="utf-8").strip()
                if text.isdigit():
                    try:
                        os.kill(int(text), 15)
                    except OSError:
                        pass
            if listener is not None:
                try:
                    listener.close()
                except Exception:
                    pass
            if app is not None:
                queue = getattr(app, "queue_window", None)
                if queue is not None:
                    queue.destroy()
                app.window.destroy()
            logging.getLogger("nomad_turbo").removeHandler(handler)
            if previous_state is None:
                os.environ.pop("NOMAD_TURBO_STATE", None)
            else:
                os.environ["NOMAD_TURBO_STATE"] = previous_state
            if previous_config is None:
                os.environ.pop("NOMAD_TURBO_CONFIG", None)
            else:
                os.environ["NOMAD_TURBO_CONFIG"] = previous_config
            state.cleanup()

    def _tick(self, app) -> float:
        started_at = time.perf_counter()
        app.on_tick()
        elapsed = time.perf_counter() - started_at
        self.assertLess(elapsed, 0.1, f"on_tick took {elapsed:.3f}s")
        if getattr(app, "_paint_failed", False):
            notes = "\n".join(getattr(self, "_log_records", [])[-8:])
            self.fail(f"window refresh raised:\n{notes}")
        return elapsed

    def _assert_reading(self, app, reading, pace: OcrPace) -> None:
        rate = pace.observe(reading.ocr_activity, now=reading.painted_at)
        if reading.painted_at == 1000.0:
            self.assertIsNone(rate)
        else:
            self.assertIsNotNone(rate)
            self.assertGreater(rate, 0)
        view = reading.view
        self.assertEqual(app.status.get_text(), view.get("error") or view.get("note"))
        self.assertEqual(app.overall.get_text(), reading.snap.overall_text)
        self.assertAlmostEqual(app.overall.get_fraction(), min(1.0, max(0.0, reading.snap.overall_fraction)))
        self.assertEqual(app.stats.get_text(), format_run_stats(view["phase"], reading.elapsed, reading.snap))
        slots = view.get("active_slots") if view.get("phase") in ("waiting", "running") else 0
        self.assertEqual(app.server_label.get_text(), server_status_text(slots or 0, reading.ocr_workers))
        free = max(0, int(reading.gpu[1]) - int(reading.gpu[2]))
        advice = tune_advice(
            app._embed_samples(reading.embed_calls),
            int(view.get("chunks", view.get("percent", 0)) or 0),
            auto=bool(reading.tune_auto),
            running=view.get("phase") in ("waiting", "running") and reading.facts.thread_alive,
            optimizing=bool(view.get("optimize_hold")),
            gpu_free_mib=free,
            hold=app._tune_hold,
            call_count=reading.embed_call_count,
            now=reading.painted_at,
            memory=app._pace_memory,
            kind=content_kind(reading.running_source) if reading.running_source else "",
        )
        self.assertEqual(app.tune_label.get_text(), advice.text)
        level = qdrant_level(
            reading.qdrant_text,
            ready=reading.qdrant_ready,
            collection_status=reading.collection_status,
            optimizer_error=reading.optimizer_error,
            collection_missing=reading.collection_missing,
            unreadable=reading.qdrant_unreadable,
        )
        facts = reading.facts
        repair_note = reading.repair.note if facts.repair_alive and reading.repair is not None else ""
        shown, _attention = activity_status(
            phase=view.get("phase") or "",
            note=view.get("note") or "",
            error=view.get("error") or "",
            optimize_hold=view.get("optimize_hold") or "",
            scanning=bool(reading.scanning),
            repairing=facts.repair_alive,
            repair_note=repair_note,
            qdrant_ready=bool(reading.qdrant_ready),
            qdrant_attention=level,
            work_alive=bool(
                facts.thread_alive or facts.delete_alive or facts.repair_alive or facts.repair_starting or facts.optimize_busy
            ),
        )
        self.assertEqual(app.qdrant_label.get_text(), reading.qdrant_text)
        self.assertEqual(app.activity_label.get_text(), shown)
        pause = track_optimize_pause(getattr(self, "_pause", None), view.get("optimize_hold") or "", reading.painted_at)
        self._pause = pause
        text, _fraction = pause_caption(pause, reading.painted_at)
        self.assertEqual(app.index_optimize.get_text(), text)
        self.assertTrue(app.index_optimize.get_visible())
        live_text, live_detail, live_pulse = reading.optimize_live
        self.assertEqual(app.repair_optimize_bar.get_text(), live_text)
        self.assertEqual(app.repair_optimize_bar.get_tooltip_text(), live_detail)
        if not live_pulse:
            self.assertEqual(app.repair_optimize_bar.get_fraction(), 0)
        title, fraction, bar_text = format_repair_current(reading.repair)
        self.assertEqual(app.repair_current_label.get_text(), title)
        self.assertEqual(app.repair_current_bar.get_text(), bar_text)
        self.assertAlmostEqual(app.repair_current_bar.get_fraction(), fraction)
        self.assertEqual(app.repair_overall.get_text(), reading.repair.overall)
        self.assertAlmostEqual(app.repair_overall.get_fraction(), min(1.0, max(0.0, reading.repair.fraction)))
        self.assertEqual(app.repair_stats.get_text(), reading.repair.stats)
        self.assertEqual(app.repair_status.get_text(), reading.repair.note)
        active = next((row for row in reading.repair.rows if row.pulse >= 0), None)
        if active is not None:
            self.assertEqual(app.repair_specs.get_text(), f"{active.name}: {active.detail}")
        self.assertEqual(_store_rows(app.repair_store), [repair_store_values(item) for item in reading.repair.rows])
        pinned = [item for item in reading.snap.files if item.state == "running"]
        self.assertEqual(_store_rows(app.now_store), [_file_row(item, reading, rate) for item in pinned])
        folders = library_tree(
            reading.library_items,
            hide_finished=bool(reading.hide_finished),
            documents=tuple(reading.doc_groups or ()),
        )
        self.assertEqual(_root_rows(app.store), [folder_store_row(folder) for folder in folders])
        self.assertTrue(folders)
        boards = build_queue(
            reading.open_files,
            choice=reading.order or "library",
            priorities=set(reading.priorities or ()),
            dates=reading.zim_dates or {},
            embed_order=reading.embed_order,
            ocr_order=reading.ocr_order,
            lane=reading.lane,
            embedding=reading.embedding,
        )
        shown_embed = shown_slots(boards.embed)
        shown_ocr = shown_slots(boards.ocr)
        self.assertEqual(app.queue_embed_caption.get_text(), queue_caption("Embedding", len(shown_embed), len(boards.embed)))
        self.assertEqual(app.queue_ocr_caption.get_text(), queue_caption("OCR", len(shown_ocr), len(boards.ocr)))
        self.assertEqual(_store_rows(app.queue_embed_store), [_file_row(slot.item, reading, rate) for slot in shown_embed])
        self.assertEqual(_store_rows(app.queue_ocr_store), [_file_row(slot.item, reading, rate) for slot in shown_ocr])
        bounds = app.log_buffer.get_bounds()
        log_text = app.log_buffer.get_text(bounds[0], bounds[1], True)
        for seq, level, msg in reading.log_lines:
            if log_line_visible(
                level,
                app.info_check.get_active(),
                app.debug_check.get_active(),
                app.error_check.get_active(),
            ):
                self.assertIn(msg, log_text)
                self.assertGreater(seq, 0)
        indexing = view.get("phase") in ("waiting", "running") and facts.thread_alive
        from nomad_turbo.live import tray_indicator_label, tray_status_text

        label, guide = tray_indicator_label(
            view,
            indexing=indexing,
            repairing=facts.repair_alive,
            hidden=bool(app.hidden),
            repair_kind=facts.repair_kind,
        )
        repair_overall = reading.repair.overall if facts.repair_alive and reading.repair is not None else ""
        self.assertEqual(app._tray_label, label)
        self.assertEqual(app._tray_guide, guide)
        self.assertEqual(app._tray_status_text, tray_status_text(repair_overall, reading.snap))
        latest = reading.host_points[-1]
        panel = app.host_panel
        self.assertEqual(panel.cpu_label.get_text(), cpu_caption(latest))
        self.assertEqual(panel.ram_label.get_text(), ram_caption(latest))
        self.assertEqual(panel.gpu_label.get_text(), gpu_caption(latest))
        self.assertEqual(panel.vram_label.get_text(), vram_caption(latest))
        expected_series = {name: (reads, writes) for name, reads, writes in disk_series(list(reading.host_points))}
        self.assertEqual(panel._disk_series, expected_series)
        for disk in latest.disks:
            caption, _area = panel._disk_rows[disk.label]
            self.assertEqual(caption.get_text(), disk_line(disk))
        senses = control_senses(view.get("phase") or "", facts)
        buttons = {
            "start": app.start_button,
            "delete": app.delete_button,
            "rescan": app.rescan_button,
            "measure": app.measure_button,
            "then_index": app.then_index_button,
            "stop": app.stop_button,
            "repair_scan": app.repair_scan,
            "repair_start": app.repair_start,
            "repair_stop": app.repair_stop_button,
            "repair_quick": app.repair_quick,
            "repair_complete": app.repair_complete,
            "optimize_segments": app.optimize_segments,
        }
        for key, button in buttons.items():
            self.assertEqual(button.get_sensitive(), bool(senses[key]), key)
        self.assertEqual(app.then_index_button.get_active(), bool(senses["then_index_armed"]))
        if facts.qdrant_ready:
            self.assertEqual(app.start_button.get_tooltip_text(), app.start_tip)
        self.assertEqual(app.live_gpu, reading.gpu)
        self.assertIsNone(app.host_panel._timer)

    def _figures(self, app) -> dict:
        bounds = app.log_buffer.get_bounds()
        senses = tuple(
            button.get_sensitive()
            for button in (
                app.start_button,
                app.stop_button,
                app.rescan_button,
                app.delete_button,
                app.measure_button,
                app.repair_scan,
                app.repair_start,
                app.repair_stop_button,
                app.optimize_segments,
            )
        )
        panel = app.host_panel
        return {
            "status": app.status.get_text(),
            "overall": app.overall.get_text(),
            "fraction": round(app.overall.get_fraction(), 4),
            "stats": app.stats.get_text(),
            "server": app.server_label.get_text(),
            "tune": app.tune_label.get_text(),
            "qdrant": app.qdrant_label.get_text(),
            "activity": app.activity_label.get_text(),
            "optimize": app.index_optimize.get_text(),
            "repair_optimize": app.repair_optimize_bar.get_text(),
            "repair_current": app.repair_current_label.get_text(),
            "repair_bar": app.repair_current_bar.get_text(),
            "repair_overall": app.repair_overall.get_text(),
            "repair_stats": app.repair_stats.get_text(),
            "repair_note": app.repair_status.get_text(),
            "repair_specs": app.repair_specs.get_text(),
            "repair_rows": tuple(_store_rows(app.repair_store)),
            "now": tuple(_store_rows(app.now_store)),
            "folders": tuple(_root_rows(app.store)),
            "queue_embed_caption": app.queue_embed_caption.get_text(),
            "queue_ocr_caption": app.queue_ocr_caption.get_text(),
            "queue_embed": tuple(_store_rows(app.queue_embed_store)),
            "queue_ocr": tuple(_store_rows(app.queue_ocr_store)),
            "tray": (app._tray_label, app._tray_guide, app._tray_status_text),
            "cpu": panel.cpu_label.get_text(),
            "ram": panel.ram_label.get_text(),
            "gpu": panel.gpu_label.get_text(),
            "vram": panel.vram_label.get_text(),
            "disks": tuple(panel._disk_rows[name][0].get_text() for name in panel._disk_rows),
            "series": tuple((name, tuple(reads), tuple(writes)) for name, (reads, writes) in panel._disk_series.items()),
            "log": app.log_buffer.get_text(bounds[0], bounds[1], True),
            "buttons": senses,
            "chunks": app.chunk_entry.get_text(),
        }


# Imported so a reader can see the stretch the child runs. The test submits it.
_ = hold_interpreter


class _RejectOne(BaseHTTPRequestHandler):
    """400 when the batch still contains the bad chunk. One vector otherwise."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if b"BADCHUNK" in raw:
            body = b'{"error":{"message":"too long"}}'
            status = 400
        else:
            payload = json.loads(raw.decode())
            inputs = payload.get("input")
            count = 1 if isinstance(inputs, str) else len(list(inputs))
            body = json.dumps({"data": [{"index": index, "embedding": [0.25]} for index in range(count)]}).encode()
            status = 200
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


class WorkloadExceptionTest(unittest.TestCase):
    """Child embed and OCR exceptions have to arrive as the exceptions the engine catches."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(prefix="nomad-turbo-exc-")
        self.root = Path(self._tmpdir.name)
        self.addCleanup(self._tmpdir.cleanup)
        self.ocr_mark = self.root / "ocr-mark"
        self._write_docker()
        self.host = Workload.install(self._child_env())
        self.addCleanup(self._shutdown)
        self.assertIsNotNone(self.host)
        self.assertTrue(self.host.proc.is_alive())
        self.assertFalse(Workload.inside())

    def _shutdown(self):
        if self.ocr_mark.is_file():
            text = self.ocr_mark.read_text(encoding="utf-8").strip()
            if text.isdigit():
                try:
                    os.kill(int(text), 15)
                except OSError:
                    pass
        current = Workload.current()
        if current is not None:
            current.close()
        Workload._current = None

    def _write_docker(self):
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        docker = bin_dir / "docker"
        docker.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "inspect" ] || [ "$1" = "rm" ] || [ "$1" = "ps" ]; then\n'
            "  exit 1\n"
            "fi\n"
            'mark="$NOMAD_TURBO_OCR_MARK"\n'
            'if [ -n "$mark" ]; then echo $$ > "$mark"; fi\n'
            "exec /bin/sleep 120\n",
            encoding="utf-8",
        )
        docker.chmod(0o755)
        self.bin_dir = bin_dir

    def _child_env(self) -> dict:
        state = self.root / "state"
        state.mkdir()
        return {
            "PATH": f"{self.bin_dir}:{os.environ.get('PATH', '')}",
            "NOMAD_TURBO_OCR_MARK": str(self.ocr_mark),
            "NOMAD_TURBO_STATE": str(state),
            "NOMAD_TURBO_CONFIG": str(self.root / "config.json"),
            "NO_PROXY": "*",
            "no_proxy": "*",
            "http_proxy": "",
            "https_proxy": "",
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
        }

    def _call(self, fn, timeout):
        box: dict = {}

        def run() -> None:
            try:
                box["value"] = fn()
            except Exception as exc:
                box["exc"] = exc

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            self._shutdown()
            self.fail(f"call did not return within {timeout}s")
        return box

    def test_embed_reraise_keeps_chunk_rejected(self):
        from nomad_turbo.runtime import ChunkRejected, embed

        server = ThreadingHTTPServer(("127.0.0.1", 0), _RejectOne)
        port = server.server_address[1]
        server_thread = threading.Thread(target=server.serve_forever, name="nomad-turbo-exc-embed", daemon=True)
        server_thread.start()

        def stop_server() -> None:
            server.shutdown()
            server.server_close()
            server_thread.join(5)

        self.addCleanup(stop_server)
        opened = []
        real_open = urllib.request.urlopen

        def spy_open(*args, **kwargs):
            opened.append(args)
            return real_open(*args, **kwargs)

        texts = ["one", "two", "three", "BADCHUNK"]
        urllib.request.urlopen = spy_open
        try:
            box = self._call(lambda: embed(f"http://127.0.0.1:{port}", "nomic-embed-text", texts), 30)
        finally:
            urllib.request.urlopen = real_open
        self.assertEqual(opened, [], "embed ran in the window process")
        self.assertTrue(self.host.proc.is_alive())
        exc = box.get("exc")
        self.assertIsInstance(exc, ChunkRejected)
        self.assertIs(type(exc), ChunkRejected)
        self.assertEqual(exc.index, 3)
        self.assertEqual(exc.detail, "too long")
        self.assertEqual(str(exc), "too long")
        self.assertNotIsInstance(exc, urllib.error.HTTPError)

    def test_read_visual_reraise_keeps_stopped(self):
        from nomad_turbo.runtime import read_visual

        storage = self.root / "storage"
        storage.mkdir()
        pdf = storage / "notes.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        install = _install(str(storage))
        stop = threading.Event()
        started = []
        real_run = subprocess.run
        real_popen = subprocess.Popen

        def spy_run(*args, **kwargs):
            started.append("run")
            return real_run(*args, **kwargs)

        def spy_popen(*args, **kwargs):
            started.append("popen")
            return real_popen(*args, **kwargs)

        subprocess.run = spy_run
        subprocess.Popen = spy_popen
        box: dict = {}

        def run() -> None:
            try:
                box["value"] = read_visual(install, str(pdf), "pdf", stop)
            except Exception as exc:
                box["exc"] = exc

        worker = threading.Thread(target=run, daemon=True)
        try:
            worker.start()
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and not self.ocr_mark.is_file():
                time.sleep(0.05)
            self.assertTrue(self.ocr_mark.is_file(), "ocr docker never started")
            pid = int(self.ocr_mark.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
            stop.set()
            worker.join(20)
        finally:
            subprocess.run = real_run
            subprocess.Popen = real_popen
            if worker.is_alive():
                self._shutdown()
                worker.join(5)
        self.assertFalse(worker.is_alive(), "read_visual did not return")
        self.assertEqual(started, [], "ocr ran in the window process")
        self.assertTrue(self.host.proc.is_alive())
        exc = box.get("exc")
        self.assertIs(type(exc), RuntimeError)
        self.assertEqual(str(exc), "stopped")
        self.assertEqual(exc.args, ("stopped",))
