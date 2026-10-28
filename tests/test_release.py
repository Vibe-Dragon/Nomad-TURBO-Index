"""Release checks: hardware call size, tooltips, and porter notes.

These tests do not open the window, pause NOMAD, or contact Qdrant.
"""

import inspect
import tempfile
import unittest
from pathlib import Path

from nomad_turbo.chunking import embed_input, fit_embed_inputs, split_chunks
from nomad_turbo.decisions import (
    finished_status_note,
    idle_handoff_note,
    nomad_embed_guard,
    offsets_are_safe,
    policy_after_stop,
    remembered_policy,
    stock_should_continue,
    stop_status_note,
)
from nomad_turbo.desktop import installed_command
from nomad_turbo.discover import Install
from nomad_turbo.pool import EmbedPool
from nomad_turbo.progress import usage_text
from nomad_turbo.session import Session
from nomad_turbo.slots import MAX_BOOST_PERCENT, embed_call_budget, install_boost_percent
from nomad_turbo.store import collection_ready
from nomad_turbo.tips import TIPS
from nomad_turbo import cli

ROOT = Path(__file__).resolve().parents[1]

# Controls on the Index and Repair pages. The window sets these tooltips.
PAGE_TIPS = (
    "qdrant",
    "slider",
    "tune",
    "tune_auto",
    "usage",
    "col_file",
    "col_kind",
    "col_status",
    "col_progress",
    "col_articles",
    "col_chunks",
    "col_size",
    "col_rate",
    "col_eta",
    "file_bar",
    "library_bar",
    "keep",
    "host",
    "errors",
    "info",
    "debug",
    "follow",
    "start",
    "stop",
    "rescan",
    "export",
    "delete",
    "repair_quick",
    "repair_complete",
    "repair_col_file",
    "repair_col_problem",
    "repair_col_points",
    "repair_col_progress",
    "repair_col_status",
    "repair_col_elapsed",
    "repair_file_bar",
    "repair_overall_bar",
    "repair_scan",
    "repair_start",
    "repair_stop",
)

LEADING_NOTES = {
    "nomad_turbo/engine.py": "outside the admin container",
    "nomad_turbo/slots.py": "single llama-server",
    "nomad_turbo/chunking.py": "without cutting a page short",
    "nomad_turbo/runtime.py": "does not patch",
    "nomad_turbo/decisions.py": "Manual",
    "nomad_turbo/plain.py": "embeds that text itself",
    "nomad_turbo/packing.py": "search_document",
    "nomad_turbo/pool.py": "small embed jobs",
    "worker/extract.mjs": "chrome",
    "worker/ocr.mjs": "does not call NOMAD",
    "worker/queuectl.mjs": "does not patch",
}


def _leading_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        body = text.lstrip()
        if body.startswith('"""') or body.startswith("'''"):
            quote = body[:3]
            end = body.find(quote, 3)
            if end > 3:
                return body[3:end]
        return ""
    lines = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            if lines:
                break
            continue
        if raw.startswith("//"):
            lines.append(raw[2:].strip())
            continue
        break
    return " ".join(lines)


class HardwareBudgetTest(unittest.TestCase):
    def test_same_inputs_return_the_same_call_size(self):
        kwargs = dict(percent=100, gpu_total_mib=8192, host_total_mib=32768, cpu_count=8)
        first = embed_call_budget(**kwargs)
        second = embed_call_budget(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(
            embed_call_budget(MAX_BOOST_PERCENT, gpu_total_mib=8192, host_total_mib=32768, cpu_count=8),
            embed_call_budget(MAX_BOOST_PERCENT, gpu_total_mib=8192, host_total_mib=32768, cpu_count=8),
        )

    def test_a_larger_gpu_budget_is_not_a_smaller_call(self):
        small = embed_call_budget(100, gpu_total_mib=4096, host_total_mib=65536, cpu_count=32)
        large = embed_call_budget(100, gpu_total_mib=16384, host_total_mib=2048, cpu_count=1)
        self.assertGreaterEqual(large, small)
        self.assertGreater(large, small)

    def test_a_larger_cpu_budget_is_not_a_smaller_call(self):
        small = embed_call_budget(100, gpu_total_mib=None, host_total_mib=4096, cpu_count=2)
        large = embed_call_budget(100, gpu_total_mib=None, host_total_mib=65536, cpu_count=16)
        none_is_zero = embed_call_budget(100, gpu_total_mib=0, host_total_mib=4096, cpu_count=2)
        self.assertGreaterEqual(large, small)
        self.assertGreater(large, small)
        self.assertEqual(small, none_is_zero)
        self.assertGreater(small, 0)
        self.assertLess(large, 100000)

    def test_overclock_is_past_the_safe_size_and_is_not_the_default(self):
        hardware = dict(gpu_total_mib=8192, host_total_mib=16384, cpu_count=8)
        safe = embed_call_budget(100, **hardware)
        mid = embed_call_budget(110, **hardware)
        over = embed_call_budget(MAX_BOOST_PERCENT, **hardware)
        self.assertGreater(mid, safe)
        self.assertGreater(over, mid)
        self.assertEqual(MAX_BOOST_PERCENT, 150)
        widths = [embed_call_budget(percent, **hardware) for percent in range(10, MAX_BOOST_PERCENT + 1)]
        self.assertEqual(widths, sorted(widths))
        self.assertIsNone(Install.__dataclass_fields__["boost_percent"].default)
        missing = type("Cfg", (), {"boost_percent": None, "reserve_ratio": 0.45})()
        saved = install_boost_percent(missing)
        self.assertLessEqual(saved, 100)
        self.assertNotEqual(saved, 110)
        self.assertNotEqual(saved, 150)
        full = install_boost_percent(type("Cfg", (), {"boost_percent": None, "reserve_ratio": 0.0})())
        self.assertEqual(full, 100)

    def test_the_planner_does_not_name_a_gpu_or_one_ceiling(self):
        blob = inspect.getsource(embed_call_budget) + inspect.getsource(cli.cmd_plan)
        self.assertNotIn("5060", blob)
        self.assertNotIn("RTX", blob)
        self.assertNotIn("GeForce", blob)
        self.assertNotIn("return 64", inspect.getsource(embed_call_budget))

    def test_the_pool_uses_the_hardware_budget(self):
        class _Install:
            slots = None

        pool = EmbedPool(
            _Install(),
            use_gpu=True,
            gpu_index=0,
            total_mib=16384,
            used_before=100,
            per_slot_mib=1024,
            cpu_count=8,
        )
        self.assertEqual(
            pool.batch_width(100),
            embed_call_budget(
                100,
                gpu_total_mib=16384,
                host_total_mib=pool.host_total_mib or 0,
                cpu_count=8,
            ),
        )
        self.assertGreater(pool.batch_width(150), pool.batch_width(110))
        self.assertGreater(pool.batch_width(110), pool.batch_width(100))
        cpu = EmbedPool(
            _Install(),
            use_gpu=False,
            gpu_index=None,
            total_mib=0,
            used_before=0,
            per_slot_mib=1024,
            cpu_count=2,
        )
        size = cpu.batch_width(100)
        self.assertGreater(size, 0)
        self.assertEqual(
            size,
            embed_call_budget(
                100,
                gpu_total_mib=None,
                host_total_mib=cpu.host_total_mib or 0,
                cpu_count=2,
            ),
        )

    def test_the_slider_can_save_150_and_a_new_session_is_not_forced_there(self):
        session = Session(55)
        self.assertEqual(session.get_percent(), 55)
        session.set_percent(110)
        self.assertEqual(session.get_percent(), 110)
        session.set_percent(150)
        self.assertEqual(session.get_percent(), 150)
        session.set_percent(500)
        self.assertEqual(session.get_percent(), 150)
        session.set_percent(1)
        self.assertEqual(session.get_percent(), 10)

    def test_the_usage_line_shows_the_chunk_count(self):
        line = usage_text(set_percent=150, chunks=70, server_up=False)
        self.assertIn("150%", line)
        self.assertIn("70 chunks per call", line)
        self.assertIn("Overclock", line)


class TooltipTest(unittest.TestCase):
    def test_every_page_control_has_the_shared_tip(self):
        window = (ROOT / "nomad_turbo" / "window.py").read_text(encoding="utf-8")
        for key in PAGE_TIPS:
            tip = TIPS[key].strip()
            self.assertTrue(tip, key)
            self.assertIn(f'TIPS["{key}"]', window, key)
        delete = TIPS["delete"]
        self.assertIn("DELETE", delete)
        self.assertIn("does not start", delete.lower())
        self.assertIn("ZIM", delete)
        self.assertIn("upload", delete.lower())
        rescan = TIPS["rescan"].lower()
        self.assertIn("does not pause nomad", rescan)
        self.assertIn("start indexing", rescan)
        stop = TIPS["stop"]
        self.assertIn("does not delete the index", stop.lower())
        self.assertIn("Manual", stop)
        self.assertIn("not handed back", stop.lower())
        self.assertIn("stock NOMAD", stop)
        slider = TIPS["slider"]
        self.assertIn("10%", slider)
        self.assertIn("100%", slider)
        self.assertIn("150%", slider)
        self.assertIn("next embed call", slider)
        self.assertIn("overclock", slider.lower())
        self.assertIn("budget", slider.lower())
        self.assertIn("chunk count", TIPS["usage"].lower())
        keep = TIPS["keep"]
        self.assertIn("found at Start", keep)
        self.assertIn("Manual", keep)
        self.assertIn("does not start embedding", keep.lower())
        self.assertNotIn("NOMAD embeds again", keep)


class PorterNoteTest(unittest.TestCase):
    def test_modules_and_workers_lead_with_an_explanation(self):
        packages = list((ROOT / "nomad_turbo").glob("*.py"))
        workers = list((ROOT / "worker").glob("*.mjs"))
        self.assertGreaterEqual(len(packages), 10)
        self.assertGreaterEqual(len(workers), 3)
        for path in packages + workers:
            self.assertTrue(_leading_text(path).strip(), path.name)

    def test_difference_notes_say_how(self):
        for relative, phrase in LEADING_NOTES.items():
            opening = " ".join((ROOT / relative).read_text(encoding="utf-8")[:1800].split())
            self.assertIn(phrase, opening, relative)

    def test_deciding_functions_have_docstrings(self):
        functions = (
            split_chunks,
            embed_input,
            fit_embed_inputs,
            policy_after_stop,
            stock_should_continue,
            remembered_policy,
            nomad_embed_guard,
            offsets_are_safe,
            embed_call_budget,
            collection_ready,
            stop_status_note,
            finished_status_note,
            idle_handoff_note,
        )
        for function in functions:
            self.assertTrue(function.__doc__ and function.__doc__.strip(), function.__name__)


class HandoffNoteTest(unittest.TestCase):
    def test_stop_and_finish_follow_the_policy_found_at_start(self):
        manual_stop = stop_status_note("Manual")
        always_stop = stop_status_note("Always")
        self.assertIn("not handed back", manual_stop.lower())
        self.assertIn("stayed Manual", manual_stop)
        self.assertNotIn("were handed back", manual_stop)
        self.assertIn("were handed back", always_stop)
        self.assertEqual(stop_status_note(None), manual_stop)
        self.assertEqual(stop_status_note(" sometimes "), manual_stop)

        manual_done = finished_status_note("Manual")
        always_done = finished_status_note("Always")
        self.assertIn("not handed back", manual_done.lower())
        self.assertIn("stayed Manual", manual_done)
        self.assertNotIn("was handed back", manual_done)
        self.assertIn("was handed back", always_done)
        self.assertEqual(finished_status_note(None), manual_done)

        manual_idle = idle_handoff_note("Manual")
        always_idle = idle_handoff_note("Always")
        self.assertNotIn("embedding again", manual_idle.lower())
        self.assertIn("stayed Manual", manual_idle)
        self.assertIn("was not started", manual_idle)
        self.assertIn("embedding again", always_idle.lower())
        self.assertEqual(idle_handoff_note("Manual"), idle_handoff_note(None))

    def test_the_window_and_the_idle_path_use_those_notes(self):
        window = (ROOT / "nomad_turbo" / "window.py").read_text(encoding="utf-8")
        engine = (ROOT / "nomad_turbo" / "engine.py").read_text(encoding="utf-8")
        cli = (ROOT / "nomad_turbo" / "cli.py").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("stop_status_note", window)
        self.assertIn("finished_status_note", window)
        self.assertNotIn("Unfinished files were handed back to NOMAD.", window)
        self.assertNotIn("Indexing was handed back to NOMAD.", window)
        self.assertIn('policy_path = engine.state_dir() / "policy"', window)
        self.assertIn("idle_handoff_note", engine)
        self.assertNotIn('set_note("Nothing left to index. NOMAD is embedding again.")', engine)
        self.assertIn("stop_status_note", cli)
        self.assertNotIn("NOMAD embeds again", readme)
        self.assertIn("## Disclaimer", readme)
        self.assertIn("## FAQ", readme)
        self.assertIn("I am not responsible", readme)
        self.assertIn("when I get to it", readme)
        self.assertIn("MIT License", readme)


class PackageSourceTest(unittest.TestCase):
    def test_desktop_entry_and_build_script_stay_generic(self):
        desktop = (ROOT / "packaging" / "nomad-turbo.desktop").read_text(encoding="utf-8")
        script = (ROOT / "packaging" / "build-deb.sh").read_text(encoding="utf-8")
        self.assertIn("Exec=/usr/bin/nomad-turbo toggle", desktop)
        self.assertNotIn("/home/nomad", desktop)
        self.assertNotIn("nvidia-driver", script)
        self.assertNotIn("5060", script)
        self.assertNotIn("RTX", script)
        self.assertNotIn("dpkg -i", script)

    def test_installed_command_is_not_a_home_path_when_the_package_is_present(self):
        source = Path("/home/nomad/nomad-turbo/nomad-turbo")
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / "nomad-turbo"
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            chosen = installed_command(binary, source)
            self.assertEqual(chosen, binary)
            self.assertNotIn("/home/nomad", str(chosen))
        self.assertEqual(installed_command(Path("/no/such/nomad-turbo-bin"), source), source)
