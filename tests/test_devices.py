"""Device choice, the OCR engine for that device, and the Qdrant CPU cap."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import ctypes

from nomad_turbo.devices import (
    ComputeDevice,
    _VkProperties,
    assignment_sentence,
    cpu_device,
    normalize_device_choice,
    present_gpu_name,
    reader_label,
    resolve_assignment,
    select_backend,
)
from nomad_turbo.discover import Install
from nomad_turbo.ocr_budget import cpu_ocr_cap, disarm, note_upsert, reset_pressure
from nomad_turbo.ocr_engine import parse_probe
from nomad_turbo.ocr_setup import reader_ready, torch_index
from nomad_turbo.session import Session


def card(kind, ident, major=0, index=0, total=16000, used=500):
    return ComputeDevice(
        f"{kind}:{ident}",
        kind,
        f"{kind} {ident}",
        index,
        total,
        used,
        major,
        0,
        ident,
    )


def load_ocr_gpu():
    path = Path(__file__).resolve().parents[1] / "worker" / "ocr_gpu.py"
    spec = importlib.util.spec_from_file_location("ocr_gpu_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChoiceTest(unittest.TestCase):
    def test_unknown_text_stays_on_auto(self):
        self.assertEqual(normalize_device_choice(""), "auto")
        self.assertEqual(normalize_device_choice("gpu"), "auto")
        self.assertEqual(normalize_device_choice("nvidia:GPU-1"), "nvidia:GPU-1")
        self.assertEqual(normalize_device_choice("cpu"), "cpu")

    def test_one_nvidia_card_indexes_there_and_reads_on_the_cpu(self):
        nvidia = card("nvidia", "GPU-1", major=12)
        assignment = resolve_assignment([cpu_device(), nvidia], "auto", "auto", {})
        self.assertEqual(assignment.index.id, nvidia.id)
        self.assertEqual(assignment.ocr.kind, "cpu")
        self.assertEqual(assignment.backend.name, "tesseract")
        self.assertFalse(assignment.share)
        self.assertIn("gives cores back", assignment_sentence(assignment))

    def test_a_second_card_with_a_reader_is_used_for_ocr(self):
        first = card("nvidia", "GPU-1", major=8, index=0, used=100)
        second = card("nvidia", "GPU-2", major=8, index=1, used=5000)
        probes = {"nvidia:GPU-2": {"cuda_works": True}}
        assignment = resolve_assignment([cpu_device(), first, second], "auto", "auto", probes)
        self.assertEqual(assignment.index.id, first.id)
        self.assertEqual(assignment.ocr.id, second.id)
        self.assertEqual(assignment.backend.name, "paddle-cuda")
        self.assertFalse(assignment.share)
        self.assertIn("at the same time", assignment_sentence(assignment))

    def test_the_same_card_waits(self):
        nvidia = card("nvidia", "GPU-1", major=12)
        probes = {"nvidia:GPU-1": {"cuda_works": True}}
        assignment = resolve_assignment([cpu_device(), nvidia], nvidia.id, nvidia.id, probes)
        self.assertEqual(assignment.backend.name, "paddle-blackwell")
        self.assertTrue(assignment.share)
        self.assertIn("waits", assignment_sentence(assignment))

    def test_a_missing_card_uses_the_cpu(self):
        nvidia = card("nvidia", "GPU-1", major=8)
        assignment = resolve_assignment([cpu_device(), nvidia], "nvidia:missing", "amd:missing", {})
        self.assertEqual(assignment.index.kind, "cpu")
        self.assertEqual(assignment.ocr.kind, "cpu")
        self.assertTrue(assignment.share)

    def test_cpu_indexing_is_explicit(self):
        nvidia = card("nvidia", "GPU-1", major=12)
        assignment = resolve_assignment([cpu_device(), nvidia], "cpu", "auto", {})
        self.assertEqual(assignment.index.kind, "cpu")
        self.assertTrue(assignment.share)

    def test_session_choice_round_trips(self):
        session = Session(16)
        session.set_index_device("nvidia:GPU-9")
        session.set_ocr_device("nope")
        self.assertEqual(session.index_device(), "nvidia:GPU-9")
        self.assertEqual(session.ocr_device(), "auto")


class BackendTest(unittest.TestCase):
    def test_blackwell_uses_the_official_build_when_it_runs(self):
        device = card("nvidia", "GPU-1", major=12)
        backend = select_backend(device, {"cuda_works": True})
        self.assertEqual(backend.name, "paddle-blackwell")
        self.assertIn("official", backend.sentence)

    def test_blackwell_uses_the_third_party_build_when_that_is_what_was_installed(self):
        device = card("nvidia", "GPU-1", major=12)
        backend = select_backend(device, {"cuda_works": True, "sm120_wheel": True})
        self.assertEqual(backend.name, "paddle-sm120")
        self.assertIn("third-party", backend.sentence)

    def test_an_older_nvidia_card_uses_official_paddle(self):
        device = card("nvidia", "GPU-1", major=8)
        backend = select_backend(device, {"cuda_works": True, "sm120_wheel": False})
        self.assertEqual(backend.name, "paddle-cuda")

    def test_a_compute_12_wheel_is_not_used_on_an_older_card(self):
        device = card("nvidia", "GPU-1", major=8)
        backend = select_backend(device, {"cuda_works": True, "sm120_wheel": True})
        self.assertEqual(backend.name, "tesseract")
        self.assertIn("12.0", backend.reason)

    def test_amd_and_intel_fall_back_until_a_runtime_is_installed(self):
        amd = select_backend(card("amd", "0000:0c:00.0"), {})
        intel = select_backend(card("intel", "0000:00:02.0"), {})
        self.assertEqual(amd.name, "tesseract")
        self.assertIn("AMD", amd.reason)
        self.assertEqual(intel.name, "tesseract")
        self.assertIn("Intel", intel.reason)
        rocm = select_backend(card("amd", "0000:0c:00.0"), {"rocm_works": True})
        self.assertEqual(rocm.name, "paddle-rocm")
        ov = select_backend(
            card("intel", "0000:00:02.0"),
            {"openvino_gpu": True, "rapid_openvino": True},
        )
        self.assertEqual(ov.name, "openvino")

    def test_onnx_is_used_when_paddle_cannot_run(self):
        device = card("nvidia", "GPU-1", major=12)
        backend = select_backend(device, {"onnx_cuda": True, "rapid_onnx": True, "cuda_error": "arch"})
        self.assertEqual(backend.name, "onnx-cuda")

    def test_easyocr_is_used_when_pytorch_runs_and_paddle_does_not(self):
        device = card("nvidia", "GPU-1", major=12)
        backend = select_backend(
            device,
            {"easyocr": True, "torch_works": True, "onnx_cuda": True, "rapid_onnx": True},
        )
        self.assertEqual(backend.name, "easyocr")
        self.assertIn("EasyOCR", backend.sentence)
        self.assertEqual(backend.reason, "")
        official = select_backend(device, {"cuda_works": True, "easyocr": True, "torch_works": True})
        self.assertEqual(official.name, "paddle-blackwell")

    def test_probe_json_keeps_the_object(self):
        raw = "loading\n" + json.dumps({"cuda_works": True, "nvidia": {"0": {"works": True}}})
        self.assertTrue(parse_probe(raw)["cuda_works"])
        self.assertEqual(parse_probe("nope"), {})


class NameTest(unittest.TestCase):
    def test_a_raw_id_or_a_processor_name_becomes_the_vendor_gpu(self):
        self.assertEqual(present_gpu_name("amd", "Device 13c0", ""), "AMD GPU")
        self.assertEqual(
            present_gpu_name("amd", "AMD Ryzen 5 9600X 6-Core Processor (RADV RAPHAEL_MENDOCINO)", "integrated"),
            "AMD Radeon Graphics",
        )
        self.assertEqual(present_gpu_name("amd", "AMD Radeon RX 7800 XT", "discrete"), "AMD Radeon RX 7800 XT")
        self.assertEqual(present_gpu_name("nvidia", "NVIDIA GeForce RTX 5060 Ti", "discrete"), "NVIDIA GeForce RTX 5060 Ti")
        self.assertEqual(present_gpu_name("intel", "Device a780", "integrated"), "Intel Graphics")
        self.assertEqual(present_gpu_name("intel", "Intel(R) Arc(TM) A770 Graphics", "discrete"), "Intel(R) Arc(TM) A770 Graphics")
        self.assertEqual(reader_label("easyocr"), "EasyOCR")
        self.assertEqual(reader_label("paddle-blackwell"), "PaddleOCR")
        self.assertEqual(reader_label("tesseract"), "Tesseract")
        self.assertEqual(ctypes.sizeof(_VkProperties), 824)

    def test_the_fallback_names_the_card(self):
        amd = select_backend(card("amd", "0000:0c:00.0"), {})
        self.assertIn("amd 0000:0c:00.0", amd.reason)
        self.assertIn("CPU", amd.reason)


class InstallTest(unittest.TestCase):
    def test_the_wheel_index_follows_the_driver(self):
        self.assertEqual(torch_index("13.0"), "cu130")
        self.assertEqual(torch_index("13.2"), "cu130")
        self.assertEqual(torch_index("12.8"), "cu128")
        self.assertEqual(torch_index("12.9"), "cu128")
        self.assertIsNone(torch_index("12.6"))
        self.assertIsNone(torch_index("11.8"))
        self.assertIsNone(torch_index(""))

    def test_gpu_ocr_stays_locked_until_a_reader_runs(self):
        nvidia = card("nvidia", "GPU-1", major=12)
        devices = [cpu_device(), nvidia]
        self.assertFalse(reader_ready(devices, {}))
        self.assertFalse(reader_ready(devices, {nvidia.id: {"cuda_error": "missing"}}))
        self.assertTrue(reader_ready(devices, {nvidia.id: {"easyocr": True, "torch_works": True}}))
        self.assertTrue(reader_ready(devices, {nvidia.id: {"cuda_works": True}}))

    def test_an_old_driver_is_refused_without_a_download(self):
        from threading import Event

        from nomad_turbo.ocr_setup import install_gpu_reader

        lines = []
        message = install_gpu_reader(lines.append, Event(), cuda_version="12.1")
        self.assertIn("12.8", message)
        self.assertEqual(lines, [])


class BudgetTest(unittest.TestCase):
    def test_qdrant_gets_the_cores_back(self):
        self.assertEqual(
            cpu_ocr_cap(12, share=False, embed=True, upserts=False, ready=True, status="green", stalled=False, index_cpu=False),
            10,
        )
        self.assertEqual(
            cpu_ocr_cap(12, share=False, embed=False, upserts=True, ready=True, status="green", stalled=False, index_cpu=False),
            2,
        )
        self.assertEqual(
            cpu_ocr_cap(12, share=False, embed=False, upserts=False, ready=True, status="yellow", stalled=False, index_cpu=False),
            1,
        )
        self.assertEqual(
            cpu_ocr_cap(12, share=False, embed=False, upserts=False, ready=False, status="green", stalled=False, index_cpu=False),
            0,
        )
        self.assertEqual(
            cpu_ocr_cap(12, share=False, embed=False, upserts=False, ready=True, status="green", stalled=True, index_cpu=False),
            0,
        )
        self.assertEqual(
            cpu_ocr_cap(12, share=True, embed=True, upserts=False, ready=True, status="green", stalled=False, index_cpu=False),
            0,
        )
        self.assertEqual(
            cpu_ocr_cap(12, share=True, embed=False, upserts=False, ready=True, status="green", stalled=False, index_cpu=False),
            10,
        )
        self.assertEqual(
            cpu_ocr_cap(12, share=False, embed=True, upserts=False, ready=True, status="green", stalled=False, index_cpu=True),
            1,
        )
        self.assertEqual(
            cpu_ocr_cap(2, share=False, embed=False, upserts=True, ready=True, status="", stalled=False, index_cpu=False),
            0,
        )

    def test_a_refused_write_is_noticed_and_cleared(self):
        from nomad_turbo.ocr_budget import current_cap, note_qdrant_fault, set_runtime

        reset_pressure()
        try:
            set_runtime(share=False, index_cpu=False, cpus=12)
            note_qdrant_fault("qdrant 429: overloaded")
            self.assertEqual(current_cap(), 0)
            note_upsert(True)
            note_upsert(False)
        finally:
            disarm()
            reset_pressure()


class PageTextTest(unittest.TestCase):
    def test_paddle_and_rapid_lines_become_page_text(self):
        gpu = load_ocr_gpu()
        paddle = [[[[0, 0], [1, 0], [1, 1], [0, 1]], ("catheter", 0.9)]]
        self.assertEqual(gpu.text_from_ocr(paddle), "catheter")
        rapid = [[[0, 0], "EMERGENCY", 0.8]]
        self.assertEqual(gpu.text_from_ocr(rapid), "EMERGENCY")
        easy = [([[0, 0], [1, 0], [1, 1], [0, 1]], "HELLO", 0.99)]
        self.assertEqual(gpu.text_from_ocr(easy), "HELLO")
        newer = {"rec_texts": ["page four", ""]}
        self.assertEqual(gpu.text_from_ocr(newer), "page four")
        self.assertEqual(len(gpu.frame(b"abc")), 7)

    def test_install_keeps_the_device_choice(self):
        install = Install(
            admin_container="a",
            admin_image="i",
            network="n",
            redis_host="r",
            redis_port=1,
            redis_container="rc",
            mysql_container="m",
            qdrant_url="q",
            ollama_image="o",
            ollama_container="oc",
            models_dir="md",
            storage_host="s",
            index_device="cpu",
            ocr_device="nvidia:GPU-1",
        )
        again = Install.from_json(install.to_json())
        self.assertEqual(again.index_device, "cpu")
        self.assertEqual(again.ocr_device, "nvidia:GPU-1")
        with tempfile.TemporaryDirectory():
            self.assertIn('"index_device"', install.to_json())


class LaunchEnvTest(unittest.TestCase):
    def test_the_budget_is_passed_without_moving_the_gc_flag(self):
        from unittest import mock

        from nomad_turbo.runtime import container_ocr_env, start_extract

        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            zim = storage / "zim" / "medicine.zim"
            zim.parent.mkdir()
            zim.write_bytes(b"zim")
            install = Install(
                admin_container="a",
                admin_image="nomad:test",
                network="n",
                redis_host="r",
                redis_port=1,
                redis_container="rc",
                mysql_container="m",
                qdrant_url="q",
                ollama_image="o",
                ollama_container="oc",
                models_dir="md",
                storage_host=str(storage),
            )
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
                    ocr_env=container_ocr_env(socket=True),
                )
            argv = popen.call_args.args[0]
            script = argv.index("/app/nomad-turbo-extract.mjs")
            self.assertEqual(argv[script - 1], "--expose-gc")
            self.assertIn("NOMAD_TURBO_OCR_BUDGET=/turbo-state/ocr-budget", argv)
            self.assertIn("NOMAD_TURBO_OCR_SOCKET=/turbo-state/ocr.sock", argv)
            self.assertLess(argv.index("NOMAD_TURBO_OCR_BUDGET=/turbo-state/ocr-budget"), script)


if __name__ == "__main__":
    unittest.main()
