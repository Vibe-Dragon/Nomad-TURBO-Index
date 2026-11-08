"""Start and stop the host GPU OCR process.

The venv is optional. When it is missing, or the build cannot run on the
chosen card, the caller keeps the CPU reader.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger("nomad_turbo")

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "worker" / "ocr_gpu.py"
_CACHE: dict = {"at": 0.0, "info": None}
_CACHE_LOCK = threading.Lock()


def ocr_python() -> str | None:
    candidate = Path.home() / ".local" / "share" / "nomad-turbo" / "ocr-venv" / "bin" / "python"
    if candidate.is_file():
        return str(candidate)
    return None


def parse_probe(text: str) -> dict:
    for line in reversed(str(text or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    return {}


def probe_info(force: bool = False) -> dict:
    """One probe for every card. Cached so a settings paint does not reload Paddle."""
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _CACHE["info"]
        if not force and cached is not None and now - float(_CACHE["at"]) < 600:
            return cached
    python = ocr_python()
    if python is None or not SCRIPT.is_file():
        info: dict = {}
    else:
        try:
            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            proc = subprocess.run(
                [python, str(SCRIPT), "probe"],
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
                env=env,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            info = {"probe_error": str(exc)[:300]}
        else:
            info = parse_probe(proc.stdout)
            if not info:
                info = {"probe_error": (proc.stderr or "GPU OCR probe returned nothing")[:300]}
    with _CACHE_LOCK:
        _CACHE["at"] = time.monotonic()
        _CACHE["info"] = info
    return info


def probe_map(devices) -> dict:
    """Per-card view of the probe. `select_backend` reads one card at a time."""
    raw = probe_info()
    found = {}
    nvidia = raw.get("nvidia") if isinstance(raw.get("nvidia"), dict) else {}
    torch_rows = raw.get("torch_nvidia") if isinstance(raw.get("torch_nvidia"), dict) else {}
    for device in devices:
        if getattr(device, "kind", "") == "cpu":
            continue
        info = dict(raw)
        if device.kind == "nvidia":
            row = nvidia.get(str(device.index)) or {}
            trow = torch_rows.get(str(device.index)) or {}
            info["cuda_works"] = bool(row.get("works"))
            info["torch_works"] = bool(trow.get("works"))
            info["cuda_error"] = str(row.get("error") or trow.get("error") or info.get("probe_error") or "")
            cap = row.get("capability") or trow.get("capability") or [0, 0]
            try:
                info["ran_major"] = int(cap[0])
            except (TypeError, ValueError, IndexError):
                info["ran_major"] = 0
        found[device.id] = info
    return found


def _socket_open(path: Path) -> bool:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.settimeout(0.2)
        client.connect(str(path))
        return True
    except OSError:
        return False
    finally:
        client.close()


class GpuOcr:
    """The long-lived reader. One model load, then a page at a time."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.backend = ""
        self.device_id = ""
        self._planned = None
        self._stderr: threading.Thread | None = None

    def plan(self, assignment) -> None:
        self._planned = assignment

    def planned(self):
        return self._planned

    def ready(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def ensure(self, assignment) -> bool:
        if assignment is None or not assignment.uses_gpu_ocr:
            self.stop()
            return False
        if ocr_python() is None:
            return False
        same = (
            self.ready()
            and self.backend == assignment.backend.name
            and self.device_id == assignment.ocr.id
        )
        if same and _socket_open(self.socket_path()):
            return True
        self.stop()
        return self._start(assignment)

    def socket_path(self) -> Path:
        from nomad_turbo.discover import state_dir

        return state_dir() / "ocr.sock"

    def _start(self, assignment) -> bool:
        python = ocr_python()
        if python is None:
            return False
        from nomad_turbo.devices import reader_label

        path = self.socket_path()
        env = os.environ.copy()
        device = assignment.ocr
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["NOMAD_TURBO_OCR_DEVICE"] = device.name
        env["NOMAD_TURBO_OCR_READER"] = reader_label(assignment.backend.name)
        if device.kind == "nvidia" and device.index is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(device.index)
        elif device.kind == "amd" and device.index is not None:
            env["HIP_VISIBLE_DEVICES"] = str(device.index)
        elif device.kind == "intel" and device.index is not None:
            env["NOMAD_TURBO_OPENVINO_DEVICE"] = f"GPU.{device.index}"
        try:
            self.proc = subprocess.Popen(
                [python, str(SCRIPT), "serve", "--socket", str(path), "--backend", assignment.backend.name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            log.warning("could not start GPU OCR: %s", exc)
            return False
        self.backend = assignment.backend.name
        self.device_id = assignment.ocr.id
        self._planned = assignment
        self._stderr = threading.Thread(target=self._log_stderr, name="nomad-turbo-ocr-gpu", daemon=True)
        self._stderr.start()
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                log.warning("GPU OCR exited %s", self.proc.returncode)
                return False
            if path.exists() and _socket_open(path):
                return True
            time.sleep(0.2)
        log.warning("GPU OCR did not open its socket")
        self.stop()
        return False

    def _log_stderr(self) -> None:
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            text = line.strip()
            if text:
                log.info("ocr: %s", text[:300])

    def stop(self) -> None:
        proc = self.proc
        self.proc = None
        self.backend = ""
        self.device_id = ""
        self._planned = None
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, 15)
        except OSError:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, 9)
            except OSError:
                proc.kill()


gpu_ocr = GpuOcr()
