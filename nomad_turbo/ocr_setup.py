"""Download the GPU reader into this user's home folder.

EasyOCR and its PyTorch build are not part of the Turbo checkout. They are
large and they have to match the NVIDIA driver. The Settings page runs this
and leaves GPU OCR off until a small check on the card succeeds.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from pathlib import Path

_CUDA_LINE = re.compile(r"CUDA Version:\s*([0-9.]+)")
_proc_lock = threading.Lock()
_proc: subprocess.Popen | None = None


def venv_dir() -> Path:
    return Path.home() / ".local" / "share" / "nomad-turbo" / "ocr-venv"


def torch_index(cuda_version: str) -> str | None:
    """The PyTorch wheel index for this driver. None when the driver is too old."""
    major_text, _, minor_text = str(cuda_version or "").strip().partition(".")
    try:
        major = int(major_text)
        minor = int(minor_text or "0")
    except ValueError:
        return None
    if major >= 13:
        return "cu130"
    if major == 12 and minor >= 8:
        return "cu128"
    return None


def read_driver_cuda() -> str:
    """CUDA version reported by nvidia-smi. Empty when the driver does not answer."""
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True, timeout=8, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return ""
    match = _CUDA_LINE.search(out)
    return match.group(1) if match else ""


def reader_ready(devices, probes: dict | None) -> bool:
    """True when some card has a reader that is not the CPU fallback."""
    from nomad_turbo.devices import select_backend

    probes = probes or {}
    for device in devices:
        if getattr(device, "kind", "") == "cpu":
            continue
        if select_backend(device, probes.get(device.id)).name != "tesseract":
            return True
    return False


def stop_install() -> None:
    """Stop a download when the window closes."""
    global _proc
    with _proc_lock:
        proc = _proc
        _proc = None
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, 15)
    except OSError:
        proc.terminate()


def install_gpu_reader(on_line, stop: threading.Event, cuda_version: str | None = None) -> str:
    """Install PyTorch and EasyOCR. Return an empty string, or a sentence on failure.

    `cuda_version` is set by tests. The real call reads nvidia-smi.
    """
    version = read_driver_cuda() if cuda_version is None else cuda_version
    if not version:
        return "The NVIDIA driver did not report a CUDA version, so the GPU reader was not installed."
    index = torch_index(version)
    if index is None:
        return "This NVIDIA driver is older than CUDA 12.8, so the GPU reader cannot be installed."
    if stop.is_set():
        return "Download cancelled."
    root = venv_dir()
    python = root / "bin" / "python"
    if not python.is_file():
        on_line("Creating the GPU reader environment.")
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", str(root)],
                check=True,
                timeout=180,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            detail = getattr(exc, "stderr", "") or str(exc)
            return f"Could not create the GPU reader environment. {detail.strip()[:200]}".strip()
    if stop.is_set():
        return "Download cancelled."
    url = f"https://download.pytorch.org/whl/{index}"
    on_line("Downloading PyTorch.")
    code, detail = _pip(python, ["install", "torch", "torchvision", "--index-url", url], on_line, stop)
    if stop.is_set():
        return "Download cancelled."
    if code != 0:
        return f"The PyTorch download failed. {detail}".strip()
    on_line("Downloading EasyOCR.")
    code, detail = _pip(python, ["install", "easyocr"], on_line, stop)
    if stop.is_set():
        return "Download cancelled."
    if code != 0:
        return f"The EasyOCR download failed. {detail}".strip()
    if not _torch_cuda_ok(python):
        on_line("Putting the CUDA build of PyTorch back.")
        code, detail = _pip(
            python,
            ["install", "torch", "torchvision", "--index-url", url, "--force-reinstall"],
            on_line,
            stop,
        )
        if stop.is_set():
            return "Download cancelled."
        if code != 0 or not _torch_cuda_ok(python):
            return f"PyTorch installed, but it does not run on this card. {detail}".strip()
    on_line("Checking the card.")
    if not _card_ok(python):
        return "The packages installed, but a small check on the card failed."
    return ""


def _pip(python: Path, args: list[str], on_line, stop: threading.Event) -> tuple[int, str]:
    global _proc
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [str(python), "-m", "pip", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    with _proc_lock:
        _proc = proc
    last = ""
    assert proc.stdout is not None
    for line in proc.stdout:
        text = line.strip()
        if text:
            last = text[:200]
            on_line(last)
        if stop.is_set():
            stop_install()
            break
    code = proc.wait()
    with _proc_lock:
        if _proc is proc:
            _proc = None
    return code, last


def _torch_cuda_ok(python: Path) -> bool:
    code, _detail = _check(
        python,
        "import torch\n"
        "cuda = torch.version.cuda or ''\n"
        "parts = [int(p) for p in cuda.split('.')[:2] if p.isdigit()]\n"
        "major = parts[0] if parts else 0\n"
        "minor = parts[1] if len(parts) > 1 else 0\n"
        "ok = torch.cuda.is_available() and (major > 12 or (major == 12 and minor >= 8))\n"
        "raise SystemExit(0 if ok else 1)\n",
    )
    return code == 0


def _card_ok(python: Path) -> bool:
    code, _detail = _check(
        python,
        "import easyocr\n"
        "import torch\n"
        "sample = torch.ones(8, 8, device='cuda')\n"
        "(sample @ sample).sum().item()\n"
        "major = torch.cuda.get_device_capability(0)[0]\n"
        "raise SystemExit(0 if major else 1)\n",
    )
    return code == 0


def _check(python: Path, script: str) -> tuple[int, str]:
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    try:
        proc = subprocess.run(
            [str(python), "-c", script],
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)[:200]
    detail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return proc.returncode, detail[-1][:200] if detail else ""
