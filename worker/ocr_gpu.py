"""GPU OCR for one page image.

The host runs this. The NOMAD container only draws the page and sends the
image over a socket. Official PaddleOCR is used when that build can run on
the card. A third-party compute 12.0 build is used when that is the wheel
that was installed. EasyOCR is used when PyTorch can run on the NVIDIA card
and Paddle cannot. AMD uses Paddle with ROCm or ONNX Runtime. Intel uses
OpenVINO. The CPU reader is not in this file.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import socket
import sys
import threading
from pathlib import Path


_ENGINE = None
_INFER = threading.Lock()


def text_from_ocr(result) -> str:
    """Lines from a PaddleOCR, EasyOCR, or RapidOCR result. Boxes are ignored."""
    lines: list[str] = []
    _collect(result, lines)
    return "\n".join(lines)


def _collect(node, lines: list[str]) -> None:
    if node is None or isinstance(node, (str, bytes, int, float)):
        return
    if isinstance(node, dict):
        texts = node.get("rec_texts")
        if isinstance(texts, list):
            for item in texts:
                if isinstance(item, str) and item.strip():
                    lines.append(item.strip())
            return
        for value in node.values():
            _collect(value, lines)
        return
    if isinstance(node, (list, tuple)):
        if len(node) >= 2 and isinstance(node[1], str) and node[1].strip():
            lines.append(node[1].strip())
            return
        if (
            len(node) >= 2
            and isinstance(node[1], (list, tuple))
            and node[1]
            and isinstance(node[1][0], str)
            and node[1][0].strip()
        ):
            lines.append(node[1][0].strip())
            return
        for item in node:
            _collect(item, lines)


def _as_array(blob: bytes):
    import numpy as np
    from PIL import Image

    image = Image.open(io.BytesIO(blob)).convert("RGB")
    return np.array(image)


def _load_engine(backend: str):
    global _ENGINE
    if _ENGINE is not None:
        return
    if backend.startswith("paddle"):
        from paddleocr import PaddleOCR

        try:
            engine = PaddleOCR(lang="en", device="gpu:0", use_textline_orientation=True)
            _ENGINE = ("paddle3", engine)
        except TypeError:
            engine = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True, show_log=False)
            _ENGINE = ("paddle2", engine)
        return
    if backend == "easyocr":
        import easyocr

        _ENGINE = ("easyocr", easyocr.Reader(["en"], gpu=True, verbose=False))
        return
    if backend == "openvino":
        from rapidocr_openvino import RapidOCR

        _ENGINE = ("rapid", RapidOCR())
        return
    from rapidocr_onnxruntime import RapidOCR

    _ENGINE = ("rapid", RapidOCR())


def recognize(blob: bytes, backend: str) -> str:
    image = _as_array(blob)
    kind, engine = _ENGINE or (None, None)
    if engine is None:
        _load_engine(backend)
        kind, engine = _ENGINE
    with _INFER:
        if kind == "paddle3":
            try:
                result = engine.predict(image)
            except AttributeError:
                result = engine.ocr(image)
        elif kind == "paddle2":
            result = engine.ocr(image, cls=True)
        elif kind == "easyocr":
            result = engine.readtext(image)
        else:
            result, _elapsed = engine(image)
    return text_from_ocr(result)


def frame(payload: bytes) -> bytes:
    return len(payload).to_bytes(4, "big") + payload


def recv_exact(conn: socket.socket, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = conn.recv(size - len(buf))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return bytes(buf)


def _probe() -> dict:
    info: dict = {
        "paddle": False,
        "cuda": False,
        "rocm": False,
        "rocm_works": False,
        "sm120_wheel": False,
        "nvidia": {},
        "onnx_cuda": False,
        "onnx_migraphx": False,
        "onnx_rocm": False,
        "onnx_openvino": False,
        "openvino_gpu": False,
        "rapid_onnx": False,
        "rapid_openvino": False,
        "torch": False,
        "easyocr": False,
        "torch_nvidia": {},
    }
    try:
        import paddle

        info["paddle"] = True
        info["cuda"] = bool(paddle.is_compiled_with_cuda())
        rocm = getattr(paddle, "is_compiled_with_rocm", None)
        info["rocm"] = bool(rocm()) if callable(rocm) else False
        marker = Path(paddle.__file__).resolve().parent / ".nomad_turbo_sm120"
        info["sm120_wheel"] = marker.is_file()
        if info["cuda"]:
            count = 0
            try:
                count = int(paddle.device.cuda.device_count())
            except Exception:
                count = 1
            for index in range(max(1, count)):
                row = {"works": False, "capability": [0, 0], "error": ""}
                try:
                    paddle.device.set_device(f"gpu:{index}")
                    sample = paddle.ones([8, 8])
                    paddle.matmul(sample, sample).numpy()
                    cap = paddle.device.cuda.get_device_capability(index)
                    row["works"] = True
                    row["capability"] = [int(cap[0]), int(cap[1])]
                except Exception as exc:
                    row["error"] = str(exc)[:300]
                info["nvidia"][str(index)] = row
        if info["rocm"]:
            try:
                paddle.device.set_device("gpu:0")
                sample = paddle.ones([8, 8])
                paddle.matmul(sample, sample).numpy()
                info["rocm_works"] = True
            except Exception as exc:
                info["rocm_error"] = str(exc)[:300]
    except Exception as exc:
        info["paddle_error"] = str(exc)[:300]
    try:
        import onnxruntime as ort

        providers = set(ort.get_available_providers())
        info["onnx_cuda"] = "CUDAExecutionProvider" in providers
        info["onnx_migraphx"] = "MIGraphXExecutionProvider" in providers
        info["onnx_rocm"] = "ROCMExecutionProvider" in providers
        info["onnx_openvino"] = "OpenVINOExecutionProvider" in providers
    except Exception:
        pass
    try:
        import rapidocr_onnxruntime  # noqa: F401

        info["rapid_onnx"] = True
    except Exception:
        pass
    try:
        import rapidocr_openvino  # noqa: F401

        info["rapid_openvino"] = True
    except Exception:
        pass
    try:
        from openvino import Core

        devices = set(Core().available_devices)
        info["openvino_gpu"] = any(str(item).startswith("GPU") for item in devices)
    except Exception:
        pass
    try:
        import torch

        info["torch"] = True
        if torch.cuda.is_available():
            count = int(torch.cuda.device_count())
            for index in range(max(1, count)):
                row = {"works": False, "capability": [0, 0], "error": ""}
                try:
                    torch.cuda.set_device(index)
                    sample = torch.ones(8, 8, device=f"cuda:{index}")
                    (sample @ sample).cpu()
                    cap = torch.cuda.get_device_capability(index)
                    row["works"] = True
                    row["capability"] = [int(cap[0]), int(cap[1])]
                except Exception as exc:
                    row["error"] = str(exc)[:300]
                info["torch_nvidia"][str(index)] = row
    except Exception as exc:
        info["torch_error"] = str(exc)[:300]
    try:
        import easyocr  # noqa: F401

        info["easyocr"] = True
    except Exception:
        pass
    return info


def _reader_where(backend: str) -> str:
    device = os.environ.get("NOMAD_TURBO_OCR_DEVICE", "").strip()
    reader = os.environ.get("NOMAD_TURBO_OCR_READER", "").strip() or backend
    if device:
        return f"{reader} on {device}"
    return reader


def serve(path: str, backend: str) -> None:
    socket_path = Path(path)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass
    where = _reader_where(backend)
    print(f"loading {where}", file=sys.stderr, flush=True)
    _load_engine(backend)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    os.chmod(socket_path, 0o666)
    server.listen(4)
    print(f"{where} is ready", file=sys.stderr, flush=True)
    try:
        while True:
            conn, _addr = server.accept()
            threading.Thread(target=_client, args=(conn, backend), daemon=True).start()
    finally:
        server.close()
        try:
            socket_path.unlink()
        except OSError:
            pass


def _client(conn: socket.socket, backend: str) -> None:
    with conn:
        try:
            size = int.from_bytes(recv_exact(conn, 4), "big")
            if size <= 0 or size > 80 * 1024 * 1024:
                body = {"text": "", "error": "page image is the wrong size"}
            else:
                blob = recv_exact(conn, size)
                body = {"text": recognize(blob, backend), "error": ""}
        except Exception as exc:
            body = {"text": "", "error": str(exc)[:300]}
        try:
            conn.sendall(frame(json.dumps(body).encode("utf-8")))
        except OSError:
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("probe", "serve"))
    parser.add_argument("--socket", default="")
    parser.add_argument("--backend", default="paddle-cuda")
    args = parser.parse_args()
    if args.mode == "probe":
        print(json.dumps(_probe()))
        return
    if not args.socket:
        raise SystemExit("serve needs --socket")
    serve(args.socket, args.backend)


if __name__ == "__main__":
    main()
