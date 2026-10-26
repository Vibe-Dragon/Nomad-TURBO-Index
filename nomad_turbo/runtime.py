"""Docker, Redis, and the embed server. Only containers this tool created are stopped.

Stock NOMAD embeds inside the admin container. Turbo starts its own
llama-server (`nomad-turbo-embed`) from the same Ollama image and does not
patch the admin container. Worker scripts live beside this package so a
system install and a source checkout both find them.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from nomad_turbo.discover import Install

REPO = Path(__file__).resolve().parent.parent
QUEUECTL = REPO / "worker" / "queuectl.mjs"
EXTRACT = REPO / "worker" / "extract.mjs"
OCR = REPO / "worker" / "ocr.mjs"
TESSDATA = REPO / "worker" / "tessdata"
OCR_NAME = "nomad-turbo-ocr"


class RuntimeError_(RuntimeError):
    pass


log = logging.getLogger("nomad_turbo")


def redis_int(install: Install, key: str) -> int:
    try:
        out = subprocess.check_output(
            ["docker", "exec", install.redis_container, "redis-cli", "LLEN", key],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError_(f"redis LLEN {key} failed") from exc
    return int(out.strip() or "0")


def active_embeds(install: Install) -> int:
    return redis_int(install, f"bull:{install.queue}:active")


def queue_is_paused(install: Install) -> bool:
    """True when BullMQ has the file-embeddings queue paused."""
    try:
        out = subprocess.check_output(
            [
                "docker",
                "exec",
                install.redis_container,
                "redis-cli",
                "HGET",
                f"bull:{install.queue}:meta",
                "paused",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError_(f"redis HGET bull:{install.queue}:meta paused failed") from exc
    return out.strip() == "1"


def queuectl(install: Install, *args: str, stdin: str = "") -> str:
    command = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--network",
        install.network,
        "--pull",
        "never",
        "-e",
        f"REDIS_HOST={install.redis_host}",
        "-e",
        f"REDIS_PORT={install.redis_port}",
        "-e",
        f"QUEUE_NAME={install.queue}",
        "-v",
        f"{QUEUECTL}:/app/nomad-turbo-queuectl.mjs:ro",
        "-w",
        "/app",
        "--entrypoint",
        "node",
        install.admin_image,
        "/app/nomad-turbo-queuectl.mjs",
        *args,
    ]
    try:
        result = subprocess.run(command, input=stdin, text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip().splitlines()
        raise RuntimeError_("queue control failed: " + (detail[-1] if detail else "unknown")) from exc
    return result.stdout


def list_jobs(install: Install) -> list[dict]:
    return json.loads(queuectl(install, "jobs") or "[]")


def gpu_rows() -> list[tuple[int, int, int]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        rows.append((int(parts[0]), int(float(parts[1])), int(float(parts[2]))))
    return rows


def _remove_ours(name: str) -> None:
    if not name.startswith("nomad-turbo-"):
        raise RuntimeError_(f"refusing to remove {name}")
    try:
        subprocess.run(
            ["docker", "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        log.warning("timed out removing %s", name)


def release_extract(proc: subprocess.Popen) -> None:
    """Kill the attached `docker run` before `docker rm`.

    `docker rm -f` waits for that client to disconnect, and the client waits
    for the remove to finish. Handoff has to be able to move on.
    """
    for stream in (proc.stdout, proc.stderr):
        if stream is None:
            continue
        try:
            stream.close()
        except Exception:
            pass
    if proc.poll() is None:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            log.warning("extract client did not exit yet")
    stop_named(["nomad-turbo-extract"])


def model_blob_from_manifest(data: dict) -> str:
    """Container path of the GGUF blob named by an Ollama manifest."""
    for layer in data.get("layers") or []:
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            digest = str(layer["digest"]).replace(":", "-", 1)
            return f"/root/.ollama/models/blobs/{digest}"
    raise RuntimeError_("ollama manifest has no model layer")


def embed_model_path(install: Install) -> str:
    name, sep, tag = install.embed_model.partition(":")
    if not sep:
        tag = "latest"
    manifest = Path(install.models_dir) / "manifests" / "registry.ollama.ai" / "library" / name / tag
    try:
        data = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError_(f"could not read the embed model manifest {manifest}") from exc
    return model_blob_from_manifest(data)


def start_embedder(install: Install, index: int, use_gpu: bool, gpu_index: int | None) -> tuple[str, str]:
    """One llama-server. Ollama itself forces embedding to one stream per process.

    A wider server (`-np` above 1) made short chunks faster and long chunks
    several times slower, so the server stays at one slot with a wide batch.
    The model is loaded once. Callers send many chunks per request.
    """
    del index  # one server, whatever the pool asks for
    name = "nomad-turbo-embed"
    _remove_ours(name)
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--label",
        "nomad-turbo=embed",
        "--network",
        install.network,
        "--pull",
        "never",
        "--restart",
        "no",
        "-p",
        "127.0.0.1::11434",
        "-v",
        f"{install.models_dir}:/root/.ollama/models:ro",
    ]
    if use_gpu and gpu_index is not None:
        command += [
            "--gpus",
            f"device={gpu_index}",
            "-e",
            "LD_LIBRARY_PATH=/usr/lib/ollama:/usr/lib/ollama/cuda_v13:/usr/lib/ollama/mlx_cuda_v13",
            "-e",
            "GGML_BACKEND_PATH=/usr/lib/ollama/cuda_v13/libggml-cuda.so",
            "-e",
            "CUDA_VISIBLE_DEVICES=0",
        ]
    command += [
        "--entrypoint",
        "/usr/lib/ollama/llama-server",
        install.ollama_image,
        "--model",
        embed_model_path(install),
        "--host",
        "0.0.0.0",
        "--port",
        "11434",
        "--no-webui",
        "--offline",
        "--embedding",
        "--flash-attn",
        "on",
        "-ngl",
        "999" if use_gpu and gpu_index is not None else "0",
        "-c",
        "2048",
        "-np",
        "1",
        "-b",
        "8192",
        "-ub",
        "8192",
        "--context-shift",
        "--keep",
        "4",
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip().splitlines()
        raise RuntimeError_("could not start an embed container: " + (detail[-1] if detail else name)) from exc
    mapping = subprocess.check_output(["docker", "port", name, "11434/tcp"], text=True).strip().splitlines()
    if not mapping:
        raise RuntimeError_(f"{name} did not publish a port")
    port = mapping[0].rsplit(":", 1)[-1]
    return f"http://127.0.0.1:{port}", name


def stop_named(names: list[str]) -> None:
    for name in names:
        _remove_ours(name)


def stop_ours() -> None:
    """Remove every container this tool created, including ones left by an old run."""
    try:
        out = subprocess.check_output(
            ["docker", "ps", "-a", "--filter", "label=nomad-turbo", "--format", "{{.Names}}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return
    names = [name.strip() for name in out.splitlines() if name.strip().startswith("nomad-turbo-")]
    stop_named(names)


class ChunkRejected(RuntimeError_):
    """One input in a batch was rejected. index is into that batch."""

    def __init__(self, index: int, detail: str):
        super().__init__(detail)
        self.index = int(index)
        self.detail = detail


class _BatchRejected(Exception):
    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


_RETRY_HTTP = {408, 429, 500, 502, 503, 504}


def embed_call_plan(status: int | None, attempt: int, limit: int = 3) -> str:
    """split, retry, or raise. A 400 is one bad input, not a dead server."""
    if status == 400:
        return "split"
    if status not in _RETRY_HTTP and status is not None:
        return "raise"
    if attempt + 1 >= limit:
        return "raise"
    return "retry"


def http_error_detail(exc: urllib.error.HTTPError) -> str:
    raw = ""
    try:
        raw = exc.read().decode("utf-8", "replace")
    except Exception:
        raw = str(getattr(exc, "reason", "") or "")
    text = raw.strip()
    if text.startswith("{"):
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            body = None
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict) and err.get("message"):
                text = str(err["message"])
            elif isinstance(err, str):
                text = err
    text = " ".join(text.split())
    if len(text) > 240:
        text = text[:240]
    return text or f"HTTP {exc.code}"


def vectors_from_openai(body: dict, expected: int) -> list[list[float]]:
    data = body.get("data") or []
    ordered = sorted(data, key=lambda item: int(item.get("index") or 0))
    vectors = [list(item.get("embedding") or []) for item in ordered]
    if len(vectors) != expected or any(not vector for vector in vectors):
        raise RuntimeError_(f"embedder returned {len(vectors)} vectors for {expected} texts")
    return vectors


def _post_embeddings(base_url: str, texts: list[str], timeout: float) -> dict:
    payload = json.dumps({"input": texts if len(texts) > 1 else texts[0]}).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def wait_embedder(base_url: str, model: str, timeout: float = 180) -> None:
    del model  # llama-server is already bound to the one model
    deadline = time.time() + timeout
    last = "embedder did not become ready"
    while time.time() < deadline:
        try:
            body = _post_embeddings(base_url, ["search_document: ready"], 30)
            vectors = vectors_from_openai(body, 1)
            if vectors and vectors[0]:
                return
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError, RuntimeError_) as exc:
            message = str(exc)
            if message != last:
                log.info("embedder not ready yet: %s", message)
            last = message
        time.sleep(2)
    raise RuntimeError_(last)


def _post_with_retry(post, texts: list[str], attempts: int, pause) -> list[list[float]]:
    last = "embedder did not answer"
    for attempt in range(attempts):
        try:
            vectors = post(texts)
            if len(vectors) != len(texts) or any(not vector for vector in vectors):
                raise RuntimeError_(f"embedder returned {len(vectors)} vectors for {len(texts)} texts")
            return vectors
        except _BatchRejected:
            raise
        except urllib.error.HTTPError as exc:
            detail = http_error_detail(exc)
            last = f"HTTP {exc.code}: {detail}"
            plan = embed_call_plan(exc.code, attempt, attempts)
            if plan == "split":
                raise _BatchRejected(detail) from exc
            if plan == "retry":
                log.warning("embedder call failed (%s). Trying again.", last)
                pause(min(4, 2**attempt))
                continue
            raise RuntimeError_(f"embedder {last}") from exc
        except (TimeoutError, urllib.error.URLError, OSError, json.JSONDecodeError, RuntimeError_) as exc:
            last = str(getattr(exc, "reason", exc))
            if embed_call_plan(None, attempt, attempts) == "retry":
                log.warning("embedder call failed (%s). Trying again.", last)
                pause(min(4, 2**attempt))
                continue
            raise RuntimeError_(f"embedder {last}") from exc
    raise RuntimeError_(f"embedder {last}")


def collect_embeddings(post, texts: list[str], attempts: int = 3, pause=time.sleep) -> list[list[float]]:
    """One vector per text. A 400 is split until the bad input is alone.

    That input raises ChunkRejected. The caller splits or skips it and continues
    the file. Timeouts and connection drops are tried again.
    """
    if not texts:
        return []
    return _collect(post, list(texts), attempts, pause)


def _collect(post, texts: list[str], attempts: int, pause) -> list[list[float]]:
    try:
        return _post_with_retry(post, texts, attempts, pause)
    except _BatchRejected as exc:
        if len(texts) == 1:
            raise ChunkRejected(0, exc.detail) from exc
        mid = max(1, len(texts) // 2)
        try:
            left = _collect(post, texts[:mid], attempts, pause)
        except ChunkRejected as err:
            raise ChunkRejected(err.index, err.detail) from err
        try:
            right = _collect(post, texts[mid:], attempts, pause)
        except ChunkRejected as err:
            raise ChunkRejected(err.index + mid, err.detail) from err
        return left + right


def embed(base_url: str, model: str, texts: list[str]) -> list[list[float]]:
    del model
    if not texts:
        return []

    def post(batch: list[str]) -> list[list[float]]:
        body = _post_embeddings(base_url, batch, 180)
        return vectors_from_openai(body, len(batch))

    return collect_embeddings(post, texts)


def start_extract(install: Install, host_zim: str, skip: int, after_path: str | None, index_start: int) -> subprocess.Popen:
    name = "nomad-turbo-extract"
    _remove_ours(name)
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--label",
        "nomad-turbo=extract",
        "--network",
        "none",
        "--pull",
        "never",
        "-v",
        f"{install.storage_host}:{install.container_storage}:ro",
        "-v",
        f"{EXTRACT}:/app/nomad-turbo-extract.mjs:ro",
        "-w",
        "/app",
        "--entrypoint",
        "node",
        install.admin_image,
        "/app/nomad-turbo-extract.mjs",
        "--zim",
        host_to_container(install, host_zim),
        "--skip",
        str(skip),
        "--index-start",
        str(index_start),
    ]
    if after_path:
        command += ["--after-path", after_path]
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def visual_text(returncode: int, stdout: str) -> str | None:
    """A failed read is not an empty document. Empty text is a real read."""
    if int(returncode) != 0:
        return None
    return stdout


def read_visual(
    install: Install,
    host_path: str,
    kind: str,
    stop: threading.Event | None = None,
) -> str | None:
    """Text of one uploaded PDF or picture, using the NOMAD image's OCR libraries.

    Every PDF page keeps its text layer and is also read from the page image.
    Turbo embeds the result itself.
    """
    if kind not in ("pdf", "image"):
        return None
    if not (TESSDATA / "eng.traineddata.gz").is_file():
        log.warning("english ocr data is missing from %s", TESSDATA)
        return None
    _remove_ours(OCR_NAME)
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        OCR_NAME,
        "--label",
        "nomad-turbo=ocr",
        "--network",
        "none",
        "--pull",
        "never",
        "-v",
        f"{install.storage_host}:{install.container_storage}:ro",
        "-v",
        f"{OCR}:/app/nomad-turbo-ocr.mjs:ro",
        "-v",
        f"{TESSDATA}:/tessdata:ro",
        "-w",
        "/app",
        "--entrypoint",
        "node",
        install.admin_image,
        "/app/nomad-turbo-ocr.mjs",
        "--file",
        host_to_container(install, host_path),
        "--kind",
        kind,
    ]
    try:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError as exc:
        log.warning("could not start ocr: %s", exc)
        return None
    box: dict[str, tuple[str, str]] = {}

    def _wait() -> None:
        box["out"] = proc.communicate()

    thread = threading.Thread(target=_wait, name="nomad-turbo-ocr-wait")
    thread.start()
    stopped = False
    try:
        while thread.is_alive():
            if stop is not None and stop.is_set():
                stopped = True
                proc.kill()
                thread.join(10)
                raise RuntimeError("stopped")
            thread.join(0.2)
    finally:
        if proc.poll() is None:
            proc.kill()
            thread.join(10)
        _remove_ours(OCR_NAME)
    if stopped:
        raise RuntimeError("stopped")
    stdout, stderr = box.get("out", ("", ""))
    if stderr.strip():
        for line in stderr.splitlines()[-20:]:
            log.info("ocr: %s", line[:300])
    text = visual_text(proc.returncode or 0, stdout)
    if text is None:
        log.warning("ocr failed for %s", host_path)
    return text


def host_to_container(install: Install, host_path: str) -> str:
    host_root = Path(install.storage_host).resolve()
    path = Path(host_path).resolve()
    relative = path.relative_to(host_root)
    return str(Path(install.container_storage) / relative)
