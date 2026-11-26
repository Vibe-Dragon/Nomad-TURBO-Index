"""Assistant models for the Models page.

The menu names live in the running chat bundle. A NOMAD update recreates
the admin container from its image and drops that edit. Opening Turbo
writes it again. This module does not edit compose and does not change
the admin image.

Context sizes follow NOMAD's ladder: the model maximum, the context
written on the model, the saved cap, and what the card can hold. The
smallest of those wins, then the size steps down to a rung.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
from pathlib import Path

from nomad_turbo.discover import state_dir

LABEL_MARKER = "FRIENDLY_CHAT_LABELS_V1"
THINK_MARKER = "MODEL_THINK_V1"
BENCHMARK_MODEL = "llama3.1:8b"
# A tasks model stays loaded. Larger than this is refused.
TASKS_MAX_BILLIONS = 4.0
TASKS_MAX_BYTES = 3 * 1024 ** 3
LABEL_LIMIT = 80
NAME_LIMIT = 256

CONTEXT_LADDER = (4096, 8192, 16384, 32768, 65536, 131072)
MIN_CONTEXT = 4096
UNKNOWN_BACKEND_CONTEXT = 8192
MEMORY_SAFETY_FRACTION = 0.8
KV_BYTES_PER_ELEMENT = 2

INDEX_WAIT = (
    "Indexing is running. Install, test, and delete wait until it stops "
    "so the card stays on the knowledge base."
)
PARAM_WAIT = (
    "Indexing is running. Context, sampling, and the renderer wait until it stops. "
    "The menu name, order, thinking default, and job are saved."
)

THINK_OFF_PROMPT = "Reply with the single word pong."
THINK_ON_PROMPT = "What is 17 times 23? Reply with the number only."

SAMPLER_KEYS = ("temperature", "top_p", "top_k", "min_p")
SAMPLER_BOUNDS = {
    "temperature": (0.0, 2.0, False),
    "top_p": (0.0, 1.0, False),
    "top_k": (0, 200, True),
    "min_p": (0.0, 1.0, False),
    "num_ctx": (1, 1048576, True),
}

OLD_OPTIONS = (
    'children:U.map(T=>p.jsxDEV("option",{value:T.name,children:[T.name,T.size>0?` (${tf(T.size)})`:""]},T.name,!0,'
    '{fileName:"/app/inertia/components/chat/index.tsx",lineNumber:684,columnNumber:23},this))'
)
OPTIONS_END = '{fileName:"/app/inertia/components/chat/index.tsx",lineNumber:684,columnNumber:23},this))})()'
THINK_OLD = "Fe=M.useCallback(T=>T in we?we[T]:j,[we,j])"
OLD_CLASS = 'id:"model-select",value:b??f,onChange:T=>pn(T.target.value),className:"min-w-0 max-w-44 sm:max-w-none '
NEW_CLASS = 'id:"model-select",value:b??f,onChange:T=>pn(T.target.value),className:"min-w-0 max-w-96 sm:max-w-none '
OLD_MODEL_ROW = 'className:"flex items-center gap-2 min-w-0",children:[p.jsxDEV("label",{htmlFor:"model-select"'
NEW_MODEL_ROW = 'className:"flex items-center gap-2",children:[p.jsxDEV("label",{htmlFor:"model-select"'
OLD_HEADER = "flex items-center gap-2 sm:gap-4 min-w-0"
NEW_HEADER = "flex flex-wrap items-center gap-2 sm:gap-4 min-w-0"

_NAME_RE = re.compile(rf"^[A-Za-z0-9][A-Za-z0-9._:/-]{{0,{NAME_LIMIT - 1}}}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
_PARAM_RE = re.compile(r"^PARAMETER (temperature|top_p|top_k|min_p|num_ctx) (\S+)\s*$")
_RENDERER_RE = re.compile(r"^RENDERER (\S+)\s*$")
_PARSER_RE = re.compile(r"^PARSER (\S+)\s*$")
_FORK_MTP_RE = re.compile(r"(?:^|[-_])mtp(?:$|[-_.])", re.IGNORECASE)
_QUANT_RE = re.compile(
    r"PTQ1_0|PQ2_0|Q8_0|Q6_K|Q5_K_M|Q5_K_S|Q5_0|Q4_K_M|Q4_K_S|Q4_0|Q3_K_M|Q2_K|Q2_0|IQ4_XS|BF16|F16|F32",
    re.IGNORECASE,
)
_BUSY = frozenset({"waiting", "running"})
_JOBS = frozenset({"chat", "tasks", "benchmark"})
_THINK = frozenset({"follow", "on", "off"})
_publish_lock = threading.Lock()

# The menu as it stood when the Models page replaced the handwritten label list.
_SEED_LABELS = (
    ("qwen3.5:9b-q4_K_M", "[Everyday / fast / photos] Qwen 3.5 9B", 1),
    ("huihui_ai/qwen3.5-abliterated:9b", "[Uncensored] Qwen 3.5 9B", 2),
    ("gemma4:12b", "[Smarter / medium / photos, audio] Gemma 4 12B", 3),
    ("hf.co/TrevorJS/gemma-4-12B-it-uncensored-GGUF:Q4_K_M", "[Uncensored] Gemma 4 12B", 4),
    ("gemma4:26b", "[Stronger / slow / photos] Gemma 4 26B", 5),
    ("hf.co/TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF:Q4_K_M", "[Uncensored] Gemma 4 26B", 6),
    ("bonsai-2.5:27b", "[Smartest / slower / photos] Bonsai 2.5 27B", 7),
    ("eriktonon/bonsai-2-uncensored:27b", "[Uncensored] Bonsai 2 27B", 8),
    ("gpt-oss:20b", "[Strong / slow / text only] GPT 20B", 9),
)
_SEED_HIDDEN = (
    (BENCHMARK_MODEL, "benchmark", 100),
    ("qwen2.5:3b", "chat", 101),
    ("huihui_ai/qwen3.5-abliterated:2b", "tasks", 102),
)

_GGUF_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    12: "Q3_K",
    13: "Q4_K",
    14: "Q5_K",
    15: "Q6_K",
}


class MenuPatchError(RuntimeError):
    """The chat bundle did not contain the dropdown code exactly once."""


class OllamaError(RuntimeError):
    """Ollama refused a request or the reply could not be read."""


def catalog_path() -> Path:
    return state_dir() / "chat-models.json"


def friendly_patch_dir() -> Path | None:
    import os

    override = os.environ.get("NOMAD_TURBO_CHAT_PATCH")
    if override:
        return Path(override)
    candidate = Path("/opt/project-nomad/patches/friendly-chat-models")
    if (candidate / "apply.py").is_file():
        return candidate
    return None


def _blank_pref(order: int = 1000, job: str = "chat", hidden: bool = False, label: str = "") -> dict:
    return {"label": label, "order": int(order), "hidden": bool(hidden), "think": "follow", "job": job}


def default_catalog() -> dict:
    """The labels, order, and hidden jobs the menu already uses."""
    models: dict[str, dict] = {}
    for name, label, order in _SEED_LABELS:
        models[name] = _blank_pref(order=order, label=label)
    for name, job, order in _SEED_HIDDEN:
        models[name] = _blank_pref(order=order, job=job, hidden=True)
    return {"models": models}


def _pref(raw: object, fallback_order: int) -> dict:
    data = raw if isinstance(raw, dict) else {}
    label = data.get("label") if isinstance(data.get("label"), str) else ""
    if "\n" in label or "\r" in label:
        label = label.replace("\r", " ").replace("\n", " ")
    label = label.strip()[:LABEL_LIMIT]
    try:
        order = int(data.get("order"))
    except (TypeError, ValueError):
        order = fallback_order
    order = min(9999, max(0, order))
    think = data.get("think") if data.get("think") in _THINK else "follow"
    job = data.get("job") if data.get("job") in _JOBS else "chat"
    return {
        "label": label,
        "order": order,
        "hidden": bool(data.get("hidden")),
        "think": think,
        "job": job,
    }


def normalize_catalog(data: object) -> dict:
    raw = data.get("models") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        raw = {}
    models = {}
    for index, (name, value) in enumerate(raw.items()):
        if not isinstance(name, str) or not name.strip() or "embed" in name.lower():
            continue
        models[name] = _pref(value, 1000 + index)
    if BENCHMARK_MODEL in models:
        models[BENCHMARK_MODEL]["job"] = "benchmark"
        models[BENCHMARK_MODEL]["hidden"] = True
    return {"models": models}


def read_catalog(path: Path) -> dict:
    """A missing file is the seeded menu. A broken file is left for the caller to see."""
    if not path.is_file():
        return default_catalog()
    return normalize_catalog(json.loads(path.read_text(encoding="utf-8")))


def write_catalog(path: Path, catalog: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = normalize_catalog(catalog)
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


def clean_label(text: str) -> str:
    if "\n" in text or "\r" in text:
        raise ValueError("The menu name cannot contain a new line.")
    text = text.strip()
    if len(text) > LABEL_LIMIT:
        raise ValueError(f"The menu name is limited to {LABEL_LIMIT} characters.")
    return text


def clean_name(text: str) -> str:
    name = text.strip()
    if not _NAME_RE.match(name):
        raise ValueError("The model name cannot contain spaces or brackets.")
    if "embed" in name.lower():
        raise ValueError("The embedding model is not changed from this page.")
    return name


def clean_token(text: str, label: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    if not _TOKEN_RE.match(value):
        raise ValueError(f"{label} has an unsupported character.")
    return value


def is_embed(name: str) -> bool:
    return "embed" in (name or "").lower()


def busy_reason(phase: object) -> str | None:
    if phase in _BUSY:
        return INDEX_WAIT
    return None


def action_block(action: str, phase: object) -> str | None:
    """Menu edits stay available while indexing. The card stays on the knowledge base."""
    if action == "menu":
        return None
    if action in ("install", "test", "delete", "parameters"):
        return busy_reason(phase)
    raise ValueError(f"unknown action {action}")


def forked_reason(*names: str) -> str | None:
    found: list[str] = []
    for name in names:
        low = (name or "").lower()
        if not low:
            continue
        if "ptq1_0" in low:
            found.append("PTQ1_0")
        if "pq2_0" in low:
            found.append("PQ2_0")
        if _FORK_MTP_RE.search(low):
            found.append("MTP")
    if not found:
        return None
    kinds = ", ".join(dict.fromkeys(found))
    return (
        f"{kinds} needs a forked runtime. Stock Ollama will not run it correctly. "
        "Confirm that before installing."
    )


def parse_parameter_billions(parameter_size: object, model_name: object = None) -> float | None:
    details = str(parameter_size or "")
    match = re.search(r"([\d.]+)\s*B", details, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    named = str(model_name or "")
    match = re.search(r"(\d+\.?\d*)\s*[bB](?![a-zA-Z])", named)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def tasks_refused(parameter_size: object, model_name: str, size_bytes: int = 0) -> str | None:
    billions = parse_parameter_billions(parameter_size, model_name)
    if billions is not None and billions > TASKS_MAX_BILLIONS:
        return "The tasks model stays small so it can remain loaded. This model is too large."
    if billions is None and size_bytes and size_bytes > TASKS_MAX_BYTES:
        return "The tasks model stays small so it can remain loaded. This file is too large."
    return None


def _pos_num(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value <= 0:
        return None
    return float(value)


def _info_map(model_info: object) -> list[tuple[str, object]]:
    if isinstance(model_info, dict):
        return list(model_info.items())
    return []


def read_context_length(model_info: object) -> int | None:
    entries = _info_map(model_info)
    if not entries:
        return None
    lookup = dict(entries)
    arch = lookup.get("general.architecture")
    if isinstance(arch, str):
        exact = _pos_num(lookup.get(f"{arch}.context_length"))
        if exact:
            return int(exact)
    for key, value in entries:
        if str(key).endswith(".context_length"):
            exact = _pos_num(value)
            if exact:
                return int(exact)
    return None


def read_num_ctx(parameters: object, modelfile: str = "") -> int | None:
    for source in (modelfile or "", str(parameters or "")):
        match = re.search(r"^\s*(?:PARAMETER\s+)?num_ctx\s+(\d+)\s*$", source, re.MULTILINE)
        if not match:
            continue
        value = int(match.group(1))
        if value > 0:
            return value
    return None


def compute_kv_bytes_per_token(model_info: object) -> float | None:
    entries = _info_map(model_info)
    if not entries:
        return None
    info = dict(entries)
    arch = info.get("general.architecture")
    if not isinstance(arch, str):
        return None

    def num(suffix: str) -> float | None:
        return _pos_num(info.get(f"{arch}.{suffix}"))

    layers = num("block_count")
    kv_heads = num("attention.head_count_kv") or num("attention.head_count")
    if not layers or not kv_heads:
        return None
    head_dim = num("attention.key_length")
    if not head_dim:
        embedding = num("embedding_length")
        heads = num("attention.head_count")
        if embedding and heads:
            head_dim = embedding / heads
    if not head_dim:
        return None
    return 2 * layers * kv_heads * head_dim * KV_BYTES_PER_ELEMENT


def estimate_kv_bytes_per_token(param_billions: float | None) -> int:
    billions = 8 if param_billions is None else param_billions
    if billions <= 1.5:
        return 32 * 1024
    if billions <= 4:
        return 64 * 1024
    if billions <= 9:
        return 128 * 1024
    if billions <= 15:
        return 192 * 1024
    if billions <= 35:
        return 256 * 1024
    return 512 * 1024


def snap_to_ladder(tokens: float) -> int:
    chosen = MIN_CONTEXT
    for rung in CONTEXT_LADDER:
        if rung <= tokens:
            chosen = rung
    return chosen


def resolve_context_window(
    *,
    model_max: int | None = None,
    num_ctx: int | None = None,
    user_cap: int | None = None,
    available_bytes: int | None = None,
    model_bytes: int | None = None,
    kv_bytes_per_token: float | None = None,
) -> dict:
    """The context NOMAD will use. The lowest ceiling wins, then it snaps to the ladder."""
    ceilings: list[tuple[int, str]] = []
    if model_max and model_max > 0:
        ceilings.append((int(model_max), "model"))
    if num_ctx and num_ctx > 0:
        ceilings.append((int(num_ctx), "model"))
    if user_cap and user_cap > 0:
        ceilings.append((int(user_cap), "user"))
    affordable = None
    if available_bytes and available_bytes > 0 and kv_bytes_per_token and kv_bytes_per_token > 0:
        for_kv = available_bytes * MEMORY_SAFETY_FRACTION - (model_bytes or 0)
        affordable = max(0, int(for_kv // kv_bytes_per_token))
        ceilings.append((affordable, "memory"))
    if not ceilings:
        return {"context": UNKNOWN_BACKEND_CONTEXT, "limited_by": "default", "affordable": affordable}
    value, source = min(ceilings, key=lambda item: item[0])
    snapped = snap_to_ladder(value)
    hard = [item for item in (model_max, num_ctx, user_cap) if isinstance(item, int) and item > 0]
    hard_ceiling = min(hard) if hard else snapped
    if snapped <= MIN_CONTEXT:
        floored = min(MIN_CONTEXT, hard_ceiling)
        return {
            "context": floored,
            "limited_by": "model" if floored < MIN_CONTEXT else "floor",
            "affordable": affordable,
        }
    return {"context": snapped, "limited_by": source, "affordable": affordable}


def kv_bytes_for(model_info: object, parameter_size: object, model_name: str) -> float:
    exact = compute_kv_bytes_per_token(model_info)
    if exact:
        return exact
    return float(estimate_kv_bytes_per_token(parse_parameter_billions(parameter_size, model_name)))


def context_advice(
    *,
    model_max: int | None,
    num_ctx: int | None,
    user_cap: int | None,
    vram_mib: int | None,
    model_bytes: int,
    kv_bytes_per_token: float,
) -> dict:
    """What fits, what NOMAD will use, and which rungs the page may offer."""
    available = None if not vram_mib else int(vram_mib) * 1024 * 1024
    fit = resolve_context_window(
        model_max=model_max,
        user_cap=user_cap,
        available_bytes=available,
        model_bytes=model_bytes,
        kv_bytes_per_token=kv_bytes_per_token,
    )
    used = resolve_context_window(
        model_max=model_max,
        num_ctx=num_ctx,
        user_cap=user_cap,
        available_bytes=available,
        model_bytes=model_bytes,
        kv_bytes_per_token=kv_bytes_per_token,
    )
    caps = [item for item in (model_max, user_cap) if item and item > 0]
    if vram_mib and fit["context"]:
        caps.append(int(fit["context"]))
    ceiling = min(caps) if caps else None
    rungs = [rung for rung in CONTEXT_LADDER if ceiling is None or rung <= ceiling]
    return {"fit": fit, "used": used, "rungs": rungs, "vram_known": bool(vram_mib)}


def context_sentence(advice: dict, model_max: int | None, user_cap: int | None) -> str:
    parts = []
    if model_max:
        parts.append(f"This model can hold {int(model_max)} tokens.")
    if user_cap:
        parts.append(f"NOMAD's cap is {int(user_cap)}.")
    fit = advice.get("fit") or {}
    if advice.get("vram_known"):
        amount = int(fit.get("context") or 0)
        reason = fit.get("limited_by")
        if reason == "floor":
            parts.append(f"This file leaves too little room on the card, so the list stops at {amount}.")
        elif reason == "memory":
            parts.append(f"About {amount} tokens fit beside this file on the card. The list stops there.")
        elif reason == "user":
            parts.append("The list stops at NOMAD's cap.")
        elif reason == "model":
            parts.append("The list stops at this model's own limit.")
        else:
            parts.append(f"About {amount} tokens fit on the card.")
    else:
        parts.append("Video memory was not read. NOMAD shrinks the context to what the card can hold.")
    parts.append(f"NOMAD will use {int(advice['used']['context'])}.")
    return " ".join(parts)


def format_sampler(key: str, value: object) -> str:
    if key not in SAMPLER_BOUNDS:
        raise ValueError(f"unsupported setting {key}")
    low, high, integer = SAMPLER_BOUNDS[key]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{key} is empty.")
        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError(f"{key} must be a number.") from exc
    else:
        number = float(value)
    if integer:
        if abs(number - round(number)) > 1e-6:
            raise ValueError(f"{key} must be a whole number.")
        number = int(round(number))
        if number < low or number > high:
            raise ValueError(f"{key} must be from {low:g} to {high:g}.")
        return str(int(number))
    if number < low or number > high:
        raise ValueError(f"{key} must be from {low:g} to {high:g}.")
    if number == 0:
        return "0"
    return f"{number:.6f}".rstrip("0").rstrip(".") or "0"


def sampling_from_entries(raw: dict) -> dict[str, str]:
    """Empty fields stay unchanged. A filled field is checked against the range."""
    found = {}
    for key in SAMPLER_KEYS:
        text = str(raw.get(key, "")).strip()
        if not text:
            continue
        found[key] = format_sampler(key, text)
    return found


def parse_samplers(parameters: object, modelfile: str = "") -> dict[str, str]:
    found: dict[str, str] = {}
    for line in (modelfile or "").splitlines():
        match = _PARAM_RE.match(line.strip())
        if match and match.group(1) in SAMPLER_KEYS:
            found[match.group(1)] = format_sampler(match.group(1), match.group(2))
    for line in str(parameters or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in SAMPLER_KEYS and parts[0] not in found:
            try:
                found[parts[0]] = format_sampler(parts[0], parts[-1])
            except ValueError:
                continue
    return found


def rewrite_modelfile(
    text: str,
    *,
    sampling: dict | None = None,
    num_ctx: int | None = None,
    clear_num_ctx: bool = False,
    renderer: str = "",
    parser: str = "",
) -> str:
    """Replace sampler lines. LICENSE quotes and both FROM lines stay as they are."""
    if clear_num_ctx and num_ctx is not None:
        raise ValueError("num_ctx cannot be set and cleared together.")
    wanted: dict[str, str] = {}
    for key, value in (sampling or {}).items():
        wanted[key] = value if isinstance(value, str) else format_sampler(key, value)
        format_sampler(key, wanted[key])
    if num_ctx is not None:
        wanted["num_ctx"] = format_sampler("num_ctx", num_ctx)
    renderer = clean_token(renderer, "Renderer")
    parser = clean_token(parser, "Parser")

    source = text.replace("\r\n", "\n")
    trailing = source.endswith("\n")
    lines = source.split("\n")
    if trailing and lines and lines[-1] == "":
        lines = lines[:-1]

    seen: set[str] = set()
    renderer_seen = False
    parser_seen = False
    out: list[str] = []
    last_param = None
    for line in lines:
        match = _PARAM_RE.match(line)
        if match:
            key = match.group(1)
            if key == "num_ctx" and clear_num_ctx:
                continue
            if key in wanted:
                line = f"PARAMETER {key} {wanted[key]}"
                seen.add(key)
            out.append(line)
            last_param = len(out) - 1
            continue
        if renderer and _RENDERER_RE.match(line):
            out.append(f"RENDERER {renderer}")
            renderer_seen = True
            continue
        if parser and _PARSER_RE.match(line):
            out.append(f"PARSER {parser}")
            parser_seen = True
            continue
        out.append(line)

    missing = [key for key in (*SAMPLER_KEYS, "num_ctx") if key in wanted and key not in seen]
    insert_at = (last_param + 1) if last_param is not None else _insert_at(out)
    for offset, key in enumerate(missing):
        out.insert(insert_at + offset, f"PARAMETER {key} {wanted[key]}")
        last_param = insert_at + offset
    if renderer and not renderer_seen:
        at = _insert_at(out)
        out.insert(at, f"RENDERER {renderer}")
    if parser and not parser_seen:
        at = _insert_at(out)
        out.insert(at, f"PARSER {parser}")
    body = "\n".join(out)
    if trailing or not body.endswith("\n"):
        body += "\n"
    return body


def _insert_at(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        if line.startswith("LICENSE ") or line.startswith("LICENSE\t"):
            return index
    return len(lines)


def settings_changed(
    *,
    modelfile: str,
    parameters: str,
    renderer: str,
    parser: str,
    sampling: dict[str, str],
    num_ctx: int | None,
    num_ctx_set: bool,
    clear_num_ctx: bool,
    renderer_edit: str,
    parser_edit: str,
) -> bool:
    current = parse_samplers(parameters, modelfile)
    for key, value in sampling.items():
        if current.get(key) != value:
            return True
    if num_ctx_set:
        have = read_num_ctx(parameters, modelfile)
        if clear_num_ctx:
            if have is not None:
                return True
        elif have != num_ctx:
            return True
    if renderer_edit and renderer_edit != (renderer or ""):
        return True
    if parser_edit and parser_edit != (parser or ""):
        return True
    return False


def merge_live(catalog: dict, installed: list[str], tasks_model: str | None) -> dict:
    """Add models that are installed and let the saved tasks model win."""
    body = normalize_catalog(catalog)
    models = body["models"]
    orders = [int(pref["order"]) for pref in models.values()] or [0]
    nxt = max(orders) + 1
    for name in installed:
        if is_embed(name) or name in models:
            continue
        job = "benchmark" if name == BENCHMARK_MODEL else "chat"
        hidden = name == BENCHMARK_MODEL
        models[name] = _blank_pref(order=nxt, job=job, hidden=hidden)
        nxt += 1
    if BENCHMARK_MODEL in models:
        models[BENCHMARK_MODEL]["job"] = "benchmark"
        models[BENCHMARK_MODEL]["hidden"] = True
    tasks = (tasks_model or "").strip()
    if tasks and not is_embed(tasks) and tasks in models and tasks != BENCHMARK_MODEL:
        for name, pref in models.items():
            if pref["job"] == "tasks" and name != tasks:
                pref["job"] = "chat"
        models[tasks]["job"] = "tasks"
    return {"models": models}


def assign_job(catalog: dict, name: str, job: str, *, parameter_size: object = "", size_bytes: int = 0) -> dict:
    if job not in _JOBS:
        raise ValueError("Choose chat or tasks.")
    body = normalize_catalog(catalog)
    models = body["models"]
    if name not in models:
        raise ValueError("That model is not on this page.")
    if name == BENCHMARK_MODEL or job == "benchmark":
        if name != BENCHMARK_MODEL or job != "benchmark":
            raise ValueError("Llama 3.1 8B stays on the benchmark.")
        return body
    current = models[name]["job"]
    if job == "chat" and current == "tasks":
        others = [item for item, pref in models.items() if pref["job"] == "tasks" and item != name]
        if not others:
            raise ValueError("Choose another tasks model before this one returns to chat.")
    if job == "tasks":
        refused = tasks_refused(parameter_size, name, size_bytes)
        if refused:
            raise ValueError(refused)
        for item, pref in models.items():
            if pref["job"] == "tasks" and item != name:
                pref["job"] = "chat"
    models[name]["job"] = job
    return body


def menu_hidden(pref: dict) -> bool:
    return bool(pref.get("hidden")) or pref.get("job") in ("tasks", "benchmark")


def menu_pieces(catalog: dict) -> tuple[dict, dict, dict, dict]:
    """Hide map, labels, ranks, and the think map. Follow is omitted from think."""
    models = normalize_catalog(catalog)["models"]
    hide = {name: 1 for name, pref in models.items() if menu_hidden(pref)}
    labels = {name: pref["label"] for name, pref in models.items() if pref.get("label")}
    ranks = {name: int(pref["order"]) for name, pref in models.items()}
    think = {}
    for name, pref in models.items():
        if pref.get("think") == "on":
            think[name] = True
        elif pref.get("think") == "off":
            think[name] = False
    return hide, labels, ranks, think


def visible_menu(catalog: dict, installed: list[str]) -> list[str]:
    hide, _labels, ranks, _think = menu_pieces(catalog)
    names = [name for name in installed if name not in hide and not is_embed(name)]
    return sorted(names, key=lambda name: (ranks.get(name, 100000), name))


def page_menu(catalog: dict, installed: list[str]) -> list[str]:
    """Shown chat models first, by order. Hidden, tasks, and the benchmark follow.

    The assistant menu leaves the second group out. This list keeps those models
    so the page can still open them, under the models people can choose.
    """
    models = normalize_catalog(catalog)["models"]

    def key(name: str) -> tuple:
        pref = models.get(name)
        if not pref:
            return (0, 100000, name)
        buried = 1 if menu_hidden(pref) else 0
        return (buried, int(pref["order"]), name)

    names: list[str] = []
    seen: set[str] = set()
    for name in installed:
        if not name or is_embed(name) or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return sorted(names, key=key)


def _js(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def render_options(catalog: dict) -> str:
    hide, labels, ranks, _think = menu_pieces(catalog)
    return (
        "children:(()=>{/*"
        + LABEL_MARKER
        + f"*/const H={_js(hide)};"
        + f"const L={_js(labels)};"
        + f"const R={_js(ranks)};"
        + "return [...U].filter(T=>!H[T.name]).sort((a,b)=>{const d=(R[a.name]??100000)-(R[b.name]??100000);return d||(a.name<b.name?-1:a.name>b.name?1:0)}).map(T=>p.jsxDEV("
        + '"option",{value:T.name,title:T.name,children:L[T.name]||[T.name,T.size>0?` (${tf(T.size)})`:""]},T.name,!0,'
        + OPTIONS_END
    )


def render_think(catalog: dict) -> str:
    _hide, _labels, _ranks, think = menu_pieces(catalog)
    return (
        "Fe=M.useCallback(T=>{/*"
        + THINK_MARKER
        + "*/const D="
        + _js(think)
        + ";return T in we?we[T]:(Object.prototype.hasOwnProperty.call(D,T)?D[T]:j)},[we,j])"
    )


def _replace_once(data: str, old: str, new: str, label: str) -> str:
    count = data.count(old)
    if count != 1:
        raise MenuPatchError(
            f"The {label} snippet was not found exactly once, so nothing was written. "
            "A NOMAD update may have changed the page."
        )
    return data.replace(old, new, 1)


def _replace_marked(data: str, marker: str, start: str, end: str, new: str, label: str) -> str:
    count = data.count(marker)
    if count > 1:
        raise MenuPatchError(f"The {label} marker appeared more than once. Nothing was written.")
    if count == 0:
        return data
    begin = data.find(start)
    if begin < 0:
        raise MenuPatchError(f"Found the {label} marker but not the code around it. Nothing was written.")
    finish = data.find(end, begin)
    if finish < 0:
        raise MenuPatchError(f"Found the {label} marker but not the end of that code. Nothing was written.")
    finish += len(end)
    return data[:begin] + new + data[finish:]


def patch_bundle(source: str, catalog: dict) -> str:
    """Labels, order, the think default, and the row layout. Exact-once, then idempotent."""
    options = render_options(catalog)
    think = render_think(catalog)
    data = source
    if LABEL_MARKER not in data:
        data = _replace_once(data, OLD_OPTIONS, options, "chat dropdown")
    else:
        data = _replace_marked(
            data,
            LABEL_MARKER,
            "children:(()=>{/*" + LABEL_MARKER,
            OPTIONS_END,
            options,
            "label",
        )
    if THINK_MARKER not in data:
        data = _replace_once(data, THINK_OLD, think, "thinking default")
    else:
        data = _replace_marked(
            data,
            THINK_MARKER,
            "Fe=M.useCallback(T=>{/*" + THINK_MARKER,
            "},[we,j])",
            think,
            "thinking default",
        )
    swaps = (
        (OLD_CLASS, NEW_CLASS, "model dropdown width"),
        (OLD_MODEL_ROW, NEW_MODEL_ROW, "model row layout"),
        (OLD_HEADER, NEW_HEADER, "chat header layout"),
    )
    for old, new, label in swaps:
        if old in data:
            if data.count(old) != 1:
                raise MenuPatchError(f"The {label} snippet matched more than once. Nothing was written.")
            data = data.replace(old, new, 1)
        elif new not in data:
            raise MenuPatchError(f"The {label} snippet was not found. Nothing was written.")
    return data


def backup_should_save(original: str, backup_exists: bool) -> bool:
    """A patched bundle is never written over the stock backup."""
    return LABEL_MARKER not in original and not backup_exists


def delete_block_reason(name: str, *, tasks_model: str | None) -> str | None:
    if is_embed(name):
        return "The embedding model stays installed. Search uses it."
    if name == BENCHMARK_MODEL:
        return "Llama 3.1 8B stays installed. The benchmark uses it."
    if tasks_model and name == tasks_model:
        return "Choose another tasks model before deleting this one."
    return None


def next_chat_model(catalog: dict, installed: list[str], removed: str) -> str | None:
    """The next assistant model. A shown model comes before a hidden one."""
    body = normalize_catalog(catalog)
    models = body["models"]
    visible = []
    hidden = []
    for name in installed:
        if name == removed or is_embed(name):
            continue
        pref = models.get(name) or _blank_pref()
        if pref.get("job") != "chat":
            continue
        bucket = hidden if menu_hidden(pref) else visible
        bucket.append(name)
    pool = visible or hidden
    if not pool:
        return None

    def sort_key(item: str) -> tuple:
        return (int(models.get(item, _blank_pref())["order"]), item)

    return sorted(pool, key=sort_key)[0]


def in_use_sentence(name: str, last_model: str | None, tasks_model: str | None) -> str:
    parts = []
    if last_model and last_model == name:
        parts.append("The assistant is using this model.")
    if tasks_model and tasks_model == name:
        parts.append("Titles and rewrites use this model.")
    return " ".join(parts)


def plan_install(*, tag: str, gguf: str, mmproj: str, name: str, confirm: bool, installed: list[str]) -> dict:
    tag = (tag or "").strip()
    gguf = (gguf or "").strip()
    mmproj = (mmproj or "").strip()
    name = (name or "").strip()
    if tag and gguf:
        raise ValueError("Install a library tag or a GGUF file, one at a time.")
    if not tag and not gguf:
        raise ValueError("Enter a library tag or choose a GGUF file.")
    if mmproj and not gguf:
        raise ValueError("A projector file is used with a GGUF file.")
    if tag:
        tag = clean_name(tag)
        reason = forked_reason(tag)
        if reason and not confirm:
            raise ValueError(reason)
        return {"kind": "pull", "name": tag, "warning": reason or ""}
    name = clean_name(name)
    if not gguf.lower().endswith(".gguf"):
        raise ValueError("Choose a .gguf file.")
    if mmproj and not mmproj.lower().endswith(".gguf"):
        raise ValueError("The projector file must be a .gguf file.")
    if name in installed:
        raise ValueError("That name already exists. Delete it before creating it again.")
    reason = forked_reason(gguf, mmproj, name)
    if reason and not confirm:
        raise ValueError(reason)
    return {"kind": "gguf", "name": name, "gguf": gguf, "mmproj": mmproj, "warning": reason or ""}


def slug_for(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "model"
    return slug[:80]


def incoming_paths(models_dir: str, name: str) -> tuple[Path, str]:
    """Host directory beside the models folder, and the path Ollama sees."""
    host = Path(models_dir).parent / "incoming" / slug_for(name)
    return host, "/root/.ollama/incoming/" + slug_for(name)


def stage_commands(pairs: list[tuple[str, str]]) -> list[list[str]]:
    parents = []
    for _src, dest in pairs:
        parent = str(Path(dest).parent)
        if parent not in parents:
            parents.append(parent)
    commands = [["sudo", "-n", "mkdir", "-p", "--", parent] for parent in parents]
    for src, dest in pairs:
        commands.append(["sudo", "-n", "cp", "--", src, dest])
    return commands


def cleanup_command(directory: str) -> list[str]:
    return ["sudo", "-n", "rm", "-rf", "--", directory]


def bytes_needed(sizes: list[int]) -> int:
    return sum(int(size) for size in sizes) * 2


def space_error(free_bytes: int, need_bytes: int) -> str | None:
    if free_bytes < need_bytes:
        return f"Need about {need_bytes} bytes free and {free_bytes} bytes are free."
    return None


def gguf_modelfile(container_model: str, container_mmproj: str = "", renderer: str = "", parser: str = "") -> str:
    lines = [f"FROM {container_model}"]
    if container_mmproj:
        lines.append(f"FROM {container_mmproj}")
    renderer = clean_token(renderer, "Renderer")
    parser = clean_token(parser, "Parser")
    if renderer:
        lines.append(f"RENDERER {renderer}")
    if parser:
        lines.append(f"PARSER {parser}")
    return "\n".join(lines) + "\n"


def quant_from_name(name: str) -> str:
    match = _QUANT_RE.search(name or "")
    return match.group(0) if match else ""


def _gguf_u(fmt: str, data: bytes) -> int:
    import struct

    return struct.unpack("<" + fmt, data)[0]


def read_gguf_meta(path: Path, limit: int = 2 * 1024 * 1024) -> dict:
    """Name, parameter count, and quant from the header. A short or odd file returns nothing."""
    import struct

    try:
        with path.open("rb") as handle:
            magic = handle.read(4)
            if magic != b"GGUF":
                return {}
            version = _gguf_u("I", handle.read(4))
            if version < 2:
                return {}
            handle.read(8)  # tensor count
            count = _gguf_u("Q", handle.read(8))
            found: dict[str, object] = {}
            for _ in range(min(count, 64)):
                if handle.tell() > limit:
                    break
                try:
                    key, value = _gguf_pair(handle)
                except ValueError:
                    break
                if key in (
                    "general.architecture",
                    "general.name",
                    "general.parameter_count",
                    "general.file_type",
                    "general.size_label",
                ):
                    found[key] = value
            return _describe_gguf(found, path.name)
    except (OSError, ValueError, struct.error):
        return {}


def _gguf_string(handle) -> str:
    length = _gguf_u("Q", handle.read(8))
    if length > 1024 * 1024:
        handle.seek(length, 1)
        return ""
    return handle.read(length).decode("utf-8", "replace")


def _skip_value(handle, kind: int) -> None:
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if kind in sizes:
        handle.seek(sizes[kind], 1)
        return
    if kind == 8:
        _gguf_string(handle)
        return
    if kind == 9:
        inner = _gguf_u("I", handle.read(4))
        count = _gguf_u("Q", handle.read(8))
        if count > 4096:
            raise ValueError("gguf metadata array is too large to preview")
        for _ in range(count):
            _skip_value(handle, inner)
        return
    raise ValueError(f"unsupported gguf type {kind}")


def _gguf_pair(handle) -> tuple[str, object]:
    key = _gguf_string(handle)
    kind = _gguf_u("I", handle.read(4))
    if kind == 8:
        return key, _gguf_string(handle)
    if kind in (4, 5, 10, 11):
        fmt = {4: "I", 5: "i", 10: "Q", 11: "q"}[kind]
        return key, _gguf_u(fmt, handle.read(8 if kind >= 10 else 4))
    if kind == 7:
        return key, handle.read(1) != b"\x00"
    _skip_value(handle, kind)
    return key, None


def _describe_gguf(found: dict, filename: str) -> dict:
    count = found.get("general.parameter_count")
    billions = None
    if isinstance(count, int) and count > 0:
        billions = count / 1_000_000_000
    file_type = found.get("general.file_type")
    quant = _GGUF_TYPES.get(file_type) if isinstance(file_type, int) else ""
    if not quant:
        quant = quant_from_name(filename)
    return {
        "architecture": found.get("general.architecture") or "",
        "name": found.get("general.name") or "",
        "billions": billions,
        "quant": quant or "",
    }


def format_bytes(size: int) -> str:
    number = float(max(0, int(size)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if number < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(number)} B"
            return f"{number:.1f} {unit}"
        number /= 1024
    return f"{int(size)} B"


def file_preview(path: Path) -> str:
    size = path.stat().st_size
    meta = read_gguf_meta(path)
    parts = [format_bytes(size)]
    if meta.get("billions"):
        parts.append(f"{meta['billions']:.1f}B")
    elif meta.get("name"):
        parts.append(str(meta["name"]))
    quant = meta.get("quant") or quant_from_name(path.name)
    if quant:
        parts.append(str(quant))
    return ", ".join(parts)


def registry_manifest_url(tag: str) -> str | None:
    """An Ollama library tag. A host such as hf.co is left to the download itself."""
    text = (tag or "").strip()
    if not text or " " in text:
        return None
    if ":" not in text:
        text += ":latest"
    repo, _, version = text.rpartition(":")
    if not repo or not version or "/" in version:
        return None
    if "." in repo.split("/")[0]:
        return None
    if "/" not in repo:
        repo = "library/" + repo
    return f"https://registry.ollama.ai/v2/{repo}/manifests/{version}"


def sum_manifest_bytes(manifest: dict) -> int:
    total = 0
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    if config.get("size"):
        total += int(config["size"])
    for layer in manifest.get("layers") or []:
        if isinstance(layer, dict) and layer.get("size"):
            total += int(layer["size"])
    return total


def parse_tags(payload: object) -> list[dict]:
    models = []
    items = payload.get("models") if isinstance(payload, dict) else None
    for item in items or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        try:
            size = int(item.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        models.append(
            {
                "name": str(item["name"]),
                "size": size,
                "parameter_size": str(details.get("parameter_size") or ""),
                "quantization": str(details.get("quantization_level") or ""),
                "family": str(details.get("family") or ""),
            }
        )
    return models


def parse_show(payload: object) -> dict:
    data = payload if isinstance(payload, dict) else {}
    details = data.get("details") if isinstance(data.get("details"), dict) else {}
    caps = [str(item) for item in data.get("capabilities") or []]
    modelfile = str(data.get("modelfile") or "")
    parameters = str(data.get("parameters") or "")
    info = data.get("model_info") if isinstance(data.get("model_info"), dict) else {}
    return {
        "capabilities": caps,
        "thinking": "thinking" in caps,
        "vision": "vision" in caps,
        "tools": "tools" in caps,
        "parameter_size": str(details.get("parameter_size") or ""),
        "quantization": str(details.get("quantization_level") or ""),
        "family": str(details.get("family") or ""),
        "context_length": read_context_length(info),
        "modelfile": modelfile,
        "parameters": parameters,
        "renderer": str(data.get("renderer") or ""),
        "parser": str(data.get("parser") or ""),
        "samplers": parse_samplers(parameters, modelfile),
        "num_ctx": read_num_ctx(parameters, modelfile),
        "model_info": info,
    }


def chat_names(models: list[dict]) -> list[dict]:
    return [item for item in models if not is_embed(item.get("name") or "")]


def ollama_base_from_port_text(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        port = line.rsplit(":", 1)[-1]
        if port.isdigit():
            return f"http://127.0.0.1:{port}"
    raise OllamaError("Ollama's port was not published.")


def discover_ollama_base(container: str) -> str:
    try:
        result = subprocess.run(
            ["docker", "port", container, "11434"],
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "http://127.0.0.1:11434"
    try:
        return ollama_base_from_port_text(result.stdout)
    except OllamaError:
        return "http://127.0.0.1:11434"


def chat_payload(name: str, content: str, think: bool) -> dict:
    return {
        "model": name,
        "messages": [{"role": "user", "content": content}],
        "think": bool(think),
        "stream": False,
        "keep_alive": 0,
        "options": {"num_predict": 64},
    }


def create_payload(name: str, modelfile: str) -> dict:
    return {"model": name, "modelfile": modelfile, "stream": True}


def reply_ok(payload: object) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "The model did not answer."
    if payload.get("error"):
        return False, str(payload["error"])
    message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
    content = str(message.get("content") or "").strip()
    if content:
        return True, content
    return False, "The model returned an empty answer."


def progress_from_status(event: object) -> tuple[str, float | None]:
    if not isinstance(event, dict):
        return "", None
    if event.get("error"):
        raise OllamaError(str(event["error"]))
    status = str(event.get("status") or "")
    total = event.get("total")
    completed = event.get("completed")
    try:
        if total:
            return status, max(0.0, min(1.0, float(completed or 0) / float(total)))
    except (TypeError, ValueError, ZeroDivisionError):
        return status, None
    return status, None


def largest_nvidia_mib(devices: list) -> int | None:
    totals = []
    for device in devices:
        if getattr(device, "kind", None) != "nvidia":
            continue
        total = getattr(device, "total_mib", 0) or 0
        if total:
            totals.append(int(total))
    return max(totals) if totals else None


def http_request(url: str, payload: dict | None):
    """GET when there is no body. The model list refuses POST."""
    import urllib.request

    data = None if payload is None else json.dumps(payload).encode("utf-8")
    return urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )


def _request(url: str, payload: dict | None, timeout: float | None):
    import urllib.error
    import urllib.request

    req = http_request(url, payload)
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise OllamaError(detail or f"Ollama returned {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise OllamaError(f"Ollama did not answer: {exc.reason}") from exc


def _decode_body(body: str) -> dict:
    if not body.strip():
        return {}
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OllamaError("Ollama returned a reply that was not JSON.") from exc
    if isinstance(data, dict) and data.get("error"):
        raise OllamaError(str(data["error"]))
    return data if isinstance(data, dict) else {"value": data}


def get_json(base: str, path: str, timeout: float = 60) -> dict:
    with _request(base.rstrip("/") + path, None, timeout) as response:
        return _decode_body(response.read().decode("utf-8"))


def post_json(base: str, path: str, payload: dict, timeout: float = 60) -> dict:
    with _request(base.rstrip("/") + path, payload, timeout) as response:
        return _decode_body(response.read().decode("utf-8"))


def post_stream(base: str, path: str, payload: dict, timeout: float | None = None):
    with _request(base.rstrip("/") + path, payload, timeout) as response:
        for raw in response:
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise OllamaError("Ollama returned a progress line that was not JSON.") from exc
            if isinstance(event, dict) and event.get("error"):
                raise OllamaError(str(event["error"]))
            yield event


def delete_model(base: str, name: str) -> None:
    import urllib.error
    import urllib.request

    url = base.rstrip("/") + "/api/delete"
    payload = json.dumps({"model": name}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise OllamaError(detail or f"Ollama returned {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise OllamaError(f"Ollama did not answer: {exc.reason}") from exc


def _run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, text=True, **kwargs)


def _chat_bundle(container: str) -> str:
    found = _run(
        ["docker", "exec", container, "sh", "-c", 'grep -l \'id:"model-select"\' /app/public/assets/*.js'],
        capture_output=True,
    ).stdout.split()
    if len(found) != 1:
        raise MenuPatchError(f"Expected one chat bundle with the model dropdown, found {found or 'none'}.")
    return found[0]


def publish_menu(install, catalog: dict) -> str:
    """Write the menu into the running admin container. An unchanged file is left alone."""
    with _publish_lock:
        container = install.admin_container
        remote = _chat_bundle(container)
        local = Path("/tmp") / f"nomad-turbo-{Path(remote).name}"
        _run(["docker", "cp", f"{container}:{remote}", str(local)])
        original = local.read_text(encoding="utf-8")
        patched = patch_bundle(original, catalog)
        if patched == original:
            return "The chat menu already matches this list."
        patch_dir = friendly_patch_dir()
        if patch_dir is not None:
            backup = patch_dir / "original" / Path(remote).name
            if backup_should_save(original, backup.is_file()):
                backup.parent.mkdir(parents=True, exist_ok=True)
                backup.write_text(original, encoding="utf-8")
        local.write_text(patched, encoding="utf-8")
        check = "/tmp/nomad-turbo-chat-check.js"
        _run(["docker", "cp", str(local), f"{container}:{check}"])
        try:
            _run(["docker", "exec", container, "node", "--check", check])
        except subprocess.CalledProcessError as exc:
            subprocess.run(["docker", "exec", container, "rm", "-f", check], check=False)
            raise MenuPatchError(
                "The chat menu edit failed a syntax check. The live file was left unchanged."
            ) from exc
        _run(["docker", "cp", str(local), f"{container}:{remote}"])
        subprocess.run(["docker", "exec", container, "rm", "-f", check], check=False)
        return "The chat menu was updated. Refresh the assistant page to see it."
