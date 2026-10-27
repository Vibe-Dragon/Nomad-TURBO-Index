"""Find a running stock NOMAD install. No paths or GPU models are hardcoded.

Turbo reads the containers Docker already started. It does not edit compose
or patch the admin image. A missing boost_percent stays at or below 100.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


class DiscoverError(RuntimeError):
    pass


@dataclass
class Install:
    admin_container: str
    admin_image: str
    network: str
    redis_host: str
    redis_port: int
    redis_container: str
    mysql_container: str
    qdrant_url: str
    ollama_image: str
    ollama_container: str
    models_dir: str
    storage_host: str
    queue: str = "file-embeddings"
    collection: str = "nomad_knowledge_base"
    embed_model: str = "nomic-embed-text:v1.5"
    document_prefix: str = "search_document: "
    dimension: int = 768
    reserve_ratio: float = 0.45
    keep_control: bool = False
    per_slot_mib: int = 1024
    max_slots: int = 64
    slots: int | None = None
    container_storage: str = "/app/storage"
    log_open: bool = True
    log_follow: bool = True
    # Saved column widths, keyed by "index:File" or "repair:Points". 0 means unset.
    column_widths: dict | None = None
    log_pane: int = 0
    # Dragged height of the area above the log bar. 0 keeps the older log_pane height.
    log_split: int = 0
    # File-list height on each page. 0 leaves the spare space with the list.
    index_pane: int = 0
    repair_pane: int = 0
    # 10–110 slider. None means "use reserve_ratio", which is at most 100.
    # 110 is an overclock and is never the default for a new install.
    boost_percent: int | None = None
    # Separate host-meter window. None leaves the place to the window manager.
    host_x: int | None = None
    host_y: int | None = None
    host_width: int = 0
    host_height: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Install":
        data = json.loads(text)
        known = {field for field in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in known})


def _run(args: list[str]) -> str:
    if shutil.which(args[0]) is None:
        raise DiscoverError(f"{args[0]} is not installed")
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        raise DiscoverError(f"{' '.join(args[:2])} failed") from exc


def _containers() -> list[dict]:
    ids = [line for line in _run(["docker", "ps", "-q"]).split() if line]
    if not ids:
        raise DiscoverError("no containers are running")
    return json.loads(_run(["docker", "inspect", *ids]))


def _env(info: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in info.get("Config", {}).get("Env") or []:
        if "=" in item:
            key, value = item.split("=", 1)
            out[key] = value
    return out


def _image(info: dict) -> str:
    return info.get("Config", {}).get("Image") or ""


def _name(info: dict) -> str:
    return (info.get("Name") or "").lstrip("/")


def _is_admin(info: dict) -> bool:
    image = _image(info)
    if "project-nomad" not in image:
        return False
    if any(skip in image for skip in ("disk-collector", "sidecar", "updater")):
        return False
    mounts = info.get("Mounts") or []
    return any(mount.get("Destination") == "/app/storage" for mount in mounts)


def _published(info: dict, port: str) -> str | None:
    binding = (info.get("NetworkSettings", {}).get("Ports") or {}).get(port) or []
    if not binding:
        return None
    host_port = binding[0].get("HostPort")
    host_ip = binding[0].get("HostIp") or "127.0.0.1"
    if host_ip in ("0.0.0.0", "::", ""):
        host_ip = "127.0.0.1"
    if not host_port:
        return None
    return f"http://{host_ip}:{host_port}"


def _shared_network(admin: dict, others: list[dict]) -> str:
    admin_nets = set((admin.get("NetworkSettings", {}).get("Networks") or {}))
    for info in others:
        image = _image(info)
        if image.startswith("redis") or image.startswith("mysql"):
            shared = admin_nets & set((info.get("NetworkSettings", {}).get("Networks") or {}))
            shared.discard("bridge")
            if shared:
                return sorted(shared)[0]
    non_bridge = [name for name in admin_nets if name != "bridge"]
    if non_bridge:
        return sorted(non_bridge)[0]
    if admin_nets:
        return sorted(admin_nets)[0]
    raise DiscoverError("NOMAD admin container is not on a docker network")


def _models_dir(info: dict) -> str:
    for mount in info.get("Mounts") or []:
        dest = mount.get("Destination") or ""
        source = mount.get("Source") or ""
        if dest.rstrip("/") == "/root/.ollama/models":
            return source
        if dest.rstrip("/") == "/root/.ollama" and source:
            models = os.path.join(source, "models")
            if os.path.isdir(models):
                return models
            return source
    raise DiscoverError("could not find the Ollama models directory from its mounts")


def discover() -> Install:
    infos = _containers()
    admins = [info for info in infos if _is_admin(info)]
    if not admins:
        raise DiscoverError("no running Project NOMAD admin container with /app/storage was found")
    admins.sort(key=lambda info: 0 if _published(info, "8080/tcp") else 1)
    admin = admins[0]
    env = _env(admin)
    network = _shared_network(admin, infos)

    def on_network(info: dict) -> bool:
        return network in (info.get("NetworkSettings", {}).get("Networks") or {})

    redis = next((info for info in infos if on_network(info) and _image(info).startswith("redis")), None)
    mysql = next((info for info in infos if on_network(info) and (_image(info).startswith("mysql") or _image(info).startswith("mariadb"))), None)
    qdrant = next((info for info in infos if "qdrant" in _image(info)), None)
    ollama = next((info for info in infos if "ollama/ollama" in _image(info) or _image(info).startswith("ollama")), None)
    if redis is None or mysql is None or qdrant is None or ollama is None:
        missing = [
            label
            for label, info in (("redis", redis), ("mysql", mysql), ("qdrant", qdrant), ("ollama", ollama))
            if info is None
        ]
        raise DiscoverError("missing NOMAD dependency containers: " + ", ".join(missing))
    qdrant_url = _published(qdrant, "6333/tcp")
    if not qdrant_url:
        raise DiscoverError("Qdrant does not publish port 6333 on the host")
    storage = ""
    for mount in admin.get("Mounts") or []:
        if mount.get("Destination") == "/app/storage":
            storage = mount.get("Source") or ""
    if not storage:
        raise DiscoverError("admin container does not mount /app/storage")
    return Install(
        admin_container=_name(admin),
        admin_image=_image(admin),
        network=network,
        redis_host=env.get("REDIS_HOST") or "redis",
        redis_port=int(env.get("REDIS_PORT") or "6379"),
        redis_container=_name(redis),
        mysql_container=_name(mysql),
        qdrant_url=qdrant_url,
        ollama_image=_image(ollama),
        ollama_container=_name(ollama),
        models_dir=_models_dir(ollama),
        storage_host=storage,
    )


def config_path() -> Path:
    override = os.environ.get("NOMAD_TURBO_CONFIG")
    if override:
        return Path(override)
    return Path.home() / ".config" / "nomad-turbo" / "config.json"


def state_dir() -> Path:
    override = os.environ.get("NOMAD_TURBO_STATE")
    path = Path(override) if override else Path.home() / ".local" / "state" / "nomad-turbo"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_install() -> Install:
    path = config_path()
    found = discover()
    if not path.exists():
        return found
    saved = Install.from_json(path.read_text())
    # Container names are rediscovered. The file only overrides tuning.
    for key in (
        "reserve_ratio",
        "keep_control",
        "log_open",
        "log_follow",
        "column_widths",
        "log_pane",
        "log_split",
        "index_pane",
        "repair_pane",
        "per_slot_mib",
        "max_slots",
        "slots",
        "embed_model",
        "document_prefix",
        "collection",
        "dimension",
        "boost_percent",
        "host_x",
        "host_y",
        "host_width",
        "host_height",
    ):
        setattr(found, key, getattr(saved, key))
    return found


def save_install(install: Install) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(install.to_json() + "\n")
    return path
