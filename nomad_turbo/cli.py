"""Desktop and command line entry points.

`plan` prints a hardware-derived call size. It does not pause NOMAD and it
does not start indexing. The window command only opens the control window.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from nomad_turbo import __version__
from nomad_turbo.decisions import file_is_embeddable, stop_status_note
from nomad_turbo.desktop import install_autostart, shortcut_path, write_shortcut, wrapper_path
from nomad_turbo.discover import DiscoverError, config_path, discover, load_install, save_install, state_dir
from nomad_turbo.engine import execute
from nomad_turbo.prompt import display_available
from nomad_turbo.runtime import gpu_rows
from nomad_turbo.slots import embed_call_budget, host_memory_mib, install_boost_percent, pick_gpu


def _log_path() -> Path:
    return state_dir() / "turbo.log"


def _pid_path() -> Path:
    return state_dir() / "pid"


def _configure_log() -> None:
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler(sys.stdout)],
    )


def _notify(title: str, body: str) -> None:
    if shutil.which("notify-send"):
        subprocess.run(["notify-send", title, body], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _running_pid() -> int | None:
    path = _pid_path()
    if not path.exists():
        return None
    try:
        pid = int(path.read_text().strip())
    except ValueError:
        return None
    if _pid_alive(pid):
        return pid
    return None


def cmd_install() -> int:
    found = discover()
    path = save_install(found)
    write_shortcut(False)
    install_autostart(str(wrapper_path()))
    print(f"NOMAD Turbo {__version__}")
    print(f"config: {path}")
    print(f"shortcut: {shortcut_path()}")
    print("Click the desktop icon to open the indexing window.")
    print("It does not start or stop indexing. The tray icon changes colour while indexing or repairing.")
    return 0


def cmd_plan() -> int:
    install = load_install()
    rows = gpu_rows()
    chosen = pick_gpu(rows)
    if chosen:
        gpu_index, total, _used = chosen
        has_gpu = True
    else:
        gpu_index, total, has_gpu = None, 0, False
    percent = install_boost_percent(install)
    memory = host_memory_mib()
    host_total = memory[0] if memory else 0
    cpus = os.cpu_count() or 1
    gpu_total = total if has_gpu else None
    width = embed_call_budget(
        percent,
        gpu_total_mib=gpu_total,
        host_total_mib=host_total,
        cpu_count=cpus,
    )
    full = embed_call_budget(
        100,
        gpu_total_mib=gpu_total,
        host_total_mib=host_total,
        cpu_count=cpus,
    )
    over = embed_call_budget(
        110,
        gpu_total_mib=gpu_total,
        host_total_mib=host_total,
        cpu_count=cpus,
    )
    root = Path(install.storage_host)
    counts = {"zim": 0, "other": 0}
    for folder in ("zim", "kb_uploads"):
        base = root / folder
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            kind = file_is_embeddable(path.name) if path.is_file() else None
            if not kind:
                continue
            counts["zim" if kind == "zim" else "other"] += 1
    print(f"admin: {install.admin_container} ({install.admin_image})")
    print(f"storage: {install.storage_host}")
    print(f"qdrant: {install.qdrant_url}")
    print(f"ollama image: {install.ollama_image}")
    print(f"models: {install.models_dir}")
    print(f"embed model: {install.embed_model}")
    if has_gpu:
        print(f"gpu index {gpu_index} total {total} MiB")
    else:
        print("no gpu detected")
    print(f"system ram {host_total} MiB, cpus {cpus}")
    where = "GPU" if has_gpu else "CPU"
    print(f"embed server: one llama-server on {where}")
    print(f"hardware-derived chunks per call at {percent}%: {width}")
    print(f"safe 100% call size from detected hardware: {full} chunks")
    print(f"110% overclock call size: {over} chunks")
    print("The slider changes only that call size. The server stays up.")
    if install.slots:
        print(f"config slots={install.slots} is ignored")
    print(f"files: {counts['zim']} zim, {counts['other']} other embeddable")
    print("This plan does not pause NOMAD or start containers.")
    return 0


def cmd_run() -> int:
    _configure_log()
    stop = threading.Event()

    def _handle(signum, _frame):
        logging.getLogger("nomad_turbo").info("signal %s, finishing the current articles", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    # A desktop click raises a window. On a headless run there is no window to raise.
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    try:
        code = execute(stop)
    except Exception:
        logging.getLogger("nomad_turbo").exception("turbo failed")
        _notify("NOMAD Turbo", "Turbo stopped because of an error. NOMAD's embed queue was handed back if it had been paused.")
        write_shortcut(False)
        return 1
    result = (state_dir() / "result").read_text().strip() if (state_dir() / "result").exists() else "done"
    if stop.is_set() or result == "partial":
        saved_policy = None
        policy_path = state_dir() / "policy"
        if policy_path.is_file():
            saved_policy = policy_path.read_text(encoding="utf-8").strip()
        _notify("NOMAD Turbo", stop_status_note(saved_policy))
    else:
        _notify("NOMAD Turbo", "Turbo finished and turned itself off.")
    pid = _pid_path()
    if pid.exists():
        pid.unlink()
    return code


def cmd_window() -> int:
    try:
        from nomad_turbo.window import open_window

        return open_window()
    except Exception:
        logging.getLogger("nomad_turbo").exception("window failed")
        write_shortcut(False)
        return 1


def _turbo_running() -> bool:
    """Indexing has been started. An open window by itself does not count."""
    status = state_dir() / "status.json"
    if not status.exists():
        return False
    try:
        import json

        phase = json.loads(status.read_text()).get("phase")
    except (OSError, json.JSONDecodeError):
        return False
    return phase in ("waiting", "running")


def _raise_window(pid: int) -> None:
    os.kill(pid, signal.SIGUSR1)


def cmd_start() -> int:
    recover_quiet()
    pid = _running_pid()
    if pid:
        if _turbo_running():
            print("turbo is already running")
            write_shortcut(True)
            return 0
        _raise_window(pid)
        write_shortcut(False)
        print("window is already open")
        return 0
    if display_available():
        log_file = _log_path().open("a")
        proc = subprocess.Popen(
            [str(wrapper_path()), "window"],
            start_new_session=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        _pid_path().write_text(str(proc.pid))
        print(f"window started, pid {proc.pid}")
        return 0
    install = load_install()
    percent = install_boost_percent(install)
    print(f"no display, starting at the saved {percent}% embed batch")
    write_shortcut(True)
    log_file = _log_path().open("a")
    proc = subprocess.Popen(
        [str(wrapper_path()), "run"],
        start_new_session=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )
    _pid_path().write_text(str(proc.pid))
    print(f"turbo started at {percent}% of the GPU, pid {proc.pid}")
    return 0


def cmd_stop() -> int:
    pid = _running_pid()
    if pid:
        os.kill(pid, signal.SIGTERM)
        for _ in range(60):
            if not _pid_alive(pid):
                break
            time.sleep(1)
        else:
            os.kill(pid, signal.SIGKILL)
    else:
        # The supervisor died. Put NOMAD back if this tool had paused it.
        cmd_recover()
    write_shortcut(False)
    print("turbo is off")
    return 0


def cmd_toggle() -> int:
    """Open or raise the window. A click never stops indexing."""
    pid = _running_pid()
    if pid:
        _raise_window(pid)
        write_shortcut(False)
        return 0
    return cmd_start()


def cmd_recover() -> int:
    if _running_pid():
        return 0
    status = state_dir() / "status.json"
    paused = False
    if status.exists():
        try:
            import json

            paused = bool(json.loads(status.read_text()).get("paused_queue"))
        except json.JSONDecodeError:
            paused = False
    if not paused:
        write_shortcut(False)
        return 0
    _configure_log()
    logging.getLogger("nomad_turbo").info("recovering a turbo run that died while the embed queue was paused")
    try:
        from nomad_turbo.engine import recover_paused

        resumed = recover_paused()
    except Exception:
        logging.getLogger("nomad_turbo").exception("recover failed")
        return 1
    write_shortcut(False)
    if resumed:
        _notify("NOMAD Turbo", "Turbo had stopped uncleanly. NOMAD embedding is resumed and the ingest policy is put back.")
    else:
        _notify("NOMAD Turbo", "A repair delete may still be running. Embedding stays paused.")
    return 0


def recover_quiet() -> None:
    try:
        cmd_recover()
    except DiscoverError:
        return


def cmd_status() -> int:
    pid = _running_pid()
    if _turbo_running():
        print("turbo on")
    elif pid:
        print("window open, turbo off")
    else:
        print("turbo off")
    status = state_dir() / "status.json"
    if status.exists():
        print(status.read_text().strip())
    print(f"shortcut: {shortcut_path()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "status"
    try:
        if command == "install":
            return cmd_install()
        if command == "plan":
            return cmd_plan()
        if command == "run":
            return cmd_run()
        if command == "start":
            return cmd_start()
        if command == "window":
            return cmd_window()
        if command == "stop":
            return cmd_stop()
        if command == "toggle":
            return cmd_toggle()
        if command == "recover":
            return cmd_recover()
        if command == "status":
            return cmd_status()
        print("commands: install, plan, toggle, start, stop, status, recover, run, window")
        return 2
    except DiscoverError as exc:
        print(f"nomad-turbo: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
