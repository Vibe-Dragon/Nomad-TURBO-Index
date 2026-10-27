"""Host meters for the separate window.

CPU, memory, GPU, video memory, and each disk the library uses come from the
machine Turbo is running on. Nothing here names a GPU model or a disk model.
A missing temperature or a missing GPU stays blank. It is not drawn as zero.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from nomad_turbo.slots import pick_gpu

log = logging.getLogger("nomad_turbo")

HISTORY = 180
_GPU_QUERY = (
    "nvidia-smi",
    "--query-gpu=index,utilization.gpu,temperature.gpu,memory.total,memory.used",
    "--format=csv,noheader,nounits",
)
_PREFERRED_CHIPS = ("k10temp", "coretemp", "zenpower")
_PREFERRED_LABELS = ("tctl", "tdie", "package id 0")


@dataclass
class GpuRow:
    index: int
    util: float | None
    temp: float | None
    total_mib: int
    used_mib: int


@dataclass
class HostPoint:
    cpu: float | None = None
    cpu_temp: float | None = None
    ram_percent: float | None = None
    ram_used: int | None = None
    ram_total: int | None = None
    gpu_present: bool = False
    gpu: float | None = None
    gpu_temp: float | None = None
    vram_percent: float | None = None
    vram_used: int | None = None
    vram_total: int | None = None
    disks: tuple["DiskMeter", ...] = ()


@dataclass(frozen=True)
class DiskMeter:
    """One filesystem the library uses. Numbers, not a graph."""

    label: str
    used: int | None = None
    total: int | None = None
    read_mbs: float | None = None
    write_mbs: float | None = None


def _number(text: str) -> float | None:
    token = text.strip()
    if not token or token.startswith("["):
        return None
    try:
        return float(token)
    except ValueError:
        return None


def cpu_times(text: str) -> tuple[int, int] | None:
    """Total and idle jiffies from /proc/stat. Idle includes iowait."""
    for line in text.splitlines():
        if not line.startswith("cpu "):
            continue
        nums = []
        for part in line.split()[1:]:
            try:
                nums.append(int(part))
            except ValueError:
                return None
        if len(nums) < 4:
            return None
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        return sum(nums), idle
    return None


def cpu_percent(previous: tuple[int, int] | None, current: tuple[int, int] | None) -> float | None:
    if previous is None or current is None:
        return None
    total = current[0] - previous[0]
    idle = current[1] - previous[1]
    if total <= 0:
        return None
    return max(0.0, min(100.0, (total - idle) * 100.0 / total))


def mem_bytes(text: str) -> tuple[int, int] | None:
    """Used and total bytes. Used is total minus available."""
    total = available = None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            total = int(line.split()[1]) * 1024
        elif line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
    if not total:
        return None
    used = total - (available if available is not None else 0)
    return max(0, used), total


def parse_gpu_csv(text: str) -> list[GpuRow]:
    rows = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            index = int(float(parts[0]))
            total = int(float(parts[3]))
            used = int(float(parts[4]))
        except ValueError:
            continue
        rows.append(GpuRow(index, _number(parts[1]), _number(parts[2]), total, used))
    return rows


def choose_gpu(rows: list[GpuRow]) -> GpuRow | None:
    """The card with the most free memory, matching the indexer."""
    if not rows:
        return None
    picked = pick_gpu([(row.index, row.total_mib, row.used_mib) for row in rows])
    if picked is None:
        return rows[0]
    for row in rows:
        if row.index == picked[0]:
            return row
    return rows[0]


def pick_cpu_temp(chips: list[tuple[str, list[tuple[str, int]]]]) -> float | None:
    """Celsius from hwmon chips. Prefer a CPU package sensor. Skip the rest."""
    ordered = sorted(chips, key=lambda item: 0 if item[0] in _PREFERRED_CHIPS else 1)
    for _name, sensors in ordered:
        for label, milli in sensors:
            if label.strip().lower() in _PREFERRED_LABELS:
                return milli / 1000.0
    for name, sensors in ordered:
        if name in _PREFERRED_CHIPS and sensors:
            return sensors[0][1] / 1000.0
    return None


def _unescape_mount(text: str) -> str:
    return (
        text.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _mount_rows(mountinfo: str) -> list[tuple[str, str, str]]:
    """Mount point, major:minor, and device basename from /proc/self/mountinfo."""
    rows = []
    for line in mountinfo.splitlines():
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        fields = left.split()
        if len(fields) < 5 or ":" not in fields[2]:
            continue
        source = right.split()
        if len(source) < 2:
            continue
        point = _unescape_mount(fields[4])
        device = _unescape_mount(source[1]).rsplit("/", 1)[-1]
        if point and device:
            rows.append((point, fields[2], device))
    return rows


def mount_table(mountinfo: str) -> list[tuple[str, str]]:
    """Mount point and kernel device name from /proc/self/mountinfo."""
    return [(point, device) for point, _ident, device in _mount_rows(mountinfo)]


def device_for_mount(path: str, mountinfo: str) -> str | None:
    """Kernel device name for the longest mount that holds path."""
    best = ""
    device = None
    for point, name in mount_table(mountinfo):
        if point != path and not path.startswith(point.rstrip("/") + "/"):
            continue
        if len(point) < len(best):
            continue
        best = point
        device = name
    return device


def storage_disks(storage: str, mountinfo: str) -> list[tuple[str, str, str]]:
    """Label, path, and major:minor for each filesystem under the library.

    The storage directory is Library. A folder mounted from a different
    device gets its own line, named for that folder. Folders on one device
    share the line. One disk for the whole library stays one line. The device
    key is major:minor so the name in mountinfo does not have to match
    /proc/diskstats.
    """
    root = storage.rstrip("/") or "/"
    table = _mount_rows(mountinfo)
    root_ident = None
    best = ""
    for point, ident, _device in table:
        if point != root and not root.startswith(point.rstrip("/") + "/"):
            continue
        if len(point) >= len(best):
            best = point
            root_ident = ident
    prefix = "/" if root == "/" else root + "/"
    grouped: dict[str, list[tuple[str, str]]] = {}
    for point, ident, _device in table:
        if ident == root_ident or not point.startswith(prefix):
            continue
        name = point[len(prefix):].split("/")[0]
        if name:
            grouped.setdefault(ident, []).append((name, point))
    rows: list[tuple[str, str, str]] = []
    if root_ident:
        rows.append(("Library", root, root_ident))
    others = []
    for ident, items in grouped.items():
        names = sorted({label for label, _path in items})
        path = sorted(item[1] for item in items)[0]
        others.append((", ".join(names), path, ident))
    others.sort(key=lambda item: item[0])
    return rows + others


def disk_sectors(text: str, name: str) -> tuple[int, int] | None:
    """Read and write sectors for one device in /proc/diskstats.

    name is the diskstats device name, or major:minor from mountinfo.
    """
    major = minor = None
    if ":" in name:
        left, right = name.split(":", 1)
        if left.isdigit() and right.isdigit():
            major, minor = left, right
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 10:
            continue
        if parts[2] != name and not (major is not None and parts[0] == major and parts[1] == minor):
            continue
        try:
            return int(parts[5]), int(parts[9])
        except ValueError:
            return None
    return None


def io_megabytes(
    previous: tuple[int, int] | None,
    current: tuple[int, int] | None,
    elapsed: float | None,
) -> tuple[float, float] | None:
    """Read and write MB/s. Sectors are 512 bytes. The first pair has no rate."""
    if previous is None or current is None or not elapsed or elapsed <= 0:
        return None
    read = max(0, current[0] - previous[0]) * 512 / elapsed / 1_000_000
    write = max(0, current[1] - previous[1]) * 512 / elapsed / 1_000_000
    return read, write


def disk_usage(path: str) -> tuple[int, int] | None:
    try:
        stat = os.statvfs(path)
    except OSError:
        return None
    if not stat.f_blocks or not stat.f_frsize:
        return None
    total = stat.f_blocks * stat.f_frsize
    used = total - stat.f_bfree * stat.f_frsize
    return max(0, used), total


def format_bytes(size: int) -> str:
    if size >= 1 << 40:
        return f"{size / (1 << 40):.1f} TiB"
    if size >= 1 << 30:
        return f"{size / (1 << 30):.1f} GiB"
    if size >= 1 << 20:
        return f"{size / (1 << 20):.0f} MiB"
    return f"{size} B"


def _percent(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{round(value)}%"


def _rate(value: float) -> str:
    if value >= 100:
        return f"{value:.0f} MB/s"
    if value >= 10:
        return f"{value:.1f} MB/s"
    return f"{value:.2f} MB/s"


def cpu_caption(point: HostPoint | None) -> str:
    if point is None:
        return "CPU"
    text = f"CPU {_percent(point.cpu)}"
    if point.cpu_temp is not None:
        text += f" · {round(point.cpu_temp)}°C"
    return text


def ram_caption(point: HostPoint | None) -> str:
    if point is None or point.ram_used is None or point.ram_total is None:
        return "RAM"
    return f"RAM {format_bytes(point.ram_used)} / {format_bytes(point.ram_total)}"


def gpu_caption(point: HostPoint | None) -> str:
    if point is None:
        return "GPU"
    if not point.gpu_present:
        return "No GPU detected. The embed server will use the CPU."
    text = f"GPU {_percent(point.gpu)}"
    if point.gpu_temp is not None:
        text += f" · {round(point.gpu_temp)}°C"
    return text


def vram_caption(point: HostPoint | None) -> str:
    if point is None or not point.gpu_present or point.vram_used is None or point.vram_total is None:
        return "vRAM —"
    return f"vRAM {point.vram_used:,} / {point.vram_total:,} MiB"


def disk_caption(point: HostPoint | None) -> str:
    """One line per filesystem: name, used / size, read, and write."""
    if point is None or not point.disks:
        return "Disk"
    lines = []
    for disk in point.disks:
        if disk.used is None or disk.total is None:
            capacity = "—"
        else:
            capacity = f"{format_bytes(disk.used)} / {format_bytes(disk.total)}"
        if disk.read_mbs is None or disk.write_mbs is None:
            speed = "Read — · Write —"
        else:
            speed = f"Read {_rate(disk.read_mbs)} · Write {_rate(disk.write_mbs)}"
        lines.append(f"{disk.label}  {capacity}   {speed}")
    return "\n".join(lines)


def measure_disks(
    rows: list[tuple[str, str, str]],
    diskstats: str,
    previous: dict[str, tuple[int, int]],
    elapsed: float | None,
) -> tuple[tuple[DiskMeter, ...], dict[str, tuple[int, int]]]:
    """Capacity and speed for each filesystem. The first sample has no speed."""
    current: dict[str, tuple[int, int]] = {}
    meters = []
    for label, path, device in rows:
        usage = disk_usage(path)
        sectors = disk_sectors(diskstats, device)
        if sectors is not None:
            current[device] = sectors
        rates = io_megabytes(previous.get(device), sectors, elapsed)
        meters.append(
            DiskMeter(
                label=label,
                used=None if usage is None else usage[0],
                total=None if usage is None else usage[1],
                read_mbs=None if rates is None else rates[0],
                write_mbs=None if rates is None else rates[1],
            )
        )
    return tuple(meters), current


def plot_segments(
    values: list[float | None],
    width: int,
    height: int,
    low: float,
    high: float,
) -> list[list[tuple[float, float]]]:
    """Map a series onto the widget. The newest sample is at the right.

    A missing sample breaks the line. Values are kept inside low and high.
    """
    if width < 4 or height < 4 or not values:
        return []
    span = high - low
    if span <= 0:
        span = 1.0
    count = len(values)
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for index, value in enumerate(values):
        if value is None:
            if current:
                segments.append(current)
                current = []
            continue
        clamped = min(high, max(low, float(value)))
        if count == 1:
            x = width - 2
        else:
            x = 2 + (width - 4) * index / (count - 1)
        y = 2 + (height - 4) * (1 - (clamped - low) / span)
        current.append((x, y))
    if current:
        segments.append(current)
    return segments


def assemble(
    *,
    previous_cpu: tuple[int, int] | None,
    cpu_text: str,
    mem_text: str,
    cpu_temp: float | None,
    gpus: list[GpuRow],
    disks: tuple[DiskMeter, ...] = (),
) -> tuple[HostPoint, tuple[int, int] | None]:
    """One meter reading, plus the CPU counters the next reading needs."""
    cpu_now = cpu_times(cpu_text)
    memory = mem_bytes(mem_text)
    card = choose_gpu(gpus)
    ram_percent = None
    ram_used = ram_total = None
    if memory is not None and memory[1] > 0:
        ram_used, ram_total = memory
        ram_percent = ram_used * 100.0 / ram_total
    point = HostPoint(
        cpu=cpu_percent(previous_cpu, cpu_now),
        cpu_temp=cpu_temp,
        ram_percent=ram_percent,
        ram_used=ram_used,
        ram_total=ram_total,
        gpu_present=card is not None,
        gpu=None if card is None else card.util,
        gpu_temp=None if card is None else card.temp,
        vram_percent=None
        if card is None or card.total_mib <= 0
        else card.used_mib * 100.0 / card.total_mib,
        vram_used=None if card is None else card.used_mib,
        vram_total=None if card is None else card.total_mib,
        disks=tuple(disks),
    )
    return point, cpu_now


def read_cpu_temp() -> float | None:
    root = Path("/sys/class/hwmon")
    if not root.is_dir():
        return None
    chips: list[tuple[str, list[tuple[str, int]]]] = []
    for node in root.iterdir():
        try:
            name = (node / "name").read_text(encoding="ascii").strip()
        except OSError:
            continue
        sensors: list[tuple[str, int]] = []
        for label_path in sorted(node.glob("temp*_label")):
            stem = label_path.name[: -len("_label")]
            try:
                label = label_path.read_text(encoding="ascii").strip()
                milli = int((node / f"{stem}_input").read_text(encoding="ascii").strip())
            except (OSError, ValueError):
                continue
            sensors.append((label, milli))
        if not sensors:
            for input_path in sorted(node.glob("temp*_input")):
                try:
                    milli = int(input_path.read_text(encoding="ascii").strip())
                except (OSError, ValueError):
                    continue
                sensors.append(("", milli))
                break
        chips.append((name, sensors))
    return pick_cpu_temp(chips)


def query_gpus() -> list[GpuRow]:
    try:
        out = subprocess.check_output(list(_GPU_QUERY), text=True, stderr=subprocess.DEVNULL, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_gpu_csv(out)


class HostSampler:
    """One background reading a second. The window copies the history."""

    def __init__(self, storage: str) -> None:
        self.storage = storage
        self.history: deque[HostPoint] = deque(maxlen=HISTORY)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu: tuple[int, int] | None = None
        self._sectors: dict[str, tuple[int, int]] = {}
        self._stamp: float | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="nomad-turbo-host", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> list[HostPoint]:
        with self._lock:
            return list(self.history)

    def _loop(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._once()
            except Exception:
                log.debug("host sample failed", exc_info=True)
            remaining = 1.0 - (time.monotonic() - started)
            if self._stop.wait(max(0.0, remaining)):
                return

    def _once(self) -> None:
        now = time.monotonic()
        elapsed = None if self._stamp is None else now - self._stamp
        self._stamp = now
        try:
            cpu_text = Path("/proc/stat").read_text(encoding="ascii")
        except OSError:
            cpu_text = ""
        try:
            mem_text = Path("/proc/meminfo").read_text(encoding="ascii")
        except OSError:
            mem_text = ""
        try:
            mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
        except OSError:
            mountinfo = ""
        try:
            diskstats = Path("/proc/diskstats").read_text(encoding="ascii")
        except OSError:
            diskstats = ""
        disks, self._sectors = measure_disks(
            storage_disks(self.storage, mountinfo),
            diskstats,
            self._sectors,
            elapsed,
        )
        point, self._cpu = assemble(
            previous_cpu=self._cpu,
            cpu_text=cpu_text,
            mem_text=mem_text,
            cpu_temp=read_cpu_temp(),
            gpus=query_gpus(),
            disks=disks,
        )
        with self._lock:
            self.history.append(point)
