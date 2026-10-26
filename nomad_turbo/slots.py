"""Call size for one embed server, plus the older process-count math.

Stock NOMAD embeds inside the admin container, a few chunks at a time.
Turbo keeps a single llama-server and sizes the next call from the hardware
it detects. embed_call_budget is that size. plan_slots is the older process
count. The indexer does not start one server per slot.

Nothing here names a GPU model. The same inputs always return the same count,
on every machine.
"""

from __future__ import annotations

from dataclasses import dataclass

# Held back even when indexing is allowed to use the whole card.
# Covers CUDA context growth and allocator fragmentation on top of memory
# that is already in use. The floor matters on small cards; the ratio matters
# on large ones. Neither number is a particular GPU.
OVERHEAD_FLOOR_MIB = 512
OVERHEAD_RATIO = 0.04

# Each embed process keeps its own copy of the model in system RAM, on top of
# the GPU copy. Leave this much RAM for NOMAD, Qdrant, and the file reader.
# The floor matters on small machines; the ratio matters on large ones.
HOST_FLOOR_MIN_MIB = 2048
HOST_FLOOR_RATIO = 0.25

# MiB kept for the embed model beside the batch. This is the weight file
# and a little runtime, not a graphics card.
MODEL_RESERVE_MIB = 768

# Smallest call worth sending. It is a floor. It is not a ceiling.
# A larger card still gets a larger 100% call.
MIN_CALL_CHUNKS = 8

# CPU path: host RAM one extra chunk needs, and how many chunks one core
# can chew at the safe 100% size. Memory and cores both cap the call.
CPU_MIB_PER_CHUNK = 256
CPU_CHUNKS_PER_CORE = 4


@dataclass(frozen=True)
class SlotPlan:
    slots: int
    use_gpu: bool
    gpu_index: int | None
    reason: str


def gpu_overhead_mib(total_mib: int) -> int:
    total = max(0, int(total_mib))
    return max(OVERHEAD_FLOOR_MIB, int(total * OVERHEAD_RATIO))


def dedicate_percent(reserve_ratio: float) -> int:
    """Share of the card indexing may use, as a percent from 10 to 100."""
    ratio = min(0.9, max(0.0, float(reserve_ratio)))
    return int(round((1.0 - ratio) * 100))


def reserve_ratio_from_percent(percent: int) -> float:
    """Chat reserve left over after the indexing share. 100% leaves no reserve.

    The embed-call slider can go to 110%. That overclock is stored on
    Install.boost_percent, not as a negative reserve. This helper still
    clamps to 10–100 for the older process-count math.
    """
    pct = min(100, max(10, int(percent)))
    return (100 - pct) / 100.0


def clamp_boost_percent(percent: int) -> int:
    """Slider percent. 10 through 100 stay inside the budget. 110 is the overclock."""
    return min(110, max(10, int(percent)))


def install_boost_percent(install: object) -> int:
    """Saved slider percent.

    A config that never stored boost_percent stays at or below 100.
    110% is not the default for a new install.
    """
    raw = getattr(install, "boost_percent", None)
    if raw is None:
        ratio = getattr(install, "reserve_ratio", 0.45)
        try:
            return dedicate_percent(float(ratio))
        except (TypeError, ValueError):
            return dedicate_percent(0.45)
    try:
        return clamp_boost_percent(int(raw))
    except (TypeError, ValueError):
        return dedicate_percent(0.45)


def _isqrt(value: int) -> int:
    value = max(0, int(value))
    if value <= 0:
        return 0
    guess = int(value ** 0.5)
    while (guess + 1) * (guess + 1) <= value:
        guess += 1
    while guess * guess > value:
        guess -= 1
    return guess


def safe_call_chunks(*, gpu_total_mib: int | None, host_total_mib: int, cpu_count: int) -> int:
    """Safe 100% chunk count for this machine.

    GPU memory wins when a GPU is present. Otherwise the count comes from
    system RAM and the CPU count. The result is finite, and a bigger memory
    budget is never a smaller call.
    """
    gpu = 0 if gpu_total_mib is None else int(gpu_total_mib)
    if gpu > 0:
        usable = max(0, gpu - MODEL_RESERVE_MIB - gpu_overhead_mib(gpu))
        # Half the square root of the memory left after the model. The call
        # grows with the card and does not stop at one shared ceiling.
        return max(MIN_CALL_CHUNKS, _isqrt(usable) // 2)
    total = max(0, int(host_total_mib or 0))
    cores = max(1, int(cpu_count or 1))
    usable = max(0, total - host_floor_mib(total))
    from_ram = max(MIN_CALL_CHUNKS, usable // CPU_MIB_PER_CHUNK)
    from_cpu = max(MIN_CALL_CHUNKS, cores * CPU_CHUNKS_PER_CORE)
    return min(from_ram, from_cpu)


def embed_call_budget(
    percent: int,
    *,
    gpu_total_mib: int | None,
    host_total_mib: int,
    cpu_count: int,
) -> int:
    """Chunks one embed call should carry on this machine.

    100% is the safe size from GPU memory when a GPU is present, otherwise
    from system RAM and the CPU count. 10% through 100% scale that size.
    110% is an overclock: strictly more chunks than 100% on the same inputs.
    The same inputs always return the same count. There is no single chunk
    ceiling for every machine, and no GPU model name is used.
    """
    full = safe_call_chunks(
        gpu_total_mib=gpu_total_mib,
        host_total_mib=host_total_mib,
        cpu_count=cpu_count,
    )
    pct = clamp_boost_percent(percent)
    if pct > 100:
        extra = max(1, full * (pct - 100) // 100)
        return full + extra
    if full <= MIN_CALL_CHUNKS:
        return full
    return MIN_CALL_CHUNKS + (pct - 10) * (full - MIN_CALL_CHUNKS) // 90


def host_floor_mib(total_mib: int) -> int:
    """RAM kept back from embed processes, for the rest of NOMAD."""
    total = max(0, int(total_mib))
    return max(HOST_FLOOR_MIN_MIB, int(total * HOST_FLOOR_RATIO))


def host_memory_mib() -> tuple[int, int] | None:
    """Current (total, available) system RAM in MiB, or None if it cannot be read."""
    total = available = None
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        if parts[0] == "MemTotal:":
            total = int(parts[1]) // 1024
        elif parts[0] == "MemAvailable:":
            available = int(parts[1]) // 1024
    if not total or available is None:
        return None
    return total, available


def host_slot_limit(total_mib: int, available_mib: int, per_slot_mib: int) -> int:
    """How many embed processes fit in system RAM while leaving the floor free.

    The reading is memory available before those processes start. Their own
    cost is per_slot_mib, which must not be replaced by the smaller GPU reading.
    """
    room = int(available_mib) - host_floor_mib(total_mib)
    per = max(256, int(per_slot_mib))
    if room < per:
        return 1
    return max(1, room // per)


def _apply_host_cap(
    count: int,
    reason: str,
    *,
    per_slot_mib: int,
    host_total_mib: int | None,
    host_available_mib: int | None,
    host_per_slot_mib: int | None,
) -> tuple[int, str]:
    if host_total_mib is None or host_available_mib is None:
        return count, reason
    per = per_slot_mib if host_per_slot_mib is None else host_per_slot_mib
    cap = host_slot_limit(int(host_total_mib), int(host_available_mib), int(per))
    if count <= cap:
        return count, reason
    floor = host_floor_mib(int(host_total_mib))
    return cap, f"{reason}, capped at {cap} by system RAM ({floor} MiB kept free)"


def plan_slots(
    *,
    total_mib: int,
    used_mib: int,
    reserve_ratio: float,
    per_slot_mib: int,
    cpu_count: int,
    max_slots: int,
    forced: int | None = None,
    has_gpu: bool = True,
    gpu_index: int | None = 0,
    host_total_mib: int | None = None,
    host_available_mib: int | None = None,
    host_per_slot_mib: int | None = None,
) -> SlotPlan:
    host = dict(
        per_slot_mib=per_slot_mib,
        host_total_mib=host_total_mib,
        host_available_mib=host_available_mib,
        host_per_slot_mib=host_per_slot_mib,
    )
    if forced is not None and int(forced) > 0:
        count, reason = _apply_host_cap(int(forced), "config slots", **host)
        return SlotPlan(count, has_gpu and total_mib > 0, gpu_index, reason)

    cpus = max(1, int(cpu_count or 1))
    cap = max(1, int(max_slots or 1))
    if not has_gpu or total_mib <= 0:
        cpu_slots = max(1, min(cap, max(1, cpus // 2)))
        count, reason = _apply_host_cap(cpu_slots, "no gpu, cpu slots", **host)
        return SlotPlan(count, False, None, reason)

    ratio = min(0.9, max(0.0, float(reserve_ratio)))
    reserve = int(total_mib * ratio)
    overhead = gpu_overhead_mib(total_mib)
    per = max(256, int(per_slot_mib))
    free_now = int(total_mib) - int(used_mib)
    # CPU count does not cap GPU slots. Embedding is limited by video memory,
    # and a 100% choice has to be able to fill whatever card is installed.
    budget = free_now - reserve - overhead
    dedicate = 1.0 - ratio
    if free_now < per or budget < per:
        if ratio > 0 and free_now >= per:
            count, reason = _apply_host_cap(
                1, "one slot, the chat reserve does not leave room for another", **host
            )
            return SlotPlan(count, True, gpu_index, reason)
        cpu_slots = max(1, min(cap, max(1, cpus // 2)))
        why = "gpu is too full for another embed model" if free_now < per else "gpu headroom is inside the overhead margin"
        count, reason = _apply_host_cap(cpu_slots, why, **host)
        return SlotPlan(count, False, None, reason)
    raw = budget // per
    count = max(1, min(cap, raw))
    reason = f"{count} slots, {dedicate:.0%} of the GPU for indexing, {overhead} MiB held for overhead"
    if raw > cap:
        reason += ", capped by max_slots"
    count, reason = _apply_host_cap(count, reason, **host)
    return SlotPlan(count, True, gpu_index, reason)


def slots_after_measure(
    *,
    planned: int,
    total_mib: int,
    used_before_mib: int,
    used_after_mib: int,
    reserve_ratio: float,
    max_slots: int,
) -> int:
    """Fit slots once the first process has loaded.

    The budget uses memory from before that process started, so its own
    allocation is counted once. A tiny or negative delta means the reading
    is not usable and the original plan stands.
    """
    cap = max(1, int(max_slots or 1))
    planned_count = max(1, min(cap, int(planned)))
    delta = int(used_after_mib) - int(used_before_mib)
    if delta < 256 or int(total_mib) <= 0:
        return planned_count
    ratio = min(0.9, max(0.0, float(reserve_ratio)))
    reserve = int(total_mib * ratio)
    overhead = gpu_overhead_mib(total_mib)
    budget = int(total_mib) - int(used_before_mib) - reserve - overhead
    if budget < delta:
        return 1
    return max(1, min(cap, budget // delta))


def pick_gpu(rows: list[tuple[int, int, int]]) -> tuple[int, int, int] | None:
    """Choose the GPU with the most free memory.

    Each row is (index, total_mib, used_mib).
    """
    best: tuple[int, int, int] | None = None
    best_free = -1
    for index, total, used in rows:
        free = int(total) - int(used)
        if free > best_free:
            best_free = free
            best = (index, int(total), int(used))
    return best
