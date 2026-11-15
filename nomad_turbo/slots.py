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

# Older percent slider. 10 through 100 stay inside the detected budget.
# 150 is only the old top of that slider. A saved percent still converts
# through it. The chunk count Auto uses is not limited to 150.
MIN_BOOST_PERCENT = 10
MAX_BOOST_PERCENT = 150
# Typed or saved chunk counts. One chunk is the floor. The high cap only
# rejects an absurd number. It is not a performance ceiling.
MIN_EMBED_CHUNKS = 1
MAX_EMBED_CHUNKS = 100_000

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

# One llama-server. A second sequence is another slot in that server, not
# another process. Each slot keeps this context. The total context is the
# slot count times this number, so a long chunk is not squeezed into a
# smaller window. 256 MiB is the room one extra context must have. It is
# not a graphics card.
SEQUENCE_CONTEXT = 2048
SEQUENCE_MIB = 256
# A call this long is the one that must not get slower. Shorter calls are
# the ones a second sequence can share a forward with.
LONG_CALL_TOKENS = 256
SEQUENCE_SAMPLES = 6
LONG_CALL_SAMPLES = 3


@dataclass(frozen=True)
class SlotPlan:
    slots: int
    use_gpu: bool
    gpu_index: int | None
    reason: str


def embed_server_shape(sequences: int, ctx_per_sequence: int = SEQUENCE_CONTEXT) -> tuple[int, int]:
    """(slot count, total context). Each slot keeps the full context."""
    count = 2 if int(sequences) >= 2 else 1
    ctx = max(1, int(ctx_per_sequence))
    return count, count * ctx


def embed_sequences(use_gpu: bool, free_mib: int, per_sequence_mib: int = SEQUENCE_MIB) -> int:
    """One sequence, or two when the free memory can hold another context.

    CPU stays at one. The count is not a particular card.
    """
    if not use_gpu:
        return 1
    cost = int(per_sequence_mib)
    if cost <= 0 or int(free_mib) < cost:
        return 1
    return 2


class SequencePace:
    """When a second sequence may stay.

    Long calls have to be at least as fast as they were with one sequence.
    Three long calls at two sequences, and none at one, ask once for a
    one-sequence baseline. When both widths have three long calls, the faster
    width is kept for the rest of the run. Later files do not rebuild the
    server. Rebuilding reloads the model, and llama-server answers 503 until
    that load finishes.
    """

    def __init__(self) -> None:
        self.one_long: list[float] = []
        self.many_long: list[float] = []
        self.one_short: list[float] = []
        self.many_short: list[float] = []
        self.calls_at_width = 0
        self.locked_at_one = False
        self.tried_two = False
        self._need_baseline = False
        self._asked_baseline = False
        self._asked_up = False
        self._asked_slower = False
        self._held: int | None = None
        self._file: str | None = None

    def begin_file(self, source: str) -> None:
        """Remember this file. A new file keeps the measurement already underway.

        Wiping it on every file reloaded llama-server twice per document.
        Each reload answers 503 until the model is back, and a short file
        ends before the comparison can finish, so the next file starts it again.
        """
        self._file = str(source or "")

    def note(self, sequences: int, mean_tokens: int, chunks: int, seconds: float) -> None:
        self.calls_at_width += 1
        if seconds <= 0 or chunks <= 0:
            return
        pace = float(chunks) / float(seconds)
        long = int(mean_tokens) >= LONG_CALL_TOKENS
        if long:
            bucket = self.one_long if int(sequences) <= 1 else self.many_long
        else:
            bucket = self.one_short if int(sequences) <= 1 else self.many_short
        bucket.append(pace)

    def mark_resized(self) -> None:
        self.calls_at_width = 0
        self._asked_baseline = False
        self._asked_up = False
        self._asked_slower = False

    def decision(self, current: int, fits: bool) -> tuple[int, str]:
        # A failed restart sets this. Leave the server where it is.
        if self.locked_at_one:
            return 1, ""
        current_n = int(current)
        if len(self.one_long) >= LONG_CALL_SAMPLES:
            self._need_baseline = False
        if (
            current_n >= 2
            and len(self.many_long) >= LONG_CALL_SAMPLES
            and len(self.one_long) < LONG_CALL_SAMPLES
        ):
            self._need_baseline = True
            if not self._asked_baseline:
                self._asked_baseline = True
                return 1, "baseline"
            return current_n, ""
        if self._need_baseline and len(self.one_long) < LONG_CALL_SAMPLES:
            return 1, ""
        self._freeze()
        if self._held is not None:
            return self._held_decision(current_n, fits)
        if current_n <= 1:
            if fits and self.calls_at_width >= SEQUENCE_SAMPLES and not self._asked_up:
                self._asked_up = True
                self.tried_two = True
                return 2, "up"
            return 1, ""
        return current_n, ""

    def _freeze(self) -> None:
        """Keep the faster long-call width once both widths have three samples."""
        if self._held is not None:
            return
        if len(self.one_long) < LONG_CALL_SAMPLES or len(self.many_long) < LONG_CALL_SAMPLES:
            return
        one = sum(self.one_long[:LONG_CALL_SAMPLES]) / LONG_CALL_SAMPLES
        many = sum(self.many_long[:LONG_CALL_SAMPLES]) / LONG_CALL_SAMPLES
        self._held = 1 if many < one else 2

    def _held_decision(self, current_n: int, fits: bool) -> tuple[int, str]:
        held = int(self._held or 1)
        if current_n >= 2 and held == 1:
            if not self._asked_slower:
                self._asked_slower = True
                return 1, "slower"
            return 1, ""
        if current_n <= 1 and held == 2:
            if fits and not self._asked_up:
                self._asked_up = True
                self.tried_two = True
                return 2, "up"
            return 1, ""
        return current_n, ""


def gpu_overhead_mib(total_mib: int) -> int:
    total = max(0, int(total_mib))
    return max(OVERHEAD_FLOOR_MIB, int(total * OVERHEAD_RATIO))


def dedicate_percent(reserve_ratio: float) -> int:
    """Share of the card indexing may use, as a percent from 10 to 100."""
    ratio = min(0.9, max(0.0, float(reserve_ratio)))
    return int(round((1.0 - ratio) * 100))


def reserve_ratio_from_percent(percent: int) -> float:
    """Chat reserve left over after the indexing share. 100% leaves no reserve.

    The old percent slider could go to 150%. That extra share is stored on
    Install.boost_percent, not as a negative reserve. This helper still
    clamps to 10–100 for the older process-count math. It does not change
    GPU clocks.
    """
    pct = min(100, max(10, int(percent)))
    return (100 - pct) / 100.0


def clamp_boost_percent(percent: int) -> int:
    """Old slider percent, from 10 through 150. Used only to read a saved percent."""
    return min(MAX_BOOST_PERCENT, max(MIN_BOOST_PERCENT, int(percent)))


def clamp_embed_chunks(count: int) -> int:
    """Chunk count for one embed call.

    The floor is one chunk. The high cap rejects an absurd typed number.
    It is not a limit Auto stops at when a larger count is still faster.
    """
    try:
        value = int(count)
    except (TypeError, ValueError):
        return MIN_EMBED_CHUNKS
    return min(MAX_EMBED_CHUNKS, max(MIN_EMBED_CHUNKS, value))


def install_boost_percent(install: object) -> int:
    """Saved slider percent.

    A config that never stored boost_percent stays at or below 100.
    150% is not the default for a new install.
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
    """Safe 100% chunk count for the computer running Turbo.

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
    """Chunks one embed call should carry on the computer running Turbo.

    100% is the safe size from GPU memory when a GPU is present, otherwise
    from system RAM and the CPU count. 10% through 100% scale that size.
    Above 100% adds chunks past that safe call on the same inputs. It does
    not change GPU clocks. 150% adds half of the safe call, rounded down to
    a whole number. The same inputs always return the same count. There is
    no single chunk ceiling for every machine, and no GPU model name is used.
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


def install_embed_chunks(
    install: object,
    *,
    gpu_total_mib: int | None,
    host_total_mib: int,
    cpu_count: int,
) -> int:
    """Chunk count saved for the next embed call.

    A stored chunk count is used as itself. An older percent is converted
    through the safe-call formula, so the percent figure is not used as a
    raw chunk count.
    """
    raw = getattr(install, "embed_chunks", None)
    if raw is not None:
        try:
            return clamp_embed_chunks(int(raw))
        except (TypeError, ValueError):
            pass
    percent = install_boost_percent(install)
    return embed_call_budget(
        percent,
        gpu_total_mib=gpu_total_mib,
        host_total_mib=host_total_mib,
        cpu_count=cpu_count,
    )


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
