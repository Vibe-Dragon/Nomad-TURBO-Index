"""One llama-server for the whole run.

Stock NOMAD queues small embed jobs inside the admin container. Turbo does
not do that. The slider changes how many chunks the next embed call sends.
It does not start or stop servers. A batch that is already in flight
finishes on the same server. The width comes from embed_call_budget.
"""

from __future__ import annotations

import logging
import time

from nomad_turbo.discover import Install
from nomad_turbo.runtime import gpu_rows, start_embedder, stop_named, wait_embedder
from nomad_turbo.session import Session
from nomad_turbo.slots import embed_call_budget, host_memory_mib, plan_slots, reserve_ratio_from_percent

log = logging.getLogger("nomad_turbo")


def should_change_slots(batch_open: bool, active: int, desired: int) -> bool:
    """Older helper. The live indexer does not add or remove servers mid-run."""
    if batch_open:
        return False
    return int(active) != int(desired)


def target_slots(
    *,
    percent: int,
    total_mib: int,
    used_before: int,
    per_slot_mib: int,
    cpu_count: int,
    max_slots: int,
    forced: int | None,
    use_gpu: bool,
    gpu_index: int | None,
    already_running: bool,
    host_total_mib: int | None = None,
    host_available_mib: int | None = None,
    host_per_slot_mib: int | None = None,
) -> int:
    """Older process count. Kept so the planner tests still describe a card."""
    plan = plan_slots(
        total_mib=total_mib,
        used_mib=used_before,
        reserve_ratio=reserve_ratio_from_percent(percent),
        per_slot_mib=per_slot_mib,
        cpu_count=cpu_count,
        max_slots=max_slots,
        forced=forced,
        has_gpu=use_gpu and total_mib > 0,
        gpu_index=gpu_index,
        host_total_mib=host_total_mib,
        host_available_mib=host_available_mib,
        host_per_slot_mib=host_per_slot_mib,
    )
    if already_running and plan.use_gpu != use_gpu:
        return 1
    return max(1, plan.slots)


class EmbedPool:
    def __init__(
        self,
        install: Install,
        *,
        use_gpu: bool,
        gpu_index: int | None,
        total_mib: int,
        used_before: int,
        per_slot_mib: int,
        cpu_count: int,
    ):
        self.install = install
        self.use_gpu = use_gpu
        self.gpu_index = gpu_index
        self.total_mib = int(total_mib)
        self.used_before = int(used_before)
        self.per_slot_mib = int(per_slot_mib)
        self.cpu_count = int(cpu_count)
        self.measured: int | None = None
        self.urls: list[str] = []
        self.names: list[str] = []
        memory = host_memory_mib()
        if memory is None:
            self.host_total_mib: int | None = None
            self.host_available_mib: int | None = None
        else:
            self.host_total_mib, self.host_available_mib = memory
        if install.slots:
            log.info("config slots=%s is ignored. Turbo keeps one embed server.", install.slots)

    def batch_width(self, percent: int) -> int:
        """Chunks the next call sends. One server. The width follows the computer running Turbo."""
        gpu_total = self.total_mib if self.use_gpu and self.total_mib > 0 else None
        return embed_call_budget(
            percent,
            gpu_total_mib=gpu_total,
            host_total_mib=self.host_total_mib or 0,
            cpu_count=self.cpu_count,
        )

    def apply_target(self, percent: int, batch_open: bool) -> tuple[int, int]:
        """Start the one server if it is down. Never scale it with the slider."""
        del batch_open
        width = self.batch_width(percent)
        if not self.urls:
            self._start_one()
            where = "GPU" if self.use_gpu and self.gpu_index is not None else "CPU"
            log.info("one embed server is up on %s, %s chunks per call at %s%%", where, width, percent)
        return len(self.urls), width

    def publish(self, session: Session | None) -> None:
        if session is None:
            return
        session.configure_gpu(
            total=self.total_mib,
            used_before=self.used_before,
            per_slot=self.per_slot_mib,
            measured=self.measured,
            use_gpu=self.use_gpu,
            gpu_index=self.gpu_index,
            active=len(self.urls),
            forced=self.install.slots,
            max_slots=self.install.max_slots,
            host_total=self.host_total_mib or 0,
            host_available=self.host_available_mib or 0,
        )

    def stop_all(self) -> None:
        stop_named(list(self.names))
        self.names.clear()
        self.urls.clear()

    def _start_one(self) -> None:
        url, name = start_embedder(self.install, 1, self.use_gpu, self.gpu_index)
        try:
            wait_embedder(url, self.install.embed_model)
        except Exception:
            stop_named([name])
            raise
        self.urls.append(url)
        self.names.append(name)
        if self.use_gpu and self.measured is None and self.gpu_index is not None:
            time.sleep(1)
            for row_index, row_total, row_used in gpu_rows():
                if row_index == self.gpu_index:
                    delta = int(row_used) - self.used_before
                    if delta >= 256:
                        self.measured = delta
                        self.total_mib = int(row_total)
                        log.info("embed server uses %s MiB of video memory", delta)
                    break
