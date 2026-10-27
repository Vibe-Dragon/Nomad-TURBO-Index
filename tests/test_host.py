"""Host meter readings. These tests do not open a window or query a GPU."""

import inspect
import json
import unittest

from nomad_turbo.discover import Install, load_install
from nomad_turbo.host import (
    DiskMeter,
    GpuRow,
    HostPoint,
    assemble,
    choose_gpu,
    cpu_caption,
    cpu_percent,
    cpu_times,
    device_for_mount,
    disk_caption,
    disk_sectors,
    gpu_caption,
    io_megabytes,
    measure_disks,
    mem_bytes,
    parse_gpu_csv,
    pick_cpu_temp,
    plot_segments,
    storage_disks,
    vram_caption,
)


MOUNTINFO = """\
57 34 9:0 / /mnt/nomad rw - xfs /dev/md0 rw
61 34 9:0 /storage /opt/project-nomad/storage rw - xfs /dev/md0 rw
62 61 9:0 /zim /opt/project-nomad/storage/zim rw - xfs /dev/md0 rw
65 61 259:2 /opt/project-nomad/ollama /opt/project-nomad/storage/ollama rw - ext4 /dev/nvme0n1p2 rw
66 61 259:2 /opt/project-nomad/qdrant /opt/project-nomad/storage/qdrant rw - ext4 /dev/mapper/data rw
80 61 8:1 /maps /opt/project-nomad/storage/maps rw - ext4 /dev/sda1 rw
"""


def _install() -> Install:
    return Install(
        admin_container="nomad_admin",
        admin_image="project-nomad:test",
        network="nomad",
        redis_host="redis",
        redis_port=6379,
        redis_container="nomad_redis",
        mysql_container="nomad_mysql",
        qdrant_url="http://127.0.0.1:9",
        ollama_image="ollama/ollama:test",
        ollama_container="nomad_ollama",
        models_dir="/tmp/nomad-turbo-test-models",
        storage_host="/opt/project-nomad/storage",
    )


class HostReadingTest(unittest.TestCase):
    def test_cpu_percent_uses_the_delta_and_skips_the_first_sample(self):
        self.assertEqual(cpu_times("cpu  10 0 10 80 0 0 0\ncpu0 1 0 1 8 0\n"), (100, 80))
        self.assertIsNone(cpu_percent(None, (100, 80)))
        self.assertEqual(cpu_percent((100, 80), (200, 130)), 50.0)
        self.assertEqual(cpu_percent((100, 80), (100, 80)), None)

    def test_memory_is_total_minus_available(self):
        used, total = mem_bytes("MemTotal: 1000 kB\nMemAvailable: 250 kB\n")
        self.assertEqual((used, total), (750 * 1024, 1000 * 1024))

    def test_a_missing_gpu_reading_is_not_zero(self):
        rows = parse_gpu_csv("0, [N/A], [N/A], 16311, 549\n1, 80, 48, 8192, 100\n")
        self.assertIsNone(rows[0].util)
        self.assertIsNone(rows[0].temp)
        self.assertEqual(rows[0].used_mib, 549)
        chosen = choose_gpu(rows)
        self.assertEqual(chosen.index, 0)
        point, _cpu = assemble(
            previous_cpu=(100, 50),
            cpu_text="cpu  20 0 20 60 0\n",
            mem_text="MemTotal: 2048 kB\nMemAvailable: 1024 kB\n",
            cpu_temp=None,
            gpus=[rows[0]],
            disks=(),
        )
        self.assertIsNone(point.gpu)
        self.assertIsNone(point.gpu_temp)
        self.assertEqual(point.vram_used, 549)
        self.assertNotIn("°", gpu_caption(point))
        self.assertNotIn("0°", cpu_caption(point))
        self.assertIn("549", vram_caption(point))
        absent, _cpu = assemble(
            previous_cpu=None,
            cpu_text="cpu  1 0 1 1 0\n",
            mem_text="",
            cpu_temp=None,
            gpus=[],
        )
        self.assertFalse(absent.gpu_present)
        self.assertIn("No GPU detected", gpu_caption(absent))
        self.assertEqual(vram_caption(absent), "vRAM —")

    def test_cpu_temperature_prefers_the_package_sensor(self):
        chips = [
            ("nvme", [("Composite", 38000)]),
            ("k10temp", [("Tccd1", 50000), ("Tctl", 62000)]),
            ("amdgpu", [("edge", 40000)]),
        ]
        self.assertEqual(pick_cpu_temp(chips), 62.0)
        self.assertIsNone(pick_cpu_temp([("nvme", [("Composite", 38000)])]))

    def test_the_library_mount_wins_over_a_directory_inside_it(self):
        self.assertEqual(device_for_mount("/opt/project-nomad/storage", MOUNTINFO), "md0")
        self.assertEqual(
            device_for_mount("/opt/project-nomad/storage/ollama", MOUNTINFO),
            "nvme0n1p2",
        )
        stats = " 9 0 md0 1 0 100 0 1 0 400 0\n 259 2 nvme0n1p2 1 0 8 0 1 0 16 0\n"
        self.assertEqual(disk_sectors(stats, "md0"), (100, 400))
        self.assertIsNone(io_megabytes(None, (100, 400), 1))
        read, write = io_megabytes((100, 400), (100 + 2000, 400 + 1000), 1)
        self.assertAlmostEqual(read, 2000 * 512 / 1_000_000)
        self.assertAlmostEqual(write, 1000 * 512 / 1_000_000)

    def test_the_newest_sample_is_at_the_right_and_a_gap_breaks_the_line(self):
        segments = plot_segments([10.0, None, 90.0], 100, 40, 0, 100)
        self.assertEqual(len(segments), 2)
        self.assertLess(segments[0][0][0], segments[1][0][0])
        self.assertLess(segments[1][0][1], segments[0][0][1])

    def test_a_folder_on_another_device_gets_its_own_line(self):
        rows = storage_disks("/opt/project-nomad/storage", MOUNTINFO)
        self.assertEqual(rows[0], ("Library", "/opt/project-nomad/storage", "9:0"))
        labels = [row[0] for row in rows]
        self.assertEqual(labels, ["Library", "maps", "ollama, qdrant"])
        self.assertEqual(rows[1][2], "8:1")
        self.assertEqual(rows[2][2], "259:2")
        one = "61 34 9:0 / /data rw - xfs /dev/md0 rw\n"
        self.assertEqual(storage_disks("/data", one), [("Library", "/data", "9:0")])
        stats = " 9 0 md0 1 0 100 0 1 0 400 0\n 259 2 dm-0 1 0 8 0 1 0 16 0\n"
        self.assertEqual(disk_sectors(stats, "9:0"), (100, 400))
        self.assertEqual(disk_sectors(stats, "259:2"), (8, 16))
        meters, current = measure_disks(
            [("Library", "/no/such/library", "9:0"), ("qdrant", "/no/such/qdrant", "259:2")],
            stats,
            {},
            None,
        )
        self.assertIsNone(meters[0].used)
        self.assertIsNone(meters[0].read_mbs)
        later = " 9 0 md0 1 0 2100 0 1 0 1400 0\n 259 2 dm-0 1 0 8 0 1 0 20016 0\n"
        meters, _current = measure_disks(
            [("Library", "/no/such/library", "9:0"), ("qdrant", "/no/such/qdrant", "259:2")],
            later,
            current,
            1,
        )
        self.assertAlmostEqual(meters[0].read_mbs, 2000 * 512 / 1_000_000)
        self.assertAlmostEqual(meters[1].write_mbs, 20000 * 512 / 1_000_000)
        point = HostPoint(
            disks=(
                DiskMeter("Library", 10 << 30, 40 << 30, 1.5, 2.0),
                DiskMeter("qdrant", None, None, None, None),
            )
        )
        text = disk_caption(point)
        self.assertEqual(
            text.splitlines()[0],
            "Library  10.0 GiB / 40.0 GiB   Read 1.50 MB/s · Write 2.00 MB/s",
        )
        self.assertIn("qdrant  —   Read — · Write —", text.splitlines()[1])
        self.assertEqual(disk_caption(None), "Disk")
        self.assertEqual(disk_caption(HostPoint()), "Disk")

    def test_the_host_window_place_round_trips_and_is_not_a_gpu_name(self):
        install = _install()
        install.host_x = 30
        install.host_y = 40
        install.host_width = 500
        install.host_height = 700
        again = Install.from_json(install.to_json())
        self.assertEqual((again.host_x, again.host_y, again.host_width, again.host_height), (30, 40, 500, 700))
        data = json.loads(install.to_json())
        del data["host_width"]
        del data["host_x"]
        restored = Install.from_json(json.dumps(data))
        self.assertEqual(restored.host_width, 0)
        self.assertIsNone(restored.host_x)
        source = inspect.getsource(load_install)
        for key in ("host_x", "host_y", "host_width", "host_height"):
            self.assertIn(f'"{key}"', source)
        blob = inspect.getsource(assemble) + inspect.getsource(choose_gpu)
        self.assertNotIn("5060", blob)
        self.assertNotIn("RTX", blob)
        self.assertIsInstance(HostPoint(), HostPoint)
        self.assertIsInstance(GpuRow(0, None, None, 1, 0), GpuRow)
