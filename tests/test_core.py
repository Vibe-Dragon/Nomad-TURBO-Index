import io
import json
import os
import struct
import tempfile
import unittest
import urllib.error
import zlib
from pathlib import Path

from nomad_turbo.chunking import (
    DOCUMENT_PREFIX,
    SAFE_CONTEXT_TOKENS,
    embed_input,
    fit_embed_inputs,
    split_chunks,
    token_upper_bound,
)
from nomad_turbo.decisions import (
    choose_resume,
    continuation_keep,
    drop_should_wait,
    file_is_embeddable,
    hold_queue_on_exit,
    offsets_are_safe,
    parse_io_pressure,
    partial_row_state,
    policy_after_stop,
    qdrant_write_held,
    remembered_policy,
    stock_should_continue,
    write_hold,
)
from nomad_turbo.desktop import desktop_text
from nomad_turbo.icons import icon_bytes, write_icons
from nomad_turbo.prompt import ask_gpu_share
from nomad_turbo.store import active_from_cell, collection_action
from nomad_turbo.discover import Install
from nomad_turbo.packing import ArticleFeed, PreparedArticle, prepare_article
from nomad_turbo.runtime import (
    ChunkRejected,
    RuntimeError_,
    collect_embeddings,
    model_blob_from_manifest,
    vectors_from_openai,
    visual_text,
)
from nomad_turbo.slots import (
    dedicate_percent,
    embed_call_budget,
    gpu_overhead_mib,
    host_floor_mib,
    host_slot_limit,
    pick_gpu,
    plan_slots,
    reserve_ratio_from_percent,
    slots_after_measure,
)
from nomad_turbo import store


class DecisionsTest(unittest.TestCase):
    def test_offsets_wait_for_an_idle_queue(self):
        self.assertFalse(offsets_are_safe(4))
        self.assertFalse(offsets_are_safe(1))
        self.assertTrue(offsets_are_safe(0))

    def test_a_startup_failure_does_not_release_the_next_batches(self):
        self.assertTrue(hold_queue_on_exit(False, True))
        self.assertFalse(hold_queue_on_exit(False, False))
        self.assertFalse(hold_queue_on_exit(True, True))

    def test_stop_restores_the_policy_turbo_found(self):
        self.assertEqual(policy_after_stop("Manual", False), "Manual")
        self.assertEqual(policy_after_stop("Manual", True), "Manual")
        self.assertEqual(policy_after_stop("Always", False), "Always")
        self.assertEqual(policy_after_stop("Always", True), "Manual")
        self.assertEqual(policy_after_stop(None, False), "Manual")
        self.assertEqual(policy_after_stop(None, True), "Manual")
        self.assertEqual(policy_after_stop("sometimes", False), "Manual")
        self.assertEqual(remembered_policy("Always"), "Always")
        self.assertEqual(remembered_policy(" Manual "), "Manual")
        self.assertEqual(remembered_policy(""), "Manual")
        self.assertEqual(remembered_policy(None), "Manual")
        self.assertTrue(stock_should_continue("Always"))
        self.assertFalse(stock_should_continue("Manual"))
        self.assertFalse(stock_should_continue(None))

    def test_a_qdrant_write_timeout_holds_the_current_file(self):
        self.assertTrue(qdrant_write_held("qdrant unreachable: timed out"))
        self.assertTrue(qdrant_write_held("qdrant unreachable: The read operation timed out"))
        self.assertFalse(qdrant_write_held("embedder call failed (timed out)"))
        self.assertFalse(qdrant_write_held("extractor exited 1"))
        self.assertFalse(qdrant_write_held("qdrant 400: bad request"))
        self.assertFalse(qdrant_write_held(""))

    def test_a_download_holds_knowledge_base_writes(self):
        self.assertEqual(
            write_hold(
                downloads_pending=True,
                seconds_since_download=0,
                io_some_avg10=0,
                io_full_avg10=0,
            ),
            "download",
        )
        self.assertEqual(
            write_hold(
                downloads_pending=False,
                seconds_since_download=30,
                io_some_avg10=0,
                io_full_avg10=0,
            ),
            "settle",
        )
        self.assertEqual(
            write_hold(
                downloads_pending=False,
                seconds_since_download=120,
                io_some_avg10=0,
                io_full_avg10=0,
            ),
            "",
        )
        self.assertEqual(
            write_hold(
                downloads_pending=False,
                seconds_since_download=None,
                io_some_avg10=10,
                io_full_avg10=50,
            ),
            "disk",
        )
        self.assertEqual(
            write_hold(
                downloads_pending=True,
                seconds_since_download=0,
                io_some_avg10=90,
                io_full_avg10=90,
            ),
            "download",
        )
        self.assertTrue(drop_should_wait("download", 5000))
        self.assertTrue(drop_should_wait("disk", 10))
        self.assertFalse(drop_should_wait("disk", 600))
        self.assertFalse(drop_should_wait("", 0))
        some, full = parse_io_pressure("some avg10=1.5 avg60=0.0 avg300=0.0 total=1\nfull avg10=9.0 avg60=0.0 avg300=0.0 total=2\n")
        self.assertEqual((some, full), (1.5, 9.0))

    def test_partial_row_blocks_backfill_without_pretending_it_is_done(self):
        self.assertEqual(partial_row_state(None), "pending_decision")
        self.assertIsNone(partial_row_state("indexed"))

    def test_resume_uses_the_later_offset(self):
        self.assertEqual(choose_resume([500, 1000], 750), 1000)
        self.assertEqual(choose_resume([500], 900), 900)
        self.assertEqual(choose_resume([], None), 0)

    def test_one_continuation_keeps_the_highest_offset(self):
        keep, remove = continuation_keep([("a", 10), ("b", 40), ("c", 40)])
        self.assertEqual(keep, "c")
        self.assertEqual(remove, ["a", "b"])

    def test_stock_file_types_ignore_downloads(self):
        self.assertEqual(file_is_embeddable("wikipedia.zim"), "zim")
        self.assertEqual(file_is_embeddable("notes.md"), "text")
        self.assertIsNone(file_is_embeddable("wikipedia.zim.tmp"))
        self.assertIsNone(file_is_embeddable("table.csv"))


class SlotsTest(unittest.TestCase):
    def test_reserve_scales_with_whatever_card_is_installed(self):
        small = plan_slots(
            total_mib=8192,
            used_mib=500,
            reserve_ratio=0.45,
            per_slot_mib=1000,
            cpu_count=8,
            max_slots=8,
        )
        large = plan_slots(
            total_mib=24576,
            used_mib=500,
            reserve_ratio=0.45,
            per_slot_mib=1000,
            cpu_count=16,
            max_slots=8,
        )
        self.assertGreater(large.slots, small.slots)
        self.assertLessEqual(large.slots, 8)
        self.assertTrue(small.use_gpu)

    def test_full_card_falls_back_to_cpu(self):
        plan = plan_slots(
            total_mib=8192,
            used_mib=8000,
            reserve_ratio=0.45,
            per_slot_mib=1000,
            cpu_count=8,
            max_slots=8,
        )
        self.assertFalse(plan.use_gpu)
        self.assertGreaterEqual(plan.slots, 1)

    def test_no_gpu(self):
        plan = plan_slots(
            total_mib=0,
            used_mib=0,
            reserve_ratio=0.45,
            per_slot_mib=1000,
            cpu_count=4,
            max_slots=8,
            has_gpu=False,
        )
        self.assertFalse(plan.use_gpu)
        self.assertEqual(plan.slots, 2)

    def test_pick_the_freer_gpu(self):
        chosen = pick_gpu([(0, 8000, 7000), (1, 16000, 1000)])
        self.assertEqual(chosen[0], 1)

    def test_hundred_percent_keeps_overhead_and_uses_the_rest(self):
        total, used, per = 16384, 400, 1000
        full = plan_slots(
            total_mib=total,
            used_mib=used,
            reserve_ratio=0.0,
            per_slot_mib=per,
            cpu_count=2,
            max_slots=64,
        )
        half = plan_slots(
            total_mib=total,
            used_mib=used,
            reserve_ratio=0.45,
            per_slot_mib=per,
            cpu_count=2,
            max_slots=64,
        )
        overhead = gpu_overhead_mib(total)
        budget = total - used - overhead
        self.assertGreater(overhead, 0)
        self.assertEqual(full.slots, min(64, budget // per))
        self.assertGreater(full.slots, half.slots)
        self.assertTrue(full.use_gpu)
        self.assertLessEqual(full.slots * per, budget)
        self.assertIn("100%", full.reason)

    def test_share_percent_round_trip(self):
        self.assertEqual(dedicate_percent(0.45), 55)
        self.assertEqual(reserve_ratio_from_percent(55), 0.45)
        self.assertEqual(reserve_ratio_from_percent(100), 0.0)
        self.assertEqual(dedicate_percent(0.0), 100)
        self.assertEqual(reserve_ratio_from_percent(10), 0.9)

    def test_measure_counts_the_first_process_once(self):
        # overhead for 10000 MiB is 512. Budget is 10000 - 1000 - 512.
        count = slots_after_measure(
            planned=4,
            total_mib=10000,
            used_before_mib=1000,
            used_after_mib=2000,
            reserve_ratio=0.0,
            max_slots=64,
        )
        self.assertEqual(count, 8)

    def test_system_ram_caps_a_wide_gpu_plan(self):
        # A measured GPU copy of 503 MiB would ask for many processes.
        # System RAM still has to hold a 1024 MiB copy of each one.
        plan = plan_slots(
            total_mib=16384,
            used_mib=2000,
            reserve_ratio=0.2,
            per_slot_mib=503,
            cpu_count=12,
            max_slots=64,
            host_total_mib=31239,
            host_available_mib=18000,
            host_per_slot_mib=1024,
        )
        self.assertEqual(host_floor_mib(31239), 7809)
        self.assertEqual(host_slot_limit(31239, 18000, 1024), 9)
        self.assertEqual(plan.slots, 9)
        self.assertIn("system RAM", plan.reason)
        self.assertTrue(plan.use_gpu)

    def test_plenty_of_ram_leaves_the_gpu_plan_alone(self):
        plan = plan_slots(
            total_mib=8192,
            used_mib=0,
            reserve_ratio=0.0,
            per_slot_mib=1000,
            cpu_count=4,
            max_slots=8,
            host_total_mib=128000,
            host_available_mib=100000,
        )
        self.assertGreater(plan.slots, 1)
        self.assertNotIn("system RAM", plan.reason)

    def test_a_full_host_still_keeps_one_process(self):
        plan = plan_slots(
            total_mib=16384,
            used_mib=0,
            reserve_ratio=0.0,
            per_slot_mib=1000,
            cpu_count=8,
            max_slots=64,
            host_total_mib=32000,
            host_available_mib=3000,
        )
        self.assertEqual(plan.slots, 1)
        self.assertIn("system RAM", plan.reason)

    def test_a_forced_count_still_has_to_fit_in_ram(self):
        plan = plan_slots(
            total_mib=16384,
            used_mib=0,
            reserve_ratio=0.0,
            per_slot_mib=1000,
            cpu_count=8,
            max_slots=64,
            forced=40,
            host_total_mib=32000,
            host_available_mib=10000,
        )
        self.assertEqual(plan.slots, 2)
        self.assertIn("config slots", plan.reason)
        self.assertIn("system RAM", plan.reason)

    def test_a_bad_measurement_keeps_the_plan(self):
        self.assertEqual(
            slots_after_measure(
                planned=4,
                total_mib=10000,
                used_before_mib=1000,
                used_after_mib=1000,
                reserve_ratio=0.0,
                max_slots=64,
            ),
            4,
        )

    def test_slider_needs_a_display(self):
        saved = os.environ.copy()
        os.environ.pop("DISPLAY", None)
        os.environ.pop("WAYLAND_DISPLAY", None)
        try:
            status, value = ask_gpu_share(55, lambda _percent: "preview")
        finally:
            os.environ.clear()
            os.environ.update(saved)
        self.assertEqual(status, "unavailable")
        self.assertIsNone(value)


class ChunkTest(unittest.TestCase):
    def test_prefix_is_on_the_vector_input_only(self):
        prefixed, payload = embed_input("Alpha team cache")
        self.assertTrue(prefixed.startswith(DOCUMENT_PREFIX))
        self.assertNotIn(DOCUMENT_PREFIX.strip(), payload)
        self.assertEqual(payload, "Alpha team cache")

    def test_long_text_splits_with_overlap(self):
        text = " ".join(f"word{i}" for i in range(4000))
        parts = split_chunks(text)
        self.assertGreater(len(parts), 1)
        self.assertLessEqual(len(parts[0]), 3000)

    def test_ordinary_english_stays_one_embed_input(self):
        text = ("The quick brown fox jumps over the lazy dog. " * 60).strip()
        pairs = fit_embed_inputs(text)
        self.assertEqual(len(pairs), 1)
        prefixed, payload = pairs[0]
        self.assertTrue(prefixed.startswith(DOCUMENT_PREFIX))
        self.assertNotIn(DOCUMENT_PREFIX.strip(), payload)
        self.assertLessEqual(token_upper_bound(prefixed), SAFE_CONTEXT_TOKENS)

    def test_dense_text_is_split_under_the_context_bound(self):
        text = "α" * 3000
        pairs = fit_embed_inputs(text)
        self.assertGreater(len(pairs), 1)
        payloads = []
        for prefixed, payload in pairs:
            self.assertTrue(prefixed.startswith(DOCUMENT_PREFIX))
            self.assertFalse(payload.startswith(DOCUMENT_PREFIX))
            self.assertLessEqual(token_upper_bound(prefixed), SAFE_CONTEXT_TOKENS)
            payloads.append(payload)
        self.assertEqual("".join(payloads), text)


class CollectionTest(unittest.TestCase):
    def test_a_timeout_on_a_live_qdrant_keeps_the_collection(self):
        self.assertEqual(
            collection_action("qdrant unreachable: timed out", None, "nomad_knowledge_base", True),
            "ready",
        )

    def test_a_missing_collection_is_created(self):
        self.assertEqual(collection_action("qdrant 404: not found", None, "nomad_knowledge_base", True), "create")

    def test_a_dead_qdrant_is_not_treated_as_ready(self):
        self.assertEqual(
            collection_action("qdrant unreachable: timed out", None, "nomad_knowledge_base", False),
            "raise",
        )


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def rgba_icon_stats(data: bytes) -> dict:
    """Color type, corner alpha, transparent fraction, and how far the glyph reaches."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError("not a png")
    pos = 8
    width = height = color = depth = interlace = None
    idat = []
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, depth, color, _comp, _filt, interlace = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"IDAT":
            idat.append(chunk)
        elif kind == b"IEND":
            break
    if color != 6 or depth != 8 or interlace or not width or not height:
        return {"color": color, "corner": 255, "clear": 0.0, "fill": 0.0}
    raw = zlib.decompress(b"".join(idat))
    stride = width * 4
    prev = bytearray(stride)
    minx, miny, maxx, maxy = width, height, 0, 0
    clear = 0
    corner = 255
    offset = 0
    for y in range(height):
        filt = raw[offset]
        row = bytearray(raw[offset + 1 : offset + 1 + stride])
        offset += stride + 1
        if filt == 1:
            for i in range(stride):
                left = row[i - 4] if i >= 4 else 0
                row[i] = (row[i] + left) & 255
        elif filt == 2:
            for i in range(stride):
                row[i] = (row[i] + prev[i]) & 255
        elif filt == 3:
            for i in range(stride):
                left = row[i - 4] if i >= 4 else 0
                row[i] = (row[i] + ((left + prev[i]) // 2)) & 255
        elif filt == 4:
            for i in range(stride):
                a = row[i - 4] if i >= 4 else 0
                b = prev[i]
                c = prev[i - 4] if i >= 4 else 0
                row[i] = (row[i] + _paeth(a, b, c)) & 255
        elif filt != 0:
            raise AssertionError(f"unknown png filter {filt}")
        if y == 0:
            corner = row[3]
        for x in range(width):
            if row[x * 4 + 3] == 0:
                clear += 1
                continue
            if x < minx:
                minx = x
            if x > maxx:
                maxx = x
            if y < miny:
                miny = y
            if y > maxy:
                maxy = y
        prev = row
    fill = 0.0
    if maxx >= minx:
        fill = max((maxx - minx + 1) / width, (maxy - miny + 1) / height)
    return {"color": color, "corner": corner, "clear": clear / (width * height), "fill": fill}


class IconTest(unittest.TestCase):
    def test_icons_are_png_and_differ(self):
        idle = icon_bytes("idle")
        indexing = icon_bytes("indexing")
        repair = icon_bytes("repair")
        for data in (idle, indexing, repair):
            self.assertTrue(data.startswith(b"\x89PNG"))
        self.assertNotEqual(idle, indexing)
        self.assertNotEqual(indexing, repair)
        self.assertNotEqual(idle, repair)

    def test_legacy_names_keep_the_app_icon(self):
        with tempfile.TemporaryDirectory() as folder:
            written = write_icons(Path(folder))
            fire = icon_bytes("indexing")
            self.assertEqual((Path(folder) / "turbo-on.png").read_bytes(), fire)
            self.assertEqual((Path(folder) / "turbo-off.png").read_bytes(), fire)
            self.assertEqual(written["repair"].read_bytes(), icon_bytes("repair"))

    def test_icons_are_transparent_and_fill_the_tile(self):
        for kind in ("idle", "indexing", "repair"):
            stats = rgba_icon_stats(icon_bytes(kind))
            self.assertEqual(stats["color"], 6, kind)
            self.assertEqual(stats["corner"], 0, kind)
            self.assertGreater(stats["clear"], 0.15, kind)
            self.assertGreaterEqual(stats["fill"], 0.75, kind)

    def test_launcher_names_the_window_class(self):
        text = desktop_text(Path("/tmp/turbo-index.png"), "/usr/local/bin/nomad-turbo toggle")
        self.assertIn("StartupWMClass=nomad-turbo", text)
        self.assertIn("Icon=/tmp/turbo-index.png", text)
        self.assertIn("Exec=/usr/local/bin/nomad-turbo toggle", text)


class VisualReadTest(unittest.TestCase):
    def test_a_failed_ocr_is_not_stored_as_an_empty_document(self):
        self.assertEqual(visual_text(0, "page one"), "page one")
        self.assertEqual(visual_text(0, ""), "")
        self.assertIsNone(visual_text(1, ""))
        self.assertIsNone(visual_text(1, "partial"))


class SpeedTest(unittest.TestCase):
    def test_slider_only_changes_the_call_width(self):
        hardware = dict(gpu_total_mib=8192, host_total_mib=16384, cpu_count=8)
        low = embed_call_budget(10, **hardware)
        high = embed_call_budget(100, **hardware)
        over = embed_call_budget(110, **hardware)
        self.assertLessEqual(low, high)
        self.assertGreater(over, high)
        widths = [embed_call_budget(percent, **hardware) for percent in range(10, 101)]
        self.assertEqual(widths, sorted(widths))

    def test_manifest_names_the_model_blob(self):
        path = model_blob_from_manifest(
            {
                "layers": [
                    {"mediaType": "application/vnd.ollama.image.params", "digest": "sha256:abc"},
                    {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:deadbeef"},
                ]
            }
        )
        self.assertEqual(path, "/root/.ollama/models/blobs/sha256-deadbeef")

    def test_openai_vectors_come_back_in_input_order(self):
        body = {"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]}
        self.assertEqual(vectors_from_openai(body, 2), [[1.0], [2.0]])
        with self.assertRaises(RuntimeError_):
            vectors_from_openai({"data": [{"index": 0, "embedding": []}]}, 1)
        with self.assertRaises(RuntimeError_):
            vectors_from_openai({"data": [{"index": 0, "embedding": [1.0]}]}, 2)

    def test_prefix_stays_off_the_stored_text(self):
        prepared = prepare_article(
            {
                "title": "Title",
                "path": "A",
                "sections": [
                    {"heading": "H", "text": "body", "level": 2},
                    {"heading": "", "text": "", "level": 2},
                    {"heading": "I", "text": "more", "level": 3},
                ],
            },
            "/app/storage/zim/a.zim",
            {"title": "Wiki"},
            "doc-1",
        )
        self.assertEqual(prepared.section_for_input, [0, 1])
        self.assertTrue(all(text.startswith("search_document: ") for text in prepared.inputs))
        self.assertNotIn("search_document:", prepared.sections[0].payloads[0])
        self.assertEqual(prepared.sections[0].payloads[0], "body")
        self.assertEqual(prepared.sections[0].meta["source"], "/app/storage/zim/a.zim")
        self.assertEqual(prepared.sections[0].meta["section_title"], "H")
        self.assertEqual(prepared.sections[1].meta["section_level"], 3)
        self.assertIs(prepared.sections[0].meta["active"], True)

    def test_a_stored_point_always_has_the_active_flag(self):
        """NOMAD rewrites every point that is missing this field."""
        self.assertTrue(active_from_cell(None))
        self.assertTrue(active_from_cell(""))
        self.assertTrue(active_from_cell("1"))
        self.assertFalse(active_from_cell("0"))
        self.assertFalse(active_from_cell("1\n0"))
        from nomad_turbo.engine import FileWork, _points_for, _prepare_article

        install = Install(
            admin_container="admin",
            admin_image="admin",
            network="net",
            redis_host="redis",
            redis_port=6379,
            redis_container="redis",
            mysql_container="mysql",
            qdrant_url="http://127.0.0.1:6333",
            ollama_image="ollama",
            ollama_container="ollama",
            models_dir="/models",
            storage_host="/storage",
        )
        work = FileWork(
            host=Path("a.zim"),
            kind="zim",
            source="/app/storage/zim/a.zim",
            point_active=False,
        )
        prepared = _prepare_article(
            install,
            work,
            {"title": "Title", "path": "A", "sections": [{"heading": "", "text": "body", "level": 2}]},
            set(),
        )
        self.assertIs(prepared.sections[0].meta["active"], False)
        off = _points_for(install, [[0.0] * 768], ["body"], prepared.sections[0].meta, 0)
        self.assertIs(off[0]["payload"]["active"], False)
        on_meta = dict(prepared.sections[0].meta)
        on_meta["active"] = True
        on = _points_for(install, [[0.0] * 768], ["body"], on_meta, 0)
        self.assertIs(on[0]["payload"]["active"], True)
        bare = dict(on_meta)
        del bare["active"]
        defaulted = _points_for(install, [[0.0] * 768], ["body"], bare, 0)
        self.assertIs(defaulted[0]["payload"]["active"], True)

    def test_a_partial_article_is_not_emitted(self):
        feed = ArticleFeed()
        feed.add(
            PreparedArticle(
                article={"index": 1, "path": "a"},
                sections=[],
                inputs=["a1", "a2"],
                section_for_input=[0, 0],
            )
        )
        feed.add(
            PreparedArticle(
                article={"index": 2, "path": "b"},
                sections=[],
                inputs=["b1", "b2"],
                section_for_input=[0, 0],
            )
        )
        self.assertEqual(feed.next_inputs(3), ["a1", "a2", "b1"])
        done = feed.push_vectors([[1.0], [2.0], [3.0]])
        self.assertEqual([item.article["path"] for item, _vectors in done], ["a"])
        self.assertEqual(done[0][1], [[1.0], [2.0]])
        self.assertEqual(feed.pending(), 1)
        self.assertEqual(feed.next_inputs(8), ["b2"])
        done = feed.push_vectors([[4.0]])
        self.assertEqual([item.article["path"] for item, _vectors in done], ["b"])

    def test_upsert_sends_128_points_at_a_time(self):
        install = Install(
            admin_container="admin",
            admin_image="admin",
            network="net",
            redis_host="redis",
            redis_port=6379,
            redis_container="redis",
            mysql_container="mysql",
            qdrant_url="http://127.0.0.1:6333",
            ollama_image="ollama",
            ollama_container="ollama",
            models_dir="/models",
            storage_host="/storage",
        )
        calls = []

        def fake_request(url, payload=None, timeout=60, method=None):
            self.assertIn("/points?wait=true", url)
            self.assertEqual(method, "PUT")
            calls.append(payload["points"])
            return {}

        original = store._request
        store._request = fake_request
        try:
            store.upsert(install, [{"id": index} for index in range(129)])
        finally:
            store._request = original
        self.assertEqual([len(batch) for batch in calls], [128, 1])

    def test_a_timed_out_qdrant_write_is_tried_again(self):
        self.assertTrue(store.store_write_again("qdrant unreachable: timed out", 0))
        self.assertTrue(store.store_write_again("qdrant unreachable: timed out", 1))
        self.assertFalse(store.store_write_again("qdrant unreachable: timed out", 2))
        self.assertFalse(store.store_write_again("qdrant 400: bad request", 0))

    def test_a_missing_segment_cap_is_four_gigabytes(self):
        cap = 4 * 1024 * 1024
        update = store.segment_cap_update(None)
        self.assertEqual(update["optimizers_config"]["max_segment_size"], cap)
        self.assertIsNotNone(store.segment_cap_update(True))
        self.assertIsNotNone(store.segment_cap_update(cap + 1))
        self.assertIsNone(store.segment_cap_update(cap))
        self.assertIsNone(store.segment_cap_update(1024))

    def test_the_same_chunk_keeps_the_same_point_id(self):
        first = store.point_id("/zim/a.zim", "Alpha", "0:Title", 3, "body")
        second = store.point_id("/zim/a.zim", "Alpha", "0:Title", 3, "body")
        self.assertEqual(first, second)
        self.assertNotEqual(first, store.point_id("/zim/a.zim", "Alpha", "0:Title", 4, "body"))
        self.assertEqual(len(first), 36)


class EmbedRetryTest(unittest.TestCase):
    def _reject(self, message: str):
        body = json.dumps({"error": {"message": message}}).encode()
        return urllib.error.HTTPError("http://embed", 400, "Bad Request", None, io.BytesIO(body))

    def test_a_400_points_at_the_one_bad_input(self):
        def post(texts):
            if "bad" in texts:
                raise self._reject("input (2177 tokens) is larger than the max context size (2048 tokens)")
            return [[1.0] for _text in texts]

        with self.assertRaises(ChunkRejected) as caught:
            collect_embeddings(post, ["a", "b", "bad", "c"], pause=lambda _seconds: None)
        self.assertEqual(caught.exception.index, 2)
        self.assertIn("2048", caught.exception.detail)

    def test_one_bad_input_is_not_retried(self):
        calls = []

        def post(_texts):
            calls.append(1)
            raise self._reject("too long")

        with self.assertRaises(ChunkRejected):
            collect_embeddings(post, ["only"], pause=lambda _seconds: None)
        self.assertEqual(calls, [1])

    def test_a_timeout_is_tried_again(self):
        calls = []

        def post(_texts):
            calls.append(1)
            if len(calls) == 1:
                raise TimeoutError("timed out")
            return [[0.5]]

        self.assertEqual(collect_embeddings(post, ["ok"], pause=lambda _seconds: None), [[0.5]])
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
