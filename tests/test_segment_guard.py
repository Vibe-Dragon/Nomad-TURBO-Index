"""Hold and force decisions for segments Qdrant has not compacted.

Facts are built in the test. Nothing here opens the live collection.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from nomad_turbo.discover import Install
from nomad_turbo import engine, segment_guard, store
from nomad_turbo.segment_guard import (
    ForceResult,
    SegmentFacts,
    apply_force_optimization,
    cached_write_facts,
    clear_write_cache,
    default_segment_cap_bytes,
    force_optimization,
    index_ram_bytes,
    measure_repair_facts,
    measure_write_facts,
    optimizer_settings,
    segment_upsert_hold,
)
from nomad_turbo.store import MAX_SEGMENT_KB


def _install(storage: str) -> Install:
    return Install(
        admin_container="admin",
        admin_image="admin",
        network="net",
        redis_host="redis",
        redis_port=6379,
        redis_container="redis",
        mysql_container="mysql",
        qdrant_url="http://127.0.0.1:9",
        ollama_image="ollama",
        ollama_container="ollama",
        models_dir="/models",
        storage_host=storage,
        collection="nomad_knowledge_base",
    )


class Recorder:
    """Stands in for Qdrant. Records pauses, patches, and optimize calls."""

    def __init__(self, config: dict, boom: bool = False) -> None:
        self.original = dict(config)
        self.current = dict(config)
        self.patches: list[dict] = []
        self.events: list[str] = []
        self.boom = boom

    def pause_embeds(self) -> None:
        self.events.append("pause")

    def resume_embeds(self) -> None:
        self.events.append("resume")

    def read_optimizer_config(self) -> dict:
        self.events.append("read")
        return dict(self.original)

    def patch_optimizer_config(self, settings: dict) -> None:
        self.events.append("patch")
        self.patches.append(dict(settings))
        self.current = dict(settings)

    def request_optimize(self) -> None:
        self.events.append("optimize")
        if self.boom:
            raise RuntimeError("optimize failed")

    def delete_collection(self) -> None:
        self.events.append("delete")


def _optimizer() -> dict:
    return {
        "deleted_threshold": 0.2,
        "vacuum_min_vector_number": 1000,
        "default_segment_number": 0,
        "max_segment_size": MAX_SEGMENT_KB,
        "indexing_threshold": 10000,
        "flush_interval_sec": 5,
    }


def _kept(result: ForceResult, recorder: Recorder) -> None:
    if result.delete_points or result.delete_collection:
        raise AssertionError("points or the collection were deleted")
    if "delete" in recorder.events:
        raise AssertionError(recorder.events)
    for patch in recorder.patches:
        if "hnsw_config" in patch or "on_disk" in patch:
            raise AssertionError(patch)


class HoldDecisionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cap = default_segment_cap_bytes()

    def test_cap_matches_the_segment_cap_turbo_already_applies(self):
        self.assertEqual(self.cap, MAX_SEGMENT_KB * 1024)
        self.assertEqual(self.cap, 4 * 1024 * 1024 * 1024)

    def test_a_segment_past_the_cap_holds_and_does_not_delete_points(self):
        decision = segment_upsert_hold(
            SegmentFacts(
                largest_plain_vector_bytes=self.cap + 1,
                largest_plain_disk_bytes=100,
                free_bytes=10**15,
                spare_ram_bytes=10**15,
                optimization_running=False,
            )
        )
        self.assertEqual(decision.reason, "segment")
        self.assertFalse(decision.delete_points)
        self.assertFalse(decision.delete_collection)

    def test_under_the_cap_with_free_space_under_twice_the_segment_holds(self):
        disk = 1000
        decision = segment_upsert_hold(
            SegmentFacts(
                largest_plain_vector_bytes=self.cap,
                largest_plain_disk_bytes=disk,
                free_bytes=(2 * disk) - 1,
            )
        )
        self.assertEqual(decision.reason, "space")
        self.assertFalse(decision.delete_points)
        self.assertFalse(decision.delete_collection)

    def test_both_conditions_cleared_allows_upserts(self):
        disk = 1000
        decision = segment_upsert_hold(
            SegmentFacts(
                largest_plain_vector_bytes=self.cap,
                largest_plain_disk_bytes=disk,
                free_bytes=2 * disk,
            )
        )
        self.assertEqual(decision.reason, "")
        self.assertFalse(decision.delete_points)
        self.assertFalse(decision.delete_collection)

    def test_optimization_already_running_does_not_hold_upserts(self):
        disk = 1000
        decision = segment_upsert_hold(
            SegmentFacts(
                largest_plain_vector_bytes=self.cap,
                largest_plain_disk_bytes=disk,
                free_bytes=2 * disk,
                optimization_running=True,
                spare_ram_bytes=1,
            )
        )
        self.assertEqual(decision.reason, "")
        self.assertFalse(decision.delete_points)
        self.assertFalse(decision.delete_collection)

    def test_unreadable_free_space_holds_when_a_plain_segment_exists(self):
        decision = segment_upsert_hold(
            SegmentFacts(largest_plain_vector_bytes=10, largest_plain_disk_bytes=10, free_bytes=None)
        )
        self.assertEqual(decision.reason, "space")
        self.assertFalse(decision.delete_points)


class ForceDecisionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cap = default_segment_cap_bytes()
        self.config = _optimizer()

    def _run(self, facts: SegmentFacts, boom: bool = False, config: dict | None = None) -> tuple[ForceResult, Recorder]:
        recorder = Recorder(self.config if config is None else config, boom=boom)
        result = apply_force_optimization(facts, recorder)
        _kept(result, recorder)
        self.assertEqual(recorder.events[0], "pause")
        self.assertEqual(recorder.events[-1], "resume")
        self.assertEqual(recorder.events.count("resume"), 1)
        return result, recorder

    def test_already_running_refuses_with_no_collection_change(self):
        facts = SegmentFacts(
            largest_plain_vector_bytes=self.cap + 1,
            largest_plain_disk_bytes=1000,
            largest_plain_vector_count=10,
            free_bytes=10**15,
            spare_ram_bytes=10**15,
            optimization_running=True,
        )
        decision = force_optimization(facts)
        result, recorder = self._run(facts)
        self.assertFalse(decision.allow)
        self.assertIn("already in progress", decision.reason)
        self.assertEqual(result.reason, decision.reason)
        self.assertFalse(result.requested)
        self.assertFalse(result.ok)
        self.assertEqual(recorder.patches, [])
        self.assertNotIn("read", recorder.events)
        self.assertNotIn("optimize", recorder.events)
        self.assertFalse(decision.delete_points)
        self.assertFalse(decision.delete_collection)

    def test_free_space_under_twice_the_segment_refuses(self):
        disk = 5000
        facts = SegmentFacts(
            largest_plain_vector_bytes=self.cap + 1,
            largest_plain_disk_bytes=disk,
            largest_plain_vector_count=1,
            free_bytes=(2 * disk) - 1,
            spare_ram_bytes=10**15,
            optimization_running=False,
        )
        result, recorder = self._run(facts)
        self.assertFalse(force_optimization(facts).allow)
        self.assertIn("free space", result.reason.lower())
        self.assertFalse(result.requested)
        self.assertEqual(recorder.patches, [])
        self.assertNotIn("optimize", recorder.events)
        self.assertNotIn("read", recorder.events)

    def test_insufficient_spare_ram_refuses_even_when_links_alone_would_fit(self):
        vector_bytes = self.cap + 1
        count = 4
        facts = SegmentFacts(
            largest_plain_vector_bytes=vector_bytes,
            largest_plain_disk_bytes=100,
            largest_plain_vector_count=count,
            free_bytes=10**15,
            spare_ram_bytes=index_ram_bytes(vector_bytes, count) - 1,
            optimization_running=False,
        )
        self.assertGreater(facts.spare_ram_bytes, count * 64)
        result, recorder = self._run(facts)
        self.assertFalse(force_optimization(facts).allow)
        self.assertIn("spare memory", result.reason.lower())
        self.assertFalse(result.requested)
        self.assertEqual(recorder.patches, [])
        self.assertNotIn("optimize", recorder.events)

    def test_safe_case_requests_optimization_and_restores_settings(self):
        facts = SegmentFacts(
            largest_plain_vector_bytes=1024,
            largest_plain_disk_bytes=2048,
            largest_plain_vector_count=1,
            largest_deleted_disk_bytes=5000,
            free_bytes=5000 * 2,
            spare_ram_bytes=0,
            optimization_running=False,
        )
        self.assertEqual(segment_upsert_hold(facts).reason, "")
        self.assertTrue(force_optimization(facts).allow)
        result, recorder = self._run(facts)
        self.assertTrue(result.ok)
        self.assertTrue(result.requested)
        self.assertIn("settings are restored", result.reason.lower())
        self.assertEqual(recorder.events.count("optimize"), 1)
        self.assertEqual(recorder.patches, [self.config])
        for patch in recorder.patches:
            self.assertNotEqual(patch.get("indexing_threshold"), 0)
        self.assertEqual(recorder.current, self.config)
        self.assertNotIn("hnsw_config", recorder.patches[-1])
        self.assertLess(recorder.events.index("optimize"), recorder.events.index("patch"))

    def test_a_raised_optimize_call_still_restores_settings_and_resumes(self):
        facts = SegmentFacts(
            largest_plain_vector_bytes=1024,
            largest_plain_disk_bytes=2048,
            largest_deleted_disk_bytes=5000,
            free_bytes=5000 * 2,
            spare_ram_bytes=0,
            optimization_running=False,
        )
        result, recorder = self._run(facts, boom=True)
        self.assertFalse(result.ok)
        self.assertTrue(result.requested)
        self.assertIn("failed", result.reason.lower())
        self.assertIn("settings are restored", result.reason.lower())
        self.assertEqual(recorder.current, self.config)
        self.assertEqual(recorder.patches, [self.config])
        for patch in recorder.patches:
            self.assertNotEqual(patch.get("indexing_threshold"), 0)
        self.assertEqual(recorder.events.count("optimize"), 1)
        self.assertLess(recorder.events.index("optimize"), recorder.events.index("patch"))

    def test_hnsw_on_disk_is_not_written(self):
        raw = dict(self.config)
        raw["hnsw_config"] = {"on_disk": True}
        raw["on_disk"] = True
        facts = SegmentFacts(
            largest_deleted_disk_bytes=100,
            free_bytes=200,
            spare_ram_bytes=0,
            optimization_running=False,
        )
        result, recorder = self._run(facts, config=raw)
        self.assertTrue(result.requested)
        self.assertEqual(recorder.current, optimizer_settings(raw))
        self.assertNotIn("hnsw_config", recorder.current)
        self.assertNotIn("on_disk", recorder.current)
        for patch in recorder.patches:
            self.assertNotIn("hnsw_config", patch)
            self.assertNotIn("on_disk", patch)
        self.assertFalse(result.delete_collection)

    def test_a_saved_threshold_of_zero_is_written_back_and_not_used_to_wake(self):
        config = dict(self.config)
        config["indexing_threshold"] = 0
        facts = SegmentFacts(
            largest_deleted_disk_bytes=100,
            free_bytes=200,
            spare_ram_bytes=0,
        )
        result, recorder = self._run(facts, config=config)
        self.assertTrue(result.requested)
        self.assertIn("settings are restored", result.reason.lower())
        # The only optimizer write is the saved config. Nothing sends 0 to wake indexing.
        self.assertEqual(recorder.patches, [config])
        self.assertEqual(recorder.current, config)
        self.assertEqual(recorder.events.count("optimize"), 1)
        self.assertLess(recorder.events.index("optimize"), recorder.events.index("patch"))

    def test_a_failed_restore_does_not_claim_success_or_leave_threshold_zero(self):
        class BoomRestore(Recorder):
            def patch_optimizer_config(self, settings: dict) -> None:
                if settings.get("indexing_threshold") == 0:
                    raise AssertionError("threshold 0 was sent")
                raise RuntimeError("restore failed")

        facts = SegmentFacts(
            largest_plain_vector_bytes=1024,
            largest_plain_disk_bytes=2048,
            largest_deleted_disk_bytes=5000,
            free_bytes=5000 * 2,
            spare_ram_bytes=0,
            optimization_running=False,
        )
        recorder = BoomRestore(self.config)
        result = apply_force_optimization(facts, recorder)
        _kept(result, recorder)
        self.assertFalse(result.ok)
        self.assertTrue(result.requested)
        self.assertNotIn("settings are restored", result.reason.lower())
        self.assertIn("could not be put back", result.reason.lower())
        self.assertNotEqual(recorder.current.get("indexing_threshold"), 0)
        self.assertEqual(recorder.current.get("indexing_threshold"), self.config["indexing_threshold"])
        self.assertEqual(recorder.events[0], "pause")
        self.assertEqual(recorder.events[-1], "resume")
        self.assertIn("optimize", recorder.events)
        self.assertNotIn("patch", recorder.events)

    def test_an_oversized_plain_is_the_rewrite_target(self):
        vector_bytes = self.cap + 50
        count = 3
        needed = index_ram_bytes(vector_bytes, count)
        facts = SegmentFacts(
            largest_plain_vector_bytes=vector_bytes,
            largest_plain_disk_bytes=800,
            largest_plain_vector_count=count,
            largest_deleted_disk_bytes=100,
            free_bytes=800 * 2,
            spare_ram_bytes=needed - 1,
            optimization_running=False,
        )
        decision = force_optimization(facts)
        self.assertFalse(decision.allow)
        self.assertIn("spare memory", decision.reason.lower())
        self.assertFalse(decision.delete_collection)

    def test_no_stuck_segments_refuses_without_a_write(self):
        facts = SegmentFacts(
            largest_plain_vector_bytes=100,
            largest_plain_disk_bytes=100,
            free_bytes=10**9,
            spare_ram_bytes=10**9,
        )
        result, recorder = self._run(facts)
        self.assertIn("No stuck segments", result.reason)
        self.assertFalse(result.requested)
        self.assertEqual(recorder.patches, [])


class WriteLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cap = default_segment_cap_bytes()
        self.tmp = Path(self.id().replace(".", "_"))

    def _facts_then(self, first: SegmentFacts, later: SegmentFacts):
        state = {"n": 0}

        def facts(_install):
            state["n"] += 1
            return first if state["n"] == 1 else later

        return facts, state

    def test_over_cap_waits_until_the_segment_clears_and_does_not_delete(self):
        over = SegmentFacts(
            largest_plain_vector_bytes=self.cap + 1,
            largest_plain_disk_bytes=100,
            free_bytes=10**15,
        )
        clear = SegmentFacts(
            largest_plain_vector_bytes=self.cap,
            largest_plain_disk_bytes=100,
            free_bytes=10**15,
        )
        facts, state = self._facts_then(over, clear)
        install = _install("/tmp/nomad-turbo-segment-hold")
        stop = threading.Event()
        with (
            mock.patch.object(segment_guard, "cached_write_facts", facts),
            mock.patch.object(engine, "_library_hold", return_value=""),
            mock.patch.object(store, "delete_collection", side_effect=AssertionError("deleted collection")),
            mock.patch.object(store, "delete_source_points", side_effect=AssertionError("deleted points")),
            mock.patch.object(engine, "delete_collection", side_effect=AssertionError("deleted collection")),
        ):
            allowed = engine._wait_to_write(install, stop, None, interval=0)
        self.assertTrue(allowed)
        self.assertGreaterEqual(state["n"], 2)
        self.assertFalse(stop.is_set())

    def test_low_free_space_waits_and_a_running_optimization_does_not(self):
        disk = 4000
        tight = SegmentFacts(
            largest_plain_vector_bytes=self.cap,
            largest_plain_disk_bytes=disk,
            free_bytes=(2 * disk) - 1,
            optimization_running=True,
        )
        room = SegmentFacts(
            largest_plain_vector_bytes=100,
            largest_plain_disk_bytes=100,
            free_bytes=10**15,
            optimization_running=True,
        )
        self.assertEqual(segment_upsert_hold(tight).reason, "space")
        self.assertEqual(segment_upsert_hold(room).reason, "")
        facts, state = self._facts_then(tight, room)
        install = _install("/tmp/nomad-turbo-segment-space")
        with (
            mock.patch.object(segment_guard, "cached_write_facts", facts),
            mock.patch.object(engine, "_library_hold", return_value="disk"),
        ):
            allowed = engine._wait_to_write(install, threading.Event(), None, interval=0)
        self.assertTrue(allowed)
        self.assertGreaterEqual(state["n"], 2)

    def test_a_disk_stall_does_not_skip_a_segment_hold(self):
        calls = {"n": 0}
        stop = threading.Event()

        def facts(_install):
            calls["n"] += 1
            if calls["n"] >= 3:
                stop.set()
            return SegmentFacts(
                largest_plain_vector_bytes=self.cap + 1,
                largest_plain_disk_bytes=10,
                free_bytes=10**15,
            )

        install = _install("/tmp/nomad-turbo-segment-disk")
        with (
            mock.patch.object(segment_guard, "cached_write_facts", facts),
            mock.patch.object(engine, "_library_hold", return_value="disk"),
        ):
            allowed = engine._wait_to_write(install, stop, None, interval=0)
        self.assertFalse(allowed)
        self.assertGreaterEqual(calls["n"], 3)

    def test_steady_state_upsert_after_push_vectors_waits(self):
        """One short article takes the push_vectors submit, not the drain submit.

        An over-cap segment must stop that upsert. The hold decision is the
        shipped one; only the measured facts are supplied.
        """
        cap = self.cap
        stop = threading.Event()
        calls = {"facts": 0, "upsert": 0, "embed": 0}
        returned: dict[str, SegmentFacts] = {}
        install = _install("/tmp/nomad-turbo-consume-hold")
        install.boost_percent = 50
        article = {
            "title": "Hello",
            "path": "A/Hello",
            "index": 1,
            "sections": [{"heading": "", "text": "hello world", "level": 2}],
        }
        items = [article, engine._EXTRACT_DONE]

        def pull(_timeout):
            if not items:
                return None
            return items.pop(0)

        def facts(_install):
            calls["facts"] += 1
            stop.set()
            measured = SegmentFacts(
                largest_plain_vector_bytes=cap + 1,
                largest_plain_disk_bytes=100,
                free_bytes=10**15,
            )
            returned["value"] = measured
            return measured

        def fake_embed(_url, _model, texts):
            calls["embed"] += 1
            self.assertEqual(len(texts), 1)
            return [[0.0] * install.dimension for _text in texts]

        def fake_upsert(_install, points):
            calls["upsert"] += 1
            self.assertTrue(points)

        class _Pool:
            urls = ["http://127.0.0.1:9/embed"]

            def batch_width(self, _percent: int) -> int:
                return 1

        work = engine.FileWork(host=Path("/tmp/nomad-turbo-consume-hold.zim"), kind="zim", source="/app/storage/zim/hold.zim")
        previous = engine._hold_announced
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with (
                    mock.patch.dict(os.environ, {"NOMAD_TURBO_STATE": tmp}),
                    mock.patch.object(segment_guard, "cached_write_facts", facts),
                    mock.patch.object(engine, "embed", fake_embed),
                    mock.patch.object(engine, "upsert", fake_upsert),
                    mock.patch.object(engine, "delete_collection", side_effect=AssertionError("deleted collection")),
                    mock.patch.object(store, "delete_collection", side_effect=AssertionError("deleted collection")),
                    mock.patch.object(store, "delete_source_points", side_effect=AssertionError("deleted points")),
                ):
                    with self.assertRaises(RuntimeError) as caught:
                        engine._consume_articles(
                            install,
                            _Pool(),
                            work,
                            stop,
                            None,
                            set(),
                            pull,
                        )
            self.assertIn("stopped", str(caught.exception).lower())
            self.assertEqual(calls["embed"], 1)
            self.assertGreaterEqual(calls["facts"], 1)
            self.assertEqual(calls["upsert"], 0)
            decision = segment_upsert_hold(returned["value"])
            self.assertEqual(decision.reason, "segment")
            self.assertFalse(decision.delete_points)
            self.assertFalse(decision.delete_collection)
        finally:
            engine._hold_announced = previous


class MeasureAndRepairPageTest(unittest.TestCase):
    def test_repair_measure_sees_a_fully_deleted_segment_and_a_temp_builder(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install = _install(str(root))
            plain = _segment(root, "plain-one", "plain", b"v" * 40, 2, None)
            _segment(root, "dead-one", "hnsw", b"d" * 16, 8, bytes([0xFF]))
            builder = (
                root
                / "qdrant"
                / "collections"
                / "nomad_knowledge_base"
                / "0"
                / "temp_segments"
                / "segment_builder_test"
            )
            builder.mkdir(parents=True)
            (builder / "marker").write_bytes(b"x")
            repair = measure_repair_facts(install)
            write = measure_write_facts(install)
            self.assertTrue(repair.optimization_running)
            self.assertEqual(repair.largest_plain_vector_bytes, 40)
            self.assertGreater(repair.largest_deleted_disk_bytes, 0)
            self.assertEqual(write.largest_deleted_disk_bytes, 0)
            self.assertEqual(write.largest_plain_vector_bytes, 40)
            self.assertEqual(segment_upsert_hold(repair).reason, "")
            refused = force_optimization(repair)
            self.assertFalse(refused.allow)
            self.assertIn("already in progress", refused.reason)
            self.assertFalse(refused.delete_collection)
            # The write cache reuses sizes, and a later read sees the new size.
            clear_write_cache()
            (plain / "vector_storage" / "vectors" / "chunk_0.mmap").write_bytes(b"v" * 40)
            first = cached_write_facts(install, now=0)
            (plain / "vector_storage" / "vectors" / "chunk_0.mmap").write_bytes(b"v" * 90)
            second = cached_write_facts(install, now=10)
            third = cached_write_facts(install, now=20)
            self.assertEqual(first.largest_plain_vector_bytes, 40)
            self.assertEqual(second.largest_plain_vector_bytes, 40)
            self.assertEqual(third.largest_plain_vector_bytes, 90)

    def test_a_fully_deleted_plain_segment_is_not_an_open_plain_segment(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _segment(root, "gone", "plain", b"v" * 8, 8, bytes([0xFF]))
            repair = measure_repair_facts(_install(str(root)))
            self.assertEqual(repair.largest_plain_vector_bytes, 0)
            self.assertGreater(repair.largest_deleted_disk_bytes, 0)
            self.assertFalse(repair.optimization_running)

    def test_repair_page_forces_optimization_without_a_file_list(self):
        import nomad_turbo.window as window

        source = Path(window.__file__).read_text()
        page = source[source.index("def _build_repair_page"): source.index("def _build_about_page")]
        self.assertIn('label="Optimize stuck segments"', page)
        self.assertIn("repair_optimize", page)
        self.assertNotIn("len(self.repair_store)", page)
        handler = source[source.index("def on_optimize_segments"): source.index("def on_repair_scan")]
        self.assertIn("apply_force_optimization", handler)
        self.assertIn("report_optimize", handler)
        self.assertNotIn("repair_store", handler)
        buttons = source[source.index("def _buttons"): source.index("def refresh_rows")]
        self.assertNotIn("len(self.repair_store)", buttons)
        self.assertIn("optimize_segments.set_sensitive(idle_repair)", buttons)

    def test_live_boundary_patches_optimizer_config_and_does_not_delete(self):
        install = _install("/tmp/nomad-turbo-boundary")
        calls = []

        def fake_request(url, payload=None, timeout=60, method=None):
            calls.append((url, payload, method))
            if method == "DELETE":
                raise AssertionError("collection deleted")
            return {
                "result": {
                    "config": {
                        "hnsw_config": {"on_disk": False},
                        "optimizer_config": {
                            "indexing_threshold": 10000,
                            "max_segment_size": MAX_SEGMENT_KB,
                        },
                    }
                }
            }

        with mock.patch("nomad_turbo.store._request", fake_request):
            boundary = segment_guard.LiveOptimizeBoundary(install)
            found = boundary.read_optimizer_config()
            boundary.patch_optimizer_config(found)
            boundary.request_optimize()
        self.assertEqual(found["indexing_threshold"], 10000)
        self.assertNotIn("hnsw_config", found)
        self.assertNotIn("on_disk", found)
        self.assertEqual(calls[1][2], "PATCH")
        self.assertEqual(set(calls[1][1]), {"optimizers_config"})
        self.assertNotIn("hnsw_config", calls[1][1]["optimizers_config"])
        self.assertNotIn("on_disk", calls[1][1]["optimizers_config"])
        self.assertEqual(calls[2][2], "PATCH")
        self.assertEqual(calls[2][1], {"optimizers_config": {}})
        self.assertNotIn("indexing_threshold", calls[2][1]["optimizers_config"])
        self.assertNotIn("hnsw_config", calls[2][1])
        self.assertNotEqual(calls[2][2], "DELETE")
        self.assertNotIn("hnsw_config", calls[0][0])
        for _url, payload, method in calls:
            if method == "DELETE":
                raise AssertionError("collection deleted")
            config = (payload or {}).get("optimizers_config") or {}
            self.assertNotEqual(config.get("indexing_threshold"), 0)
            self.assertNotIn("hnsw_config", payload or {})

    def test_live_boundary_restores_a_queue_it_paused_and_leaves_a_paused_queue(self):
        install = _install("/tmp/nomad-turbo-queue")
        boundary = segment_guard.LiveOptimizeBoundary(install)
        with (
            mock.patch("nomad_turbo.runtime.queue_is_paused", return_value=False),
            mock.patch("nomad_turbo.runtime.queuectl") as queue,
        ):
            boundary.pause_embeds()
            boundary.resume_embeds()
        self.assertEqual(queue.call_args_list[0].args, (install, "pause"))
        self.assertEqual(queue.call_args_list[1].args, (install, "resume"))

        boundary = segment_guard.LiveOptimizeBoundary(install)
        with (
            mock.patch("nomad_turbo.runtime.queue_is_paused", return_value=True),
            mock.patch("nomad_turbo.runtime.queuectl") as queue,
        ):
            boundary.pause_embeds()
            boundary.resume_embeds()
        queue.assert_not_called()

    def test_live_apply_schedules_a_rewrite_and_restores_settings(self):
        install = _install("/tmp/nomad-turbo-boundary-apply")
        calls = []

        def fake_request(url, payload=None, timeout=60, method=None):
            calls.append((url, payload, method))
            if method == "DELETE":
                raise AssertionError("collection deleted")
            config = (payload or {}).get("optimizers_config") or {}
            if config.get("indexing_threshold") == 0:
                raise AssertionError("threshold 0 disables indexing")
            return {
                "result": {
                    "config": {
                        "hnsw_config": {"on_disk": False},
                        "optimizer_config": {
                            "indexing_threshold": 10000,
                            "max_segment_size": MAX_SEGMENT_KB,
                            "deleted_threshold": 0.2,
                        },
                    }
                }
            }

        facts = SegmentFacts(
            largest_deleted_disk_bytes=100,
            free_bytes=200,
            spare_ram_bytes=0,
            optimization_running=False,
        )
        with (
            mock.patch("nomad_turbo.store._request", fake_request),
            mock.patch("nomad_turbo.runtime.queue_is_paused", return_value=False),
            mock.patch("nomad_turbo.runtime.queuectl") as queue,
        ):
            result = apply_force_optimization(facts, segment_guard.LiveOptimizeBoundary(install))
        self.assertTrue(result.ok)
        self.assertTrue(result.requested)
        self.assertIn("settings are restored", result.reason.lower())
        self.assertFalse(result.delete_collection)
        bodies = [payload for _url, payload, method in calls if method == "PATCH"]
        self.assertEqual(bodies[0], {"optimizers_config": {}})
        self.assertEqual(bodies[-1]["optimizers_config"]["indexing_threshold"], 10000)
        self.assertNotIn("hnsw_config", bodies[-1])
        self.assertNotIn("on_disk", bodies[-1]["optimizers_config"])
        for payload in bodies:
            self.assertNotEqual((payload or {}).get("optimizers_config", {}).get("indexing_threshold"), 0)
        self.assertEqual(queue.call_args_list[0].args, (install, "pause"))
        self.assertEqual(queue.call_args_list[-1].args, (install, "resume"))

    def test_live_apply_reports_a_failed_restore_and_never_sends_threshold_zero(self):
        install = _install("/tmp/nomad-turbo-boundary-restore")
        calls = []

        def fake_request(url, payload=None, timeout=60, method=None):
            calls.append((url, payload, method))
            if method == "DELETE":
                raise AssertionError("collection deleted")
            config = (payload or {}).get("optimizers_config") or {}
            if config.get("indexing_threshold") == 0:
                raise AssertionError("threshold 0 disables indexing")
            if method == "PATCH" and "indexing_threshold" in config:
                raise RuntimeError("restore failed")
            return {
                "result": {
                    "config": {
                        "optimizer_config": {
                            "indexing_threshold": 10000,
                            "max_segment_size": MAX_SEGMENT_KB,
                        }
                    }
                }
            }

        facts = SegmentFacts(
            largest_deleted_disk_bytes=100,
            free_bytes=200,
            spare_ram_bytes=0,
            optimization_running=False,
        )
        with (
            mock.patch("nomad_turbo.store._request", fake_request),
            mock.patch("nomad_turbo.runtime.queue_is_paused", return_value=True),
            mock.patch("nomad_turbo.runtime.queuectl") as queue,
        ):
            result = apply_force_optimization(facts, segment_guard.LiveOptimizeBoundary(install))
        self.assertFalse(result.ok)
        self.assertTrue(result.requested)
        self.assertNotIn("settings are restored", result.reason.lower())
        self.assertIn("could not be put back", result.reason.lower())
        self.assertFalse(result.delete_collection)
        bodies = [payload for _url, payload, method in calls if method == "PATCH"]
        self.assertEqual(bodies[0], {"optimizers_config": {}})
        self.assertTrue(any("indexing_threshold" in (payload.get("optimizers_config") or {}) for payload in bodies[1:]))
        for payload in bodies:
            self.assertNotEqual((payload or {}).get("optimizers_config", {}).get("indexing_threshold"), 0)
            self.assertNotIn("hnsw_config", payload or {})
        queue.assert_not_called()

    def test_live_apply_restores_when_the_schedule_call_raises(self):
        install = _install("/tmp/nomad-turbo-boundary-schedule")
        calls = []

        def fake_request(url, payload=None, timeout=60, method=None):
            calls.append((url, payload, method))
            config = (payload or {}).get("optimizers_config") or {}
            if config.get("indexing_threshold") == 0:
                raise AssertionError("threshold 0 disables indexing")
            if payload == {"optimizers_config": {}}:
                raise RuntimeError("schedule failed")
            return {
                "result": {
                    "config": {
                        "optimizer_config": {"indexing_threshold": 10000, "deleted_threshold": 0.2}
                    }
                }
            }

        facts = SegmentFacts(
            largest_deleted_disk_bytes=100,
            free_bytes=200,
            spare_ram_bytes=0,
            optimization_running=False,
        )
        with (
            mock.patch("nomad_turbo.store._request", fake_request),
            mock.patch("nomad_turbo.runtime.queue_is_paused", return_value=False),
            mock.patch("nomad_turbo.runtime.queuectl") as queue,
        ):
            result = apply_force_optimization(facts, segment_guard.LiveOptimizeBoundary(install))
        self.assertFalse(result.ok)
        self.assertTrue(result.requested)
        self.assertIn("failed", result.reason.lower())
        self.assertIn("settings are restored", result.reason.lower())
        bodies = [payload for _url, payload, method in calls if method == "PATCH"]
        self.assertEqual(bodies[0], {"optimizers_config": {}})
        self.assertEqual(bodies[-1]["optimizers_config"]["indexing_threshold"], 10000)
        self.assertEqual(queue.call_args_list[-1].args, (install, "resume"))


def _segment(root: Path, name: str, index_type: str, vectors: bytes, count: int, deleted: bytes | None) -> Path:
    segment = (
        root
        / "qdrant"
        / "collections"
        / "nomad_knowledge_base"
        / "0"
        / "segments"
        / name
    )
    vectors_dir = segment / "vector_storage" / "vectors"
    vectors_dir.mkdir(parents=True)
    (vectors_dir / "chunk_0.mmap").write_bytes(vectors)
    (vectors_dir / "status.dat").write_bytes(int(count).to_bytes(8, "little"))
    (vectors_dir / "config.json").write_text("{}\n")
    (segment / "segment.json").write_text(
        json.dumps(
            {
                "config": {
                    "vector_data": {
                        "": {"size": 768, "distance": "Cosine", "index": {"type": index_type}}
                    }
                }
            }
        )
    )
    if deleted is not None:
        (segment / "id_tracker.deleted").write_bytes(deleted)
    return segment


class OptimizeMessageTest(unittest.TestCase):
    def test_a_click_is_written_to_the_log(self):
        with self.assertLogs("nomad_turbo", level="INFO") as caught:
            line = segment_guard.report_optimize(
                "refused",
                "Optimization is already in progress. This click did not start another one.",
            )
        self.assertIn("Optimize stuck segments: refused.", line)
        self.assertIn("did not start another one", caught.output[0])

    def test_each_pause_starts_the_bar_over(self):
        first = segment_guard.track_optimize_pause(None, "segment", 10)
        same = segment_guard.track_optimize_pause(first, "segment", 25)
        self.assertEqual(same.generation, 1)
        self.assertEqual(same.started, 10)
        text, fraction = segment_guard.pause_caption(same, 10)
        self.assertIsNone(fraction)
        self.assertIn("Pause 1", text)
        self.assertIn("0s", text)
        self.assertIn("not reported a percent", text)
        ended = segment_guard.track_optimize_pause(same, "", 40)
        done, fraction = segment_guard.pause_caption(ended, 40)
        self.assertEqual(fraction, 1.0)
        self.assertIn("finished", done.lower())
        again = segment_guard.track_optimize_pause(ended, "space", 50)
        self.assertEqual(again.generation, 2)
        text, fraction = segment_guard.pause_caption(again, 50)
        self.assertIsNone(fraction)
        self.assertIn("Pause 2 · 0s", text)

    def test_an_open_build_is_already_running_and_has_no_percent(self):
        code, bar, detail, pulse = segment_guard.describe_optimizer_state("yellow", "", True)
        self.assertEqual(code, "running")
        self.assertEqual(bar, "Already optimizing")
        self.assertTrue(pulse)
        self.assertIn("already in progress", detail.lower())
        self.assertIn("not reported a percent", detail.lower())
        self.assertIn("will not start another", detail.lower())
        level, line = segment_guard.optimize_watch_line(code, detail)
        self.assertEqual(level, logging.INFO)
        self.assertEqual(line, "Optimize: a segment build is open. No percent reported.")
        self.assertNotIn("\n", line)
        error_level, error_line = segment_guard.optimize_watch_line(
            "error",
            "Qdrant reported an optimizer error: disk full. A segment build is open.",
        )
        self.assertEqual(error_level, logging.WARNING)
        self.assertIn("disk full", error_line)
        self.assertIn("segment build is open", error_line)
        waiting = segment_guard.describe_optimizer_state("grey", "", False)
        self.assertEqual(waiting[0], "waiting")
        self.assertFalse(waiting[3])
        self.assertIn("until an update is sent", waiting[2])
        text, pulse = segment_guard.optimize_bar_after_click(False, False, detail)
        self.assertEqual(text, "Already optimizing")
        self.assertTrue(pulse)

    def test_a_refused_copy_names_the_free_space(self):
        facts = SegmentFacts(
            largest_plain_vector_bytes=default_segment_cap_bytes() + 1,
            largest_plain_disk_bytes=5000,
            free_bytes=1000,
            spare_ram_bytes=10**15,
        )
        reason = force_optimization(facts).reason
        self.assertIn("free space", reason.lower())
        self.assertIn(segment_guard.format_bytes(1000), reason)
        self.assertIn("left unchanged", reason.lower())

    def test_a_build_folder_counts_as_optimization_already_open(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install = _install(str(root))
            self.assertFalse(segment_guard.optimization_open(install))
            folder = (
                root
                / "qdrant"
                / "collections"
                / "nomad_knowledge_base"
                / "0"
                / "temp_segments"
                / "segment_builder_test"
            )
            folder.mkdir(parents=True)
            self.assertTrue(segment_guard.optimization_open(install))


if __name__ == "__main__":
    unittest.main()
