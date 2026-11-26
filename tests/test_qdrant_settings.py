"""Storage choices for the knowledge-base collection.

These tests stay off the live Qdrant. Requests are faked, and the segment
fixtures live in a temporary directory.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nomad_turbo.discover import Install, load_install
from nomad_turbo.live import ButtonFacts, control_senses
from nomad_turbo.qdrant_settings import (
    accepted_datatype,
    apply_note,
    apply_plan,
    bytes_per_vector,
    canonical_datatype,
    changed_options,
    create_body,
    parse_version,
    quantization_enabled,
    graph_auto_wants_disk,
    rebuild_warning,
    vectors_auto_wants_disk,
    snapshot_from_info,
    status_text,
    storage_choices,
    update_body,
    vector_bytes_from_config,
)
from nomad_turbo.segment_guard import _describe_segment, _vector_width, measure_write_facts
from nomad_turbo.store import StoreError, collection_ready, read_collection_view, update_collection_options
from nomad_turbo import store


ROOT = Path(__file__).resolve().parents[1]


def _install(**overrides) -> Install:
    values = dict(
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
        collection="nomad_knowledge_base",
    )
    values.update(overrides)
    return Install(**values)


class StorageChoiceTest(unittest.TestCase):
    def test_version_gates_and_vector_widths(self):
        self.assertIsNone(parse_version(""))
        self.assertIsNone(parse_version("dev"))
        self.assertEqual(parse_version("1.16.3"), (1, 16, 3))
        self.assertEqual(parse_version("1.20.0-dev"), (1, 20, 0))
        self.assertEqual(storage_choices(None), ("float32", "float16"))
        self.assertEqual(storage_choices((1, 8, 9)), ("float32",))
        self.assertEqual(storage_choices((1, 16, 3)), ("float32", "float16"))
        self.assertEqual(
            storage_choices((1, 20, 0)),
            ("float32", "float16", "turbo16", "turbo8", "turbo4"),
        )
        self.assertEqual(canonical_datatype(None), "float32")
        self.assertEqual(canonical_datatype(" Float16 "), "float16")
        self.assertEqual(canonical_datatype("uint8"), "uint8")
        self.assertEqual(accepted_datatype("float16", ("float32", "float16")), "float16")
        with self.assertRaises(ValueError):
            accepted_datatype("uint8", ("float32", "float16", "uint8"))
        with self.assertRaises(ValueError):
            accepted_datatype("turbo16", ("float32", "float16"))
        self.assertEqual(bytes_per_vector("float32", 768), 3072)
        self.assertEqual(bytes_per_vector("float16", 768), 1536)
        self.assertEqual(bytes_per_vector("turbo16", 768), 1536)
        self.assertEqual(bytes_per_vector("turbo8", 768), 768)
        self.assertEqual(bytes_per_vector("turbo4", 768), 384)

    def test_a_missing_datatype_is_float32_and_a_named_vector_still_counts(self):
        unnamed = {"params": {"vectors": {"size": 768, "distance": "Cosine"}}}
        self.assertEqual(vector_bytes_from_config(unnamed), 3072)
        named = {"params": {"vectors": {"text": {"size": 768, "datatype": "float16"}}}}
        self.assertEqual(vector_bytes_from_config(named), 1536)
        self.assertFalse(quantization_enabled({"quantization_config": None}))
        self.assertFalse(quantization_enabled({}))
        self.assertTrue(quantization_enabled({"quantization_config": {"product": {"compression": "x32"}}}))
        snap = snapshot_from_info(
            {
                "points_count": 86,
                "config": {
                    "params": {"vectors": {"size": 768, "distance": "Cosine"}},
                    "hnsw_config": {"on_disk": False, "m": 16},
                    "quantization_config": None,
                },
            }
        )
        self.assertEqual(snap["datatype"], "float32")
        self.assertFalse(snap["quantization"])
        self.assertFalse(snap["hnsw_on_disk"])
        self.assertEqual(snap["points"], 86)
        self.assertEqual(snap["size"], 768)

    def test_plans_and_bodies_keep_nomads_vector(self):
        live = {"present": True, "datatype": "float32", "quantization": False, "hnsw_on_disk": False, "points": 10}
        self.assertEqual(apply_plan(None, "float16", False, False), "create")
        self.assertEqual(apply_plan({"present": False}, "float16", False, False), "create")
        self.assertEqual(apply_plan(live, "float16", True, True), "rebuild")
        self.assertEqual(apply_plan(live, "float32", True, False), "update")
        self.assertEqual(apply_plan(live, "float32", False, True), "update")
        self.assertEqual(apply_plan(live, "float32", False, False), "none")
        self.assertEqual(apply_plan(live, "float32", False, False, True), "update")
        self.assertEqual(changed_options(live, False, False, True), {"vectors_on_disk": True})
        product = dict(live, quantization=True)
        self.assertEqual(apply_plan(product, "float32", True, False), "none")
        self.assertEqual(changed_options(product, True, True), {"hnsw_on_disk": True})
        body = create_body(768, "float32", False, False)
        self.assertEqual(body["vectors"], {"size": 768, "distance": "Cosine"})
        self.assertNotIn("datatype", body["vectors"])
        self.assertNotIn("quantization_config", body)
        self.assertEqual(body["hnsw_config"], {"on_disk": False})
        self.assertTrue(body["on_disk_payload"])
        half = create_body(768, "float16", True, True)
        self.assertEqual(half["vectors"]["datatype"], "float16")
        self.assertEqual(half["vectors"]["distance"], "Cosine")
        self.assertEqual(half["quantization_config"]["scalar"]["type"], "int8")
        self.assertEqual(half["quantization_config"]["scalar"]["quantile"], 0.99)
        self.assertIs(half["quantization_config"]["scalar"]["always_ram"], False)
        self.assertIs(half["vectors"]["on_disk"], True)
        turned_on = update_body({"quantization": True})
        self.assertEqual(turned_on["vectors"], {"": {"on_disk": True}})
        self.assertIs(turned_on["quantization_config"]["scalar"]["always_ram"], False)
        self.assertNotIn("uint8", json.dumps(half))
        with self.assertRaises(ValueError):
            create_body(768, "uint8", False, False)
        self.assertEqual(update_body({"quantization": False}), {"quantization_config": None})
        self.assertEqual(update_body({"vectors_on_disk": True}), {"vectors": {"": {"on_disk": True}}})
        forced_off = update_body({"quantization": True, "vectors_on_disk": False})
        self.assertEqual(forced_off["vectors"], {"": {"on_disk": False}})
        self.assertEqual(update_body({"hnsw_on_disk": True}), {"hnsw_config": {"on_disk": True}})
        self.assertNotIn("m", update_body({"hnsw_on_disk": False})["hnsw_config"])
        self.assertEqual(update_body({}), {})
        title, warning = rebuild_warning(12, "float16")
        self.assertIn("DELETE", warning)
        self.assertIn("Manual", warning)
        self.assertIn("Float16", warning)
        self.assertIn("Replace", title)
        self.assertIn("Stop indexing", apply_note("rebuild", "float16", 12, busy=True))
        self.assertIn("Points stay", apply_note("update", "float32", 12, busy=False))
        self.assertIn("already uses", apply_note("none", "float32", 12, busy=False))
        self.assertIn("opened", status_text({"loaded": False}, "nomad_knowledge_base"))
        gib = 1024 ** 3
        self.assertFalse(
            graph_auto_wants_disk(automatic=False, on_disk=False, points=10**9, spare_ram_bytes=1)
        )
        self.assertFalse(
            graph_auto_wants_disk(automatic=True, on_disk=True, points=10**9, spare_ram_bytes=1)
        )
        self.assertFalse(
            graph_auto_wants_disk(automatic=True, on_disk=False, points=1_000_000, spare_ram_bytes=20 * gib)
        )
        self.assertFalse(
            graph_auto_wants_disk(automatic=True, on_disk=False, points=40_000_000, spare_ram_bytes=None)
        )
        self.assertTrue(
            graph_auto_wants_disk(automatic=True, on_disk=False, points=40_000_000, spare_ram_bytes=6 * gib)
        )
        self.assertTrue(
            graph_auto_wants_disk(automatic=True, on_disk=False, points=0, spare_ram_bytes=3 * gib)
        )
        self.assertFalse(
            vectors_auto_wants_disk(
                automatic=False, on_disk=False, quantization=True, points=10**9, dimension=768, spare_ram_bytes=1
            )
        )
        self.assertFalse(
            vectors_auto_wants_disk(
                automatic=True, on_disk=True, quantization=True, points=10**9, dimension=768, spare_ram_bytes=1
            )
        )
        self.assertFalse(
            vectors_auto_wants_disk(
                automatic=True, on_disk=False, quantization=False, points=10**9, dimension=768, spare_ram_bytes=1
            )
        )
        self.assertFalse(
            vectors_auto_wants_disk(
                automatic=True,
                on_disk=False,
                quantization=True,
                points=1_000_000,
                dimension=768,
                spare_ram_bytes=20 * gib,
            )
        )
        self.assertTrue(
            vectors_auto_wants_disk(
                automatic=True,
                on_disk=False,
                quantization=True,
                points=40_000_000,
                dimension=768,
                spare_ram_bytes=6 * gib,
            )
        )


class CollectionRequestTest(unittest.TestCase):
    def test_a_new_collection_uses_the_saved_storage_choices(self):
        install = _install()
        install.qdrant_datatype = "float16"
        install.qdrant_quantization = True
        install.qdrant_hnsw_on_disk = True
        calls = []
        created = {"done": False}

        def fake_request(url, payload=None, timeout=60, method=None):
            calls.append((method, url, payload))
            if str(url).rstrip("/").endswith("/collections"):
                return {"result": {"collections": []}}
            if str(url).endswith("/index"):
                return {"result": True}
            if method == "PUT":
                created["done"] = True
                return {"result": True}
            if "collections/" in str(url):
                if not created["done"]:
                    raise StoreError("qdrant 404: missing")
                return {"result": {"config": {"optimizer_config": {"max_segment_size": 4 * 1024 * 1024}}}}
            raise AssertionError(url)

        original = store._request
        store._request = fake_request
        try:
            collection_ready(install)
        finally:
            store._request = original
        bodies = [payload for method, _url, payload in calls if method == "PUT" and payload and "vectors" in payload]
        self.assertEqual(len(bodies), 1)
        body = bodies[0]
        self.assertEqual(
            body["vectors"],
            {"size": 768, "distance": "Cosine", "datatype": "float16", "on_disk": True},
        )
        self.assertEqual(body["hnsw_config"], {"on_disk": True})
        self.assertTrue(body["on_disk_payload"])
        self.assertIs(body["quantization_config"]["scalar"]["always_ram"], False)

    def test_the_default_create_is_still_float32_without_quantization(self):
        install = _install()
        body = create_body(
            install.dimension,
            install.qdrant_datatype,
            install.qdrant_quantization,
            install.qdrant_hnsw_on_disk,
        )
        self.assertEqual(body["vectors"], {"size": 768, "distance": "Cosine"})
        self.assertNotIn("quantization_config", body)
        self.assertFalse(body["hnsw_config"]["on_disk"])

    def test_a_patch_sends_only_the_changed_field(self):
        install = _install()
        calls = []

        def fake_request(url, payload=None, timeout=60, method=None):
            calls.append((method, url, payload))
            return {"result": True}

        original = store._request
        store._request = fake_request
        try:
            update_collection_options(install)
            update_collection_options(install, quantization=False)
            update_collection_options(install, hnsw_on_disk=True)
        finally:
            store._request = original
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], "PATCH")
        self.assertEqual(calls[0][2], {"quantization_config": None})
        self.assertEqual(calls[1][2], {"hnsw_config": {"on_disk": True}})
        self.assertNotIn("vectors", calls[0][2])
        self.assertNotIn("vectors", calls[1][2])

    def test_automatic_patches_the_graph_once_when_memory_cannot_hold_it(self):
        install = _install()
        install.qdrant_graph_auto = True
        install.qdrant_hnsw_on_disk = False
        store._graph_auto_checked = 0.0
        view = {"ok": True, "present": True, "hnsw_on_disk": False, "points": 40_000_000}
        with (
            mock.patch.object(store, "read_collection_view", return_value=view),
            mock.patch("nomad_turbo.segment_guard._spare_ram_bytes", return_value=6 * 1024**3),
            mock.patch.object(store, "update_collection_options") as patch,
            mock.patch("nomad_turbo.discover.save_install") as save,
        ):
            self.assertTrue(store.consider_graph_auto(install))
            self.assertFalse(store.consider_graph_auto(install))
        patch.assert_called_once_with(install, hnsw_on_disk=True)
        self.assertTrue(install.qdrant_hnsw_on_disk)
        save.assert_called_once_with(install)
        install.qdrant_graph_auto = False
        store._graph_auto_checked = 0.0
        with mock.patch.object(store, "update_collection_options") as blocked:
            self.assertFalse(store.consider_graph_auto(install))
        blocked.assert_not_called()

    def test_automatic_patches_the_vectors_once_when_the_int8_copy_would_not_fit(self):
        install = _install()
        self.assertTrue(install.qdrant_vectors_auto)
        self.assertFalse(install.qdrant_graph_auto)
        install.qdrant_vectors_on_disk = False
        store._vectors_auto_checked = 0.0
        view = {
            "ok": True,
            "present": True,
            "vectors_on_disk": False,
            "quantization": True,
            "points": 40_000_000,
            "size": 768,
        }
        with (
            mock.patch.object(store, "read_collection_view", return_value=view),
            mock.patch("nomad_turbo.segment_guard._spare_ram_bytes", return_value=6 * 1024**3),
            mock.patch.object(store, "update_collection_options") as patch,
            mock.patch("nomad_turbo.discover.save_install") as save,
        ):
            self.assertTrue(store.consider_vectors_auto(install))
            self.assertFalse(store.consider_vectors_auto(install))
        patch.assert_called_once_with(install, vectors_on_disk=True)
        self.assertTrue(install.qdrant_vectors_on_disk)
        save.assert_called_once_with(install)
        install.qdrant_vectors_auto = False
        store._vectors_auto_checked = 0.0
        with mock.patch.object(store, "update_collection_options") as blocked:
            self.assertFalse(store.consider_vectors_auto(install))
        blocked.assert_not_called()

    def test_the_page_shows_the_live_type_and_a_missing_collection_shows_the_saved_one(self):
        install = _install()
        install.qdrant_datatype = "float16"
        install.qdrant_quantization = True

        def present(url, payload=None, timeout=60, method=None):
            if "collections/" in str(url):
                return {
                    "result": {
                        "points_count": 4,
                        "config": {
                            "params": {"vectors": {"size": 768, "distance": "Cosine"}},
                            "hnsw_config": {"on_disk": False},
                            "quantization_config": None,
                        },
                    }
                }
            return {"version": "1.16.3"}

        original = store._request
        store._request = present
        try:
            view = read_collection_view(install)
        finally:
            store._request = original
        self.assertTrue(view["ok"])
        self.assertTrue(view["present"])
        self.assertEqual(view["datatype"], "float32")
        self.assertFalse(view["quantization"])
        self.assertEqual(view["choices"], ("float32", "float16"))
        self.assertEqual(view["version"], "1.16.3")

        def missing(url, payload=None, timeout=60, method=None):
            if "collections/" in str(url):
                raise StoreError("qdrant 404: missing")
            return {"version": "1.16.3"}

        store._request = missing
        try:
            empty = read_collection_view(install)
        finally:
            store._request = original
        self.assertTrue(empty["ok"])
        self.assertFalse(empty["present"])
        self.assertEqual(empty["datatype"], "float16")
        self.assertTrue(empty["quantization"])

        def ancient(url, payload=None, timeout=60, method=None):
            if "collections/" in str(url):
                raise StoreError("qdrant 404: missing")
            return {"version": "1.8.2"}

        store._request = ancient
        try:
            old = read_collection_view(install)
        finally:
            store._request = original
        self.assertEqual(old["choices"], ("float32",))
        self.assertEqual(old["datatype"], "float32")

        def down(url, payload=None, timeout=60, method=None):
            raise StoreError("qdrant unreachable: timed out")

        store._request = down
        try:
            failed = read_collection_view(install)
        finally:
            store._request = original
        self.assertFalse(failed["ok"])
        self.assertTrue(failed["loaded"])
        self.assertIn("timed out", failed["error"])


class SavedChoiceTest(unittest.TestCase):
    def test_saved_storage_choices_survive_a_reload(self):
        discovered = _install()
        saved = _install()
        saved.qdrant_datatype = "float16"
        saved.qdrant_quantization = True
        saved.qdrant_hnsw_on_disk = True
        saved.qdrant_graph_auto = False
        saved.qdrant_vectors_on_disk = True
        saved.qdrant_vectors_auto = False
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(saved.to_json())
            with (
                mock.patch.dict(os.environ, {"NOMAD_TURBO_CONFIG": str(path)}),
                mock.patch("nomad_turbo.discover.discover", return_value=discovered),
            ):
                loaded = load_install()
            self.assertEqual(loaded.qdrant_datatype, "float16")
            self.assertTrue(loaded.qdrant_quantization)
            self.assertTrue(loaded.qdrant_hnsw_on_disk)
            self.assertFalse(loaded.qdrant_graph_auto)
            self.assertTrue(loaded.qdrant_vectors_on_disk)
            self.assertFalse(loaded.qdrant_vectors_auto)
            old = json.loads(saved.to_json())
            del old["qdrant_datatype"]
            del old["qdrant_quantization"]
            del old["qdrant_hnsw_on_disk"]
            del old["qdrant_graph_auto"]
            del old["qdrant_vectors_on_disk"]
            del old["qdrant_vectors_auto"]
            path.write_text(json.dumps(old))
            again = _install()
            with (
                mock.patch.dict(os.environ, {"NOMAD_TURBO_CONFIG": str(path)}),
                mock.patch("nomad_turbo.discover.discover", return_value=again),
            ):
                legacy = load_install()
        self.assertEqual(legacy.qdrant_datatype, "float32")
        self.assertFalse(legacy.qdrant_quantization)
        self.assertFalse(legacy.qdrant_hnsw_on_disk)
        self.assertFalse(legacy.qdrant_graph_auto)
        self.assertFalse(legacy.qdrant_vectors_on_disk)
        self.assertTrue(legacy.qdrant_vectors_auto)


class SegmentWidthTest(unittest.TestCase):
    def test_a_missing_status_file_uses_the_stored_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection = root / "qdrant" / "collections" / "nomad_knowledge_base"
            segment = collection / "0" / "segments" / "one"
            vectors = segment / "vector_storage" / "vectors"
            vectors.mkdir(parents=True)
            (vectors / "chunk_0.mmap").write_bytes(b"\0" * (1536 * 3))
            (segment / "segment.json").write_text(
                json.dumps({"config": {"vector_data": {"": {"index": {"type": "plain"}}}}})
            )
            (collection / "config.json").write_text(
                json.dumps(
                    {"params": {"vectors": {"size": 768, "distance": "Cosine", "datatype": "float16"}}}
                )
            )
            self.assertEqual(_vector_width(collection), 1536)
            described = _describe_segment(segment, count_deleted=False, vector_width=1536)
            self.assertEqual(described[3], 3)
            wide = _describe_segment(segment, count_deleted=False, vector_width=3072)
            self.assertEqual(wide[3], 1)
            install = _install(storage_host=str(root))
            facts = measure_write_facts(install)
            self.assertEqual(facts.largest_plain_vector_count, 3)
            self.assertEqual(facts.largest_plain_vector_bytes, 1536 * 3)
            self.assertEqual(_vector_width(root / "missing"), 3072)


class PageSourceTest(unittest.TestCase):
    def test_the_page_sits_between_settings_and_about_and_keeps_a_failed_apply(self):
        text = (ROOT / "nomad_turbo" / "window.py").read_text(encoding="utf-8")
        self.assertLess(text.index("_build_settings_page()"), text.index("_build_qdrant_page()"))
        self.assertLess(text.index("_build_qdrant_page()"), text.index("_build_about_page()"))
        self.assertIn('add_titled(page, "qdrant", "Qdrant")', text)
        self.assertIn("Qdrant settings", text)
        builder = text.split("def _build_qdrant_page", 1)[1].split("\n    def ", 1)[0]
        self.assertNotIn("read_collection_view", builder)
        self.assertNotIn("_refresh_qdrant_page", builder)
        for key in (
            "qdrant_page",
            "qdrant_storage",
            "qdrant_quant",
            "qdrant_vectors",
            "qdrant_vectors_auto",
            "qdrant_graph",
            "qdrant_graph_auto",
            "qdrant_apply",
        ):
            self.assertIn(f'TIPS["{key}"]', builder)
        self.assertLess(builder.index('label="Vectors on disk"'), builder.index('label="Automatic"'))
        self.assertLess(builder.index('label="Automatic"'), builder.index('label="Graph on disk"'))
        self.assertLess(builder.index('label="Graph on disk"'), builder.index('label="Graph automatic"'))
        self.assertIn("self.qdrant_vectors_disk", builder)
        self.assertNotIn("self.qdrant_vectors =", builder)
        watch = text.split("def _watch_qdrant", 1)[1].split("\n    def ", 1)[0]
        self.assertIn("previous_vectors=self.qdrant_vectors", watch)
        show = text.split("def _show_qdrant_page", 1)[1].split("\n    def ", 1)[0]
        self.assertLess(show.index("_paint_qdrant_note()"), show.index("qdrant_apply_note.set_text(error)"))
        finished = text.split("def _qdrant_finished", 1)[1].split("\n    def ", 1)[0]
        self.assertLess(finished.index("_qdrant_error"), finished.index("_refresh_qdrant_page()"))
        rebuild = text.split("def _qdrant_rebuild", 1)[1].split("\n    def ", 1)[0]
        self.assertIn("not removed", rebuild)
        self.assertIn("delete_index_only", rebuild)
        write = text.split("def _qdrant_write", 1)[1].split("\n    def ", 1)[0]
        self.assertLess(write.index("collection_ready"), write.index("save_install(self.install)"))
        self.assertIn('if "quantization" in changes else None', write)
        tips = (ROOT / "nomad_turbo" / "tips.py").read_text(encoding="utf-8")
        self.assertIn("uint8", tips)
        self.assertIn("DELETE", tips)
        self.assertIn("768", tips)
        self.assertNotIn("this machine", tips.lower())

    def test_a_quantization_write_blocks_start_and_does_not_enable_stop(self):
        idle = ButtonFacts(rows_ready=True, qdrant_ready=True)
        ready = control_senses("off", idle)
        self.assertTrue(ready["start"])
        self.assertTrue(ready["delete"])
        self.assertFalse(ready["stop"])
        busy = ButtonFacts(rows_ready=True, qdrant_ready=True, qdrant_busy=True)
        writing = control_senses("off", busy)
        self.assertFalse(writing["start"])
        self.assertFalse(writing["delete"])
        self.assertFalse(writing["measure"])
        self.assertFalse(writing["repair_scan"])
        self.assertFalse(writing["repair_start"])
        self.assertFalse(writing["optimize_segments"])
        self.assertFalse(writing["stop"])
        deleting = ButtonFacts(rows_ready=True, qdrant_ready=True, delete_alive=True)
        rebuild = control_senses("off", deleting)
        self.assertTrue(rebuild["stop"])
        self.assertFalse(rebuild["start"])
        self.assertFalse(rebuild["delete"])


if __name__ == "__main__":
    unittest.main()
