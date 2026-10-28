import unittest
from datetime import datetime, timezone

import nomad_turbo.qdrant_status as qdrant_status
from nomad_turbo.qdrant_status import (
    QdrantView,
    activity_from_text,
    changes_allowed,
    describe_qdrant,
    growth_line,
    indexing_problem_line,
    indexing_qdrant_problem,
    read_vector_count,
    uptime_text,
)


class QdrantStatusTest(unittest.TestCase):
    def test_the_newest_log_line_drops_the_level(self):
        text = "\n".join(
            [
                "2026-10-24T00:00:00.000Z INFO qdrant::startup: starting",
                "2026-10-24T00:00:01.000Z INFO qdrant::collection: Collection is ready",
            ]
        )
        self.assertEqual(activity_from_text(text), "Collection is ready")
        self.assertEqual(activity_from_text(""), "")
        self.assertEqual(len(activity_from_text("INFO " + ("x" * 400))), 220)

    def test_uptime_reads_docker_timestamps(self):
        now = datetime(2026, 10, 24, 1, 2, 3, tzinfo=timezone.utc)
        self.assertEqual(uptime_text("2026-10-24T00:00:00.123456789Z", now), "1h 2m")
        self.assertEqual(uptime_text("2026-10-24T01:00:00Z", now), "2m 3s")
        self.assertEqual(uptime_text("2026-10-24T01:02:00Z", now), "3s")
        self.assertEqual(uptime_text("", now), "")
        self.assertEqual(uptime_text("0001-01-01T00:00:00Z", now), "")
        self.assertEqual(uptime_text("not-a-time", now), "")

    def test_ready_requires_the_collection(self):
        common = {
            "running": True,
            "status": "running",
            "collection": "nomad_knowledge_base",
            "activity": "opening segments",
            "uptime": "2m 1s",
        }
        starting = describe_qdrant(
            health_ok=False,
            ready_ok=False,
            collection_ok=False,
            **common,
        )
        self.assertFalse(starting.ready)
        self.assertIn("has not passed the health check", starting.text)
        self.assertIn("was not asked", starting.text)
        self.assertIn("Last log: opening segments", starting.text)

        listed = describe_qdrant(
            health_ok=True,
            ready_ok=True,
            collection_ok=False,
            vectors=1000,
            previous_vectors=800,
            **common,
        )
        self.assertFalse(listed.ready)
        self.assertIn("Do not restart it or delete the index.", listed.text)
        self.assertIn("did not answer", listed.text)
        self.assertIn("grew by 200 vectors", listed.text)
        self.assertIn("now holds 1,000", listed.text)
        self.assertIn("A restart would replay this write.", listed.text)
        stalled = describe_qdrant(
            health_ok=True,
            ready_ok=True,
            collection_ok=False,
            vectors=1000,
            previous_vectors=1000,
            **common,
        )
        self.assertIn("still holds 1,000 vectors", stalled.text)
        self.assertIn("Leave Qdrant running until the collection answers.", stalled.text)

        ready = describe_qdrant(
            health_ok=True,
            ready_ok=True,
            collection_ok=True,
            **common,
        )
        self.assertTrue(ready.ready)
        self.assertIn("Qdrant is ready for changes.", ready.text)
        self.assertIn("Collection nomad_knowledge_base answered.", ready.text)

        down = describe_qdrant(
            running=False,
            status="exited",
            health_ok=False,
            ready_ok=False,
            collection_ok=False,
            collection="nomad_knowledge_base",
            activity="",
            uptime="",
        )
        self.assertFalse(down.ready)
        self.assertIn("Qdrant is exited.", down.text)
        self.assertIn("Not ready for changes.", down.text)

    def test_the_segment_counter_and_its_wording(self):
        self.assertIsNone(read_vector_count(b""))
        self.assertIsNone(read_vector_count(b"\x01\x02"))
        self.assertEqual(read_vector_count((24556160).to_bytes(8, "little") + b"\x00"), 24556160)
        self.assertIn("could not be read", growth_line(10, None))
        self.assertIn("holds 4 vectors", growth_line(None, 4))

    def test_a_qdrant_problem_during_indexing_is_logged_once(self):
        ready = QdrantView(text="Qdrant is ready for changes.", ready=True, collection_status="green")
        self.assertEqual(indexing_qdrant_problem(ready), "")
        yellow = QdrantView(
            text="Qdrant is ready for changes.",
            ready=True,
            collection_status="yellow",
        )
        self.assertEqual(indexing_qdrant_problem(yellow), "")
        failed = QdrantView(
            text="Qdrant is still writing. Do not restart it or delete the index.\nCollection did not answer.",
            ready=False,
        )
        self.assertEqual(
            indexing_qdrant_problem(failed),
            "Qdrant is still writing. Do not restart it or delete the index.",
        )
        broken = QdrantView(
            text="Qdrant is ready for changes.",
            ready=True,
            optimizer_error="segment build failed",
        )
        self.assertEqual(indexing_qdrant_problem(broken), "Optimizer error: segment build failed.")
        red = QdrantView(text="Qdrant is ready for changes.", ready=True, collection_status="red")
        self.assertIn("error state", indexing_qdrant_problem(red))

        quiet, remembered = indexing_problem_line("", False, "Optimizer error: segment build failed.")
        self.assertEqual((quiet, remembered), ("", ""))
        first, remembered = indexing_problem_line("", True, "Optimizer error: segment build failed.")
        self.assertIn("Qdrant problem while indexing:", first)
        self.assertIn("segment build failed", first)
        again, remembered = indexing_problem_line(remembered, True, "Optimizer error: segment build failed.")
        self.assertEqual(again, "")
        self.assertIn("segment build failed", remembered)
        cleared, remembered = indexing_problem_line(remembered, True, "")
        self.assertIn("Qdrant problem cleared", cleared)
        self.assertEqual(remembered, "")

    def test_changes_wait_until_the_collection_answers(self):
        self.assertFalse(changes_allowed(False))
        self.assertTrue(changes_allowed(True))

    def test_a_timeout_after_store_is_reloaded_is_not_an_error(self):
        # importlib.reload(store) builds a new StoreError. The probe must still
        # treat that as "the collection did not answer" instead of a traceback.
        class StoreError(Exception):
            pass

        original = qdrant_status._request

        def explode(url, timeout=3):
            raise StoreError("qdrant unreachable: timed out")

        qdrant_status._request = explode
        try:
            self.assertFalse(
                qdrant_status._collection_answered("http://127.0.0.1:6333/collections/nomad_knowledge_base")
            )
        finally:
            qdrant_status._request = original

    def test_an_answer_from_the_collection_counts(self):
        original = qdrant_status._request
        qdrant_status._request = lambda url, timeout=3: {"result": {}}
        try:
            self.assertTrue(qdrant_status._collection_answered("http://127.0.0.1:6333/collections/x"))
        finally:
            qdrant_status._request = original


if __name__ == "__main__":
    unittest.main()
