import unittest
from datetime import datetime, timezone

import nomad_turbo.qdrant_status as qdrant_status
from nomad_turbo.qdrant_status import activity_from_text, changes_allowed, describe_qdrant, uptime_text


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
            **common,
        )
        self.assertFalse(listed.ready)
        self.assertIn("not answering", listed.text)
        self.assertIn("did not answer", listed.text)

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
