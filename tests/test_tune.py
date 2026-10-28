"""Batch advice follows finished embed calls. It does not open a window or start indexing."""

import unittest

from nomad_turbo.session import Session
from nomad_turbo.tune import (
    FAIL_RETRY_CALLS,
    FAIL_RETRY_SECONDS,
    RETRY_CALLS,
    RETRY_SECONDS,
    EmbedSample,
    TuneHold,
    tune_advice,
)


def _calls(percent: int, count: int, seconds: float, ok: bool = True) -> list[EmbedSample]:
    return [EmbedSample(percent, 10, seconds, ok) for _ in range(count)]


def _advice(samples, percent, **kwargs):
    values = dict(
        auto=True,
        running=True,
        optimizing=False,
        gpu_free_mib=4000,
        hold=None,
        call_count=0,
    )
    values.update(kwargs)
    return tune_advice(samples, percent, **values)


class TuneAdviceTest(unittest.TestCase):
    def test_auto_raises_the_call_while_finished_calls_keep_up(self):
        advice = _advice(_calls(55, 3, 1.0), 55)
        self.assertTrue(advice.move)
        self.assertEqual(advice.percent, 60)
        self.assertIn("larger batch", advice.text)
        self.assertIsNone(advice.hold)

    def test_a_slower_larger_call_steps_down_and_waits_before_trying_again(self):
        samples = _calls(60, 3, 1.0) + _calls(65, 3, 1.3)
        advice = _advice(samples, 65, call_count=6)
        self.assertTrue(advice.move)
        self.assertEqual(advice.percent, 60)
        self.assertIsNotNone(advice.hold)
        self.assertEqual(advice.hold.above, 60)
        self.assertEqual(advice.hold.until_calls, 6 + RETRY_CALLS)
        self.assertEqual(advice.hold.until_time, RETRY_SECONDS)
        self.assertIn("slower", advice.text)
        stayed = samples + _calls(60, 3, 1.0)
        held = _advice(stayed, 60, hold=advice.hold, call_count=9)
        self.assertFalse(held.move)
        self.assertEqual(held.percent, 60)
        self.assertIn("short wait", held.text)
        early = _advice(stayed, 60, hold=advice.hold, call_count=advice.hold.until_calls, now=0)
        self.assertFalse(early.move)
        retried = _advice(
            stayed,
            60,
            hold=advice.hold,
            call_count=advice.hold.until_calls,
            now=advice.hold.until_time,
        )
        self.assertTrue(retried.move)
        self.assertEqual(retried.percent, 65)
        self.assertIn("again", retried.text)
        self.assertIsNotNone(retried.hold)
        self.assertEqual(retried.hold.kind, "settle")
        self.assertEqual(retried.hold.above, 65)
        settled = _advice(
            stayed + _calls(65, 3, 1.0),
            65,
            hold=retried.hold,
            call_count=retried.hold.until_calls,
            now=0,
        )
        self.assertFalse(settled.move)
        self.assertEqual(settled.percent, 65)

    def test_a_small_slowdown_does_not_drop_the_batch(self):
        advice = _advice(_calls(60, 3, 1.0) + _calls(65, 3, 1.1), 65)
        self.assertNotEqual(advice.percent, 60)
        self.assertNotIn("smaller", advice.text)

    def test_one_slow_call_does_not_drop_the_batch(self):
        advice = _advice(_calls(60, 3, 1.0) + _calls(65, 1, 1.3), 65, call_count=4)
        self.assertFalse(advice.move)
        self.assertEqual(advice.percent, 65)

    def test_a_new_visit_is_not_judged_by_the_older_slow_calls(self):
        samples = _calls(60, 3, 1.0) + _calls(65, 3, 1.3) + _calls(60, 3, 1.0)
        advice = _advice(samples, 65, call_count=30)
        self.assertFalse(advice.move)
        self.assertEqual(advice.percent, 65)
        self.assertIn("Watching", advice.text)

    def test_a_changed_pace_ends_the_wait_early(self):
        hold = TuneHold(above=60, until_calls=1000, pace=10.0)
        advice = _advice(_calls(60, 3, 2.0), 60, hold=hold, call_count=10)
        self.assertTrue(advice.move)
        self.assertEqual(advice.percent, 65)

    def test_a_failed_call_steps_down_without_waiting_for_three(self):
        advice = _advice([EmbedSample(80, 10, 1.0, False)], 80, call_count=10)
        self.assertEqual(advice.percent, 75)
        self.assertTrue(advice.move)
        self.assertIn("struggled", advice.text)
        self.assertIsNotNone(advice.hold)
        self.assertEqual(advice.hold.until_calls, 10 + FAIL_RETRY_CALLS)
        self.assertEqual(advice.hold.until_time, FAIL_RETRY_SECONDS)
        self.assertGreater(FAIL_RETRY_CALLS, RETRY_CALLS)
        self.assertGreater(FAIL_RETRY_SECONDS, RETRY_SECONDS)

    def test_manual_recommends_and_does_not_move_the_slider(self):
        advice = _advice(_calls(100, 3, 1.0), 100, auto=False)
        self.assertFalse(advice.move)
        self.assertEqual(advice.percent, 100)
        self.assertIn("higher setting", advice.text)

    def test_a_segment_wait_is_not_blamed_on_the_batch(self):
        advice = _advice(_calls(100, 3, 1.0), 100, optimizing=True)
        self.assertFalse(advice.move)
        self.assertIn("waiting on a segment", advice.text)

    def test_the_slider_top_is_not_called_the_card_maximum(self):
        advice = _advice(_calls(150, 3, 1.0), 150)
        self.assertFalse(advice.move)
        self.assertEqual(advice.percent, 150)
        self.assertIn("slider goes no higher", advice.text)
        self.assertNotIn("absolute", advice.text.lower())

    def test_little_free_card_memory_stops_the_climb(self):
        advice = _advice(_calls(100, 3, 1.0), 100, gpu_free_mib=100)
        self.assertFalse(advice.move)
        self.assertIn("nearly full", advice.text)

    def test_the_session_keeps_recent_calls_only(self):
        session = Session(55)
        for _index in range(60):
            session.note_embed_call(55, 8, 1.0, True)
        self.assertEqual(len(session.embed_calls()), 48)
        self.assertEqual(session.embed_calls()[-1][0], 55)
        self.assertEqual(session.embed_call_count(), 60)
