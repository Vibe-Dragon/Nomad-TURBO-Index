"""Batch advice follows finished embed calls. It does not open a window or start indexing."""

import unittest

from nomad_turbo.session import Session
from nomad_turbo.tune import EmbedSample, tune_advice


def _calls(percent: int, count: int, seconds: float, ok: bool = True) -> list[EmbedSample]:
    return [EmbedSample(percent, 10, seconds, ok) for _ in range(count)]


class TuneAdviceTest(unittest.TestCase):
    def test_auto_raises_the_call_while_finished_calls_keep_up(self):
        advice = tune_advice(
            _calls(55, 3, 1.0),
            55,
            auto=True,
            running=True,
            optimizing=False,
            gpu_free_mib=4000,
            ceiling=None,
        )
        self.assertTrue(advice.move)
        self.assertEqual(advice.percent, 60)
        self.assertIn("larger batch", advice.text)

    def test_a_slower_larger_call_steps_back_to_the_measured_line(self):
        samples = _calls(60, 3, 1.0) + _calls(65, 3, 1.3)
        advice = tune_advice(
            samples,
            65,
            auto=True,
            running=True,
            optimizing=False,
            gpu_free_mib=4000,
            ceiling=None,
        )
        self.assertTrue(advice.move)
        self.assertEqual(advice.percent, 60)
        self.assertEqual(advice.ceiling, 60)
        self.assertIn("slower", advice.text)
        held = tune_advice(
            samples,
            60,
            auto=True,
            running=True,
            optimizing=False,
            gpu_free_mib=4000,
            ceiling=advice.ceiling,
        )
        self.assertFalse(held.move)
        self.assertEqual(held.percent, 60)
        self.assertIn("fastest", held.text)

    def test_a_failed_call_steps_down_without_waiting_for_three(self):
        advice = tune_advice(
            [EmbedSample(80, 10, 1.0, False)],
            80,
            auto=True,
            running=True,
            optimizing=False,
            gpu_free_mib=4000,
            ceiling=None,
        )
        self.assertEqual(advice.percent, 75)
        self.assertTrue(advice.move)
        self.assertIn("struggled", advice.text)

    def test_manual_recommends_and_does_not_move_the_slider(self):
        advice = tune_advice(
            _calls(100, 3, 1.0),
            100,
            auto=False,
            running=True,
            optimizing=False,
            gpu_free_mib=4000,
            ceiling=None,
        )
        self.assertFalse(advice.move)
        self.assertEqual(advice.percent, 100)
        self.assertIn("higher setting", advice.text)

    def test_a_segment_wait_is_not_blamed_on_the_batch(self):
        advice = tune_advice(
            _calls(100, 3, 1.0),
            100,
            auto=True,
            running=True,
            optimizing=True,
            gpu_free_mib=4000,
            ceiling=None,
        )
        self.assertFalse(advice.move)
        self.assertIn("waiting on a segment", advice.text)

    def test_the_slider_top_is_not_called_the_card_maximum(self):
        advice = tune_advice(
            _calls(150, 3, 1.0),
            150,
            auto=True,
            running=True,
            optimizing=False,
            gpu_free_mib=4000,
            ceiling=None,
        )
        self.assertFalse(advice.move)
        self.assertEqual(advice.percent, 150)
        self.assertIn("slider goes no higher", advice.text)
        self.assertNotIn("absolute", advice.text.lower())

    def test_little_free_card_memory_stops_the_climb(self):
        advice = tune_advice(
            _calls(100, 3, 1.0),
            100,
            auto=True,
            running=True,
            optimizing=False,
            gpu_free_mib=100,
            ceiling=None,
        )
        self.assertFalse(advice.move)
        self.assertIn("nearly full", advice.text)

    def test_the_session_keeps_recent_calls_only(self):
        session = Session(55)
        for index in range(60):
            session.note_embed_call(55, 8, 1.0, True)
        self.assertEqual(len(session.embed_calls()), 48)
        self.assertEqual(session.embed_calls()[-1][0], 55)
