"""Batch advice follows finished embed calls. It does not open a window or start indexing."""

import unittest

from nomad_turbo.session import Session
from nomad_turbo.slots import (
    MAX_EMBED_CHUNKS,
    embed_call_budget,
    install_embed_chunks,
    safe_call_chunks,
)
from nomad_turbo.tune import (
    FAIL_RETRY_CALLS,
    FAIL_RETRY_SECONDS,
    RETRY_CALLS,
    article_band,
    chunk_step,
    content_kind,
    EmbedSample,
    tune_advice,
)


def _calls(
    size: int,
    count: int,
    seconds: float,
    ok: bool = True,
    *,
    chunks: int = 10,
    kind: str = "notes",
    article_size: int = 100,
) -> list[EmbedSample]:
    return [EmbedSample(size, chunks, seconds, ok, kind, article_size) for _ in range(count)]


def _advice(samples, count, **kwargs):
    values = dict(
        auto=True,
        running=True,
        optimizing=False,
        gpu_free_mib=4000,
        hold=None,
        call_count=0,
        kind=kwargs.get("kind", samples[-1].kind if samples else ""),
        article_size=kwargs.get("article_size", samples[-1].article_size if samples else 0),
    )
    values.update(kwargs)
    return tune_advice(samples, count, **values)


class TuneAdviceTest(unittest.TestCase):
    def test_auto_raises_the_call_while_the_same_work_keeps_up(self):
        advice = _advice(_calls(55, 6, 1.0), 55)
        self.assertTrue(advice.move)
        self.assertEqual(advice.count, 55 + chunk_step(55))
        self.assertGreater(advice.count, 55)
        self.assertIn("larger batch", advice.text)
        self.assertIsNone(advice.hold)

    def test_a_larger_count_still_faster_on_the_same_work_passes_the_old_ceiling(self):
        hardware = dict(gpu_total_mib=8192, host_total_mib=16384, cpu_count=8)
        ceiling = embed_call_budget(150, **hardware)
        below = ceiling - chunk_step(ceiling)
        self.assertGreater(ceiling, safe_call_chunks(**hardware))
        self.assertNotEqual(ceiling, 150)
        samples = _calls(below, 6, 1.2) + _calls(ceiling, 6, 1.0)
        advice = _advice(samples, ceiling)
        self.assertTrue(advice.move)
        self.assertGreater(advice.count, ceiling)
        self.assertNotIn("slower", advice.text.lower())

    def test_a_tie_with_the_size_just_tried_does_not_climb(self):
        step = chunk_step(80)
        samples = _calls(80, 6, 1.0) + _calls(80 + step, 6, 1.0)
        advice = _advice(samples, 80 + step)
        self.assertFalse(advice.move)
        self.assertEqual(advice.count, 80 + step)
        self.assertIn("tie", advice.text.lower())
        self.assertNotIn("larger", advice.text.lower())

    def test_a_short_dip_that_later_calls_do_not_repeat_leaves_the_count(self):
        step = chunk_step(80)
        samples = _calls(80, 6, 1.0) + _calls(80 + step, 3, 2.0) + _calls(80 + step, 3, 1.0)
        advice = _advice(samples, 80 + step)
        self.assertFalse(advice.move)
        self.assertEqual(advice.count, 80 + step)
        self.assertNotIn("smaller", advice.text.lower())

    def test_a_larger_count_that_stays_slower_on_the_same_work_steps_down(self):
        step = chunk_step(80)
        current = 80 + step
        samples = _calls(80, 6, 1.0) + _calls(current, 6, 2.0)
        advice = _advice(samples, current, call_count=12)
        self.assertTrue(advice.move)
        self.assertEqual(advice.count, current - chunk_step(current))
        self.assertIn("slower", advice.text.lower())
        self.assertIsNotNone(advice.hold)
        self.assertEqual(advice.hold.above, advice.count)
        self.assertEqual(advice.hold.until_calls, 12 + RETRY_CALLS)
        held = _advice(
            samples + _calls(advice.count, 3, 1.0),
            advice.count,
            hold=advice.hold,
            call_count=15,
            now=0,
        )
        self.assertFalse(held.move)
        self.assertEqual(held.count, advice.count)

    def test_a_different_content_kind_does_not_step_the_count_down(self):
        band = article_band(100)
        memory = {("alpha", band, 40): 50.0}
        samples = _calls(80, 6, 5.0, kind="beta", article_size=100)
        advice = _advice(samples, 80, memory=memory, kind="beta", article_size=100)
        self.assertGreaterEqual(advice.count, 80)
        self.assertNotIn("slower", advice.text.lower())

    def test_two_long_pages_do_not_step_down_and_a_timed_slow_stretch_does(self):
        self.assertNotEqual(article_band(20), article_band(400))
        short_band = {("book", article_band(20), 40): 50.0}
        early = _advice(
            _calls(80, 2, 5.0, kind="book", article_size=400),
            80,
            memory=dict(short_band),
            kind="book",
            article_size=400,
        )
        self.assertEqual(early.count, 80)
        self.assertFalse(early.move)
        self.assertNotIn("slower", early.text.lower())
        other_length = _advice(
            _calls(80, 6, 5.0, kind="book", article_size=400),
            80,
            memory=dict(short_band),
            kind="book",
            article_size=400,
        )
        self.assertFalse(other_length.move)
        self.assertEqual(other_length.count, 80)
        self.assertNotIn("slower", other_length.text.lower())
        same_band = {("book", article_band(400), 40): 50.0}
        advice = _advice(
            _calls(80, 6, 5.0, kind="book", article_size=400),
            80,
            memory=same_band,
            kind="book",
            article_size=400,
        )
        self.assertTrue(advice.move)
        self.assertLess(advice.count, 80)
        self.assertIn("slower", advice.text.lower())

    def test_a_modest_article_change_on_the_same_kind_is_still_the_same_work(self):
        self.assertEqual(article_band(100), article_band(120))
        samples = _calls(40, 6, 1.0, kind="book", article_size=100) + _calls(
            80, 6, 2.0, kind="book", article_size=120
        )
        advice = _advice(samples, 80, kind="book", article_size=120)
        self.assertLess(advice.count, 80)
        self.assertTrue(advice.move)
        self.assertIn("slower", advice.text.lower())

    def test_a_longer_stretch_keeps_the_count(self):
        self.assertNotEqual(article_band(20), article_band(400))
        settled = _calls(40, 6, 1.0, kind="book", article_size=20) + _calls(
            80, 6, 1.0, kind="book", article_size=20
        )
        two = settled + _calls(80, 2, 1.0, kind="book", article_size=400)
        early = _advice(two, 80, kind="book", article_size=400)
        self.assertFalse(early.move)
        self.assertEqual(early.count, 80)
        self.assertNotIn("longer articles", early.text.lower())
        self.assertNotIn("slower", early.text.lower())
        stretched = settled + _calls(80, 6, 1.0, kind="book", article_size=400)
        advice = _advice(stretched, 80, kind="book", article_size=400)
        self.assertFalse(advice.move)
        self.assertEqual(advice.count, 80)
        self.assertIn("article", advice.text.lower())
        self.assertNotIn("longer articles", advice.text.lower())
        self.assertNotIn("slower", advice.text.lower())
        later = stretched + _calls(80, 1, 1.0, kind="book", article_size=400)
        again = _advice(later, 80, kind="book", article_size=400)
        self.assertFalse(again.move)
        self.assertEqual(again.count, 80)

    def test_one_long_page_does_not_step_the_count_down(self):
        samples = _calls(40, 6, 1.0, kind="wiki", article_size=20) + _calls(
            80, 5, 1.0, kind="wiki", article_size=20
        ) + [EmbedSample(80, 10, 8.0, True, "wiki", 400)]
        advice = _advice(samples, 80, kind="wiki", article_size=400)
        self.assertFalse(advice.move)
        self.assertEqual(advice.count, 80)
        self.assertNotIn("longer articles", advice.text.lower())
        self.assertNotIn("smaller", advice.text.lower())

    def test_mixed_page_lengths_do_not_walk_the_count_down(self):
        count = 61
        samples: list[EmbedSample] = []
        climbs = 0
        for index in range(36):
            article = 4 if index % 2 == 0 else 400
            seconds = 1.0 if article == 4 else 5.0
            samples.append(EmbedSample(count, 10, seconds, True, "wiki", article))
            advice = _advice(samples, count, kind="wiki", article_size=article)
            self.assertNotIn("longer articles", advice.text.lower())
            self.assertNotIn("shorter articles", advice.text.lower())
            if advice.move:
                self.assertGreater(advice.count, count)
                climbs += 1
                count = advice.count
        self.assertEqual(climbs, 1)
        self.assertEqual(count, 61 + chunk_step(61))
        self.assertGreater(count, 50)

    def test_a_returning_publication_waits_for_a_stretch_before_a_remembered_size(self):
        memory: dict = {}
        prior = _calls(40, 6, 1.0, kind="beta", article_size=100)
        _advice(prior, 40, memory=memory, kind="beta", article_size=100)
        self.assertIn(("beta", article_band(100), 40), memory)
        self.assertFalse(any(isinstance(key, tuple) and len(key) == 2 for key in memory))
        early_samples = _calls(80, 6, 1.0, kind="alpha", article_size=100) + _calls(
            80, 2, 5.0, kind="beta", article_size=100
        )
        early = _advice(early_samples, 80, memory=memory, kind="beta", article_size=100)
        self.assertFalse(early.move)
        self.assertEqual(early.count, 80)
        self.assertIn("content", early.text.lower())
        self.assertNotIn("slower", early.text.lower())
        slow = _calls(80, 6, 1.0, kind="alpha", article_size=100) + _calls(
            80, 6, 2.0, kind="beta", article_size=100
        )
        advice = _advice(slow, 80, memory=memory, kind="beta", article_size=100)
        self.assertTrue(advice.move)
        self.assertLess(advice.count, 80)
        self.assertIn("slower", advice.text.lower())

    def test_a_new_kind_with_no_memory_is_not_scored_against_the_other_work(self):
        samples = _calls(80, 6, 1.0, kind="alpha", article_size=100) + _calls(
            80, 2, 5.0, kind="beta", article_size=100
        )
        advice = _advice(samples, 80, kind="beta", article_size=100)
        self.assertEqual(advice.count, 80)
        self.assertFalse(advice.move)
        self.assertIn("content", advice.text.lower())
        self.assertNotIn("slower", advice.text.lower())
        self.assertNotIn("faster", advice.text.lower())

    def test_content_kind_drops_a_trailing_month_and_keeps_the_publication(self):
        self.assertEqual(content_kind("reports_2026-08.zim"), content_kind("reports_2026-09.zim"))
        self.assertNotEqual(content_kind("reports_2026-08.zim"), content_kind("notes_2026-08.zim"))
        self.assertNotEqual(content_kind("reports_2026-08.zim"), "")

    def test_a_small_slowdown_does_not_drop_the_batch(self):
        step = chunk_step(80)
        advice = _advice(_calls(80, 6, 1.0) + _calls(80 + step, 6, 1.1), 80 + step)
        self.assertEqual(advice.count, 80 + step)
        self.assertFalse(advice.move)
        self.assertNotIn("smaller", advice.text.lower())

    def test_one_slow_call_does_not_drop_the_batch(self):
        step = chunk_step(80)
        samples = _calls(80, 6, 1.0) + _calls(80 + step, 5, 1.0) + _calls(80 + step, 1, 3.0)
        advice = _advice(samples, 80 + step)
        self.assertFalse(advice.move)
        self.assertEqual(advice.count, 80 + step)

    def test_a_shorter_stretch_does_not_climb_when_card_memory_is_low(self):
        self.assertGreater(article_band(400), article_band(20))
        samples = _calls(80, 6, 1.0, kind="book", article_size=400) + _calls(
            80, 6, 1.0, kind="book", article_size=20
        )
        blocked = _advice(samples, 80, kind="book", article_size=20, gpu_free_mib=100)
        self.assertFalse(blocked.move)
        self.assertEqual(blocked.count, 80)
        self.assertIn("nearly full", blocked.text.lower())
        room = _advice(samples, 80, kind="book", article_size=20, gpu_free_mib=4000)
        self.assertTrue(room.move)
        self.assertEqual(room.count, 80 + chunk_step(80))
        self.assertIn("larger", room.text.lower())
        self.assertNotIn("shorter articles", room.text.lower())
        nxt = samples + _calls(room.count, 1, 1.0, kind="book", article_size=20)
        again = _advice(nxt, room.count, kind="book", article_size=20, gpu_free_mib=4000)
        self.assertFalse(again.move)
        self.assertEqual(again.count, room.count)

    def test_a_longer_page_does_not_lift_a_step_down_hold(self):
        kind = "wiki"
        short = 100
        longer = 400
        self.assertNotEqual(article_band(short), article_band(longer))
        samples = _calls(80, 6, 1.0, kind=kind, article_size=short) + _calls(
            84, 6, 2.0, kind=kind, article_size=short
        )
        down = _advice(samples, 84, kind=kind, article_size=short, call_count=12, now=0)
        self.assertTrue(down.move)
        self.assertEqual(down.count, 80)
        hold = down.hold
        self.assertIsNotNone(hold)
        self.assertEqual(hold.above, 80)
        self.assertGreater(hold.until_calls, 12)

        longer_samples = samples + _calls(80, 2, 1.0, kind=kind, article_size=longer)
        mid = _advice(
            longer_samples,
            80,
            hold=hold,
            kind=kind,
            article_size=longer,
            call_count=14,
            now=0,
        )
        self.assertIsNotNone(mid.hold)
        self.assertEqual(mid.hold.above, hold.above)
        self.assertEqual(mid.hold.until_calls, hold.until_calls)
        self.assertLessEqual(mid.count, hold.above)

        back_samples = longer_samples + _calls(80, 2, 1.0, kind=kind, article_size=short)
        back = _advice(
            back_samples,
            80,
            hold=hold,
            kind=kind,
            article_size=short,
            call_count=16,
            now=0,
        )
        self.assertIsNotNone(back.hold)
        self.assertEqual(back.hold.above, hold.above)
        self.assertFalse(back.move)
        self.assertEqual(back.count, 80)
        self.assertNotEqual(back.count, 84)

    def test_a_failure_on_other_work_does_not_shrink_the_count(self):
        samples = [EmbedSample(80, 10, 1.0, False, "alpha", 100)] + _calls(
            80, 6, 1.0, kind="beta", article_size=100
        )
        advice = _advice(samples, 80, kind="beta", article_size=100)
        self.assertGreaterEqual(advice.count, 80)
        self.assertNotIn("struggled", advice.text.lower())

    def test_a_failed_call_on_this_publication_steps_down_when_the_page_length_differs(self):
        self.assertNotEqual(article_band(20), article_band(100))
        samples = _calls(80, 6, 1.0, kind="book", article_size=100) + [
            EmbedSample(80, 10, 1.0, False, "book", 20)
        ]
        advice = _advice(samples, 80, kind="book", article_size=100)
        self.assertTrue(advice.move)
        self.assertEqual(advice.count, 80 - chunk_step(80))
        self.assertIn("struggled", advice.text.lower())

    def test_another_works_measured_climb_does_not_pass_an_open_failure_hold(self):
        failed = [EmbedSample(80, 10, 1.0, False, "wiki", 100)]
        down = _advice(failed, 80, kind="wiki", article_size=100, call_count=10, now=0)
        self.assertEqual(down.count, 80 - chunk_step(80))
        hold = down.hold
        self.assertIsNotNone(hold)
        self.assertEqual(hold.above, down.count)
        self.assertEqual(hold.until_calls, 10 + FAIL_RETRY_CALLS)
        self.assertGreater(hold.until_calls, 16)

        beta = failed + _calls(down.count, 6, 1.0, kind="beta", article_size=100)
        moved = _advice(
            beta,
            down.count,
            hold=hold,
            kind="beta",
            article_size=100,
            call_count=16,
            now=0,
        )
        self.assertFalse(moved.move)
        self.assertEqual(moved.count, down.count)
        self.assertNotEqual(moved.count, down.count + chunk_step(down.count))
        self.assertIsNotNone(moved.hold)
        self.assertEqual(moved.hold.above, hold.above)
        self.assertEqual(moved.hold.until_calls, hold.until_calls)

        wiki = beta + _calls(down.count, 6, 1.0, kind="wiki", article_size=100)
        back = _advice(
            wiki,
            down.count,
            hold=moved.hold,
            kind="wiki",
            article_size=100,
            call_count=22,
            now=0,
        )
        self.assertFalse(back.move)
        self.assertEqual(back.count, down.count)
        self.assertIsNotNone(back.hold)
        self.assertEqual(back.hold.above, hold.above)
        self.assertLess(22, back.hold.until_calls)

    def test_a_failed_call_steps_down_without_waiting_for_six(self):
        advice = _advice([EmbedSample(80, 10, 1.0, False, "notes", 100)], 80, call_count=10)
        self.assertEqual(advice.count, 80 - chunk_step(80))
        self.assertTrue(advice.move)
        self.assertIn("struggled", advice.text)
        self.assertIsNotNone(advice.hold)
        self.assertEqual(advice.hold.until_calls, 10 + FAIL_RETRY_CALLS)
        self.assertEqual(advice.hold.until_time, FAIL_RETRY_SECONDS)

    def test_manual_recommends_and_does_not_move_the_count(self):
        advice = _advice(_calls(80, 6, 1.0), 80, auto=False)
        self.assertFalse(advice.move)
        self.assertEqual(advice.count, 80)
        self.assertIn("higher", advice.text.lower())

    def test_a_segment_wait_is_not_blamed_on_the_batch(self):
        advice = _advice(_calls(80, 6, 1.0), 80, optimizing=True)
        self.assertFalse(advice.move)
        self.assertEqual(advice.count, 80)
        self.assertIn("segment", advice.text.lower())

    def test_little_free_card_memory_stops_the_climb(self):
        advice = _advice(_calls(80, 6, 1.0), 80, gpu_free_mib=100)
        self.assertFalse(advice.move)
        self.assertEqual(advice.count, 80)
        self.assertIn("nearly full", advice.text)

    def test_the_session_keeps_recent_calls_only(self):
        session = Session(55)
        for _index in range(60):
            session.note_embed_call(55, 8, 1.0, True)
        self.assertEqual(len(session.embed_calls()), 48)
        self.assertEqual(session.embed_calls()[-1][0], 55)
        self.assertEqual(session.embed_call_count(), 60)

    def test_a_fast_far_size_is_not_the_neighbor(self):
        band = article_band(100)
        memory = {("notes", band, 40): 50.0, ("notes", band, 80): 10.0}
        advice = _advice(_calls(84, 6, 1.0), 84, memory=memory)
        self.assertGreaterEqual(advice.count, 84)
        self.assertNotIn("slower", advice.text.lower())

    def test_a_large_drop_against_the_size_just_measured_steps_down(self):
        samples = _calls(80, 6, 1.0) + _calls(84, 6, 2.5)
        advice = _advice(samples, 84)
        self.assertLess(advice.count, 84)
        self.assertIn("slower", advice.text.lower())
        self.assertNotIn("pace of the work", advice.text)
        self.assertIsNotNone(advice.hold)
        self.assertGreater(advice.hold.pace, 0)


class RecommendedCountTest(unittest.TestCase):
    def test_two_budgets_recommend_their_own_safe_call(self):
        small = dict(gpu_total_mib=4096, host_total_mib=8192, cpu_count=4)
        large = dict(gpu_total_mib=16384, host_total_mib=32768, cpu_count=8)
        first = safe_call_chunks(**small)
        second = safe_call_chunks(**large)
        self.assertNotEqual(first, second)
        self.assertGreater(first, 0)
        self.assertGreater(second, 0)
        saved_small = type("Cfg", (), {"embed_chunks": None, "boost_percent": 100, "reserve_ratio": 0.0})()
        saved_large = type("Cfg", (), {"embed_chunks": None, "boost_percent": 100, "reserve_ratio": 0.0})()
        self.assertEqual(install_embed_chunks(saved_small, **small), first)
        self.assertEqual(install_embed_chunks(saved_large, **large), second)
        self.assertNotEqual(first, 100)
        self.assertNotEqual(second, 100)

    def test_a_saved_percent_at_the_old_top_opens_as_a_chunk_count(self):
        hardware = dict(gpu_total_mib=8192, host_total_mib=16384, cpu_count=8)
        saved = type("Cfg", (), {"embed_chunks": None, "boost_percent": 150, "reserve_ratio": 0.0})()
        opened = install_embed_chunks(saved, **hardware)
        safe = safe_call_chunks(**hardware)
        self.assertEqual(opened, embed_call_budget(150, **hardware))
        self.assertGreater(opened, safe)
        self.assertNotEqual(opened, 150)
        explicit = type("Cfg", (), {"embed_chunks": 80, "boost_percent": 150, "reserve_ratio": 0.0})()
        self.assertEqual(install_embed_chunks(explicit, **hardware), 80)

    def test_a_typed_count_is_clamped_as_chunks_not_as_a_percent(self):
        session = Session(55)
        self.assertEqual(session.get_chunks(), 55)
        session.set_chunks(200)
        self.assertEqual(session.get_chunks(), 200)
        session.set_percent(0)
        self.assertEqual(session.get_chunks(), 1)
        session.set_percent(10**9)
        self.assertEqual(session.get_chunks(), MAX_EMBED_CHUNKS)
        self.assertNotEqual(session.get_chunks(), 150)
