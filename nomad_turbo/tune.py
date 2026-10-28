"""Live embed-batch advice from finished calls, not from the card's spec sheet.

Auto raises the slider while full calls keep up, and steps down as soon as a
larger call is slower or the embed server fails one. After a step down it
waits through more calls before trying one step higher again, so it does not
bounce. If the pace of the work changes, that wait ends early. 150% is only
the top of the slider. A call already in flight is left alone. One server
stays up.
"""

from __future__ import annotations

from dataclasses import dataclass

from nomad_turbo.slots import MAX_BOOST_PERCENT, clamp_boost_percent

# Three full calls are enough to compare. One odd slow page is not.
CALLS_BEFORE_MOVE = 3
PERCENT_STEP = 5
# A larger call this far below the previous pace is slower, not noise.
# Ten percent is normal swing on short pages, so the step down waits for a fifth.
SLOW_RATIO = 0.80
# After a slower-but-successful call, wait this many recorded embed calls
# and this many seconds before probing one step higher. Both have to pass.
# A failed call waits longer. Short pages finish the call count in a few seconds,
# so the clock is what stops the bounce.
RETRY_CALLS = 24
RETRY_SECONDS = 90
FAIL_RETRY_CALLS = 48
FAIL_RETRY_SECONDS = 180
# The wait ends early when the current pace is this far from the pace that
# made the larger call look worse. The work has changed.
PACE_SHIFT = 0.60
# Spare video memory before Auto stops raising the call. This is a safety
# stop from the live reading, not a batch size computed from the card.
GPU_HEADROOM_MIB = 768


@dataclass(frozen=True)
class EmbedSample:
    """One embed call. ok is false when the server failed the call, not when one chunk was too long."""

    percent: int
    chunks: int
    seconds: float
    ok: bool


@dataclass(frozen=True)
class TuneHold:
    """Auto will not climb past `above` until both the call count and the time have passed.

    `pace` is chunks per second when the hold was set. Zero means there was no
    pace to compare. `kind` is "down" after a slower call, or "settle" after a
    probe that kept up, so the next step does not follow in the same breath.
    """

    above: int
    until_calls: int
    pace: float
    until_time: float = 0.0
    kind: str = "down"


@dataclass(frozen=True)
class TuneAdvice:
    """percent is where Auto should put the slider. move is false for a recommendation only."""

    text: str
    percent: int
    move: bool
    hold: TuneHold | None


def _pace(samples: list[EmbedSample]) -> float | None:
    chunks = 0
    seconds = 0.0
    for sample in samples:
        if sample.chunks <= 0 or sample.seconds <= 0:
            continue
        chunks += sample.chunks
        seconds += sample.seconds
    if chunks <= 0 or seconds <= 0:
        return None
    return chunks / seconds


def _stint(samples: list[EmbedSample], percent: int) -> list[EmbedSample]:
    """Calls at this percent since it was last selected. An older visit does not count."""
    if not samples or samples[-1].percent != percent:
        return []
    found: list[EmbedSample] = []
    for sample in reversed(samples):
        if sample.percent != percent:
            break
        found.append(sample)
    found.reverse()
    return found[-6:]


def _lower_pace(samples: list[EmbedSample], percent: int) -> tuple[int, float] | None:
    """The nearest smaller setting that already has enough finished calls."""
    percents = sorted({sample.percent for sample in samples if sample.percent < percent})
    found: tuple[int, float] | None = None
    for candidate in percents:
        ok = [sample for sample in samples if sample.percent == candidate and sample.ok]
        pace = _pace(ok[-CALLS_BEFORE_MOVE:])
        if len(ok) >= CALLS_BEFORE_MOVE and pace is not None:
            found = (candidate, pace)
    return found


def _hold_open(hold: TuneHold | None, pace: float | None, call_count: int, now: float) -> bool:
    """True while Auto should not climb past the hold."""
    if hold is None:
        return False
    if hold.kind == "down" and hold.pace > 0 and pace is not None and pace > 0:
        ratio = pace / hold.pace
        if ratio < PACE_SHIFT or ratio > (1 / PACE_SHIFT):
            return False
    if call_count >= hold.until_calls and now >= hold.until_time:
        return False
    return True


def _hold_text(hold: TuneHold) -> str:
    if hold.kind == "settle":
        return "Staying at this size. Auto will try one step higher after a short wait."
    return "A larger call was slower. Auto will try one step higher after a short wait."


def _step_down_hold(call_count: int, now: float, lower: int, pace: float, *, failed: bool) -> TuneHold:
    calls = FAIL_RETRY_CALLS if failed else RETRY_CALLS
    seconds = FAIL_RETRY_SECONDS if failed else RETRY_SECONDS
    return TuneHold(lower, call_count + calls, pace if pace > 0 else 0.0, now + seconds, "down")


def tune_advice(
    samples: list[EmbedSample],
    percent: int,
    *,
    auto: bool,
    running: bool,
    optimizing: bool,
    gpu_free_mib: int | None,
    hold: TuneHold | None = None,
    call_count: int = 0,
    now: float = 0.0,
) -> TuneAdvice:
    """Recommend a batch, and move it only when auto is on and a change is earned.

    A step down is immediate once three full calls are about a fifth slower, or
    on one failed call. The next step up waits on the clock and the call count,
    unless the pace of the work has changed a lot. Manual use leaves the hold unchanged.
    """
    percent = clamp_boost_percent(percent)
    now = float(now)
    if hold is not None:
        kind = hold.kind if hold.kind in {"down", "settle"} else "down"
        hold = TuneHold(
            clamp_boost_percent(hold.above),
            int(hold.until_calls),
            float(hold.pace),
            float(hold.until_time),
            kind,
        )
    if optimizing:
        return TuneAdvice(
            "Indexing is waiting on a segment. The batch size will not speed that up.",
            percent,
            False,
            hold,
        )
    if not running:
        return TuneAdvice(
            "A recommendation shows after indexing finishes a few embed calls.",
            percent,
            False,
            hold,
        )
    stint = _stint(samples, percent)
    pace = _pace([sample for sample in stint if sample.ok])
    if auto and _hold_open(hold, pace, call_count, now) and hold is not None and percent > hold.above:
        return TuneAdvice(
            "A larger call was slower. Returning to the smaller batch.",
            hold.above,
            True,
            hold,
        )
    if any(not sample.ok for sample in stint):
        lower = clamp_boost_percent(percent - PERCENT_STEP)
        text = "The embed server struggled with this call. A smaller batch is safer."
        if auto and lower != percent:
            text = "The embed server struggled with this call. Trying a smaller one."
            previous = _lower_pace(samples, percent)
            baseline = previous[1] if previous is not None else 0.0
            return TuneAdvice(
                text,
                lower,
                True,
                _step_down_hold(call_count, now, lower, baseline, failed=True),
            )
        return TuneAdvice(text, percent, False, hold)
    ok = [sample for sample in stint if sample.ok][-CALLS_BEFORE_MOVE:]
    if len(ok) < CALLS_BEFORE_MOVE or _pace(ok) is None:
        seen = len([sample for sample in stint if sample.ok])
        if auto and hold is not None and _hold_open(hold, None, call_count, now):
            return TuneAdvice(_hold_text(hold), percent, False, hold)
        return TuneAdvice(
            f"Watching finished calls ({min(seen, CALLS_BEFORE_MOVE)} of {CALLS_BEFORE_MOVE}) before recommending a change.",
            percent,
            False,
            hold,
        )
    pace = _pace(ok)
    previous = _lower_pace(samples, percent)
    if previous is not None and pace is not None and pace < previous[1] * SLOW_RATIO:
        lower = clamp_boost_percent(percent - PERCENT_STEP)
        text = f"This batch is slower than {previous[0]}%. Trying a smaller one."
        if not auto or lower == percent:
            text = f"This batch is slower than {previous[0]}%. A smaller size is faster."
            return TuneAdvice(text, percent, False, hold)
        return TuneAdvice(
            text,
            lower,
            True,
            _step_down_hold(call_count, now, lower, previous[1], failed=False),
        )
    if gpu_free_mib is not None and gpu_free_mib < GPU_HEADROOM_MIB:
        return TuneAdvice(
            "Card memory is nearly full. A larger batch may be refused.",
            percent,
            False,
            hold,
        )
    if auto and hold is not None and _hold_open(hold, pace, call_count, now):
        return TuneAdvice(_hold_text(hold), percent, False, hold)
    if percent >= MAX_BOOST_PERCENT:
        return TuneAdvice(
            "At 150%. Calls are still keeping up. The slider goes no higher.",
            percent,
            False,
            None,
        )
    nxt = clamp_boost_percent(percent + PERCENT_STEP)
    if auto:
        again = hold is not None
        text = "Trying a larger batch again." if again else "Chunks are still keeping up. Trying a larger batch."
        # A first climb can keep stepping. A probe that just earned its step waits
        # before the one after it, so a noisy stretch cannot walk the slider up.
        settled = None
        if again:
            settled = TuneHold(
                nxt,
                call_count + RETRY_CALLS,
                pace or 0.0,
                now + RETRY_SECONDS,
                "settle",
            )
        return TuneAdvice(text, nxt, True, settled)
    return TuneAdvice(
        "Chunks are keeping up. You can try a higher setting.",
        percent,
        False,
        hold,
    )
