"""Live embed-batch advice from finished calls, not from the card's spec sheet.

Auto raises the slider while full calls keep up, and steps back when a
larger call is slower or the embed server fails one. 150% is only the top
of the slider. A call already in flight is left alone. One server stays up.
"""

from __future__ import annotations

from dataclasses import dataclass

from nomad_turbo.slots import MAX_BOOST_PERCENT, clamp_boost_percent

# Three full calls are enough to compare. One odd page is not.
CALLS_BEFORE_MOVE = 3
PERCENT_STEP = 5
# A larger call this far below the previous pace is slower, not noise.
SLOW_RATIO = 0.90
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
class TuneAdvice:
    """percent is where Auto should put the slider. move is false for a recommendation only."""

    text: str
    percent: int
    move: bool
    ceiling: int | None


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


def _recent(samples: list[EmbedSample], percent: int) -> list[EmbedSample]:
    matched = [sample for sample in samples if sample.percent == percent]
    return matched[-6:]


def _lower_pace(samples: list[EmbedSample], percent: int) -> tuple[int, float] | None:
    """The nearest smaller setting that already has enough finished calls."""
    percents = sorted({sample.percent for sample in samples if sample.percent < percent})
    found: tuple[int, float] | None = None
    for candidate in percents:
        ok = [sample for sample in _recent(samples, candidate) if sample.ok]
        pace = _pace(ok[-CALLS_BEFORE_MOVE:])
        if len(ok) >= CALLS_BEFORE_MOVE and pace is not None:
            found = (candidate, pace)
    return found


def tune_advice(
    samples: list[EmbedSample],
    percent: int,
    *,
    auto: bool,
    running: bool,
    optimizing: bool,
    gpu_free_mib: int | None,
    ceiling: int | None,
) -> TuneAdvice:
    """Recommend a batch, and move it only when auto is on and a change is earned.

    ceiling is the highest percent Auto may use after a larger call was slower
    or failed. Manual use leaves that ceiling unchanged.
    """
    percent = clamp_boost_percent(percent)
    if ceiling is not None:
        ceiling = clamp_boost_percent(ceiling)
    if optimizing:
        return TuneAdvice(
            "Indexing is waiting on a segment. The batch size will not speed that up.",
            percent,
            False,
            ceiling,
        )
    if not running:
        return TuneAdvice(
            "A recommendation shows after indexing finishes a few embed calls.",
            percent,
            False,
            ceiling,
        )
    if auto and ceiling is not None and percent > ceiling:
        return TuneAdvice(
            "A larger call was slower. Returning to the measured line.",
            ceiling,
            True,
            ceiling,
        )
    recent = _recent(samples, percent)
    if any(not sample.ok for sample in recent):
        lower = clamp_boost_percent(percent - PERCENT_STEP)
        text = "The embed server struggled with this call. A smaller batch is safer."
        if auto and lower != percent:
            text = "The embed server struggled with this call. Trying a smaller one."
        return TuneAdvice(text, lower if auto else percent, bool(auto and lower != percent), lower if auto else ceiling)
    ok = [sample for sample in recent if sample.ok][-CALLS_BEFORE_MOVE:]
    if len(ok) < CALLS_BEFORE_MOVE or _pace(ok) is None:
        seen = len([sample for sample in recent if sample.ok])
        return TuneAdvice(
            f"Watching finished calls ({min(seen, CALLS_BEFORE_MOVE)} of {CALLS_BEFORE_MOVE}) before recommending a change.",
            percent,
            False,
            ceiling,
        )
    if gpu_free_mib is not None and gpu_free_mib < GPU_HEADROOM_MIB:
        return TuneAdvice(
            "Card memory is nearly full. A larger batch may be refused.",
            percent,
            False,
            ceiling,
        )
    pace = _pace(ok)
    previous = _lower_pace(samples, percent)
    top = MAX_BOOST_PERCENT
    if ceiling is not None:
        top = min(top, ceiling)
    if previous is not None and pace is not None and pace < previous[1] * SLOW_RATIO:
        lower = clamp_boost_percent(percent - PERCENT_STEP)
        text = (
            f"This batch is slower than {previous[0]}%. "
            "That smaller size is the fastest one measured."
        )
        if auto and lower != percent:
            return TuneAdvice(text, lower, True, lower)
        return TuneAdvice(text, percent, False, ceiling)
    if percent >= top:
        if percent >= MAX_BOOST_PERCENT:
            text = "At 150%. Calls are still keeping up. The slider goes no higher."
        else:
            text = "This is the fastest size measured. A larger call was slower."
        return TuneAdvice(text, percent, False, ceiling)
    nxt = clamp_boost_percent(min(top, percent + PERCENT_STEP))
    if nxt == percent:
        return TuneAdvice(
            "This is the fastest size measured. A larger call was slower.",
            percent,
            False,
            ceiling,
        )
    if auto:
        return TuneAdvice("Chunks are still keeping up. Trying a larger batch.", nxt, True, ceiling)
    return TuneAdvice(
        "Chunks are keeping up. You can try a higher setting.",
        percent,
        False,
        ceiling,
    )
