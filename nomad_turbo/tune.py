"""Live embed-batch advice from finished calls.

The count is chunks per call. Auto may raise it past the old percent ceiling
while a larger count is still faster on the same publication and the same
article length. It steps down when a larger count is clearly slower on that
work, when the embed server fails a call, or when free GPU memory is under
the headroom stop. A brief slow patch that later calls do not repeat does not
move the count. Article length does not move the count, and a pace measured
on a different article length is not a verdict. Another publication's pace is
not a verdict either. One server stays up. A call already in flight is left
alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from nomad_turbo.slots import clamp_embed_chunks

# One half of the confirmation window. Both halves have to agree.
CALLS_BEFORE_MOVE = 3
# A larger call this far below the neighbor is slower, not noise.
SLOW_RATIO = 0.80
# Both halves have to beat the smaller size by this much. A tie does not climb.
# Neighboring sizes on one stretch often differ by less than this.
FAST_RATIO = 1.05
# After a failed call, wait this many recorded embed calls and this many
# seconds before another climb. Both have to pass.
RETRY_CALLS = 24
RETRY_SECONDS = 90
FAIL_RETRY_CALLS = 48
FAIL_RETRY_SECONDS = 180
# Spare video memory before Auto stops raising the call.
GPU_HEADROOM_MIB = 768

_TRAILING_DATE = re.compile(r"(?:[._-](?:19|20)\d{2}(?:[._-]\d{2})?)+$")


def chunk_step(count: int) -> int:
    """How many chunks one Auto step moves. About a twentieth of the current count."""
    return max(1, int(count) // 20)


def article_band(article_size: int) -> int:
    """Power-of-two band for article length. A clearly different length is the next band."""
    size = max(0, int(article_size or 0))
    if size <= 0:
        return 0
    band = 1
    while band * 2 <= size:
        band *= 2
    return band


def content_kind(source: str) -> str:
    """Identity of the work being indexed. Not a table of batch sizes.

    A trailing year or year-month is removed so two monthly files of the same
    publication stay one kind. The name itself is kept.
    """
    name = Path(str(source or "")).name.lower()
    suffix = Path(name).suffix
    stem = name[: -len(suffix)] if suffix else name
    stem = _TRAILING_DATE.sub("", stem).strip("._- ")
    if suffix == ".zim":
        return stem or "zim"
    if not stem:
        return suffix.lstrip(".") or "file"
    return f"{stem}{suffix}"


@dataclass(frozen=True)
class EmbedSample:
    """One embed call. `size` is the chunk count that was asked for.

    `chunks` is how many were actually sent, used for the pace. `ok` is false
    when the server failed the call, not when one chunk was too long. `kind`
    is the content identity. `article_size` is the article length for that call,
    in chunks of that article.
    """

    size: int
    chunks: int
    seconds: float
    ok: bool
    kind: str = ""
    article_size: int = 0


@dataclass(frozen=True)
class TuneHold:
    """Auto will not climb past `above` until both the call count and the time have passed.

    `kind` is "down" after a failed call. `work_kind` limits the hold to that
    publication. `work_band` is kept so an older hold still loads. A different
    page length does not lift the hold.
    """

    above: int
    until_calls: int
    pace: float
    until_time: float = 0.0
    kind: str = "down"
    work_kind: str = ""
    work_band: int = -1


@dataclass(frozen=True)
class TuneAdvice:
    """`count` is the chunk count Auto should use. `move` is false for a note only."""

    text: str
    count: int
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


def _remember_paces(samples: list[EmbedSample], memory: dict) -> None:
    """Store a pace only after a full timed stretch at one count for one publication.

    The stretch is both halves of the confirmation window. A shorter visit does
    not become a target. The key keeps the article-length band of that stretch,
    so a long-page pace is not a neighbor for a short-page pace.
    """
    for key in list(memory):
        if not (isinstance(key, tuple) and len(key) == 3):
            memory.pop(key, None)
    need = CALLS_BEFORE_MOVE * 2
    run_key: tuple[str, int] | None = None
    run: list[EmbedSample] = []

    def finish() -> None:
        if run_key is None:
            return
        ok = [sample for sample in run if sample.ok]
        if len(ok) < need:
            return
        stretch = ok[-need:]
        pace = _pace(stretch)
        if pace is None:
            return
        memory[(run_key[0], _median_band(stretch), run_key[1])] = pace

    for sample in samples:
        key = (str(sample.kind), int(sample.size))
        if key != run_key:
            finish()
            run_key = key
            run = []
        run.append(sample)
    finish()


def _parsed_pace(key: object, remembered: object) -> tuple[str, int, int, float] | None:
    """(kind, article band, chunk count, pace). A key without a band is ignored."""
    if not isinstance(key, tuple) or len(key) != 3:
        return None
    if remembered is None or float(remembered) <= 0:
        return None
    return str(key[0]), int(key[1]), int(key[2]), float(remembered)


def _nearest(memory: dict, kind: str, size: int, *, higher: bool, band: int | None):
    """Closest same-publication size on one side. Not the fastest size ever seen.

    `band` limits the match to that article length. None accepts any length,
    which only answers whether some other length already has a smaller count.
    """
    found_size: int | None = None
    found_pace: float | None = None
    for key, remembered in memory.items():
        parsed = _parsed_pace(key, remembered)
        if parsed is None:
            continue
        key_kind, key_band, candidate, pace = parsed
        if key_kind != kind:
            continue
        if band is not None and key_band != band:
            continue
        if higher and candidate <= size:
            continue
        if not higher and candidate >= size:
            continue
        if found_size is None or (candidate < found_size if higher else candidate > found_size):
            found_size = candidate
            found_pace = pace
    if found_size is None or found_pace is None:
        return None
    return found_size, found_pace


def _halves(samples: list[EmbedSample]):
    """Two groups of finished calls. None until both groups exist."""
    ok = [sample for sample in samples if sample.ok]
    need = CALLS_BEFORE_MOVE * 2
    if len(ok) < need:
        return None
    recent = ok[-need:]
    first = recent[:CALLS_BEFORE_MOVE]
    second = recent[CALLS_BEFORE_MOVE:]
    left = _pace(first)
    right = _pace(second)
    whole = _pace(recent)
    if left is None or right is None or whole is None:
        return None
    return left, right, whole


def _both_faster(halves, baseline: float) -> bool:
    if baseline <= 0:
        return False
    left, right, _whole = halves
    return left > baseline * FAST_RATIO and right > baseline * FAST_RATIO


def _both_slower(halves, baseline: float) -> bool:
    if baseline <= 0:
        return False
    left, right, _whole = halves
    return left < baseline * SLOW_RATIO and right < baseline * SLOW_RATIO


def _trailing(samples: list[EmbedSample], size: int) -> list[EmbedSample]:
    found: list[EmbedSample] = []
    for sample in reversed(samples):
        if sample.size != size:
            break
        found.append(sample)
    found.reverse()
    return found


def _current_work(samples: list[EmbedSample], kind: str, article_size: int) -> tuple[str, int]:
    last = samples[-1] if samples else None
    live_kind = kind if kind else (last.kind if last else "")
    if article_size:
        live_article = article_size
    elif last is not None and (not kind or last.kind == live_kind):
        live_article = last.article_size
    else:
        live_article = 0
    return live_kind, article_band(live_article)


def _median_band(samples: list[EmbedSample]) -> int:
    """Band of the middle page length. One long page does not move it."""
    sizes = sorted(sample.article_size for sample in samples if sample.article_size > 0)
    if len(sizes) < CALLS_BEFORE_MOVE:
        return 0
    return article_band(sizes[len(sizes) // 2])


def _kind_just_opened(samples: list[EmbedSample], kind: str) -> bool:
    """True when this publication started after a different one in the recent calls."""
    if not samples or not kind:
        return False
    index = len(samples) - 1
    if samples[index].kind != kind:
        return True
    while index >= 0 and samples[index].kind == kind:
        index -= 1
    return index >= 0


def _hold_open(hold: TuneHold | None, call_count: int, now: float) -> bool:
    if hold is None:
        return False
    if call_count >= hold.until_calls and now >= hold.until_time:
        return False
    return True


def _hold_applies(hold: TuneHold | None, kind: str, band: int) -> bool:
    """A hold follows the publication. A different page length does not lift it."""
    del band
    if hold is None:
        return False
    if hold.work_kind and kind and hold.work_kind != kind:
        return False
    return True


def _refuse_upward(
    count: int,
    nxt: int,
    hold: TuneHold | None,
    live_kind: str,
    live_band: int,
    call_count: int,
    now: float,
    gpu_free_mib: int | None,
) -> TuneAdvice | None:
    """Block a climb when card memory is low or an open hold still applies.

    Stepping down is allowed. An open hold is kept. A climb that would pass
    `hold.above` is refused even when the publication changed.
    """
    if nxt <= count:
        return None
    if gpu_free_mib is not None and int(gpu_free_mib) < GPU_HEADROOM_MIB:
        return TuneAdvice(
            "Card memory is nearly full. A larger batch may be refused.",
            count,
            False,
            hold,
        )
    if not _hold_open(hold, call_count, now):
        return None
    applies = _hold_applies(hold, live_kind, live_band)
    if not applies and nxt <= hold.above:
        return None
    if count > hold.above:
        return TuneAdvice(
            "A larger call was slower. Returning to the smaller batch.",
            hold.above,
            True,
            hold,
        )
    return TuneAdvice(_hold_text(hold), count, False, hold)


def _step_down_hold(call_count: int, now: float, lower: int, pace: float, *, failed: bool, kind: str, band: int) -> TuneHold:
    calls = FAIL_RETRY_CALLS if failed else RETRY_CALLS
    seconds = FAIL_RETRY_SECONDS if failed else RETRY_SECONDS
    return TuneHold(
        lower,
        call_count + calls,
        pace if pace > 0 else 0.0,
        now + seconds,
        "down",
        kind,
        band,
    )


def _hold_text(hold: TuneHold) -> str:
    return "A larger call was slower. Auto will try a larger batch only after a short wait."


def tune_advice(
    samples: list[EmbedSample],
    count: int,
    *,
    auto: bool,
    running: bool,
    optimizing: bool,
    gpu_free_mib: int | None,
    hold: TuneHold | None = None,
    call_count: int = 0,
    now: float = 0.0,
    memory: dict | None = None,
    kind: str = "",
    article_size: int = 0,
) -> TuneAdvice:
    """Recommend a chunk count. Move it only when auto is on and the calls earn it.

    `memory` is updated in place. Keys are (content kind, article band, chunk count).
    A pace from another publication is not a neighbor. A pace from another
    article length is not a neighbor either.
    """
    count = clamp_embed_chunks(count)
    now = float(now)
    if hold is not None:
        hold = TuneHold(
            clamp_embed_chunks(hold.above),
            int(hold.until_calls),
            float(hold.pace),
            float(hold.until_time),
            hold.kind if hold.kind in {"down", "settle"} else "down",
            str(hold.work_kind),
            int(hold.work_band),
        )
    if optimizing:
        return TuneAdvice(
            "Indexing is waiting on a segment. The batch size will not speed that up.",
            count,
            False,
            hold,
        )
    if not running:
        return TuneAdvice(
            "A recommendation shows after indexing finishes a few embed calls.",
            count,
            False,
            hold,
        )
    if memory is None:
        memory = {}
    _remember_paces(samples, memory)
    live_kind, live_band = _current_work(samples, kind, article_size)
    trailing = _trailing(samples, count)
    # Pace is this publication at this count and this article length.
    # Another publication at the same count is not this batch getting slower.
    same = [sample for sample in trailing if sample.kind == live_kind]

    if any(not sample.ok for sample in same):
        lower = clamp_embed_chunks(count - chunk_step(count))
        text = "The embed server struggled with this call. A smaller batch is safer."
        if auto and lower != count:
            return TuneAdvice(
                "The embed server struggled with this call. Trying a smaller one.",
                lower,
                True,
                _step_down_hold(call_count, now, lower, 0.0, failed=True, kind=live_kind, band=live_band),
            )
        return TuneAdvice(text, count, False, hold)

    if _kind_just_opened(samples, live_kind):
        seen = len([sample for sample in same if sample.ok])
        if seen < CALLS_BEFORE_MOVE * 2:
            return TuneAdvice(
                "This content is different. The pace from the other work is not a verdict.",
                count,
                False,
                hold,
            )

    halves = _halves(same)
    if halves is None:
        seen = len([sample for sample in same if sample.ok])
        need = CALLS_BEFORE_MOVE * 2
        if auto and _hold_applies(hold, live_kind, live_band) and _hold_open(hold, call_count, now):
            return TuneAdvice(_hold_text(hold), count, False, hold)
        return TuneAdvice(
            f"Watching finished calls ({min(seen, need)} of {need}) before recommending a change.",
            count,
            False,
            hold,
        )

    ok_same = [sample for sample in same if sample.ok]
    judge_band = _median_band(ok_same[-(CALLS_BEFORE_MOVE * 2) :])
    lower = _nearest(memory, live_kind, count, higher=False, band=judge_band)
    if lower is not None and _both_slower(halves, lower[1]):
        step = chunk_step(count)
        nxt = clamp_embed_chunks(count - step)
        text = "This batch is slower than the smaller one on this work. Trying a smaller one."
        if not auto or nxt == count:
            note = "This batch is slower than the smaller one on this work. A smaller size is faster."
            return TuneAdvice(note, count, False, hold)
        return TuneAdvice(
            text,
            nxt,
            True,
            _step_down_hold(call_count, now, nxt, lower[1], failed=False, kind=live_kind, band=live_band),
        )

    if gpu_free_mib is not None and gpu_free_mib < GPU_HEADROOM_MIB:
        return TuneAdvice(
            "Card memory is nearly full. A larger batch may be refused.",
            count,
            False,
            hold,
        )
    if auto and _hold_applies(hold, live_kind, live_band) and _hold_open(hold, call_count, now):
        if count > hold.above:
            return TuneAdvice(
                "A larger call was slower. Returning to the smaller batch.",
                hold.above,
                True,
                hold,
            )
        return TuneAdvice(_hold_text(hold), count, False, hold)

    if lower is None and _nearest(memory, live_kind, count, higher=False, band=None) is not None:
        return TuneAdvice(
            "Article length changed. Staying at this count.",
            count,
            False,
            hold,
        )

    higher = _nearest(memory, live_kind, count, higher=True, band=judge_band)
    _left, _right, pace = halves
    higher_blocks = higher is not None and not (higher[1] > pace * FAST_RATIO)
    faster_than_lower = lower is not None and _both_faster(halves, lower[1])
    if higher_blocks or (lower is not None and not faster_than_lower):
        if faster_than_lower and higher_blocks:
            text = "This size is faster than the smaller one. A larger count was not, so Auto is staying here."
        else:
            text = "This size ties the one just tried. Staying here."
        if not auto:
            text = "This size ties the one just tried. You can leave it here."
        return TuneAdvice(text, count, False, hold)

    nxt = clamp_embed_chunks(count + chunk_step(count))
    if auto:
        if lower is None:
            text = "Trying a larger batch."
        else:
            text = "This batch is still faster. Trying a larger one."
        # A full run of different work can reach this climb. It must not step
        # past an open hold or drop that hold on the way by.
        refused = _refuse_upward(
            count,
            nxt,
            hold,
            live_kind,
            live_band,
            call_count,
            now,
            gpu_free_mib,
        )
        if refused is not None:
            return refused
        return TuneAdvice(text, nxt, True, hold)
    if lower is None:
        text = "You can try a higher count."
    else:
        text = "This batch is still faster. You can try a higher count."
    return TuneAdvice(text, count, False, hold)
