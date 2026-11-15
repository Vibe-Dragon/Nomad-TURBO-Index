"""Group small plain-text files so one embed pass stores several of them.

A file smaller than the gap between files spends more time being opened than
being embedded. Those files share a pass. Each one is still stored under its
own source, and it is complete only when its own text was stored. A ZIM, a
PDF, a picture, and a larger text file stay alone.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bytes the embedder would get through during the pause between two files.
# A plain-text file at or under this size is small relative to that gap.
GAP_BYTES = 256 * 1024
# One pass stops at this much text, so a long run of tiny files does not
# become a single unbounded embed call.
BUNDLE_CAP_BYTES = 1024 * 1024


@dataclass
class BundleMember:
    source: str
    text: str


def is_small_text(kind: str, size_bytes: int, *, gap_bytes: int = GAP_BYTES) -> bool:
    """True for plain text that is small relative to the gap between files."""
    if kind != "text":
        return False
    return int(size_bytes or 0) <= int(gap_bytes)


def group_small_text(rows, *, gap_bytes: int = GAP_BYTES, cap_bytes: int = BUNDLE_CAP_BYTES) -> list[list]:
    """Small plain-text files share a group. Everything else stays alone.

    The small files keep their relative order and fill groups up to the cap.
    A ZIM, a PDF, a picture, or a larger text file is never put in that group.
    """
    small: list = []
    alone: list[list] = []
    used = 0
    groups: list[list] = []

    def flush() -> None:
        nonlocal small, used
        if small:
            groups.append(small)
            small = []
            used = 0

    for row in rows:
        kind = str(getattr(row, "kind", "") or "")
        size = int(getattr(row, "size_bytes", 0) or 0)
        if is_small_text(kind, size, gap_bytes=gap_bytes):
            if small and used + max(size, 1) > cap_bytes:
                flush()
            small.append(row)
            used += max(size, 1)
        else:
            alone.append([row])
    flush()
    return groups + alone


def members_stored(members: list[BundleMember], stored_sources: set[str]) -> list[str]:
    """Sources whose own text was part of the stored pass.

    A member with no text was not included. A source that was not stored is
    left unfinished even if another member of the group was stored.
    """
    done: list[str] = []
    for member in members:
        if not member.text:
            continue
        if member.source in stored_sources:
            done.append(member.source)
    return done
