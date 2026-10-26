"""Chunk text the way stock NOMAD embeds it, without cutting a page short.

Stock uses 1500 estimated tokens, 2 characters per token, and 150 tokens of
overlap, then embeds `search_document: ` plus the chunk. The stored payload
text does not include that prefix. Queries stay on `search_query: ` inside
NOMAD. A different embedding model is not used.

Turbo keeps the meat of a long page and splits it into those chunks. It does
not drop the rest of a book, law, or other long page to save a call.
"""

from __future__ import annotations

import math
import re

TARGET_CHARS = 1500 * 2
OVERLAP_CHARS = 150 * 2
MAX_SAFE_TOKENS = 1600
CHAR_TO_TOKEN = 2
DOCUMENT_PREFIX = "search_document: "
# The embed server allows 2048 tokens. Dense text is about one token per
# character, so a stock 3000-character chunk can pass that. 1800 leaves a
# little room when this bound is still low. English stays one chunk.
SAFE_CONTEXT_TOKENS = 1800
MIN_FIT_CHARS = 180

# Compact English stopword list so keyword payload is useful to NOMAD's rerank.
_STOP = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "you", "your", "yours", "yourself",
    "yourselves",
}


def sanitize(text: str) -> str:
    keep = []
    for char in text:
        code = ord(char)
        if code == 0:
            continue
        if 1 <= code <= 8 or code in (11, 12) or 14 <= code <= 31 or code == 127:
            continue
        if 0xD800 <= code <= 0xDFFF:
            continue
        keep.append(char)
    return "".join(keep).strip()


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHAR_TO_TOKEN)


def split_chunks(text: str) -> list[str]:
    """Split the whole sanitized page into stock-sized chunks.

    The walk covers every character. A long page becomes many chunks instead
    of being cut off. fit_embed_inputs splits a piece that is still too long
    for the embedder. The prefix is applied later, not here.
    """
    cleaned = sanitize(text)
    if not cleaned:
        return []
    size = TARGET_CHARS
    overlap = OVERLAP_CHARS
    parts: list[str] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(length, start + size)
        if end < length:
            space = cleaned.rfind(" ", start + size // 2, end)
            if space > start:
                end = space
        piece = cleaned[start:end].strip()
        if piece:
            parts.append(piece)
        if end >= length:
            break
        nxt = end - overlap
        if nxt <= start:
            nxt = end
        start = nxt
    return parts


def embed_input(chunk: str, prefix: str = DOCUMENT_PREFIX) -> tuple[str, str]:
    """Return (prefixed model input, payload text without the prefix).

    The model sees `search_document: ` plus the chunk, which is the stock
    NOMAD document prefix. The text stored in Qdrant omits that prefix.
    NOMAD queries with `search_query: ` through its own Ollama. Turbo does
    not change either prefix.
    """
    payload = sanitize(chunk)
    prefixed = prefix + payload
    if estimate_tokens(prefixed) > MAX_SAFE_TOKENS:
        prefix_tokens = estimate_tokens(prefix)
        max_chars = max(1, (MAX_SAFE_TOKENS - prefix_tokens) * CHAR_TO_TOKEN)
        trimmed = payload[:max_chars]
        space = trimmed.rfind(" ")
        if space > int(max_chars * 0.8):
            trimmed = trimmed[:space]
        payload = trimmed.strip()
        prefixed = prefix + payload
    return prefixed, payload


def token_upper_bound(text: str) -> int:
    """A high guess used only to stay inside the embed context.

    ASCII words use the stock 2 characters per token. Symbols and non-ASCII
    count as one token each. That is where the stock guess runs over 2048.
    """
    if not text:
        return 0
    bound = 0
    word = 0
    for char in text:
        if char.isascii() and char.isalnum():
            word += 1
            continue
        if word:
            bound += (word + 1) // 2
            word = 0
        if char.isascii() and char.isspace():
            continue
        bound += 1
    if word:
        bound += (word + 1) // 2
    return max(bound, 1)


def halve_text(text: str) -> tuple[str, str] | None:
    """Split into two strictly shorter pieces, on a nearby space when there is one."""
    cleaned = text.strip()
    if len(cleaned) < 2:
        return None
    mid = len(cleaned) // 2
    space = cleaned.rfind(" ", max(1, mid - 80), min(len(cleaned) - 1, mid + 80))
    cut = space if space > 0 else mid
    left = cleaned[:cut].strip()
    right = cleaned[cut:].strip()
    if not left or not right:
        left, right = cleaned[:mid], cleaned[mid:]
    if not left or not right or len(left) >= len(cleaned) or len(right) >= len(cleaned):
        return None
    return left, right


def fit_embed_inputs(
    chunk: str,
    prefix: str = DOCUMENT_PREFIX,
    limit: int = SAFE_CONTEXT_TOKENS,
) -> list[tuple[str, str]]:
    """(model input, stored text) pairs that should fit the embed context."""
    payload = sanitize(chunk)
    if not payload:
        return []
    return _fit_payload(payload, prefix, limit, 0)


def _fit_payload(payload: str, prefix: str, limit: int, depth: int) -> list[tuple[str, str]]:
    if token_upper_bound(prefix + payload) <= limit or len(payload) <= MIN_FIT_CHARS or depth >= 12:
        prefixed, fitted = embed_input(payload, prefix)
        if not fitted:
            return []
        return [(prefixed, fitted)]
    halves = halve_text(payload)
    if halves is None:
        prefixed, fitted = embed_input(payload, prefix)
        return [(prefixed, fitted)] if fitted else []
    left, right = halves
    return _fit_payload(left, prefix, limit, depth + 1) + _fit_payload(right, prefix, limit, depth + 1)


def keywords(*parts: str) -> str:
    found: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for word in re.split(r"\s+", part or ""):
            token = re.sub(r"[^\w]", "", word).lower()
            if len(token) <= 2 or token in _STOP or token in seen:
                continue
            seen.add(token)
            found.append(token)
    return " ".join(found)
