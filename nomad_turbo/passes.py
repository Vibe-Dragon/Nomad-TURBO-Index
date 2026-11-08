"""ZIM passes, OCR quality, spelling, and duplicate text.

HTML is indexed first. Other text is next. PDFs and pictures are last.
A long passage that is already stored is not stored again. The word list
ships with Turbo. Nothing is downloaded to correct a spelling.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
from pathlib import Path

_WORD = re.compile(r"[A-Za-z0-9]+")
_BOILER = re.compile(
    r"(?:^|/)(readme|license|licence|changelog|notice|contributing|code_of_conduct)(?:[._-]|$)",
    re.IGNORECASE,
)
_HTML = {"text/html", "application/xhtml+xml"}
_PDF = {"application/pdf"}
_EPUB = {"application/epub+zip"}
_DOCX = {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
_TEXT = {"text/plain", "text/markdown", "text/x-markdown", "text/vtt"}
_SVG = {"image/svg+xml"}
_IMAGES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/tiff",
    "image/tif",
    "image/bmp",
    "image/x-ms-bmp",
}
_GENERIC = {"", "application/octet-stream", "binary/octet-stream"}
_EXT_KIND = {
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    ".pdf": "ocr",
    ".epub": "text",
    ".docx": "text",
    ".txt": "text",
    ".md": "text",
    ".markdown": "text",
    ".vtt": "text",
    ".svg": "text",
    ".jpg": "ocr",
    ".jpeg": "ocr",
    ".png": "ocr",
    ".gif": "ocr",
    ".webp": "ocr",
    ".tif": "ocr",
    ".tiff": "ocr",
    ".bmp": "ocr",
}
PASS_LABELS = {
    "html": "HTML",
    "text": "text",
    "ocr": "OCR",
    "ocr-text": "OCR text",
}
BAR_COLORS = {
    "HTML": "#2f9e44",
    "text": "#1c7ed6",
    "OCR": "#e8590c",
    "OCR text": "#7048e8",
}
PAGE_CHARS = 3000
BOILER_WORDS = 200
IMAGE_WORDS = 8
FINGERPRINT_WORDS = 200
FINGERPRINT_DISTANCE = 3
FINGERPRINT_RATIO = 0.8
CHUNK_DEDUPE_CHARS = 400
_FNV_OFFSET = 14695981039346656037
_FNV_PRIME = 1099511628211
_FNV_MASK = (1 << 64) - 1
_WORDS: set[str] | None = None
_LEDGER: "ChunkLedger | None" = None
_LEDGER_LOCK = threading.Lock()


def normalize_ocr_quality(value: object) -> str:
    """low, medium, or high. Anything else is medium."""
    text = str(value or "").strip().lower()
    if text in ("low", "medium", "high"):
        return text
    return "medium"


def bar_color(status: str) -> str:
    """Paint for one pass. An empty string keeps the ordinary bar."""
    return BAR_COLORS.get(status or "", "")


def pass_label(phase: str) -> str:
    return PASS_LABELS.get(phase or "", "")


def _is_book(mime: str, lowered: str) -> bool:
    name = (mime or "").split(";", 1)[0].strip().lower()
    if name in _EPUB or name in _DOCX:
        return True
    return Path(lowered).suffix in {".epub", ".docx"}


def classify_entry(mime: str, path: str, size: int = 0) -> str:
    """html, text, ocr, or skip.

    Inside a ZIM, pictures are left out. EPUB and Word stay. A PDF stays.
    HTML stays. Plain text in a ZIM is left out. Uploads do not use this.
    """
    del size
    lowered = (path or "").replace("\\", "/").lower()
    if "/tiles/" in f"/{lowered}" or lowered.startswith("tiles/"):
        return "skip"
    if _is_image(mime, lowered) or (mime or "").split(";", 1)[0].strip().lower() in _SVG or Path(lowered).suffix == ".svg":
        return "skip"
    kind = _mime_kind(mime)
    if kind == "":
        kind = _EXT_KIND.get(Path(lowered).suffix, "skip")
    if kind == "text" and not _is_book(mime, lowered):
        return "skip"
    return kind or "skip"


def _mime_kind(mime: str) -> str:
    name = (mime or "").split(";", 1)[0].strip().lower()
    if name in _GENERIC:
        return ""
    if name in _HTML:
        return "html"
    if name in _PDF or name in _IMAGES:
        return "ocr"
    if name in _EPUB or name in _DOCX or name in _TEXT or name in _SVG:
        return "text"
    return "skip"


def _is_image(mime: str, lowered: str) -> bool:
    name = (mime or "").split(";", 1)[0].strip().lower()
    if name in _IMAGES:
        return True
    return Path(lowered).suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif", ".tiff", ".bmp"}


def boilerplate(path: str, text: str) -> bool:
    """A short README, license, or changelog. A long one is the manual."""
    name = (path or "").replace("\\", "/").rsplit("/", 1)[-1]
    if _BOILER.search("/" + name) is None:
        return False
    return len(_WORD.findall(text or "")) < BOILER_WORDS


def meaty_image(text: str) -> bool:
    """A picture with enough words to be a diagram or a table, not a photo."""
    return len(re.findall(r"[A-Za-z0-9]{3,}", text or "")) >= IMAGE_WORDS


def join_pages(pages: list[tuple[int, str]], limit: int = PAGE_CHARS) -> list[dict]:
    """Group page texts into sections of about 3000 characters.

    The heading is `page 4` or `pages 5-7`, which is what search shows.
    """
    sections: list[dict] = []
    buf: list[str] = []
    start = end = None
    size = 0

    def flush() -> None:
        nonlocal buf, start, end, size
        if not buf or start is None or end is None:
            buf, start, end, size = [], None, None, 0
            return
        heading = f"page {start}" if start == end else f"pages {start}-{end}"
        sections.append({"heading": heading, "text": "\n\n".join(buf), "level": 2})
        buf, start, end, size = [], None, None, 0

    for number, text in pages:
        body = (text or "").strip()
        if not body:
            continue
        if buf and size + len(body) > limit:
            flush()
        if start is None:
            start = int(number)
        end = int(number)
        buf.append(body)
        size += len(body)
    flush()
    return sections


def sections_from_article(article: dict, *, drop_short_image: bool = False) -> list[dict]:
    """Sections for one extract event. Pages are merged before they are joined.

    A short picture is dropped only for a ZIM. An upload the user added is kept.
    """
    pages = article.get("pages")
    if isinstance(pages, list) and pages:
        merged: list[tuple[int, str]] = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            number = int(page.get("page") or 0)
            text = merge_page(str(page.get("layer") or ""), str(page.get("ocr") or ""))
            merged.append((number, text))
        return join_pages(merged)
    if article.get("kind") == "image":
        text = str(article.get("ocr") or "").strip()
        if not text:
            return []
        if drop_short_image and not meaty_image(text):
            return []
        return [{"heading": "", "text": text, "level": 2}]
    sections = article.get("sections")
    if isinstance(sections, list):
        return [item for item in sections if isinstance(item, dict)]
    return []


def article_plain(article: dict) -> str:
    parts = [str(item.get("text") or "") for item in sections_from_article(article)]
    if not parts:
        parts.append(str(article.get("text") or ""))
    return "\n".join(part for part in parts if part).strip()


def word_list_path() -> Path:
    return Path(__file__).resolve().parents[1] / "worker" / "words.txt"


def load_words() -> set[str]:
    global _WORDS
    if _WORDS is not None:
        return _WORDS
    path = word_list_path()
    if not path.is_file():
        _WORDS = set()
        return _WORDS
    _WORDS = {line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    return _WORDS


def merge_page(layer: str, ocr: str, words: set[str] | None = None) -> str:
    """Keep the text layer. Append OCR words it does not already have.

    A word within one edit of a layer word is dropped, so a misread does not
    sit beside the real spelling. An unknown word is kept, including a place
    name. A page with no text layer is corrected only when one dictionary
    word is within one edit.
    """
    written = (layer or "").strip()
    seen = (ocr or "").strip()
    if not seen:
        return written
    if not written:
        return correct_scan(seen, words if words is not None else load_words())
    buckets = _length_buckets(_WORD.findall(written.lower()))
    kept: list[str] = []
    for match in _WORD.finditer(seen):
        token = match.group(0)
        if len(token) < 2:
            continue
        if _near_layer(token.lower(), buckets):
            continue
        kept.append(token)
    if not kept:
        return written
    return written + "\n" + " ".join(kept)


def correct_scan(text: str, words: set[str]) -> str:
    """Fix a lowercase word on a scan when the dictionary has one neighbor."""

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if any(char.isdigit() for char in token) or token[:1].isupper() or len(token) < 5:
            return token
        lowered = token.lower()
        if lowered in words:
            return token
        found: list[str] = []
        for edit in _edits(lowered):
            if edit in words:
                found.append(edit)
                if len(found) > 1:
                    return token
        if len(found) == 1:
            return found[0]
        return token

    return _WORD.sub(replace, text or "")


def _length_buckets(tokens: list[str]) -> dict[int, list[str]]:
    buckets: dict[int, list[str]] = {}
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        buckets.setdefault(len(token), []).append(token)
    return buckets


def _near_layer(token: str, buckets: dict[int, list[str]]) -> bool:
    for length in (len(token) - 1, len(token), len(token) + 1):
        for other in buckets.get(length, ()):
            if within_one(token, other):
                return True
    return False


def within_one(left: str, right: str) -> bool:
    """True when the words are equal, one substitution, one insertion, or one swap."""
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        diffs = [index for index, (a, b) in enumerate(zip(left, right)) if a != b]
        if len(diffs) == 1:
            return True
        if len(diffs) == 2 and diffs[1] == diffs[0] + 1:
            first, second = diffs
            return left[first] == right[second] and left[second] == right[first]
        return False
    if len(left) > len(right):
        left, right = right, left
    index = other = 0
    skipped = False
    while index < len(left) and other < len(right):
        if left[index] == right[other]:
            index += 1
            other += 1
            continue
        if skipped:
            return False
        skipped = True
        other += 1
    return True


def _edits(word: str) -> set[str]:
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:index], word[index:]) for index in range(len(word) + 1)]
    deletes = [left + right[1:] for left, right in splits if right]
    swaps = [left + right[1] + right[0] + right[2:] for left, right in splits if len(right) > 1]
    replaces = [left + char + right[1:] for left, right in splits if right for char in letters]
    inserts = [left + char + right for left, right in splits for char in letters]
    return set(deletes + swaps + replaces + inserts)


def fnv64(text: str) -> int:
    value = _FNV_OFFSET
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * _FNV_PRIME) & _FNV_MASK
    return value


def fingerprint(text: str) -> tuple[int, int]:
    """64-bit simhash of the words, and how many words went into it."""
    tokens = re.findall(r"[a-z0-9]{2,}", (text or "").lower())
    if not tokens:
        return 0, 0
    bits = [0] * 64
    for token in tokens:
        hashed = fnv64(token)
        for index in range(64):
            bits[index] += 1 if (hashed >> index) & 1 else -1
    value = 0
    for index, weight in enumerate(bits):
        if weight > 0:
            value |= 1 << index
    return value, len(tokens)


def same_work(left: tuple[int, int], right: tuple[int, int]) -> bool:
    """The same long text, allowing a few words of packaging to differ."""
    left_hash, left_count = left
    right_hash, right_count = right
    if left_count < FINGERPRINT_WORDS or right_count < FINGERPRINT_WORDS:
        return False
    short, long = sorted((left_count, right_count))
    if short / long < FINGERPRINT_RATIO:
        return False
    return _popcount(left_hash ^ right_hash) <= FINGERPRINT_DISTANCE


def _popcount(value: int) -> int:
    return bin(value & _FNV_MASK).count("1")


class ArticlePrints:
    """Long articles already indexed in this file. The first copy wins.

    New prints are appended as they are kept, so a stop during HTML does not
    forget the articles already passed. The file is merged again before OCR
    so an empty memory cannot wipe it.
    """

    def __init__(self) -> None:
        self._rows: list[tuple[int, int]] = []
        self._exact: set[tuple[int, int]] = set()
        self._path: Path | None = None
        self._handle = None

    def bind(self, path: Path) -> None:
        """Append further keeps to this file. Existing lines stay where they are."""
        self.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._handle = path.open("a", encoding="utf-8")

    def close(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None:
            handle.close()

    def duplicate(self, text: str) -> bool:
        current = fingerprint(text)
        if current[1] < FINGERPRINT_WORDS:
            return False
        return any(same_work(current, old) for old in self._rows)

    def _remember(self, row: tuple[int, int]) -> bool:
        """Store one print. False when this article is already represented."""
        if row in self._exact or any(same_work(row, old) for old in self._rows):
            return False
        self._rows.append(row)
        self._exact.add(row)
        return True

    def keep(self, text: str) -> None:
        current = fingerprint(text)
        if current[1] < FINGERPRINT_WORDS:
            return
        if not self._remember(current):
            return
        handle = self._handle
        if handle is None:
            return
        hashed, count = current
        handle.write(f"{hashed:016x} {count}\n")
        handle.flush()

    def merge_file(self, path: Path) -> None:
        if not path.is_file():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                hashed = int(parts[0], 16)
                count = int(parts[1])
            except ValueError:
                continue
            if count < FINGERPRINT_WORDS:
                continue
            self._remember((hashed, count))

    def write(self, path: Path) -> None:
        """Replace the file with what is in memory, then keep appending to it."""
        self.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{hashed:016x} {count}" for hashed, count in self._rows]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self.bind(path)


def normalize_chunk(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def chunk_key(text: str) -> str | None:
    """Hash of a long chunk. Short chunks are not deduped."""
    normal = normalize_chunk(text)
    if len(normal) < CHUNK_DEDUPE_CHARS:
        return None
    return hashlib.sha256(normal.encode("utf-8")).hexdigest()


class ChunkLedger:
    """Chunks already stored for the whole library. The file lives on disk."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("create table if not exists chunks (hash text primary key)")
        self._db.commit()

    def seen(self, text: str) -> bool:
        key = chunk_key(text)
        if key is None:
            return False
        with self._lock:
            row = self._db.execute("select 1 from chunks where hash = ?", (key,)).fetchone()
        return row is not None

    def remember(self, text: str) -> None:
        key = chunk_key(text)
        if key is None:
            return
        with self._lock:
            self._db.execute("insert or ignore into chunks (hash) values (?)", (key,))
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()


def library_ledger() -> ChunkLedger:
    """One ledger for this process. Tests can replace it by setting the global."""
    global _LEDGER
    with _LEDGER_LOCK:
        if _LEDGER is None:
            from nomad_turbo.discover import state_dir

            _LEDGER = ChunkLedger(state_dir() / "chunk-hashes.sqlite")
        return _LEDGER


def reset_library_ledger() -> None:
    """Forget stored chunk hashes. A wiped index has to be allowed to store them again."""
    global _LEDGER
    from nomad_turbo.discover import state_dir

    with _LEDGER_LOCK:
        ledger = _LEDGER
        _LEDGER = None
    if ledger is not None:
        ledger.close()
    path = state_dir() / "chunk-hashes.sqlite"
    if path.is_file():
        path.unlink()
    spill = state_dir() / "spill"
    if spill.is_dir():
        for item in spill.iterdir():
            if item.is_file():
                item.unlink()
