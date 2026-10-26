"""Text extraction for uploaded documents that are already text.

Turbo reads these on the host. PDFs and pictures are not in this file:
runtime.read_visual renders them with the OCR libraries in the NOMAD image,
then Turbo chunks and embeds that text itself instead of handing the file
to stock's embedder.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree


def _strip_tags(html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_text(path: Path, kind: str) -> str | None:
    if kind == "text":
        if path.suffix.lower() == ".rtf":
            raw = path.read_text("utf-8", errors="replace")
            raw = re.sub(r"\\'[0-9a-fA-F]{2}", " ", raw)
            raw = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", raw)
            return re.sub(r"[{}]", " ", raw)
        return path.read_text("utf-8", errors="replace")
    if kind == "docx":
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml)
        parts = [node.text or "" for node in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")]
        return "\n".join(part for part in parts if part)
    if kind == "epub":
        texts: list[str] = []
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.lower().endswith((".html", ".xhtml", ".htm")):
                    texts.append(_strip_tags(archive.read(name).decode("utf-8", "replace")))
        return "\n".join(part for part in texts if part)
    return None
