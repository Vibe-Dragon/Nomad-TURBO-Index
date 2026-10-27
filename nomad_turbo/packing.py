"""Turn one article into embed inputs. No model call happens here.

Stock stores one payload per section. Turbo does the same, and it keeps
every section of a long page. The `search_document: ` prefix is added only
on the text sent to the embed server. The stored payload leaves it off.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

import nomad_turbo.chunking as chunking

# The open window reloads this module on Start and does not reload chunking.
# Reload it here so that Start still picks up the fit helpers.
chunking = importlib.reload(chunking)
from nomad_turbo.chunking import DOCUMENT_PREFIX, fit_embed_inputs, halve_text, split_chunks


@dataclass
class SectionDraft:
    meta: dict
    payloads: list[str] = field(default_factory=list)


@dataclass
class PreparedArticle:
    article: dict
    sections: list[SectionDraft]
    inputs: list[str]
    section_for_input: list[int]
    already_stored: bool = False


def prepare_article(
    article: dict,
    source: str,
    archive: dict,
    document_id: str,
    prefix: str = DOCUMENT_PREFIX,
) -> PreparedArticle:
    """Match stock NOMAD's per-section payload, with the prefix only on the model input."""
    title = article.get("title") or article.get("path") or ""
    sections_in = article.get("sections") or []
    if not sections_in:
        sections_in = [{"heading": "", "text": article.get("text") or "", "level": 2}]
    sections: list[SectionDraft] = []
    inputs: list[str] = []
    owners: list[int] = []
    for raw in sections_in:
        body = raw.get("text") or ""
        pieces = split_chunks(body)
        if not pieces:
            continue
        heading = raw.get("heading") or ""
        full = f"{title} - {heading}" if heading else title
        draft = SectionDraft(
            meta={
                "source": source,
                "content_type": "zim_article" if source.endswith(".zim") else "file",
                "article_title": title,
                "article_path": article.get("path") or source,
                "section_title": heading or title,
                "full_title": full,
                "hierarchy": f"{title} > {heading}" if heading else title,
                "section_level": raw.get("level") or 2,
                "document_id": document_id,
                # Stock search hides a point only when active is false. Leaving
                # the field off makes the knowledge-base page rewrite the index.
                "active": True,
                "archive_title": archive.get("title") or "",
                "archive_creator": archive.get("creator") or "",
                "archive_publisher": archive.get("publisher") or "",
                "archive_date": archive.get("date") or "",
                "archive_language": archive.get("language") or "",
                "archive_description": archive.get("description") or "",
            }
        )
        section_index = len(sections)
        for piece in pieces:
            for prefixed, payload in fit_embed_inputs(piece, prefix):
                draft.payloads.append(payload)
                inputs.append(prefixed)
                owners.append(section_index)
        if draft.payloads:
            sections.append(draft)
    return PreparedArticle(article=article, sections=sections, inputs=inputs, section_for_input=owners)


class ArticleFeed:
    """Hold articles until every chunk has a vector, then emit whole articles.

    A chunk from a later article may share an embed call with an earlier one.
    The earlier article is emitted only after its own chunks are filled.
    """

    def __init__(self) -> None:
        self._open: list[PreparedArticle] = []
        self._vectors: list[list[list[float] | None]] = []
        self._claim: list[tuple[int, int]] = []

    def add(self, prepared: PreparedArticle) -> None:
        if self._claim:
            raise RuntimeError("finish the open embed call before adding articles")
        self._open.append(prepared)
        self._vectors.append([None] * len(prepared.inputs))

    def pending(self) -> int:
        return sum(vector is None for row in self._vectors for vector in row)

    def next_inputs(self, width: int) -> list[str]:
        if self._claim:
            raise RuntimeError("the previous embed call has not been stored")
        chosen: list[str] = []
        claim: list[tuple[int, int]] = []
        limit = max(1, int(width))
        for item_index, prepared in enumerate(self._open):
            for input_index, vector in enumerate(self._vectors[item_index]):
                if vector is not None:
                    continue
                chosen.append(prepared.inputs[input_index])
                claim.append((item_index, input_index))
                if len(chosen) >= limit:
                    self._claim = claim
                    return chosen
        self._claim = claim
        return chosen

    def split_rejected(self, index: int) -> tuple[str, str]:
        """Split one rejected input, or drop it when it cannot get smaller.

        Returns split, dropped, or dropped-last, plus the article path.
        The claim is cleared so the caller asks for inputs again.
        """
        if not self._claim or not 0 <= index < len(self._claim):
            self._claim = []
            raise RuntimeError("embedder rejected a chunk that is not in this batch")
        item_index, input_index = self._claim[index]
        prepared = self._open[item_index]
        path = str(prepared.article.get("path") or "")
        owner, payload_pos = _payload_slot(prepared, input_index)
        section = prepared.sections[owner]
        payload = section.payloads[payload_pos]
        model_input = prepared.inputs[input_index]
        if payload and model_input.endswith(payload):
            prefix = model_input[: len(model_input) - len(payload)]
        else:
            prefix = DOCUMENT_PREFIX
        halves = halve_text(payload)
        if halves is None:
            del prepared.inputs[input_index]
            del prepared.section_for_input[input_index]
            del section.payloads[payload_pos]
            del self._vectors[item_index][input_index]
            self._claim = []
            if not prepared.inputs and (prepared.article.get("skip") or "").strip() not in ("read", "parse"):
                prepared.article["skip"] = "embed"
                return "dropped-last", path
            return "dropped", path
        left, right = halves
        prepared.inputs[input_index] = prefix + left
        prepared.inputs.insert(input_index + 1, prefix + right)
        prepared.section_for_input.insert(input_index + 1, owner)
        section.payloads[payload_pos] = left
        section.payloads.insert(payload_pos + 1, right)
        self._vectors[item_index].insert(input_index + 1, None)
        self._claim = []
        return "split", path

    def push_vectors(self, vectors: list[list[float]]) -> list[tuple[PreparedArticle, list[list[float]]]]:
        if len(vectors) != len(self._claim):
            raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(self._claim)} chunks")
        for (item_index, input_index), vector in zip(self._claim, vectors):
            self._vectors[item_index][input_index] = vector
        self._claim = []
        return self.drain_ready()

    def drain_ready(self) -> list[tuple[PreparedArticle, list[list[float]]]]:
        done: list[tuple[PreparedArticle, list[list[float]]]] = []
        while self._open:
            row = self._vectors[0]
            if any(vector is None for vector in row):
                break
            prepared = self._open.pop(0)
            vectors = self._vectors.pop(0)
            done.append((prepared, [vector for vector in vectors if vector is not None]))
        return done


def _payload_slot(prepared: PreparedArticle, input_index: int) -> tuple[int, int]:
    owner = prepared.section_for_input[input_index]
    pos = 0
    for index, item_owner in enumerate(prepared.section_for_input):
        if index == input_index:
            return owner, pos
        if item_owner == owner:
            pos += 1
    raise RuntimeError("chunk is not attached to a section")
