"""Documents inside a ZIM, read after every ZIM's pages and every upload.

The page walk records inner paths. This module turns those lists and the
checkpoint into the file-list branch and the next document to read. It does
not open a ZIM and it does not copy a PDF out.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

INTERNAL_NAME = "Internal Zim Documents"
INTERNAL_KEY = "zim/internal-zim-documents"

_CACHE: dict[str, tuple] = {}


def doc_zim_key(source: str) -> str:
    """Folder key for one ZIM's documents. It is not a library path."""
    digest = hashlib.sha256((source or "").encode()).hexdigest()[:16]
    return "doczim:" + digest


def virtual_source(zim_source: str, inner_path: str) -> str:
    """File-list source for one document. The parent ZIM is the second line."""
    return "zimdoc\n" + (zim_source or "") + "\n" + (inner_path or "")


def parent_source(source: str | None) -> str:
    """The ZIM a virtual document row belongs to. Any other source is unchanged."""
    text = source or ""
    if text.startswith("zimdoc\n"):
        parts = text.split("\n", 2)
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return text


def catalog_entries(text_paths, ocr_paths) -> tuple[tuple[str, str], ...]:
    """Unique inner paths in path order. A path listed twice keeps the PDF kind."""
    found: dict[str, str] = {}
    for path in text_paths or ():
        text = str(path or "").strip()
        if text:
            found.setdefault(text, "text")
    for path in ocr_paths or ():
        text = str(path or "").strip()
        if text:
            found[text] = "ocr"
    return tuple(sorted(found.items(), key=lambda item: item[0].casefold()))


def document_percent(closed: bool, total: int, done: int, skipped: int) -> int | None:
    """Stored documents out of the closed list. An open list has no percent."""
    if not closed or int(total) <= 0:
        return None
    return min(100, (int(done) + int(skipped)) * 100 // int(total))


@dataclass(frozen=True)
class ZimDocGroup:
    """One ZIM's inner documents. entries are path, kind, state, in path order."""

    source: str
    name: str
    closed: bool
    total: int
    done: int
    skipped: int
    running_path: str
    entries: tuple[tuple[str, str, str], ...]

    @property
    def pending(self) -> tuple[tuple[str, str, str], ...]:
        if not self.closed:
            return ()
        return tuple(entry for entry in self.entries if entry[2] == "waiting")


def pick_document(groups, priorities, index_now) -> tuple[str, str, str] | None:
    """The next closed document. Index now, then priority, then the ZIM name.

    Inside one ZIM the remaining paths stay in path order. A list that is
    still growing is not picked.
    """
    now = parent_source(index_now or "")
    ranked_priority = {parent_source(item) for item in (priorities or ())}
    ranked = []
    for group in groups or ():
        if not group.closed or not group.pending:
            continue
        if now and group.source == now:
            band = 0
        elif group.source in ranked_priority:
            band = 1
        else:
            band = 2
        ranked.append((band, (group.name or "").casefold(), group.source, group))
    if not ranked:
        return None
    ranked.sort(key=lambda row: (row[0], row[1], row[2]))
    group = ranked[0][3]
    path, kind, _state = group.pending[0]
    return group.source, path, kind


def source_has_pending_documents(root, source: str | None) -> bool:
    """True when that ZIM's page walk has closed and a document is still waiting."""
    parent = parent_source(source)
    if not parent:
        return False
    for group in load_document_groups(root):
        if group.source == parent and group.closed and group.pending:
            return True
    return False


def write_running_document(root, source: str | None, path: str | None) -> None:
    """Remember the document being read. Clearing it removes the file."""
    target = Path(root) / "document-now"
    if not source or not path:
        target.unlink(missing_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"{source}\n{path}\n", encoding="utf-8")


def document_stamp(root) -> tuple:
    """Names, mtimes, and sizes of the lists the branch is drawn from.

    A directory mtime does not move when a list is appended, so each file
    is stated on its own.
    """
    base = Path(root)
    rows = []
    checkpoints = base / "checkpoints"
    if checkpoints.is_dir():
        for path in sorted(checkpoints.glob("*.json"), key=lambda item: item.name):
            rows.append(_stamp_row(path))
    spill = base / "spill"
    if spill.is_dir():
        lists = [
            path
            for path in spill.iterdir()
            if path.name.endswith("-text.jsonl") or path.name.endswith("-ocr.jsonl")
        ]
        for path in sorted(lists, key=lambda item: item.name):
            rows.append(_stamp_row(path))
    current = base / "document-now"
    if current.is_file():
        rows.append(_stamp_row(current))
    return tuple(row for row in rows if row is not None)


def invalidate() -> None:
    """Drop the parsed groups. The next load states the files again anyway."""
    _CACHE.clear()


def load_document_groups(root) -> tuple[ZimDocGroup, ...]:
    """Groups whose pages are done, or whose list is still being written.

    An indexed ZIM with no checkpoint and no lists is left out. A pass name
    from the older OCR queue is not treated as closed here.
    """
    base = Path(root)
    key = str(base)
    stamp = document_stamp(base)
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    groups = _read_groups(base)
    _CACHE[key] = (stamp, groups)
    return groups


def document_branch(groups, hide_finished: bool = False):
    """Internal Zim Documents, or None when there is nothing left to show."""
    from nomad_turbo.schedule import TreeFolder, ZIM_ROOT

    zims = []
    for group in groups or ():
        folder = _zim_folder(group, hide_finished, TreeFolder, ZIM_ROOT)
        if folder is not None:
            zims.append(folder)
    if not zims:
        return None
    zims.sort(key=lambda folder: folder.name.casefold())
    count = sum(folder.file_count for folder in zims)
    return TreeFolder(
        INTERNAL_KEY,
        INTERNAL_NAME,
        ZIM_ROOT,
        count,
        0,
        tuple(zims),
        (),
    )


def place_document_branch(roots, groups, hide_finished: bool = False):
    """Pin the branch last under zim. A library with no zim root still shows it."""
    branch = document_branch(groups, hide_finished=hide_finished)
    if branch is None:
        return tuple(roots)
    from nomad_turbo.schedule import TreeFolder, ZIM_ROOT

    built = tuple(roots)
    for index, folder in enumerate(built):
        if folder.key == ZIM_ROOT:
            updated = replace(folder, folders=tuple(folder.folders) + (branch,))
            return built[:index] + (updated,) + built[index + 1 :]
    cover = TreeFolder(ZIM_ROOT, ZIM_ROOT, ZIM_ROOT, branch.file_count, 0, (branch,), ())
    return built + (cover,)


def document_file_row(
    zim_source: str,
    inner_path: str,
    kind: str,
    state: str,
    *,
    priority: bool = False,
    indexing: bool = False,
) -> tuple:
    """One inner document, in the same 20 columns as a library file."""
    from nomad_turbo.catalog import row_color
    from nomad_turbo.progress import UNKNOWN_ETA_SECONDS
    from nomad_turbo.schedule import EM_DASH, file_type_label, row_mark

    name = (inner_path or "").replace("\\", "/").rsplit("/", 1)[-1] or inner_path or "document"
    file_type = file_type_label(name)
    if not file_type:
        file_type = "pdf" if kind == "ocr" else "docx"
    status = {"running": "indexing", "indexed": "indexed", "skipped": "left out"}.get(state, "waiting")
    percent = 100 if state == "indexed" else 0
    color = row_color("zim", "indexed" if state == "indexed" else "pending")
    return (
        virtual_source(zim_source, inner_path),
        row_mark(priority=priority, indexing=indexing or state == "running"),
        name,
        file_type,
        EM_DASH,
        status,
        percent,
        EM_DASH,
        EM_DASH,
        EM_DASH,
        EM_DASH,
        "",
        color,
        0,
        0,
        0,
        0,
        0,
        UNKNOWN_ETA_SECONDS,
        -1,
    )


def _stamp_row(path: Path):
    try:
        st = path.stat()
    except OSError:
        return None
    return (path.name, st.st_mtime_ns, st.st_size)


def _read_list(path: Path) -> list[str]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            rows.append(line)
            continue
        if isinstance(value, str) and value:
            rows.append(value)
    return rows


def _names(data: dict, key: str, legacy: str | None = None) -> list[str]:
    if key in data:
        raw = data.get(key) or []
    elif legacy:
        raw = data.get(legacy) or []
    else:
        raw = []
    return [str(item) for item in raw if item]


def _running_now(base: Path) -> tuple[str, str]:
    path = base / "document-now"
    if not path.is_file():
        return "", ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return "", ""
    source = lines[0].strip() if lines else ""
    inner = lines[1].strip() if len(lines) > 1 else ""
    return source, inner


def _read_groups(base: Path) -> tuple[ZimDocGroup, ...]:
    folder = base / "checkpoints"
    if not folder.is_dir():
        return ()
    running_source, running_path = _running_now(base)
    groups = []
    for path in sorted(folder.glob("*.json"), key=lambda item: item.name):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        source = str(data.get("source") or "")
        if not source:
            continue
        closed = bool(data.get("documents_closed"))
        spill_id = hashlib.sha256(source.encode()).hexdigest()[:16]
        spill = base / "spill"
        text = _read_list(spill / f"{spill_id}-text.jsonl")
        ocr = _read_list(spill / f"{spill_id}-ocr.jsonl")
        entries = list(catalog_entries(text, ocr))
        if not entries and closed:
            legacy = _names(data, "ocr_pending") + _names(data, "doc_done", "ocr_done")
            legacy += _names(data, "doc_skipped", "ocr_skipped")
            entries = list(catalog_entries([], legacy))
        if not entries and not closed:
            continue
        done_names = set(_names(data, "doc_done", "ocr_done"))
        skipped_names = set(_names(data, "doc_skipped", "ocr_skipped"))
        current = running_path if running_source == source else ""
        built = []
        for inner, kind in entries:
            if inner in skipped_names:
                state = "skipped"
            elif inner in done_names:
                state = "indexed"
            elif current and inner == current:
                state = "running"
            else:
                state = "waiting"
            built.append((inner, kind, state))
        total = len(built)
        if closed:
            stored_total = int(data.get("doc_total") or 0)
            if stored_total > total:
                total = stored_total
        done = sum(1 for _path, _kind, state in built if state == "indexed")
        skipped = sum(1 for _path, _kind, state in built if state == "skipped")
        groups.append(
            ZimDocGroup(
                source=source,
                name=Path(source).name or source,
                closed=closed,
                total=total,
                done=done,
                skipped=skipped,
                running_path=current if any(inner == current for inner, _kind, _state in built) else "",
                entries=tuple(built),
            )
        )
    groups.sort(key=lambda group: group.name.casefold())
    return tuple(groups)


class _Node:
    def __init__(self) -> None:
        self.dirs: dict[str, _Node] = {}
        self.files: list[tuple[str, str]] = []


def _zim_folder(group: ZimDocGroup, hide_finished: bool, tree_folder, zim_root):
    if group.closed and group.total > 0 and group.done + group.skipped >= group.total:
        return None
    if group.closed and group.total <= 0 and not group.entries:
        return None
    root = _Node()
    shown = 0
    for inner, _kind, state in group.entries:
        if hide_finished and state in ("indexed", "skipped"):
            continue
        if group.closed and state in ("indexed", "skipped") and group.done + group.skipped >= group.total:
            continue
        parts = [part for part in inner.replace("\\", "/").split("/") if part]
        if not parts:
            continue
        *dirs, name = parts
        node = root
        for part in dirs:
            node = node.dirs.setdefault(part, _Node())
        node.files.append((name, virtual_source(group.source, inner)))
        shown += 1
    folders, files = _freeze_docs(doc_zim_key(group.source), root, tree_folder, zim_root)
    if shown <= 0:
        return None
    percent = document_percent(group.closed, group.total, group.done, group.skipped)
    if group.closed:
        status = "documents"
        articles = f"{group.done + group.skipped} of {group.total}"
    else:
        status = "finding"
        articles = f"{len(group.entries)} found"
    return tree_folder(
        doc_zim_key(group.source),
        group.name,
        zim_root,
        shown,
        0,
        folders,
        files,
        status_text=status,
        articles_text=articles,
        progress=0 if percent is None else percent,
        show_bar=percent is not None,
        doc_source=group.source,
    )


def _freeze_docs(key: str, node: _Node, tree_folder, zim_root):
    folders = []
    for part in sorted(node.dirs, key=str.casefold):
        child_key = f"{key}/{part}"
        nested, nested_files = _freeze_docs(child_key, node.dirs[part], tree_folder, zim_root)
        count = len(nested_files) + sum(folder.file_count for folder in nested)
        if not count:
            continue
        folders.append(tree_folder(child_key, part, zim_root, count, 0, nested, nested_files))
    files = tuple(source for _name, source in sorted(node.files, key=lambda row: (row[0].casefold(), row[1])))
    return tuple(folders), files
