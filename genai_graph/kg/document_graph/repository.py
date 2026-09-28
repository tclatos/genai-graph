"""Folder abstraction for the Document Graph.

A :class:`~genai_graph.kg.nodes.document.Folder` is a base location that
source documents are read from. This module resolves a source *spec* (a local
directory, a single file, or a ``.zip`` archive) into a Folder node plus the list
of files it contains, each with a ``relative_path`` relative to the folder base.
This keeps full absolute paths out of the graph — a Document stores only its
``folder_id`` and ``relative_path``.

SharePoint (or other remote) folders are intentionally left as a future
``kind`` — the interface (``folder_node`` + ``iter_files``) is designed so a
remote backend can be plugged in without touching callers.
"""

from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from loguru import logger
from pydantic import BaseModel

from genai_graph.kg.nodes.document import Folder


class ResolvedFile(BaseModel):
    """A file discovered inside a folder."""

    abs_path: Path
    relative_path: str

    model_config = {"arbitrary_types_allowed": True}


class SourceFolder:
    """Resolves a directory, single file, or ``.zip`` source into a Folder node + its files."""

    def __init__(self, *, uri: str, kind: str, name: str, base_path: Path) -> None:
        self.uri = uri
        self.kind = kind
        self.name = name
        self.base_path = base_path

    @classmethod
    def from_source(cls, source: str, *, cache_dir: str | None = None) -> "SourceFolder":
        """Build a folder from a directory path or a ``.zip`` archive path.

        Args:
            source: Path to a directory or a ``.zip`` file (config placeholders
                such as ``${paths...}`` are resolved).
            cache_dir: Where to extract ``.zip`` archives (defaults to a sibling
                ``.document_graph_cache`` directory).
        """
        from genai_tk.config_mgmt.file_patterns import resolve_config_path

        resolved = Path(resolve_config_path(source))
        uri = str(resolved)

        if resolved.is_dir():
            return cls(uri=uri, kind="directory", name=resolved.name, base_path=resolved)

        if resolved.suffix.lower() == ".zip":
            base_path = cls._extract_zip(resolved, cache_dir)
            return cls(uri=uri, kind="zip", name=resolved.stem, base_path=base_path)

        # A single file: treat its parent directory as the folder base.
        return cls(uri=str(resolved.parent), kind="directory", name=resolved.parent.name, base_path=resolved.parent)

    @staticmethod
    def _extract_zip(zip_path: Path, cache_dir: str | None) -> Path:
        from genai_tk.utils.hashing import buffer_digest

        base_cache = Path(cache_dir) if cache_dir else zip_path.parent / ".document_graph_cache"
        digest = buffer_digest(str(zip_path.resolve()).encode("utf-8"))
        extract_dir = base_cache / f"{zip_path.stem}_{digest}"

        if extract_dir.exists():
            return extract_dir

        extract_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Extracting {} -> {}", zip_path.name, extract_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        return extract_dir

    def relative_path_of(self, abs_path: Path) -> str:
        """Return *abs_path* relative to the folder base (falls back to the name)."""
        try:
            return str(abs_path.resolve().relative_to(self.base_path.resolve()))
        except ValueError:
            return abs_path.name

    def iter_files(
        self,
        *,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        recursive: bool = True,
        single_file: Path | None = None,
    ) -> list[ResolvedFile]:
        """Discover files under the folder base matching *include*/*exclude*.

        Args:
            include: gitignore-style glob patterns to include (default ``["*.md"]``).
            exclude: glob patterns to exclude.
            recursive: recurse into sub-directories.
            single_file: when the source was a single file, restrict discovery to it.
        """
        if single_file is not None:
            return [ResolvedFile(abs_path=single_file, relative_path=self.relative_path_of(single_file))]

        from genai_tk.config_mgmt.file_patterns import resolve_files

        patterns = include or ["*.md"]
        pathspecs: list[str] = []
        for pattern in patterns:
            if recursive:
                pathspecs.append(f"**/{pattern}")
            pathspecs.append(pattern)
        pathspecs.extend(f"!{p}" for p in (exclude or []))

        discovered = resolve_files(str(self.base_path), pathspecs=pathspecs)
        return [ResolvedFile(abs_path=Path(p), relative_path=self.relative_path_of(Path(p))) for p in discovered]


class FolderTree:
    """Builds a Merkle-hashed `Folder` node for every directory level under a `SourceFolder`.

    Only directories that (transitively) contain at least one matched file get a
    Folder node — a directory's `folder_id` is a content hash of its direct
    children (matched file names + content hashes, subfolder names + folder_ids),
    so identical subtrees collapse to the same node and unchanged subtrees keep
    a stable id across re-ingestion. Two *different* sources with a structurally
    identical subtree intentionally collapse to the same Folder node, mirroring
    how `Document.content_hash` dedupes identical files.
    """

    def __init__(self, source: SourceFolder) -> None:
        self.source = source
        self.folders: dict[str, Folder] = {}
        self._chain_by_file: dict[str, list[str]] = {}

    def build(
        self,
        files: list[ResolvedFile],
        content_hashes: dict[str, str],
        acl_provider: Any = None,
    ) -> None:
        """Compute all Folder nodes for *files* bottom-up.

        Args:
            files: Files already matched by include/exclude under this source.
            content_hashes: Map of `str(abs_path)` -> content hash (precomputed by the caller).
            acl_provider: Optional BaseAccessControlProvider to populate folder allowed_principals.
        """
        from genai_tk.utils.hashing import buffer_digest

        root: dict[str, Any] = {"files": [], "dirs": {}}
        for rf in files:
            parts = PurePosixPath(rf.relative_path).parts
            dir_parts, filename = tuple(parts[:-1]), parts[-1]
            node = root
            for part in dir_parts:
                node = node["dirs"].setdefault(part, {"files": [], "dirs": {}})
            node["files"].append((filename, str(rf.abs_path)))

        self.folders = {}
        dir_folder_id: dict[tuple[str, ...], str] = {}

        def build_dir(node: dict[str, Any], dir_parts: tuple[str, ...]) -> str:
            child_ids = {name: build_dir(child, (*dir_parts, name)) for name, child in sorted(node["dirs"].items())}

            entries = sorted(f"F:{name}:{content_hashes[path]}" for name, path in node["files"])
            entries += sorted(f"D:{name}:{cid}" for name, cid in child_ids.items())
            canonical = "\n".join(entries)
            folder_id = f"folder_{buffer_digest(canonical.encode('utf-8'))}"

            is_root = not dir_parts
            folder_uri = self.source.uri if is_root else "/".join(dir_parts)
            folder_dir = self.source.base_path if is_root else (self.source.base_path / Path(*dir_parts))

            allowed_principals = ["public"]
            if acl_provider is not None:
                try:
                    res = acl_provider.get_folder_acl_sync(folder_path=folder_dir, uri=folder_uri)
                    allowed_principals = list(res.allowed_principals)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed getting ACL for folder {}: {}", folder_uri, exc)

            folder = Folder(
                folder_id=folder_id,
                parent_folder_id=None,
                uri=folder_uri,
                kind=self.source.kind if is_root else "directory",
                name=self.source.name if is_root else dir_parts[-1],
                allowed_principals=allowed_principals,
            )
            self.folders[folder_id] = folder
            for cid in child_ids.values():
                self.folders[cid].parent_folder_id = folder_id

            dir_folder_id[dir_parts] = folder_id
            return folder_id

        build_dir(root, ())

        def chain_of(folder_id: str) -> list[str]:
            chain: list[str] = []
            current: str | None = folder_id
            while current is not None:
                chain.append(current)
                current = self.folders[current].parent_folder_id
            chain.reverse()
            return chain

        for rf in files:
            dir_parts = tuple(PurePosixPath(rf.relative_path).parts[:-1])
            self._chain_by_file[str(rf.abs_path)] = chain_of(dir_folder_id[dir_parts])

    def chain_for(self, abs_path: Path) -> list[str]:
        """Return the ancestor `folder_id` chain (root-first, leaf-last) for *abs_path*."""
        return self._chain_by_file[str(abs_path)]
