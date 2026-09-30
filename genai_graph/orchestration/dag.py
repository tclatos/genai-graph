"""Import DAG resolution for KG construction.

This module resolves the import tree from KG configurations into a
topologically sorted execution plan, replacing the recursive
``_ensure_kg_exists`` pattern with a flat DAG.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ImportNode(BaseModel):
    """A single node in the import dependency graph."""

    config_name: str
    """KG config name (key in ``kg_configs``)."""

    depends_on: list[str] = Field(default_factory=list)
    """Config names this node must be built after."""


class ImportDag(BaseModel):
    """Topologically sorted import plan for a KG configuration."""

    root: str
    """The target KG config name being built."""

    execution_order: list[ImportNode] = Field(default_factory=list)
    """Import nodes in topological order (dependencies first, root excluded)."""


def resolve_import_dag(
    root_config: str,
    kg_configs: dict[str, object],
) -> ImportDag:
    """Resolve the full import tree into a flat, topologically sorted DAG.

    Walks the ``imports`` / ``import`` fields in KG configurations
    recursively, detects cycles, and returns nodes in dependency-first
    order.  The *root* config itself is **not** included in
    ``execution_order`` — only its transitive imports are.

    Args:
        root_config: Name of the target KG config.
        kg_configs: Mapping of config name → config object (must have
            a ``model_dump()`` method or be a dict with ``imports``/``import`` keys).
    """

    visited: set[str] = set()
    in_stack: set[str] = set()
    order: list[ImportNode] = []

    def _get_imports(name: str) -> list[str]:
        cfg = kg_configs.get(name)
        if cfg is None:
            return []
        d = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg  # type: ignore[assignment]
        return d.get("imports", []) or d.get("import", []) or []

    def _visit(name: str) -> None:
        if name in visited:
            return
        if name in in_stack:
            raise ValueError(f"Circular import detected: {name} is already in the import chain")
        in_stack.add(name)

        deps = _get_imports(name)
        for dep in deps:
            _visit(dep)

        in_stack.discard(name)
        visited.add(name)

        # Only add imports (not the root itself)
        if name != root_config:
            order.append(ImportNode(config_name=name, depends_on=deps))

    # Walk the root's imports
    for imp in _get_imports(root_config):
        _visit(imp)

    return ImportDag(root=root_config, execution_order=order)
