"""Central Knowledge Graph manager.

This module defines a singleton :class:`KgManager` responsible for
coordinating KG configuration, identity (profile + tag), filesystem
layout for artifacts, and high-level outcome/warning tracking.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.utils.singleton import once
from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from genai_graph.kg.ingest.lineage import MarkdownLineage


class KgOutcome(BaseModel):
    """Record of a KG operation outcome."""

    timestamp: str
    operation: str
    status: str
    message: str
    details: dict[str, Any] | None = None


class KgGraphConfig(BaseModel):
    """Configuration for a single graph entry."""

    factory: str
    initial_load: list[str] = Field(default_factory=list)

    # Allow arbitrary extra keys (db_dsn, files, pull, trigger, ...)
    model_config = {
        "extra": "allow",
    }


class KgAgentConfig(BaseModel):
    """Agent-related configuration for a KG profile."""

    mcp_servers: list[str] = Field(default_factory=list)

    model_config = {
        "extra": "allow",
    }


class KgProfileConfig(BaseModel):
    """Configuration for a single KG profile (entry in ``kg_configs``)."""

    graphs: list[KgGraphConfig] = Field(default_factory=list)
    agent: KgAgentConfig | None = None
    imports: list[str] = Field(default_factory=list, alias="import")
    """List of KG config names to import before building this KG."""

    model_config = {
        "extra": "allow",
        "populate_by_name": True,
    }


def _extract_graphs_from_workflow(
    workflow_name: str, workflows: dict[str, Any], _visited: set[str] | None = None
) -> list[dict[str, Any]]:
    """Recursively collect graph configs from a workflow's pipeline steps.

    Handles composite workflows that delegate to sub-workflows via ``run:``.
    Supports both a single ``graph:`` and a list ``graphs:`` per step.
    """
    if _visited is None:
        _visited = set()
    if workflow_name in _visited:
        return []
    _visited.add(workflow_name)

    workflow_def = workflows.get(workflow_name, {})
    graphs: list[dict[str, Any]] = []
    # v2 uses "pipeline:"; fall back to "steps:" for legacy configs
    steps = workflow_def.get("pipeline") or workflow_def.get("steps") or []
    for step in steps:
        step_with = step.get("with", {})
        if "graph" in step_with:
            graphs.append(step_with["graph"])
        if "graphs" in step_with:
            graphs.extend(step_with["graphs"])
        # v2 sub-workflow reference via "run:"
        sub = step.get("run")
        if sub and isinstance(sub, str) and sub in workflows and sub != workflow_name:
            graphs.extend(_extract_graphs_from_workflow(sub, workflows, _visited))
        # v1 sub-workflow reference via "invoke:"
        invoke = step.get("invoke", {})
        if invoke.get("kind") == "workflow":
            sub_v1 = invoke.get("target")
            if sub_v1:
                graphs.extend(_extract_graphs_from_workflow(sub_v1, workflows, _visited))
    return graphs


class KgConfig(BaseModel):
    """Top-level KG configuration loaded from ``config/ekg_workflows.yaml``."""

    kg_config: str
    kg_tag: str = "dev"
    schemas_root: str | None = None
    kg_configs: dict[str, KgProfileConfig] = Field(default_factory=dict)


class KgManager(BaseModel):
    """Singleton manager for KG configuration, identity and artifacts."""

    ekg_config: KgConfig
    profile: str
    tag: str
    warnings: list[str] = Field(default_factory=list)

    _base_path: Path | None = None
    _db_path: Path | None = None
    _html_path: Path | None = None
    _schema_path: Path | None = None
    _schema_json_path: Path | None = None
    _schema_html_path: Path | None = None
    _info_path: Path | None = None
    _outcomes_file: Path | None = None
    _warnings_file: Path | None = None
    _warnings_md_path: Path | None = None

    model_config = {
        "arbitrary_types_allowed": True,
    }

    # ------------------------------------------------------------------
    # Construction and activation
    # ------------------------------------------------------------------

    @classmethod
    def from_global_config(cls) -> KgManager:
        """Build a manager instance from the current global configuration."""

        cfg = global_config()

        tag_env = os.environ.get("KG_CONFIG_TAG")
        tag = cfg.get("kg_tag", default=tag_env or "dev")

        try:
            # Use resolve=False to avoid crashing on ${values.*} placeholders.
            # Path interpolations like ${paths.rainbow_json} remain as strings and are
            # resolved on demand by resolve_config_path() inside the factory.
            from omegaconf import OmegaConf

            _merged = OmegaConf.merge(cfg.root, cfg.selected or {})
            _wf_node = OmegaConf.select(_merged, "workflows", default=None)
            workflows: dict[str, Any] = (
                OmegaConf.to_container(_wf_node, resolve=False, throw_on_missing=False)  # type: ignore[assignment]
                if _wf_node is not None
                else {}
            )
        except Exception:
            workflows = {}

        schemas_root = cfg.get("schemas_root", default=None)

        # Build kg_configs by scanning v2 workflow pipeline steps for kg_name + graph.
        # Each step that supplies a kg_name contributes one graph entry.
        kg_configs: dict[str, KgProfileConfig] = {}
        _SKIP = frozenset({"step_templates", "definitions", "profiles"})
        for wf_name, wf_def in workflows.items():
            if not isinstance(wf_def, dict) or wf_name in _SKIP:
                continue
            steps = wf_def.get("pipeline") or wf_def.get("steps") or []
            for step in steps:
                step_with = step.get("with") or {}
                kg_name = step_with.get("kg_name")
                if not kg_name:
                    continue
                graph_raw = step_with.get("graph")
                graphs_raw = step_with.get("graphs") or []
                all_graphs: list[dict[str, Any]] = []
                if graph_raw:
                    all_graphs.append(graph_raw)
                all_graphs.extend(graphs_raw)
                if not all_graphs:
                    continue
                existing = kg_configs.get(kg_name)
                new_graphs = [KgGraphConfig(**g) for g in all_graphs]
                if existing is None:
                    kg_configs[kg_name] = KgProfileConfig(graphs=new_graphs)
                else:
                    kg_configs[kg_name] = KgProfileConfig(graphs=existing.graphs + new_graphs)

        available = sorted(kg_configs.keys())
        # Respect the active KG config set by the UI (global_config().set("kg_config", ...))
        kg_config_override = cfg.get("kg_config", default=None)
        if kg_config_override and kg_config_override in kg_configs:
            profile = kg_config_override
        elif available:
            profile = available[0]
        else:
            profile = "default"

        ekg_config = KgConfig(
            kg_config=profile,
            kg_tag=tag,
            schemas_root=schemas_root,
            kg_configs=kg_configs,
        )

        return cls(ekg_config=ekg_config, profile=profile, tag=tag)

    def reset_cached_paths(self) -> None:
        self._base_path = None
        self._db_path = None
        self._html_path = None
        self._schema_path = None
        self._schema_json_path = None
        self._schema_html_path = None
        self._info_path = None
        self._outcomes_file = None
        self._warnings_file = None
        self._warnings_md_path = None

    def activate(self) -> tuple[str, str]:
        """Validate profile and return current profile and tag.

        Returns:
            Tuple of (profile, tag) in use.
        """
        if self.profile not in self.ekg_config.kg_configs:
            logger.warning(
                f"Unknown KG profile '{self.profile}'; available={sorted(self.ekg_config.kg_configs.keys())}"
            )
        return (self.profile, self.tag)

    # ------------------------------------------------------------------
    # Configuration access
    # ------------------------------------------------------------------

    def get_profile_config(self) -> KgProfileConfig:
        """Return configuration for the active profile."""
        if self.profile not in self.ekg_config.kg_configs:
            raise KeyError(
                f"KG profile '{self.profile}' is not defined in ekg_workflows.yaml; "
                f"available: {sorted(self.ekg_config.kg_configs.keys())}"
            )
        return self.ekg_config.kg_configs[self.profile]

    def get_profile_dict(self) -> dict[str, Any]:
        """Return active profile configuration as a plain dictionary."""
        return self.get_profile_config().model_dump()

    # ------------------------------------------------------------------
    # Filesystem layout helpers - for any profile
    # ------------------------------------------------------------------

    def get_base_path_for(self, profile: str) -> Path:
        """Return the base path for any given KG profile.

        Args:
            profile: KG configuration profile name

        Returns:
            Root directory for the specified KG profile
        """
        return Path(global_config().get_dir_path("paths.kg_outputs", create_if_not_exists=True)) / profile

    def get_db_path_for(self, profile: str) -> Path:
        """Return the database path for any given KG profile."""
        return self.get_base_path_for(profile) / f"{profile}-{self.tag}.db"

    def get_html_path_for(self, profile: str) -> Path:
        """Return the HTML export path for any given KG profile."""
        return self.get_base_path_for(profile) / f"{profile}-{self.tag}.html"

    def get_schema_path_for(self, profile: str) -> Path:
        """Return the schema text file path for any given KG profile."""
        return self.get_base_path_for(profile) / f"{profile}-{self.tag}-schema.txt"

    def get_schema_json_path_for(self, profile: str) -> Path:
        """Return the schema JSON file path for any given KG profile."""
        return self.get_base_path_for(profile) / f"{profile}-{self.tag}-schema.json"

    def get_schema_html_path_for(self, profile: str) -> Path:
        """Return the schema HTML file path for any given KG profile."""
        return self.get_base_path_for(profile) / f"{profile}-{self.tag}-schema.html"

    def get_info_path_for(self, profile: str) -> Path:
        """Return the info file path for any given KG profile."""
        return self.get_base_path_for(profile) / f"{profile}-{self.tag}-info.md"

    def get_outcomes_file_for(self, profile: str) -> Path:
        """Return the outcomes log file path for any given KG profile."""
        return self.get_base_path_for(profile) / f"{profile}-{self.tag}-outcomes.jsonl"

    def get_warnings_file_for(self, profile: str) -> Path:
        """Return the warnings log file path for any given KG profile."""
        return self.get_base_path_for(profile) / f"{profile}-{self.tag}-warnings.log"

    def get_warnings_md_path_for(self, profile: str) -> Path:
        """Return the warnings markdown report path for any given KG profile."""
        return self.get_base_path_for(profile) / f"{profile}-{self.tag}-warnings.md"

    def ensure_directories_for(self, profile: str) -> None:
        """Create base directory for any given profile if it doesn't exist."""
        self.get_base_path_for(profile).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Filesystem layout helpers - for current profile (backward compatible)
    # ------------------------------------------------------------------

    @property
    def base_path(self) -> Path:
        """Root directory for this KG profile."""
        if self._base_path is None:
            self._base_path = (
                Path(global_config().get_dir_path("paths.kg_outputs", create_if_not_exists=True)) / self.profile
            )
        return self._base_path

    @property
    def db_path(self) -> Path:
        """Path to the Kuzu database file for this KG."""
        if self._db_path is None:
            self._db_path = self.base_path / f"{self.profile}-{self.tag}.db"
        return self._db_path

    @property
    def html_path(self) -> Path:
        """Path to the HTML export file for this KG."""
        if self._html_path is None:
            self._html_path = self.base_path / f"{self.profile}-{self.tag}.html"
        return self._html_path

    @property
    def schema_path(self) -> Path:
        """Path to the schema text file for this KG."""
        if self._schema_path is None:
            self._schema_path = self.base_path / f"{self.profile}-{self.tag}-schema.txt"
        return self._schema_path

    @property
    def schema_json_path(self) -> Path:
        """Path to the schema JSON file for this KG."""
        if self._schema_json_path is None:
            self._schema_json_path = self.base_path / f"{self.profile}-{self.tag}-schema.json"
        return self._schema_json_path

    @property
    def schema_html_path(self) -> Path:
        """Path to the schema HTML visualization for this KG."""
        if self._schema_html_path is None:
            self._schema_html_path = self.base_path / f"{self.profile}-{self.tag}-schema.html"
        return self._schema_html_path

    @property
    def info_path(self) -> Path:
        """Path to the info markdown file for this KG."""
        if self._info_path is None:
            self._info_path = self.base_path / f"{self.profile}-{self.tag}-info.md"
        return self._info_path

    @property
    def outcomes_file(self) -> Path:
        """Path to the outcomes log file (JSONL)."""
        if self._outcomes_file is None:
            self._outcomes_file = self.base_path / f"{self.profile}-{self.tag}-outcomes.jsonl"
        return self._outcomes_file

    @property
    def warnings_file(self) -> Path:
        """Path to the warnings log file (plain text)."""
        if self._warnings_file is None:
            self._warnings_file = self.base_path / f"{self.profile}-{self.tag}-warnings.log"
        return self._warnings_file

    @property
    def warnings_md_path(self) -> Path:
        """Path to the warnings markdown report file."""
        if self._warnings_md_path is None:
            self._warnings_md_path = self.base_path / f"{self.profile}-{self.tag}-warnings.md"
        return self._warnings_md_path

    def ensure_directories(self) -> None:
        """Create base directory if it doesn't exist."""
        self.base_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Outcome and warning management
    # ------------------------------------------------------------------

    def log_outcome(
        self,
        operation: str,
        status: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Append a structured outcome entry to the JSONL outcomes file."""
        self.base_path.mkdir(parents=True, exist_ok=True)

        outcome = KgOutcome(
            timestamp=datetime.now().isoformat(),
            operation=operation,
            status=status,
            message=message,
            details=details,
        )

        with open(str(self.outcomes_file), "a") as f:
            f.write(outcome.model_dump_json() + "\n")

        logger.debug("[KG {}@{}] outcome: {} - {}", self.profile, self.tag, operation, status)

    def log_warnings(self, warnings: list[str]) -> None:
        """Append a block of warnings to the warnings log file."""
        if not warnings:
            return

        self.base_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().isoformat()
        with open(str(self.warnings_file), "a") as f:
            f.write(f"\n=== Warnings at {timestamp} ===\n")
            f.writelines(f"{warning}\n" for warning in warnings)

        logger.debug("[KG {}@{}] logged {} warnings", self.profile, self.tag, len(warnings))

    def get_recent_outcomes(self, limit: int = 10) -> list[KgOutcome]:
        """Return the most recent outcome entries (newest first)."""
        if not self.outcomes_file.exists():
            return []

        outcomes: list[KgOutcome] = []
        with open(str(self.outcomes_file)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    outcomes.append(KgOutcome.model_validate_json(line))
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Failed to parse outcome line: {}", exc)

        return outcomes[-limit:][::-1]

    def get_recent_warnings(self, limit: int = 50) -> list[str]:
        """Return the most recent warning lines (newest first)."""
        if not self.warnings_file.exists():
            return []

        with open(str(self.warnings_file)) as f:
            lines = f.readlines()

        return [line.strip() for line in lines[-limit:][::-1]]

    def clear_all(self) -> None:
        """Remove all files and directories for this KG profile/tag."""
        import shutil

        if self.base_path.exists():
            shutil.rmtree(str(self.base_path))
            logger.info("Cleared all data for KG '{}@{}'", self.profile, self.tag)

    def get_info(self) -> dict[str, Any]:
        """Return information about this KG's artifacts and logs."""
        info: dict[str, Any] = {
            "profile": self.profile,
            "tag": self.tag,
            "base_path": str(self.base_path),
            "exists": self.base_path.exists(),
        }

        if not self.base_path.exists():
            return info

        # Database info
        if self.db_path.exists():
            info["database"] = {
                "path": str(self.db_path),
                "size_mb": self.db_path.stat().st_size / (1024 * 1024),
            }
        else:
            info["database"] = None

        # HTML export
        if self.html_path.exists():
            info["html_export"] = {
                "path": str(self.html_path),
                "size_mb": self.html_path.stat().st_size / (1024 * 1024),
            }
        else:
            info["html_export"] = None

        # Schema artifacts
        if self.schema_path.exists():
            info["schema"] = {
                "path": str(self.schema_path),
                "size_mb": self.schema_path.stat().st_size / (1024 * 1024),
            }
        else:
            info["schema"] = None

        if self.schema_json_path.exists():
            info["schema_json"] = {
                "path": str(self.schema_json_path),
                "size_mb": self.schema_json_path.stat().st_size / (1024 * 1024),
            }
        else:
            info["schema_json"] = None

        if self.schema_html_path.exists():
            info["schema_html"] = {
                "path": str(self.schema_html_path),
                "size_mb": self.schema_html_path.stat().st_size / (1024 * 1024),
            }
        else:
            info["schema_html"] = None

        # Outcomes
        if self.outcomes_file.exists():
            with open(str(self.outcomes_file)) as f:
                outcome_count = sum(1 for _ in f)
            info["outcomes"] = {
                "count": outcome_count,
                "file": str(self.outcomes_file),
            }
        else:
            info["outcomes"] = None

        # Warnings
        if self.warnings_file.exists():
            with open(str(self.warnings_file)) as f:
                warning_count = sum(1 for _ in f)
            info["warnings"] = {
                "count": warning_count,
                "file": str(self.warnings_file),
            }
        else:
            info["warnings"] = None

        # Warnings markdown report
        if self.warnings_md_path.exists():
            info["warnings_report"] = {
                "file": str(self.warnings_md_path),
                "size_bytes": self.warnings_md_path.stat().st_size,
            }
        else:
            info["warnings_report"] = None

        return info

    def get_data_lineage(self) -> tuple[list[MarkdownLineage], list[LineageImportError]]:  # noqa: F821
        """Return data lineage entries for JSON/Markdown/source artifacts.

        This delegates to :mod:`genai_graph.kg.ingest.lineage` so that
        lineage computation remains independent from this manager's other
        responsibilities while still exposing a convenient entry point for
        callers.

        Returns:
            Tuple of (lineage_entries, import_errors). ``import_errors`` contains
            details on any subgraphs that could not be imported (e.g. due to a
            BAML version mismatch), with actionable hints where available.
        """
        from genai_graph.kg.ingest.lineage import LineageImportError, MarkdownLineage, build_lineage_for_manager

        _ = LineageImportError  # re-exported for callers
        _ = MarkdownLineage  # re-exported for callers
        return build_lineage_for_manager(self)

    # ------------------------------------------------------------------
    # Warning collection helpers (for use as a context object)
    # ------------------------------------------------------------------

    def add_warning(self, message: str) -> None:
        """Record a warning message in memory (deduplicated on retrieval)."""
        if message and message not in self.warnings:
            self.warnings.append(message)

    def get_warnings(self) -> list[str]:
        """Return deduplicated warnings in order of first occurrence."""
        seen: set[str] = set()
        result: list[str] = []
        for warning in self.warnings:
            if warning not in seen:
                seen.add(warning)
                result.append(warning)
        return result

    def has_warnings(self) -> bool:
        """Return True if any warnings were collected in memory."""
        return bool(self.warnings)

    def clear_warnings(self) -> None:
        """Clear in-memory warnings (does not touch log files)."""
        self.warnings.clear()


@once
def get_kg_manager(activate: bool = True) -> KgManager:
    """Return the process-wide KgManager singleton.

    Args:
        activate: If True (default), validate profile and log warnings if needed.

    Returns:
        The singleton KgManager instance.
    """
    manager = KgManager.from_global_config()
    if activate:
        manager.activate()
    return manager


def reset_kg_manager() -> None:
    """Reset the singleton KgManager instance.

    Call this when configuration changes and a fresh instance is needed.
    """
    get_kg_manager.clear()  # type: ignore[attr-defined]
