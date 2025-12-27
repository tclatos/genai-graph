"""Central Knowledge Graph manager.

This module defines a singleton :class:`KgManager` responsible for
coordinating KG configuration, identity (profile + tag), filesystem
layout for artifacts, and high-level outcome/warning tracking.

"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from genai_tk.utils.config_mngr import global_config
from genai_tk.utils.singleton import once
from loguru import logger
from pydantic import BaseModel, Field
from upath import UPath


class KgOutcome(BaseModel):
    """Record of a KG operation outcome."""

    timestamp: str
    operation: str
    status: str
    message: str
    details: dict[str, Any] | None = None


class EkgSubgraphConfig(BaseModel):
    """Configuration for a single EKG subgraph entry."""

    factory: str
    initial_load: list[str] = Field(default_factory=list)

    # Allow arbitrary extra keys (db_dsn, files, pull, trigger, ...)
    model_config = {
        "extra": "allow",
    }


class EkgAgentConfig(BaseModel):
    """Agent-related configuration for a KG profile."""

    mcp_servers: list[str] = Field(default_factory=list)

    model_config = {
        "extra": "allow",
    }


class EkgProfileConfig(BaseModel):
    """Configuration for a single KG profile (entry in ``kg_configs``)."""

    subgraphs: list[EkgSubgraphConfig] = Field(default_factory=list)
    agent: EkgAgentConfig | None = None

    model_config = {
        "extra": "allow",
    }


class EkgConfig(BaseModel):
    """Top-level EKG configuration loaded from ``config/ekg.yaml``."""

    kg_config: str
    kg_tag: str = "dev"
    schemas_root: str | None = None
    kg_configs: dict[str, EkgProfileConfig] = Field(default_factory=dict)


class KgManager(BaseModel):
    """Singleton manager for KG configuration, identity and artifacts."""

    ekg_config: EkgConfig
    profile: str
    tag: str
    warnings: list[str] = Field(default_factory=list)

    _base_path: UPath | None = None
    _db_path: UPath | None = None
    _html_dir: UPath | None = None
    _outcomes_file: UPath | None = None
    _warnings_file: UPath | None = None

    model_config = {
        "arbitrary_types_allowed": True,
    }

    # ------------------------------------------------------------------
    # Construction and activation
    # ------------------------------------------------------------------

    @classmethod
    def from_global_config(cls) -> "KgManager":
        """Build a manager instance from the current global configuration."""

        cfg = global_config()

        # Top-level EKG config
        profile = cfg.get("kg_config", default="db_only")
        tag_env = os.environ.get("KG_CONFIG_TAG")
        tag = cfg.get("kg_tag", default=tag_env or "dev")

        try:
            kg_configs_dict = cfg.get_dict("kg_configs")
        except Exception:
            kg_configs_dict = {}

        schemas_root = cfg.get("schemas_root", default=None)

        ekg_config = EkgConfig(
            kg_config=profile,
            kg_tag=tag,
            schemas_root=schemas_root,
            kg_configs={k: EkgProfileConfig(**v) for k, v in kg_configs_dict.items()},
        )

        return cls(ekg_config=ekg_config, profile=profile, tag=tag)

    def _reset_cached_paths(self) -> None:
        self._base_path = None
        self._db_path = None
        self._html_dir = None
        self._outcomes_file = None
        self._warnings_file = None

    def activate(self) -> tuple[str, str]:
        """Validate profile and return current profile and tag.

        Returns:
            Tuple of (profile, tag) in use.
        """

        if self.profile not in self.ekg_config.kg_configs:
            logger.warning(
                "Unknown KG profile '%s'; available=%s",
                self.profile,
                sorted(self.ekg_config.kg_configs.keys()),
            )

        return (self.profile, self.tag)

    # ------------------------------------------------------------------
    # Configuration access
    # ------------------------------------------------------------------

    def get_profile_config(self) -> EkgProfileConfig:
        """Return configuration for the active profile."""

        if self.profile not in self.ekg_config.kg_configs:
            raise KeyError(
                f"KG profile '{self.profile}' is not defined in ekg.yaml; "
                f"available: {sorted(self.ekg_config.kg_configs.keys())}"
            )
        return self.ekg_config.kg_configs[self.profile]

    def get_profile_dict(self) -> dict[str, Any]:
        """Return active profile configuration as a plain dictionary."""

        return self.get_profile_config().model_dump()

    # ------------------------------------------------------------------
    # Filesystem layout helpers
    # ------------------------------------------------------------------

    @property
    def base_path(self) -> UPath:
        """Root directory for this KG profile/tag.

        Layout:
            <paths.data_root>/kg_outputs/<profile>/<tag>/
        """

        if self._base_path is None:
            cfg = global_config()
            data_root = cfg.get_dir_path("paths.data_root")
            self._base_path = data_root / "kg_outputs" / self.profile / self.tag
        return self._base_path

    @property
    def db_dir(self) -> UPath:
        """Directory for database files."""

        return self.base_path / "kuzu"

    @property
    def db_path(self) -> UPath:
        """Path to the Kuzu database file for this KG."""

        if self._db_path is None:
            self._db_path = self.db_dir / "ekg_database.db"
        return self._db_path

    @property
    def html_dir(self) -> UPath:
        """Directory for HTML exports."""

        if self._html_dir is None:
            self._html_dir = self.base_path / "html"
        return self._html_dir

    @property
    def outcomes_file(self) -> UPath:
        """Path to the outcomes log file (JSONL)."""

        if self._outcomes_file is None:
            self._outcomes_file = self.base_path / "outcomes.jsonl"
        return self._outcomes_file

    @property
    def warnings_file(self) -> UPath:
        """Path to the warnings log file (plain text)."""

        if self._warnings_file is None:
            self._warnings_file = self.base_path / "warnings.log"
        return self._warnings_file

    def ensure_directories(self) -> None:
        """Create all necessary directories if they don't exist."""

        self.base_path.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.html_dir.mkdir(parents=True, exist_ok=True)

    def get_html_export_path(self, suffix: str = "") -> UPath:
        """Return destination path for an HTML export file.

        Args:
            suffix: Optional suffix added before ``.html`` in the filename.
        """

        self.html_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.profile}-{self.tag}{suffix}_graph.html"
        return self.html_dir / filename

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

        logger.debug("[KG %s@%s] outcome: %s - %s", self.profile, self.tag, operation, status)

    def log_warnings(self, warnings: list[str]) -> None:
        """Append a block of warnings to the warnings log file."""

        if not warnings:
            return

        self.base_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().isoformat()
        with open(str(self.warnings_file), "a") as f:
            f.write(f"\n=== Warnings at {timestamp} ===\n")
            f.writelines(f"{warning}\n" for warning in warnings)

        logger.debug("[KG %s@%s] logged %d warnings", self.profile, self.tag, len(warnings))

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
                    logger.warning("Failed to parse outcome line: %s", exc)

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
            logger.info("Cleared all data for KG '%s@%s'", self.profile, self.tag)

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

        # HTML exports
        html_files = list(self.html_dir.glob("*.html")) if self.html_dir.exists() else []
        info["html_exports"] = {
            "count": len(html_files),
            "files": [f.name for f in html_files],
        }

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

        return info

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
def get_kg_manager() -> KgManager:
    """Return the process-wide KgManager singleton."""

    return KgManager.from_global_config()
