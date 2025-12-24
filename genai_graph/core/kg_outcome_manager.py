"""Knowledge Graph Outcome Manager.

This module provides a centralized manager for organizing KG outputs,
including database files, HTML exports, warnings, and other artifacts.
Each KG configuration gets its own organized folder structure.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from genai_tk.utils.config_mngr import global_config
from loguru import logger
from pydantic import BaseModel
from upath import UPath


class KgOutcome(BaseModel):
    """Record of a KG operation outcome."""

    timestamp: str
    operation: str  # e.g., "create", "add_doc", "delete", "export_html"
    status: str  # e.g., "success", "warning", "error"
    message: str
    details: dict[str, Any] | None = None


class KgOutcomeManager(BaseModel):
    """Manages organized folder structure and outcomes for KG configurations.

    Each KG configuration gets its own folder containing:
    - Database files (kuzu/*.db)
    - HTML exports (html/*.html)
    - Outcome logs (outcomes.jsonl)
    - Warnings and errors (warnings.log)
    """

    config_name: str
    _base_path: UPath | None = None
    _outcomes_file: UPath | None = None
    _warnings_file: UPath | None = None

    @property
    def base_path(self) -> UPath:
        """Get the base directory for this KG configuration.

        The path is constructed from:
        1. paths.data_root from config
        2. The config_name

        Example: /data/kg/my_project/test1
        """
        if self._base_path is None:
            config = global_config()
            data_root = config.get_dir_path("paths.data_root")
            self._base_path = data_root / "kg_outputs" / self.config_name

        return self._base_path

    @property
    def db_dir(self) -> UPath:
        """Get the directory for database files."""
        return self.base_path / "kuzu"

    @property
    def db_path(self) -> UPath:
        """Get the path to the Kuzu database file."""
        return self.db_dir / "ekg_database.db"

    @property
    def html_dir(self) -> UPath:
        """Get the directory for HTML exports."""
        return self.base_path / "html"

    @property
    def outcomes_file(self) -> UPath:
        """Get the path to the outcomes log file."""
        if self._outcomes_file is None:
            self._outcomes_file = self.base_path / "outcomes.jsonl"
        return self._outcomes_file

    @property
    def warnings_file(self) -> UPath:
        """Get the path to the warnings log file."""
        if self._warnings_file is None:
            self._warnings_file = self.base_path / "warnings.log"
        return self._warnings_file

    def ensure_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.html_dir.mkdir(parents=True, exist_ok=True)

    def get_html_export_path(self, suffix: str = "") -> UPath:
        """Get the path for an HTML export file.

        Args:
            suffix: Optional suffix to add to the filename (before .html)

        Returns:
            Full path to the HTML file
        """
        self.html_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.config_name}{suffix}_graph.html"
        return self.html_dir / filename

    def log_outcome(
        self,
        operation: str,
        status: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log an outcome to the outcomes file.

        Args:
            operation: The operation performed (e.g., "create", "add_doc")
            status: Status of the operation (e.g., "success", "warning", "error")
            message: Human-readable message describing the outcome
            details: Optional additional details as a dictionary
        """
        self.base_path.mkdir(parents=True, exist_ok=True)

        outcome = KgOutcome(
            timestamp=datetime.now().isoformat(),
            operation=operation,
            status=status,
            message=message,
            details=details,
        )

        # Append to JSONL file
        with open(self.outcomes_file, "a") as f:
            f.write(outcome.model_dump_json() + "\n")

        logger.debug(f"Logged outcome: {operation} - {status} - {message}")

    def log_warnings(self, warnings: list[str]) -> None:
        """Log warnings to the warnings file.

        Args:
            warnings: List of warning messages
        """
        if not warnings:
            return

        self.base_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().isoformat()
        with open(self.warnings_file, "a") as f:
            f.write(f"\n=== Warnings at {timestamp} ===\n")
            f.writelines(f"{warning}\n" for warning in warnings)

        logger.debug(f"Logged {len(warnings)} warnings to {self.warnings_file}")

    def get_recent_outcomes(self, limit: int = 10) -> list[KgOutcome]:
        """Get the most recent outcomes.

        Args:
            limit: Maximum number of outcomes to return

        Returns:
            List of recent outcomes, newest first
        """
        if not self.outcomes_file.exists():
            return []

        outcomes = []
        with open(self.outcomes_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        outcomes.append(KgOutcome.model_validate_json(line))
                    except Exception as e:
                        logger.warning(f"Failed to parse outcome: {e}")

        # Return newest first
        return outcomes[-limit:][::-1]

    def get_recent_warnings(self, limit: int = 50) -> list[str]:
        """Get the most recent warnings.

        Args:
            limit: Maximum number of warning lines to return

        Returns:
            List of recent warning lines, newest first
        """
        if not self.warnings_file.exists():
            return []

        with open(self.warnings_file) as f:
            lines = f.readlines()

        # Return newest first
        return [line.strip() for line in lines[-limit:][::-1]]

    def clear_all(self) -> None:
        """Remove all files and directories for this KG configuration."""
        import shutil

        if self.base_path.exists():
            shutil.rmtree(self.base_path)
            logger.info(f"Cleared all data for KG config '{self.config_name}'")

    def get_info(self) -> dict[str, Any]:
        """Get information about this KG configuration's artifacts.

        Returns:
            Dictionary with paths, file sizes, and counts
        """
        info = {
            "config_name": self.config_name,
            "base_path": str(self.base_path),
            "exists": self.base_path.exists(),
        }

        if self.base_path.exists():
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
                with open(self.outcomes_file) as f:
                    outcome_count = sum(1 for _ in f)
                info["outcomes"] = {
                    "count": outcome_count,
                    "file": str(self.outcomes_file),
                }
            else:
                info["outcomes"] = None

            # Warnings
            if self.warnings_file.exists():
                with open(self.warnings_file) as f:
                    warning_count = sum(1 for _ in f)
                info["warnings"] = {
                    "count": warning_count,
                    "file": str(self.warnings_file),
                }
            else:
                info["warnings"] = None

        return info


def get_kg_outcome_manager(config_name: str | None = None) -> KgOutcomeManager:
    """Get a KG outcome manager for the specified or default configuration.

    Args:
        config_name: KG configuration name, or None to use default

    Returns:
        KgOutcomeManager instance
    """
    if config_name is None:
        config = global_config()
        config_name = config.get("kg_config") or config.get("default_kg_config", "default")
    assert config_name is not None, "config_name must be specified or defaulted"
    return KgOutcomeManager(config_name=config_name)
