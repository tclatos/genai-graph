"""YAML-backed access control provider for file-path-based ACL matching."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from genai_graph.kg.access.base import AccessControlResult, BaseAccessControlProvider


class AclRule(BaseModel):
    """Path matching rule mapping glob pattern to allowed principals."""

    path: str = Field(
        ..., description="Glob pattern to match against relative or absolute path (e.g. 'finance/**', '*.pdf')"
    )
    allowed_principals: list[str] = Field(..., description="List of allowed security principals")
    inheritance: str = Field(default="inherit", description="Inheritance mode: inherit | override | union | none")


class YamlAccessControlProvider(BaseAccessControlProvider):
    """Extracts access control lists by evaluating glob rules from a YAML configuration file.

    Example YAML configuration:
    ```yaml
    default: ["public"]
    rules:
      - path: "finance/**"
        allowed_principals: ["group:finance", "role:cfo"]
        inheritance: "override"
      - path: "hr/confidential/**"
        allowed_principals: ["group:hr_leadership", "user:alice"]
      - path: "public/**"
        allowed_principals: ["public"]
    ```
    """

    def __init__(
        self,
        yaml_path: str | Path | None = None,
        rules: list[dict[str, Any]] | None = None,
        default_principals: list[str] | None = None,
        **options: Any,
    ) -> None:
        super().__init__(**options)
        self.yaml_path = Path(yaml_path) if yaml_path else None
        self.default_principals = default_principals or ["public"]
        self._rules: list[AclRule] = []

        if rules is not None:
            self._rules = [AclRule.model_validate(r) for r in rules]
        elif self.yaml_path and self.yaml_path.exists():
            self._load_yaml(self.yaml_path)
        elif self.yaml_path:
            logger.warning(
                "YamlAccessControlProvider: yaml_path '{}' does not exist. Defaulting to public.", self.yaml_path
            )

    def _load_yaml(self, path: Path) -> None:
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if "default" in data and isinstance(data["default"], list):
                self.default_principals = data["default"]
            if "rules" in data and isinstance(data["rules"], list):
                self._rules = [AclRule.model_validate(r) for r in data["rules"]]
        except Exception as exc:
            logger.error("Failed loading ACL YAML from '{}': {}", path, exc)

    def _match_rules(self, target_path: str) -> AccessControlResult:
        norm_path = target_path.replace("\\", "/").strip("/")
        for rule in self._rules:
            pattern = rule.path.replace("\\", "/").strip("/")
            # Support both fnmatch (standard glob) and path prefix matching
            if fnmatch.fnmatch(norm_path, pattern) or fnmatch.fnmatch(Path(norm_path).name, pattern):
                return AccessControlResult(
                    allowed_principals=list(rule.allowed_principals),
                    inheritance=rule.inheritance,  # type: ignore[arg-type]
                    metadata={"matched_pattern": rule.path},
                )
            # Handle directory wildcard matching like 'finance/**'
            if pattern.endswith("/**"):
                prefix = pattern[:-3]
                if norm_path == prefix or norm_path.startswith(f"{prefix}/"):
                    return AccessControlResult(
                        allowed_principals=list(rule.allowed_principals),
                        inheritance=rule.inheritance,  # type: ignore[arg-type]
                        metadata={"matched_pattern": rule.path},
                    )
        return AccessControlResult(
            allowed_principals=list(self.default_principals),
            inheritance="inherit",
            metadata={"matched_pattern": "default"},
        )

    async def get_document_acl(
        self,
        file_path: Path,
        relative_path: str,
        folder_chain: list[str] | None = None,
    ) -> AccessControlResult:
        # Match against relative_path first, fallback to filename
        return self._match_rules(relative_path or file_path.name)

    async def get_folder_acl(
        self,
        folder_path: Path,
        uri: str,
        parent_uri: str | None = None,
    ) -> AccessControlResult:
        return self._match_rules(uri or folder_path.name)
