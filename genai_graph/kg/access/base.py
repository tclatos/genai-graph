"""Base protocols and data models for DocGraph Access Control and Security Trimming."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class AccessControlResult(BaseModel):
    """Result of an access control query for a folder or document."""

    allowed_principals: list[str] = Field(
        default_factory=lambda: ["public"],
        description="Security principals authorized to read the resource (e.g. 'group:03d8e370', 'user:5d8a7ca1', 'public')",
    )
    inheritance: Literal["inherit", "override", "union", "intersection", "none"] = Field(
        default="inherit",
        description="How this resource's ACL interacts with parent folder ACL: "
        "'inherit' (folder ∩ item), 'override' (replaces parent), 'union' (folder ∪ item), 'intersection', 'none' (self only)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional provider-specific ACL metadata (e.g. role definitions, permission IDs)",
    )


class BaseAccessControlProvider(ABC):
    """Abstract base provider for extracting access control lists (ACLs) from file paths or external sources."""

    def __init__(self, **options: Any) -> None:
        self.options = options

    @abstractmethod
    async def get_document_acl(
        self,
        file_path: Path,
        relative_path: str,
        folder_chain: list[str] | None = None,
    ) -> AccessControlResult:
        """Asynchronously determine the access control list for a document file.

        Args:
            file_path: Absolute or local path to the source file.
            relative_path: Path relative to the source repository or folder root.
            folder_chain: Optional list of ancestor folder identifiers or paths (root-first).

        Returns:
            AccessControlResult containing allowed principals and inheritance directive.
        """

    @abstractmethod
    async def get_folder_acl(
        self,
        folder_path: Path,
        uri: str,
        parent_uri: str | None = None,
    ) -> AccessControlResult:
        """Asynchronously determine the access control list for a folder.

        Args:
            folder_path: Path to the folder directory or container.
            uri: Base URI or relative identifier for this folder.
            parent_uri: Optional URI of the parent folder.

        Returns:
            AccessControlResult containing allowed principals and inheritance directive.
        """

    def get_document_acl_sync(
        self,
        file_path: Path,
        relative_path: str,
        folder_chain: list[str] | None = None,
    ) -> AccessControlResult:
        """Synchronous wrapper for get_document_acl."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.get_document_acl(file_path, relative_path, folder_chain))
        else:
            # Running inside an active loop (e.g. jupyter, async worker)
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(lambda: asyncio.run(self.get_document_acl(file_path, relative_path, folder_chain)))
                return future.result()

    def get_folder_acl_sync(
        self,
        folder_path: Path,
        uri: str,
        parent_uri: str | None = None,
    ) -> AccessControlResult:
        """Synchronous wrapper for get_folder_acl."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.get_folder_acl(folder_path, uri, parent_uri))
        else:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(lambda: asyncio.run(self.get_folder_acl(folder_path, uri, parent_uri)))
                return future.result()


class DefaultPublicAccessControlProvider(BaseAccessControlProvider):
    """Default access control provider assigning public access to all resources."""

    async def get_document_acl(
        self,
        file_path: Path,
        relative_path: str,
        folder_chain: list[str] | None = None,
    ) -> AccessControlResult:
        return AccessControlResult(allowed_principals=["public"], inheritance="inherit")

    async def get_folder_acl(
        self,
        folder_path: Path,
        uri: str,
        parent_uri: str | None = None,
    ) -> AccessControlResult:
        return AccessControlResult(allowed_principals=["public"], inheritance="inherit")
