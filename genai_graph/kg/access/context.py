"""Runtime User Context and Security Principal Propagation."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Iterable

from pydantic import BaseModel, Field

_DEFAULT_PUBLIC_SET = frozenset({"public"})


class UserContext(BaseModel):
    """Authenticated user context carrying effective security principals for query trimming."""

    user_id: str = Field(default="anonymous", description="Authenticated user identifier")
    principals: set[str] = Field(
        default_factory=lambda: {"public"},
        description="Expanded set of user principals (e.g. {'user:5d8a7ca1', 'group:03d8e370', 'role:viewer', 'public'})",
    )
    tenant_id: str | None = Field(default=None, description="Optional tenant or organization identifier")
    is_admin: bool = Field(default=False, description="Superuser flag bypassing security trimming (for sysadmins/ETL)")

    model_config = {"frozen": True}

    @classmethod
    def create(
        cls,
        user_id: str = "anonymous",
        groups: Iterable[str] | None = None,
        roles: Iterable[str] | None = None,
        tenant_id: str | None = None,
        is_admin: bool = False,
        extra_principals: Iterable[str] | None = None,
    ) -> UserContext:
        """Helper to construct a UserContext with expanded normalized principal URIs."""
        effective: set[str] = {"public"}
        if user_id and user_id != "anonymous":
            norm_uid = user_id if ":" in user_id else f"user:{user_id}"
            effective.add(norm_uid)
        if groups:
            for g in groups:
                norm_g = g if ":" in g else f"group:{g}"
                effective.add(norm_g)
        if roles:
            for r in roles:
                norm_r = r if ":" in r else f"role:{r}"
                effective.add(norm_r)
        if tenant_id:
            effective.add(tenant_id if ":" in tenant_id else f"tenant:{tenant_id}")
        if extra_principals:
            effective.update(extra_principals)

        return cls(
            user_id=user_id,
            principals=effective,
            tenant_id=tenant_id,
            is_admin=is_admin,
        )

    def has_access(self, allowed_principals: Iterable[str] | None) -> bool:
        """Check if this user context is authorized against the resource's allowed principals."""
        if self.is_admin:
            return True
        if not allowed_principals:
            return "public" in self.principals
        allowed_set = set(allowed_principals)
        return bool(self.principals & allowed_set)


# Thread/Task local context variable for non-LangGraph callers (DeerFlow, CLI, tests)
CURRENT_USER_CONTEXT: ContextVar[UserContext | None] = ContextVar("current_user_context", default=None)


def set_active_user_context(context: UserContext | None) -> Any:
    """Set the active UserContext for the current execution thread or async task."""
    return CURRENT_USER_CONTEXT.set(context)


def get_active_user_context(runtime: Any = None) -> UserContext:
    """Resolve the active UserContext from runtime parameter or ContextVar.

    Order of precedence:
    1. LangGraph `ToolRuntime.context` (if `runtime` is passed and carries a `UserContext`).
    2. Explicit `runtime` object if it is an instance of `UserContext`.
    3. `CURRENT_USER_CONTEXT` ContextVar (used by DeerFlow, CLI, and direct Python calls).
    4. Default anonymous UserContext (public access only).
    """
    if runtime is not None:
        if isinstance(runtime, UserContext):
            return runtime
        if hasattr(runtime, "context") and isinstance(runtime.context, UserContext):
            return runtime.context
        if isinstance(runtime, dict) and "user_id" in runtime:
            return UserContext.create(
                user_id=runtime.get("user_id", "anonymous"),
                groups=runtime.get("groups"),
                roles=runtime.get("roles"),
                tenant_id=runtime.get("tenant_id"),
                is_admin=runtime.get("is_admin", False),
                extra_principals=runtime.get("principals"),
            )

    ctx = CURRENT_USER_CONTEXT.get()
    if ctx is not None:
        return ctx

    return UserContext()
