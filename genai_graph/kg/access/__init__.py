"""DocGraph Access Control and Security Trimming Module."""

from genai_graph.kg.access.base import (
    AccessControlResult,
    BaseAccessControlProvider,
    DefaultPublicAccessControlProvider,
)
from genai_graph.kg.access.context import (
    CURRENT_USER_CONTEXT,
    UserContext,
    get_active_user_context,
    set_active_user_context,
)
from genai_graph.kg.access.factory import create_access_control_provider
from genai_graph.kg.access.yaml_provider import AclRule, YamlAccessControlProvider

__all__ = [
    "AccessControlResult",
    "AclRule",
    "BaseAccessControlProvider",
    "CURRENT_USER_CONTEXT",
    "DefaultPublicAccessControlProvider",
    "UserContext",
    "YamlAccessControlProvider",
    "create_access_control_provider",
    "get_active_user_context",
    "set_active_user_context",
]
