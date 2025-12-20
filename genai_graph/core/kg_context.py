"""Context object for Knowledge Graph creation operations.

This module provides a context object that passes through the entire KG creation
process, collecting warnings and storing metadata that can be useful throughout
the pipeline.
"""

from pydantic import BaseModel, Field


class KgContext(BaseModel):
    """Context object for KG creation, carrying warnings and metadata.

    This object is passed through the entire KG creation chain to:
    - Collect warnings from all modules in one place
    - Pass configuration metadata (KG name, config dict, etc.)
    - Provide a consistent interface for warning collection

    The warnings are deduplicated when retrieved to avoid showing
    the same warning multiple times.
    """

    config_name: str = Field(default="", description="Name of the KG configuration being used")
    config_dict: dict = Field(default_factory=dict, description="Full configuration dictionary")
    warnings: list[str] = Field(default_factory=list, description="List of warning messages")

    def add_warning(self, message: str) -> None:
        """Add a warning message to the context.

        Args:
            message: Warning message to add
        """
        if message and message not in self.warnings:
            self.warnings.append(message)

    def get_warnings(self) -> list[str]:
        """Get deduplicated list of warnings.

        Returns:
            List of unique warning messages in order of first occurrence
        """
        seen = set()
        result = []
        for warning in self.warnings:
            if warning not in seen:
                seen.add(warning)
                result.append(warning)
        return result

    def has_warnings(self) -> bool:
        """Check if any warnings were collected.

        Returns:
            True if warnings exist, False otherwise
        """
        return len(self.warnings) > 0

    def clear_warnings(self) -> None:
        """Clear all warnings from the context."""
        self.warnings.clear()
