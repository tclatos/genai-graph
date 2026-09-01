"""In-process parallelism for Ladybug backends sharing one ``Database``."""

from __future__ import annotations

from genai_tk.utils.ladybug import SharedKuzuParallel, SharedLadybugParallel

__all__ = ["SharedKuzuParallel", "SharedLadybugParallel"]
