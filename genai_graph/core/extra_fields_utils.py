from __future__ import annotations

from typing import Any, Dict

from caseconverter import snakecase
from pydantic import BaseModel

from genai_graph.core.graph_schema import GraphNode


def apply_extra_fields(
    item_data: Dict[str, Any], node_info: GraphNode, model: BaseModel, item: Any, source_key: str | None
) -> None:
    """Apply extra structured fields and normalize legacy `metadata`.

    This central helper implements the extraction-time behavior for
    ExtraFields subclasses (e.g., `FileMetadata`, `WinLoss`) and the
    legacy `metadata` map fallback. It mutates `item_data` in-place.
    """
    # Determine if FileMetadata is configured for this node
    extras = getattr(node_info, "extra_classes", []) or []
    metadata_handled = any(getattr(ec, "__name__", "") == "FileMetadata" for ec in extras)

    # If FileMetadata is present, convert existing raw metadata map into the
    # structured file_metadata field (and remove the legacy `metadata` key).
    if metadata_handled:
        try:
            file_meta_cls = next((ec for ec in extras if getattr(ec, "__name__", "") == "FileMetadata"), None)
        except Exception:
            file_meta_cls = None

        if file_meta_cls is not None and "metadata" in item_data:
            try:
                ctx = {"root_model": model, "item": item, "item_data": item_data, "source_key": source_key}
                extra_val = file_meta_cls.get_data(ctx)
                if extra_val is not None:
                    if hasattr(extra_val, "model_dump"):
                        item_data[snakecase(file_meta_cls.__name__)] = extra_val.model_dump()
                    elif isinstance(extra_val, dict):
                        item_data[snakecase(file_meta_cls.__name__)] = extra_val
            except Exception:
                # Do not fail extraction on helper generation errors
                pass

            # Remove legacy metadata to avoid schema mismatches
            item_data.pop("metadata", None)
    else:
        # Legacy behavior: ensure metadata is a dict when the BAML class defines it
        try:
            if hasattr(node_info.baml_class, "model_fields") and "metadata" in node_info.baml_class.model_fields:
                if "metadata" in item_data:
                    metadata = item_data["metadata"]
                    if not isinstance(metadata, dict):
                        if isinstance(metadata, str):
                            import json

                            try:
                                item_data["metadata"] = json.loads(metadata)
                            except (json.JSONDecodeError, TypeError):
                                item_data["metadata"] = {}
                        else:
                            item_data["metadata"] = {}
                else:
                    item_data["metadata"] = {}
        except Exception:
            # be defensive - don't break extraction on unexpected structure
            pass

    # Populate extra_classes structured fields (e.g., file_metadata, win_loss)
    for extra_cls in extras:
        try:
            ctx = {"root_model": model, "item": item, "item_data": item_data, "source_key": source_key}
            extra_val = extra_cls.get_data(ctx)
            if not extra_val:
                continue
            if hasattr(extra_val, "model_dump"):
                extra_dict = extra_val.model_dump()
            elif isinstance(extra_val, dict):
                extra_dict = extra_val
            else:
                extra_dict = dict(getattr(extra_val, "__dict__", {}))

            field_name = snakecase(extra_cls.__name__)
            item_data[field_name] = extra_dict
        except Exception:
            # Defensive: do not fail extraction if extra data generation fails
            continue
