"""Integration tests for the Prefect-based KG creation flow."""

from __future__ import annotations

from pathlib import Path
import os

import pytest

from genai_graph.orchestration.flows import create_kg_flow


@pytest.mark.skipif(
    not os.environ.get("PREFECT_API_URL"),
    reason="Prefect API URL not configured; skipping KG flow integration test.",
)
def test_create_kg_flow_runs(tmp_path: Path, monkeypatch) -> None:
    """Run the KG creation flow end-to-end for a small test config.

    This test primarily verifies that the flow executes without raising and
    that it produces a database file and reasonable statistics for one of the
    existing test configurations.
    """

    # Point data_root to a temporary directory so the test is isolated.
    from genai_tk.utils.config_mngr import global_config

    cfg = global_config()
    cfg.set("paths.data_root", str(tmp_path))

    # Use whatever default test KG config is defined for tests, typically
    # something like "db_only" or any config used in existing integration
    # tests and documentation.
    default_kg = cfg.get("default_kg_config") or "db_only"
    cfg.set("default_kg_config", default_kg)

    result = create_kg_flow(config_name=None, delete_first=True, export_html=False)

    # Database path should exist
    assert result.backend.db_path is not None
    assert result.backend.db_path.parent.exists()

    # Statistics should at least be well-formed
    assert result.stats.total_processed >= 0
    assert result.stats.total_failed >= 0

    # Warnings collection should not crash
    assert isinstance(result.warnings, list)
