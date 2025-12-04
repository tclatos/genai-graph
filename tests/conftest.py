"""Pytest configuration and shared fixtures for genai-graph tests."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from pydantic import BaseModel

from genai_graph.core.graph_backend import KuzuBackend


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Provide a temporary database path for testing.

    Yields:
        Path to a temporary database file that will be cleaned up after the test.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        yield tmp.name
    # Cleanup
    Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture
def graph_backend(temp_db_path: str) -> Generator[KuzuBackend, None, None]:
    """Provide a fresh KuzuBackend instance for testing.

    Args:
        temp_db_path: Temporary database path from fixture

    Yields:
        KuzuBackend instance with temporary database
    """
    backend = KuzuBackend()
    backend.connect(temp_db_path)
    yield backend
    backend.close()


@pytest.fixture
def sample_pydantic_model() -> type[BaseModel]:
    """Provide a simple Pydantic model for testing.

    Returns:
        A basic Pydantic model class for schema testing
    """

    class SampleModel(BaseModel):
        id: str
        name: str
        value: int | None = None
        tags: list[str] = []

    return SampleModel


@pytest.fixture
def nested_pydantic_models() -> dict[str, type[BaseModel]]:
    """Provide nested Pydantic models for testing embedded structs.

    Returns:
        Dictionary of related Pydantic model classes
    """

    class Address(BaseModel):
        street: str
        city: str
        country: str

    class Metrics(BaseModel):
        score: float
        count: int
        is_active: bool = True

    class Company(BaseModel):
        name: str
        address: Address | None = None
        metrics: Metrics | None = None
        employees: list[str] = []

    return {
        "Address": Address,
        "Metrics": Metrics,
        "Company": Company,
    }
