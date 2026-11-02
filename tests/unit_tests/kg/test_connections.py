"""Unit tests for the reusable graph connection lifecycle."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from pydantic import PrivateAttr

from genai_graph.kg.backend import KgBackend, LadybugBackend
from genai_graph.kg.connections import (
    ConnectionManager,
    LadybugConnectionManager,
    close_all_connections,
    connection,
    get_connection,
    get_connection_manager,
    register_connection_manager,
    release_connection,
)


class RecordingManager(ConnectionManager):
    """Manager that hands out fake backends, for lifecycle assertions."""

    name: str = "recording"
    _opened: list[FakeBackend] = PrivateAttr(default_factory=list)

    def acquire(self, target: str, **options: object) -> KgBackend:
        backend = FakeBackend(target)
        self._opened.append(backend)
        return backend

    def release(self, target: str) -> None:
        self._opened = [b for b in self._opened if b.target != target]

    def close_all(self) -> None:
        for backend in self._opened:
            backend.closed = True
        self._opened = []

    def opened_count(self) -> int:
        """Return how many fake backends are currently handed out."""
        return len(self._opened)


class FakeBackend:
    """Minimal stand-in recording close() calls."""

    def __init__(self, target: str) -> None:
        self.target = target
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def temp_db(tmp_path: Path) -> str:
    """Path to a fresh, empty database file for the real Ladybug manager."""
    return str(tmp_path / "connections_test.db")


def test_same_thread_reuses_one_connection(temp_db: str) -> None:
    first = get_connection(temp_db)
    second = get_connection(temp_db)
    assert first is second
    release_connection(temp_db)


def test_release_closes_and_reopens(temp_db: str) -> None:
    first = get_connection(temp_db)
    release_connection(temp_db)
    manager = get_connection_manager()
    assert isinstance(manager, LadybugConnectionManager)
    assert manager.live_count() == 0

    second = get_connection(temp_db)
    assert second is not first
    release_connection(temp_db)


def test_threads_get_distinct_connections(temp_db: str) -> None:
    seen: list[KgBackend] = []
    main = get_connection(temp_db)

    def worker() -> None:
        seen.append(get_connection(temp_db))
        release_connection(temp_db)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert len(seen) == 1
    assert seen[0] is not main
    release_connection(temp_db)


def test_release_without_connection_is_a_noop(temp_db: str) -> None:
    release_connection(temp_db)


def test_context_manager_releases_on_exit(temp_db: str) -> None:
    manager = get_connection_manager()
    with connection(temp_db) as backend:
        assert isinstance(backend, LadybugBackend)
        assert manager.live_count() == 1
    assert manager.live_count() == 0


def test_close_all_connections_disposes_everything(temp_db: str) -> None:
    manager = get_connection_manager()
    get_connection(temp_db)
    assert manager.live_count() == 1
    close_all_connections()
    assert manager.live_count() == 0


def test_unknown_backend_raises() -> None:
    with pytest.raises(KeyError):
        get_connection_manager("not-a-backend")


def test_custom_manager_can_be_registered() -> None:
    manager = RecordingManager()
    register_connection_manager(manager)
    backend = get_connection("anything", backend="recording")
    assert isinstance(backend, FakeBackend)
    assert manager.opened_count() == 1
    release_connection("anything", backend="recording")
    assert manager.opened_count() == 0


def test_release_connection_is_exported_from_the_kg_package() -> None:
    import genai_graph.kg as kg

    assert kg.release_connection is release_connection
    assert "release_connection" in kg.__all__
