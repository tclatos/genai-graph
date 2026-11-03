"""Per-thread graph connection lifecycle, backend-agnostic.

Consumers ask for a connection to a *target* (a database file path for an
embedded engine, or a server URI) and release it when their unit of work is
done:

```
from genai_graph.kg.connections import connection

with connection("data/kg/docgraph.db") as backend:
    rows = backend.execute("MATCH (d:Document) RETURN d.content_hash")
```

Two rules drive the design:

- **One connection per (thread, target).** Embedded engines have no connection
  pool; a connection is a native handle that is not safe to share between
  threads. Client/server engines pool internally, so their manager hands out a
  cheap per-thread session over one process-shared driver.
- **Connections must be released.** A Ladybug ``Connection`` carries ~120 MB of
  native working set (measured). Keeping one per question grew a benchmark
  process from 151 MB to 3.25 GB until the native buffer manager could not
  allocate and every later query failed. Managers therefore track every handle
  they hand out, expose :func:`release_connection` for the owner of a unit of
  work, and dispose everything in order at interpreter shutdown.

Adding a backend means implementing :class:`ConnectionManager` (see
``docs/design/07_kg_backend_abstraction.md``); callers never change.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, PrivateAttr

from genai_graph.kg.backend import KgBackend, LadybugBackend

DEFAULT_BACKEND = "ladybug"


def resolve_buffer_pool_size() -> int | str:
    """Return the embedded-engine buffer-pool size to request.

    Resolution order: ``LADYBUG_BUFFER_POOL_SIZE`` env var, then the project's
    ``ladybug.buffer_pool_size`` config key (e.g. in ``config/app_conf.yaml``),
    then ``8GB``. The engine default (~80% of system RAM) lets the pool squeeze
    the host process on long concurrent runs, which surfaces as
    "buffer pool is full" tool errors, so an explicit bound is preferable.
    """
    env_size = os.getenv("LADYBUG_BUFFER_POOL_SIZE", "").strip()
    if env_size:
        return env_size
    try:
        from genai_tk.config_mgmt.config_mngr import global_config

        configured = global_config().get("ladybug.buffer_pool_size", None)
        if configured:
            return str(configured)
    except Exception as exc:
        logger.debug("Could not read ladybug.buffer_pool_size from config: {}", exc)
    return "8GB"


class ConnectionManager(BaseModel, ABC):
    """Owns the connection lifecycle for one graph backend kind."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = ""

    @abstractmethod
    def acquire(self, target: str, **options: Any) -> KgBackend:
        """Return a connection for *target* usable by the calling thread."""

    @abstractmethod
    def release(self, target: str) -> None:
        """Dispose the calling thread's connection for *target* (no-op when absent)."""

    @abstractmethod
    def close_all(self) -> None:
        """Dispose every connection this manager handed out, in a safe order."""


class ThreadLocalConnectionManager(ConnectionManager):
    """One connection per (thread, target), with every handle tracked for shutdown.

    Subclasses implement :meth:`_create` (open a connection) and :meth:`_dispose`
    (close it). Thread affinity and bookkeeping live here so backends only deal
    with their own driver or file semantics.
    """

    _local: threading.local = PrivateAttr(default_factory=threading.local)
    _live: set[Any] = PrivateAttr(default_factory=set)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def _cache(self) -> dict[str, KgBackend]:
        """Return the calling thread's target -> connection map."""
        cache = getattr(self._local, "backends", None)
        if cache is None:
            cache = {}
            self._local.backends = cache
        return cache

    def acquire(self, target: str, **options: Any) -> KgBackend:
        """Return the calling thread's connection for *target*, creating it once."""
        cache = self._cache()
        backend = cache.get(target)
        if backend is None:
            backend = self._create(target, **options)
            cache[target] = backend
            with self._lock:
                self._live.add(backend)
            logger.debug("{}: opened connection for {}", self.name, target)
        return backend

    def release(self, target: str) -> None:
        """Dispose the calling thread's connection for *target*."""
        backend = self._cache().pop(target, None)
        if backend is None:
            return
        with self._lock:
            self._live.discard(backend)
        self._dispose(backend)
        logger.debug("{}: released connection for {}", self.name, target)

    def close_all(self) -> None:
        """Dispose every connection handed out by this manager."""
        with self._lock:
            backends = list(self._live)
            self._live.clear()
        for backend in backends:
            self._dispose(backend)
        if backends:
            logger.debug("{}: closed {} connection(s)", self.name, len(backends))

    def live_count(self) -> int:
        """Return how many connections this manager currently holds."""
        with self._lock:
            return len(self._live)

    @abstractmethod
    def _create(self, target: str, **options: Any) -> KgBackend:
        """Open a new connection for *target*."""

    @abstractmethod
    def _dispose(self, backend: KgBackend) -> None:
        """Close *backend*, releasing its native resources in an orderly way."""


class LadybugConnectionManager(ThreadLocalConnectionManager):
    """Embedded Ladybug: one process-shared Database, one Connection per thread.

    ``ladybug.Database`` must be a singleton per file in a process, while
    ``Connection`` is not thread-safe — hence the shared Database plus a
    thread-local connection, which is what this manager wires together.
    """

    name: str = "ladybug"

    _announced: set[str] = PrivateAttr(default_factory=set)

    def _create(self, target: str, **options: Any) -> KgBackend:
        """Attach a new Connection to the process-shared Database for *target*."""
        from genai_tk.utils.ladybug import get_shared_database

        pool_size = options.pop("buffer_pool_size", None) or resolve_buffer_pool_size()
        read_only = options.pop("read_only", None)
        if read_only is None:
            # Every read tool is a pure read, and read-only handles skip WAL
            # bookkeeping entirely. A missing file keeps the read-write open so
            # the "not ingested yet" experience still works.
            read_only = Path(target).exists()
        if target not in self._announced:
            # The first connection fixes the shared Database's buffer pool for the
            # whole process; log it so a "buffer pool is full" failure can be read
            # against the size that was actually requested.
            self._announced.add(target)
            logger.info("ladybug: shared Database for {} (buffer pool {}, read_only={})", target, pool_size, read_only)
        backend = LadybugBackend()
        backend.attach(
            get_shared_database(target, read_only=read_only, buffer_pool_size=pool_size),
            num_threads=options.pop("num_threads", 0),
        )
        return backend

    def _dispose(self, backend: KgBackend) -> None:
        """Close the connection before dropping the reference.

        Letting a Connection be garbage-collected while other threads still
        query the shared Database corrupts the native heap, so teardown goes
        through the explicit ``close()`` that releases cached results and
        prepared statements first.
        """
        with contextlib.suppress(Exception):
            backend.close()


class Neo4jConnectionManager(ThreadLocalConnectionManager):
    """Neo4j: one process-shared driver, one session per thread.

    Outline only — see ``docs/design/07_kg_backend_abstraction.md``. The driver
    is thread-safe and pools TCP connections, so ``_create`` returns a
    ``Neo4jBackend`` bound to a fresh session and ``_dispose`` closes that
    session (returning the connection to the pool); the driver itself is closed
    once per process by :meth:`close_driver`.
    """

    name: str = "neo4j"

    def _create(self, target: str, **options: Any) -> KgBackend:
        """Open a session for the calling thread over the shared driver."""
        raise NotImplementedError("Neo4j support is not implemented yet — see docs/design/07_kg_backend_abstraction.md")

    def _dispose(self, backend: KgBackend) -> None:
        """Close the thread's session, returning its connection to the pool."""
        raise NotImplementedError("Neo4j support is not implemented yet — see docs/design/07_kg_backend_abstraction.md")


_MANAGERS: dict[str, ConnectionManager] = {}
_MANAGERS_LOCK = threading.Lock()


def register_connection_manager(manager: ConnectionManager) -> None:
    """Register *manager* under its ``name`` so it can be selected by name."""
    with _MANAGERS_LOCK:
        _MANAGERS[manager.name] = manager


def get_connection_manager(backend: str = DEFAULT_BACKEND) -> ConnectionManager:
    """Return the registered manager for *backend* (creating built-ins lazily)."""
    with _MANAGERS_LOCK:
        manager = _MANAGERS.get(backend)
        if manager is None:
            builtin = {"ladybug": LadybugConnectionManager, "neo4j": Neo4jConnectionManager}.get(backend)
            if builtin is None:
                raise KeyError(f"No connection manager registered for backend '{backend}'. Known: {sorted(_MANAGERS)}")
            manager = builtin()
            _MANAGERS[backend] = manager
        return manager


def get_connection(target: str, *, backend: str = DEFAULT_BACKEND, **options: Any) -> KgBackend:
    """Return the calling thread's connection for *target*, opening it once.

    Args:
        target: Database file path (embedded engines) or server URI.
        backend: Registered backend kind (``ladybug``, ``neo4j``).
        **options: Backend-specific options (``read_only``, ``buffer_pool_size``,
            ``num_threads`` for Ladybug; ``auth``/``database`` for Neo4j).
    """
    return get_connection_manager(backend).acquire(target, **options)


def release_connection(target: str, *, backend: str = DEFAULT_BACKEND) -> None:
    """Dispose the calling thread's connection for *target*.

    Call this when a unit of work finishes (a question, a CLI command, a web
    request). Embedded connections hold a large native working set, so leaving
    them to accumulate is what exhausted the engine's buffer pool on long runs.
    """
    get_connection_manager(backend).release(target)


@contextmanager
def connection(target: str, *, backend: str = DEFAULT_BACKEND, **options: Any) -> Iterator[KgBackend]:
    """Yield a connection for *target* and release it on exit."""
    yield get_connection(target, backend=backend, **options)
    release_connection(target, backend=backend)


def close_all_connections() -> None:
    """Dispose every connection held by every registered manager."""
    with _MANAGERS_LOCK:
        managers = list(_MANAGERS.values())
    for manager in managers:
        manager.close_all()


atexit.register(close_all_connections)
