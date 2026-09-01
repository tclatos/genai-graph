"""Knowledge Graph database backend abstraction layer.

This module provides an abstract interface for graph databases and concrete
implementations for different backends (Kuzu, Neo4j, etc.).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class QueryExecutor(ABC):
    """Thin abstraction for anything that can execute Cypher-like queries.

    Implemented by KgBackend and can be satisfied by raw connections that
    expose a compatible ``execute`` method.
    """

    @abstractmethod
    def execute(self, query: str, parameters: dict[str, Any] | None = None) -> Any:
        """Execute a query and return backend-specific results."""
        ...


class KgBackend(QueryExecutor, ABC):
    """Abstract base class for KG database backends."""

    @abstractmethod
    def connect(self, connection_string: str) -> None:
        """Connect to the graph database.

        Args:
            connection_string: Database connection string or path
        """
        ...

    @abstractmethod
    def execute(self, query: str, parameters: dict[str, Any] | None = None) -> Any:
        """Execute a query on the graph database.

        Args:
            query: Query string in the backend's query language
            parameters: Optional query parameters

        Returns:
            Query results
        """
        ...

    @abstractmethod
    def create_node_table(
        self,
        table_name: str,
        fields: dict[str, str],
        primary_key: str,
    ) -> None:
        """Create a node table.

        Args:
            table_name: Name of the node table
            fields: Mapping of field names to types
            primary_key: Primary key field name
        """
        ...

    @abstractmethod
    def create_relationship_table(
        self,
        rel_name: str,
        from_table: str,
        to_table: str,
        properties: dict[str, str] | None = None,
    ) -> None:
        """Create a relationship table.

        Args:
            rel_name: Relationship name/type
            from_table: Source node table
            to_table: Target node table
            properties: Optional relationship properties
        """
        ...

    @abstractmethod
    def drop_table(self, table_name: str) -> None:
        """Drop a table (node or relationship).

        Args:
            table_name: Name of the table to drop
        """
        ...

    @abstractmethod
    def insert_node(self, table_name: str, data: dict[str, Any]) -> None:
        """Insert a node.

        Args:
            table_name: Node table name
            data: Node properties
        """
        ...

    @abstractmethod
    def insert_relationship(
        self,
        rel_name: str,
        from_table: str,
        from_key: str,
        to_table: str,
        to_key: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Insert a relationship.

        Args:
            rel_name: Relationship name/type
            from_table: Source node table
            from_key: Source node key value
            to_table: Target node table
            to_key: Target node key value
            properties: Optional relationship properties
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        ...

    @abstractmethod
    def get_query_language(self) -> str:
        """Get the query language used by this backend.

        Returns:
            Query language name (e.g., 'Cypher', 'KuzuQL')
        """
        ...

    def execute_get_as_df(
        self, query: str, parameters: dict[str, Any] | None = None, union: bool = True
    ) -> pd.DataFrame:
        """Execute a query and return results as a pandas DataFrame.

        Handles both single query results and multiple results (e.g., from queries
        separated by semicolons). When multiple results are returned and union=True,
        the dataframes are concatenated.

        Args:
            query: Query string in the backend's query language
            parameters: Optional query parameters
            union: If True, merge multiple result dataframes. If False, raise error for multiple results.

        Returns:
            Query results as a pandas DataFrame

        Raises:
            ValueError: If multiple results are returned but union=False
            AttributeError: If result object doesn't support get_as_df()
        """
        result = self.execute(query, parameters)

        # Handle single result
        if not isinstance(result, list):
            if hasattr(result, "get_as_df"):
                return result.get_as_df()
            else:
                raise AttributeError(f"Result object of type {type(result).__name__} does not support get_as_df()")

        # Handle multiple results (list)
        if not result:
            # Empty list - return empty dataframe
            return pd.DataFrame()

        if not union:
            raise ValueError(
                f"Query returned {len(result)} results, but union=False. "
                "Set union=True to merge multiple results into a single dataframe."
            )

        # Merge multiple dataframes
        dataframes = []
        for i, res in enumerate(result):
            if hasattr(res, "get_as_df"):
                dataframes.append(res.get_as_df())
            else:
                raise AttributeError(f"Result object {i} of type {type(res).__name__} does not support get_as_df()")

        # Concatenate all dataframes
        if dataframes:
            return pd.concat(dataframes, ignore_index=True)
        else:
            return pd.DataFrame()


class KuzuBackend(KgBackend):
    """Ladybug graph database backend implementation.

    Ladybug is a maintained fork of Kuzu with full API compatibility.
    """

    def __init__(self) -> None:
        """Initialize Ladybug backend."""
        self.db: Any = None
        self.conn: Any = None

    def connect(self, connection_string: str, *, enable_multi_writes: bool = False) -> None:
        """Connect to a Ladybug database file, owning the underlying ``Database``.

        Opens a fresh ``ladybug.Database`` plus one ``Connection`` and loads the
        vector extension. Use ``attach`` instead when several workers must share
        a single ``Database`` (the only shape Ladybug allows for concurrent
        writers within one process).

        Args:
            connection_string: Path to the ``.lbdb`` file (or ``:memory:``).
            enable_multi_writes: Forwarded to ``ladybug.Database``. Set ``True``
                only when concurrent write transactions will run against this
                same ``Database`` object (e.g. a shared-DB worker pool); with the
                default ``False`` Ladybug rejects concurrent write transactions
                with "Only one write transaction at a time is allowed".
        """
        import ladybug

        self.db = ladybug.Database(connection_string, enable_multi_writes=enable_multi_writes)
        self.conn = ladybug.Connection(self.db)
        # Ensure the vector and FTS extensions are loaded so tables with vector
        # or full-text search indexes can accept deletes/inserts without a
        # "extension is not loaded" binder error.
        self.ensure_vector_extension()
        self.ensure_fts_extension()

    def attach(self, database: Any, *, num_threads: int = 0) -> None:
        """Reuse an already-open ``ladybug.Database`` with a fresh ``Connection``.

        Unlike ``connect``, this does **not** create a new ``Database`` object —
        Ladybug permits only one read-write ``Database`` per file in a process, so
        worker threads that need to write concurrently must all share one and each
        take their own ``Connection`` from it. The vector extension is intentionally
        **not** loaded here: attached workers are meant for plain Cypher work (the
        summarize path), and loading it on every worker is wasteful. Callers that
        need vector operations on an attached backend can run ``ensure_vector_extension``
        themselves.

        Args:
            database: An existing, open ``ladybug.Database`` object.
            num_threads: Per-connection query thread cap (0 = unlimited).
        """
        import ladybug

        self.db = database
        self.conn = ladybug.Connection(database, num_threads=num_threads)

    def execute(self, query: str, parameters: dict[str, Any] | None = None) -> Any:
        """Execute a Cypher query on Ladybug."""
        if not self.conn:
            raise RuntimeError("Not connected to database")
        return self.conn.execute(query, parameters)

    def create_node_table(
        self,
        table_name: str,
        fields: dict[str, str],
        primary_key: str,
    ) -> None:
        """Create a node table in Ladybug (idempotent)."""
        fields_str = ", ".join([f"{name} {type_}" for name, type_ in fields.items()])
        create_sql = f"CREATE NODE TABLE IF NOT EXISTS {table_name}({fields_str}, PRIMARY KEY({primary_key}))"
        self.execute(create_sql)

    def create_relationship_table(
        self,
        rel_name: str,
        from_table: str,
        to_table: str,
        properties: dict[str, str] | None = None,
    ) -> None:
        """Create a relationship table in Ladybug (idempotent)."""
        if properties:
            props_str = ", " + ", ".join([f"{name} {type_}" for name, type_ in properties.items()])
        else:
            props_str = ""
        create_rel_sql = f"CREATE REL TABLE IF NOT EXISTS {rel_name}(FROM {from_table} TO {to_table}{props_str})"
        self.execute(create_rel_sql)

    def drop_table(self, table_name: str) -> None:
        """Drop a table in Ladybug."""
        try:
            self.execute(f"DROP TABLE {table_name};")
        except Exception:
            pass

    def insert_node(self, table_name: str, data: dict[str, Any]) -> None:
        """Insert a node in Ladybug.

        DEPRECATED: Use merge_node() instead for incremental graph construction.
        This method creates duplicate nodes and should only be used for initial loads.
        """
        cleaned_data: dict[str, str] = {}
        for key, value in data.items():
            if value is None:
                cleaned_data[key] = "NULL"
            elif isinstance(value, str):
                escaped = value.replace("'", "\\'")
                cleaned_data[key] = f"'{escaped}'"
            elif isinstance(value, list):
                str_list: list[str] = []
                for v in value:
                    if hasattr(v, "value"):
                        clean_v = str(v.value)
                    elif hasattr(v, "__dict__") or isinstance(v, dict):
                        clean_v = str(v).replace("'", "\\'").replace('"', '\\"')
                    else:
                        clean_v = str(v)
                    escaped_v = clean_v.replace("'", "\\'")
                    str_list.append(f"'{escaped_v}'")
                cleaned_data[key] = f"[{','.join(str_list)}]"
            elif hasattr(value, "value"):
                escaped = str(value.value).replace("'", "\\'")
                cleaned_data[key] = f"'{escaped}'"
            elif hasattr(value, "__dict__") or isinstance(value, dict):
                import re

                clean_str = str(value).replace("'", "\\'").replace('"', '\\"')
                clean_str = re.sub(
                    r"<[^>]+>", lambda m: m.group(0).split("'")[1] if "'" in m.group(0) else m.group(0), clean_str
                )
                cleaned_data[key] = f"'{clean_str}'"
            else:
                cleaned_data[key] = str(value)

        fields = ", ".join([f"{k}: {v}" for k, v in cleaned_data.items()])
        create_sql = f"CREATE (:{table_name} {{{fields}}})"
        self.execute(create_sql)

    def insert_relationship(
        self,
        rel_name: str,
        from_table: str,
        from_key: str,
        to_table: str,
        to_key: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Insert a relationship in Ladybug."""
        from_key_escaped = from_key.replace("'", "\\'")
        to_key_escaped = to_key.replace("'", "\\'")

        # Note: This assumes the primary key field names are known
        # In practice, you'd need to track these or pass them in
        match_sql = f"""
        MATCH (from:{from_table}), (to:{to_table})
        WHERE from.{from_table.lower()}_key = '{from_key_escaped}'
          AND to.{to_table.lower()}_key = '{to_key_escaped}'
        CREATE (from)-[:{rel_name}]->(to)
        """
        self.execute(match_sql)

    def _primary_key_field(self, table_name: str) -> str:
        """Return the primary key field of a node table (defaults to 'id')."""
        try:
            for row in self.execute(f"CALL table_info('{table_name}') RETURN *"):
                # Row layout: [column_id, name, type, default, is_primary_key]
                if row[4]:
                    return str(row[1])
        except Exception:  # noqa: BLE001 - fall back to convention
            pass
        return "id"

    def merge_node(
        self, node_type: str, node_data: dict[str, Any], schema_config: Any | None = None
    ) -> tuple[bool, str]:
        """Merge a node into the graph database using MERGE semantics.

        Merges on the table's primary key (introspected via ``table_info``),
        falling back to the 'name' field when the PK cannot be determined.
        Returns a tuple indicating whether the node was created and its identifier.

        Args:
            node_type: Node label/type
            node_data: Node properties dictionary (should include the primary key field)
            schema_config: Optional schema configuration (unused for basic merge)

        Returns:
            Tuple of (was_created: bool, node_id: str)
        """
        merge_key = self._primary_key_field(node_type)
        if merge_key not in node_data:
            # Fall back to 'name' or the first available key
            merge_key = "name" if "name" in node_data else next(iter(node_data.keys()))
        merge_value = node_data[merge_key]

        # The primary key cannot appear in SET clauses — Ladybug rejects updates
        # to primary key properties ("Cannot set property ... used as primary key").
        settable = {k: v for k, v in node_data.items() if k != merge_key}

        def _literal(value: Any) -> str:
            if value is None:
                return "NULL"
            if isinstance(value, str):
                return "'" + value.replace("'", "\\'") + "'"
            return str(value)

        props = ", ".join([f"n.{k} = {_literal(v)}" for k, v in settable.items()])

        merge_query = f"MERGE (n:{node_type} {{{merge_key}: {_literal(merge_value)}}})"
        if props:
            merge_query += f" ON CREATE SET {props}"
        self.execute(merge_query)

        # Note: Simplified return - actual was_created status would require checking result
        return (True, str(merge_value))

    def ensure_vector_extension(self) -> bool:
        """Install and load the Ladybug vector extension.

        Returns:
            True if the vector extension is loaded and available, False otherwise.
        """
        from loguru import logger

        try:
            self.execute("INSTALL VECTOR;")
        except Exception as e:
            msg = str(e)
            if "already" not in msg.lower():
                logger.warning(f"Vector extension install failed (vector indexes will be unavailable): {e}")
        try:
            self.execute("LOAD EXTENSION VECTOR;")
            return True
        except Exception as e:
            msg = str(e)
            if "already loaded" in msg.lower():
                return True
            logger.warning(f"Vector extension load failed (vector indexes will be unavailable): {e}")
            return False

    def ensure_fts_extension(self) -> bool:
        """Install and load the Ladybug FTS (full-text search) extension.

        BM25 scoring (``QUERY_FTS_INDEX``) powers the Document Graph hybrid
        ``search_sections`` tool. Safe to call repeatedly.

        Returns:
            True if the FTS extension is loaded and available, False otherwise.
        """
        from loguru import logger

        try:
            self.execute("INSTALL FTS;")
        except Exception as e:
            if "already" not in str(e).lower():
                logger.warning(f"FTS extension install failed (BM25 search will be unavailable): {e}")
        try:
            self.execute("LOAD EXTENSION FTS;")
            return True
        except Exception as e:
            if "already loaded" in str(e).lower():
                return True
            logger.warning(f"FTS extension load failed (BM25 search will be unavailable): {e}")
            return False

    def create_vector_index(
        self,
        table_name: str,
        field_name: str,
        index_name: str,
        metric: str = "cosine",
        mu: int = 30,
        ml: int = 60,
        pu: float = 0.05,
        efc: int = 200,
    ) -> None:
        """Create a vector index on a FLOAT[] field in Ladybug.

        Args:
            table_name: Table containing the vector field
            field_name: FLOAT[] field to index
            index_name: Name for the created index
            metric: Distance metric (cosine, l2, l2sq, dotproduct). Default: cosine
            mu: Max degree of nodes in upper graph. Default: 30
            ml: Max degree of nodes in lower graph. Default: 60
            pu: Percentage of nodes sampled into upper graph. Default: 0.05
            efc: Number of candidate vertices during construction. Default: 200

        Raises:
            Exception: If vector extension is not available or index creation fails
        """
        from loguru import logger

        # Ensure vector extension is loaded
        if not self.ensure_vector_extension():
            logger.warning(f"Skipping vector index {index_name}: VECTOR extension not available")
            return

        try:
            cypher = (
                f"CALL CREATE_VECTOR_INDEX("
                f"    '{table_name}',"
                f"    '{index_name}',"
                f"    '{field_name}',"
                f"    mu := {mu},"
                f"    ml := {ml},"
                f"    pu := {pu},"
                f"    metric := '{metric}',"
                f"    efc := {efc},"
                f"    cache_embeddings := true"
                f");"
            )
            self.execute(cypher)
            logger.info(f"Created vector index {index_name} on {table_name}.{field_name}")
        except Exception as e:
            err = str(e)
            if "already exists" in err:
                logger.debug(f"Vector index {index_name} on {table_name} already exists, skipping")
                return
            logger.error(f"Failed to create vector index {index_name}: {e}")
            raise

    def drop_vector_index(self, table_name: str, index_name: str) -> None:
        """Drop a vector index from Ladybug.

        Args:
            table_name: Table containing the index
            index_name: Name of the index to drop
        """
        from loguru import logger

        try:
            self.execute(f"CALL DROP_VECTOR_INDEX('{table_name}', '{index_name}');")
            logger.info(f"Dropped vector index {index_name}")
        except Exception as e:
            logger.debug(f"Failed to drop vector index {index_name}: {e}")

    def query_vector_index(
        self,
        table_name: str,
        index_name: str,
        query_vector: list[float],
        k: int,
        efs: int = 200,
    ) -> Any:
        """Query a vector index for k nearest neighbors.

        Args:
            table_name: Table containing the index
            index_name: Name of the vector index
            query_vector: Query embedding vector
            k: Number of nearest neighbors to return
            efs: Number of candidate vertices to consider during search. Default: 200

        Returns:
            Ladybug query result with node and distance columns
        """
        from loguru import logger

        try:
            cypher = (
                f"CALL QUERY_VECTOR_INDEX("
                f"    '{table_name}',"
                f"    '{index_name}',"
                f"    $query_vector,"
                f"    {k},"
                f"    efs := {efs}"
                f") RETURN node, distance;"
            )
            result = self.execute(cypher, {"query_vector": query_vector})
            logger.debug("Vector index query returned results")
            return result
        except Exception as e:
            logger.error(f"Vector index query failed: {e}")
            raise

    def close(self) -> None:
        """Close Ladybug connection."""
        # Ladybug doesn't require explicit closing
        self.db = None
        self.conn = None

    def get_query_language(self) -> str:
        """Get query language."""
        return "Cypher"


class Neo4jBackend(KgBackend):
    """Neo4j graph database backend implementation (placeholder)."""

    def __init__(self) -> None:
        """Initialize Neo4j backend."""
        self.driver: Any = None

    def connect(self, connection_string: str) -> None:
        """Connect to Neo4j database."""
        raise NotImplementedError("Neo4j backend not yet implemented")

    def execute(self, query: str, parameters: dict[str, Any] | None = None) -> Any:
        """Execute a Cypher query on Neo4j."""
        raise NotImplementedError("Neo4j backend not yet implemented")

    def create_node_table(
        self,
        table_name: str,
        fields: dict[str, str],
        primary_key: str,
    ) -> None:
        """Create a node table in Neo4j."""
        raise NotImplementedError("Neo4j backend not yet implemented")

    def create_relationship_table(
        self,
        rel_name: str,
        from_table: str,
        to_table: str,
        properties: dict[str, str] | None = None,
    ) -> None:
        """Create a relationship table in Neo4j."""
        raise NotImplementedError("Neo4j backend not yet implemented")

    def drop_table(self, table_name: str) -> None:
        """Drop a table in Neo4j."""
        raise NotImplementedError("Neo4j backend not yet implemented")

    def insert_node(self, table_name: str, data: dict[str, Any]) -> None:
        """Insert a node in Neo4j."""
        raise NotImplementedError("Neo4j backend not yet implemented")

    def insert_relationship(
        self,
        rel_name: str,
        from_table: str,
        from_key: str,
        to_table: str,
        to_key: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Insert a relationship in Neo4j."""
        raise NotImplementedError("Neo4j backend not yet implemented")

    def close(self) -> None:
        """Close Neo4j connection."""
        raise NotImplementedError("Neo4j backend not yet implemented")

    def get_query_language(self) -> str:
        """Get query language."""
        return "Cypher"


def create_backend(backend_type: str = "kuzu") -> KgBackend:
    """Create a graph backend instance.

    Args:
        backend_type: Type of backend ('kuzu', 'neo4j')

    Returns:
        KgBackend instance
    """
    backends = {
        "kuzu": KuzuBackend,
        "neo4j": Neo4jBackend,
    }

    backend_class = backends.get(backend_type.lower())
    if not backend_class:
        raise ValueError(f"Unknown backend type: {backend_type}. Available: {list(backends.keys())}")
    return backend_class()


def _normalize_db_config(db_config: Any) -> dict[str, Any]:
    """Normalize a ``graph_db.<key>`` entry to a ``{type, path}`` dict.

    Supports the shorthand form (a bare path string, defaulting to the Kuzu/Ladybug
    backend) alongside the explicit ``{type: ..., path: ...}`` mapping form.
    """
    if isinstance(db_config, str):
        return {"type": "kuzu", "path": db_config}
    return dict(db_config)


def create_in_memory_backend() -> KgBackend:
    """Create an in-memory graph backend for temporary graphs/tests.

    Currently returns a Ladybug-based backend connected to an in-memory database.
    """
    backend = KuzuBackend()
    backend.connect(":memory:")
    return backend


def create_backend_from_config(config_key: str = "default", kg_config_name: str | None = None) -> KgBackend:
    """Create a graph backend from YAML configuration.

    Reads configuration from global_config()["graph_db"][config_key] and creates
    the appropriate backend with connection parameters. If kg_config_name is provided,
    uses the KG outcome manager to determine the database path.

    Args:
        config_key: Key in graph_db config section
        kg_config_name: Optional KG configuration name for organized output folders

    Returns:
        Connected KgBackend instance
    """
    from genai_tk.config_mgmt.config_mngr import global_config

    config = global_config()
    graph_db_config = config.get("graph_db", {})

    if config_key not in graph_db_config:
        raise ValueError(f"Graph database config '{config_key}' not found. Available: {list(graph_db_config.keys())}")

    db_config = _normalize_db_config(graph_db_config[config_key])
    backend_type = db_config.get("type")
    connection_path = db_config.get("path")

    if not backend_type:
        raise ValueError(f"Missing 'type' in graph_db config for '{config_key}'")

    # Use KgManager-derived path if kg_config_name is provided
    if kg_config_name and backend_type.lower() == "kuzu":
        from genai_graph.kg.manager import get_kg_manager

        manager = get_kg_manager()

        # Validate that kg_config_name exists
        if kg_config_name not in manager.ekg_config.kg_configs:
            raise ValueError(
                f"KG config '{kg_config_name}' not found. Available: {sorted(manager.ekg_config.kg_configs.keys())}"
            )

        # Construct the path for the specified kg_config without mutating the singleton
        # Use the same path construction logic as KgManager
        kg_base_path = config.get_dir_path("paths.kg_outputs", create_if_not_exists=True) / kg_config_name
        connection_path = kg_base_path / f"{kg_config_name}-{manager.tag}.db"
        kg_base_path.mkdir(parents=True, exist_ok=True)
    elif not connection_path:
        raise ValueError(f"Missing 'path' in graph_db config for '{config_key}'")

    # Create backend instance
    backend = create_backend(backend_type)

    # Convert connection_path to string for consistency
    connection_path_str = str(connection_path)

    # Handle different backend types
    if backend_type.lower() == "kuzu":
        # Ensure parent directory exists for Kuzu
        db_path = Path(connection_path_str)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        backend.connect(connection_path_str)
    elif backend_type.lower() == "neo4j":
        # Neo4j connection with credentials
        _username = db_config.get("username")
        _password = db_config.get("password")
        # Note: Neo4j backend not yet implemented, but config structure is ready
        backend.connect(connection_path_str)
    else:
        # Generic connection
        backend.connect(connection_path_str)

    return backend


def get_backend_storage_path_from_config(config_key: str = "default", kg_config_name: str | None = None) -> Path:
    """Return the filesystem path used by the configured graph backend.

    This helper reads the same configuration used by create_backend_from_config
    and returns the resolved ``path`` as a Path instance. If kg_config_name is provided,
    constructs the path using the same logic as create_backend_from_config.

    Args:
        config_key: Key in the ``graph_db`` config section.
        kg_config_name: Optional KG configuration name for organized output folders

    Returns:
        Path to the backend storage location.
    """
    from genai_tk.config_mgmt.config_mngr import global_config

    config = global_config()
    graph_db_config = config.get("graph_db", {})

    if config_key not in graph_db_config:
        raise ValueError(f"Graph database config '{config_key}' not found. Available: {list(graph_db_config.keys())}")

    db_config = _normalize_db_config(graph_db_config[config_key])
    backend_type = db_config.get("type", "")

    # Use KgManager-derived path if kg_config_name is provided for Kuzu backend
    if kg_config_name and backend_type.lower() == "kuzu":
        from genai_graph.kg.manager import get_kg_manager

        manager = get_kg_manager()
        # Construct the path for the specified kg_config without relying on manager.db_path
        # (which may be tied to a different profile)
        kg_base_path = global_config().get_dir_path("paths.kg_outputs", create_if_not_exists=True) / kg_config_name
        return kg_base_path / f"{kg_config_name}-{manager.tag}.db"

    connection_path = db_config.get("path")
    if not connection_path:
        raise ValueError(f"Missing 'path' in graph_db config for '{config_key}'")

    return Path(str(connection_path))


def delete_backend_storage_from_config(config_key: str = "default", kg_config_name: str | None = None) -> None:
    """Delete on-disk storage for the configured graph backend if it exists.

    This is primarily used by CLI commands (e.g. ``cli kg delete``) to drop the
    entire knowledge graph database in a backend-agnostic way. If kg_config_name is
    provided, uses the KG outcome manager to determine the path.

    For Kuzu databases, also removes associated WAL (write-ahead log) files.

    Args:
        config_key: Key in the ``graph_db`` config section.
        kg_config_name: Optional KG configuration name for organized output folders
    """
    import shutil

    db_path = get_backend_storage_path_from_config(config_key, kg_config_name)

    if not db_path.exists():
        # Also check for orphaned WAL files even if main db doesn't exist
        wal_path = Path(str(db_path) + ".wal")
        if wal_path.exists():
            wal_path.unlink()
        return

    if db_path.is_file():
        db_path.unlink()
        # Also remove WAL file if it exists (Kuzu write-ahead log)
        wal_path = Path(str(db_path) + ".wal")
        if wal_path.exists():
            wal_path.unlink()
    else:
        shutil.rmtree(str(db_path))
