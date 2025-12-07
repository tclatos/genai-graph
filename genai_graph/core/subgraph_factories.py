""" """

import hashlib
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Type

import pandas as pd
from genai_tk.utils.pydantic.kv_store import PydanticStore
from loguru import logger
from pydantic import BaseModel
from rich.console import Console
from sqlalchemy import Engine, text
from upath import UPath

from genai_graph.core.graph_schema import (
    GraphSchema,
)

console = Console()


class SubgraphFactory(ABC, BaseModel):
    """Abstract base class for subgraph implementations."""

    # Class constant - must be overridden by subclasses
    TOP_CLASS: Type[BaseModel]

    @property
    def name(self) -> str:
        """Name of the subgraph."""
        return self.TOP_CLASS.__name__

    @abstractmethod
    def get_struct_data_by_key(self, key: str) -> BaseModel | None:
        """Load data for the given key."""
        ...

    @abstractmethod
    def build_schema(self) -> GraphSchema:
        """Build and return the graph schema configuration."""
        ...

    def get_node_labels(self) -> dict[str, str]:
        """Get mapping of node types to human-readable descriptions from schema."""
        schema = self.build_schema()
        return {node.node_class.__name__: node.description for node in schema.nodes}

    def get_relationship_labels(self) -> dict[str, tuple[str, str]]:
        """Get mapping of relationship types to (direction, meaning) tuples from schema."""
        schema = self.build_schema()
        result = {}
        for relation in schema.relations:
            direction = f"{relation.from_node.__name__} → {relation.to_node.__name__}"
            result[relation.name] = (direction, relation.description)
        return result

    def get_sample_queries(self) -> list[str]:
        """Get list of sample Cypher queries for this subgraph."""
        return []

    # def get_entity_name_from_data(self, data: Any) -> str:
    #     """Extract a human-readable entity name from loaded data."""
    #     return "Unknown Entity"

    def register(self, registry: Any = None) -> None:  # noqa: F821 "Optional[GraphRegistry]"
        """Register this subgraph implementation.

        If ``registry`` is not provided, the global :class:`GraphRegistry`
        instance is used.
        """
        # Local import to avoid circular dependency at module import time.
        from genai_graph.core.graph_registry import register_subgraph

        register_subgraph(self.name, self, registry=registry)


class KvStoreBackedSubgraphFactory(SubgraphFactory):
    kv_store_id: str = "default"

    def get_struct_data_by_key(self, key: str) -> BaseModel | None:
        """Load graph data from the key-value store.

        Args:
            key: The identifier to load

        Returns:
            Top class instance or None if not found
        """
        try:
            store = PydanticStore(kvstore_id=self.kv_store_id, model=self.TOP_CLASS)
            opportunity = store.load_object(key)
            return opportunity
        except Exception as e:
            raise ValueError(f"[red]Error loading opportunity data: {e}[/red]") from e


class TableBackedSubgraphFactory(SubgraphFactory):
    db_dsn: str
    files: list[UPath]
    pd_read_parameters: dict[str, Any] = {}

    _db_engine: Engine | None = None

    @property
    def table_name(self) -> str:
        """Derive table name from TOP_CLASS name in snake_case."""
        import re

        # Convert PascalCase to snake_case
        name = self.TOP_CLASS.__name__
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
        return snake

    @abstractmethod
    def mapper_function(self, row: dict[str, Any]) -> BaseModel | None:
        """Map database row to model instance.

        Subclasses must implement this to convert a database row dictionary
        to an instance of their top_class model.
        """

    @abstractmethod
    def get_key_field(self) -> str:
        """Return the field name used as the unique key for data retrieval.
        Must implement by subclass.
        """

    def model_post_init(self, _context: Any) -> None:
        """Initialize the database engine and load data from files."""
        from sqlalchemy import create_engine

        logger.info(f"Initializing TableBackedSubgraphFactory with db_dsn: {self.db_dsn}")
        self._db_engine = create_engine(self.db_dsn)
        self._create_import_tracking_table()
        for file_path in self.files:
            self._process_file(file_path)

    def _create_import_tracking_table(self) -> None:
        """Create a table to track imported files with checksums and timestamps."""
        if self._db_engine is None:
            raise RuntimeError("Database engine not initialized")

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS imported_files (
            file_path TEXT PRIMARY KEY,
            checksum TEXT NOT NULL,
            import_date TIMESTAMP NOT NULL,
            row_count INTEGER NOT NULL
        )
        """
        with self._db_engine.connect() as conn:
            conn.execute(text(create_table_sql))
            conn.commit()
        logger.info("Import tracking table created or verified")

    def _calculate_file_checksum(self, file_path: UPath) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(str(file_path), "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _check_file_changed(self, file_path: UPath, checksum: str) -> bool | None:
        """Check if file has changed since last import.

        Returns:
            None: File never imported before
            False: File unchanged (same checksum)
            True: File changed (different checksum)
        """
        if self._db_engine is None:
            raise RuntimeError("Database engine not initialized")

        query = text("SELECT checksum FROM imported_files WHERE file_path = :file_path")
        with self._db_engine.connect() as conn:
            result = conn.execute(query, {"file_path": str(file_path)}).fetchone()
            if result:
                existing_checksum = result[0]
                if existing_checksum == checksum:
                    return False  # Unchanged
                else:
                    return True  # Changed
            return None  # New file

    def _delete_file_data(self, file_path: UPath) -> None:
        """Delete all data previously imported from this file."""
        if self._db_engine is None:
            raise RuntimeError("Database engine not initialized")

        table_name = self.table_name

        # Get the row count that will be deleted
        query = text("SELECT row_count FROM imported_files WHERE file_path = :file_path")
        with self._db_engine.connect() as conn:
            result = conn.execute(query, {"file_path": str(file_path)}).fetchone()
            if result:
                old_row_count = result[0]
                logger.info(f"Deleting {old_row_count} rows from previous import of {file_path}")

        # Delete all data from the table (simple approach: delete everything since we only have one file typically)
        # In multi-file scenarios, you'd want to track file_source column for selective deletion
        delete_sql = text(f"DELETE FROM {table_name}")
        delete_tracking_sql = text("DELETE FROM imported_files WHERE file_path = :file_path")

        with self._db_engine.connect() as conn:
            result = conn.execute(delete_sql)
            conn.execute(delete_tracking_sql, {"file_path": str(file_path)})
            conn.commit()
            logger.info(f"Deleted {result.rowcount} rows from table '{table_name}' for reimport")

    def _record_import(self, file_path: UPath, checksum: str, row_count: int) -> None:
        """Record file import in tracking table."""
        if self._db_engine is None:
            raise RuntimeError("Database engine not initialized")

        delete_sql = text("DELETE FROM imported_files WHERE file_path = :file_path")
        insert_sql = text("""
            INSERT INTO imported_files (file_path, checksum, import_date, row_count)
            VALUES (:file_path, :checksum, :import_date, :row_count)
        """)
        with self._db_engine.connect() as conn:
            conn.execute(delete_sql, {"file_path": str(file_path)})
            conn.execute(
                insert_sql,
                {
                    "file_path": str(file_path),
                    "checksum": checksum,
                    "import_date": datetime.now(),
                    "row_count": row_count,
                },
            )
            conn.commit()
        logger.info(f"Recorded import of {file_path} with {row_count} rows")

    def _process_file(self, file_path: UPath) -> None:
        """Process a single file: check existence, checksum, and import if needed."""
        # Check file exists
        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Processing file: {file_path}")

        checksum = self._calculate_file_checksum(file_path)
        logger.debug(f"File checksum: {checksum}")

        # Check if file was previously imported
        file_changed = self._check_file_changed(file_path, checksum)

        if file_changed is False:
            logger.info(f"Skipping already imported file: {file_path}")
            return
        elif file_changed is True:
            # File changed - delete existing data before reimport
            logger.warning(f"File {file_path} has changed (checksum differs) - will delete old data and reimport")
            self._delete_file_data(file_path)

        try:
            df = self._load_dataframe(file_path)
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            self._import_dataframe(df)
            self._record_import(file_path, checksum, len(df))

        except Exception as e:
            error_msg = f"Failed to process file {file_path}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _load_dataframe(self, file_path: UPath) -> pd.DataFrame:
        """Load data from Excel or CSV file using pandas."""
        file_suffix = file_path.suffix.lower()

        try:
            if file_suffix in [".xlsx", ".xls"]:
                logger.debug(f"Reading Excel file with parameters: {self.pd_read_parameters}")
                df = pd.read_excel(str(file_path), **self.pd_read_parameters)  # type: ignore[arg-type]
            elif file_suffix == ".csv":
                logger.debug(f"Reading CSV file with parameters: {self.pd_read_parameters}")
                df = pd.read_csv(str(file_path), **self.pd_read_parameters)  # type: ignore[arg-type]
            else:
                raise ValueError(f"Unsupported file format: {file_suffix}. Use .xlsx, .xls, or .csv")

            return df

        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {str(e)}")
            raise

    def _import_dataframe(self, df: pd.DataFrame) -> None:
        """Import dataframe to SQL database with unique index on key field."""
        if self._db_engine is None:
            raise RuntimeError("Database engine not initialized")

        table_name = self.table_name
        key_field = self.get_key_field()

        # Validate key field exists, check for null keys
        if key_field not in df.columns:
            raise ValueError(f"Key field '{key_field}' not found in dataframe columns: {df.columns.tolist()}")
        null_keys = df[key_field].isna().sum()
        if null_keys > 0:
            logger.warning(f"Found {null_keys} rows with null key field '{key_field}' - these will be skipped")
            df = df[df[key_field].notna()]

        # Remove duplicates based on key field
        initial_rows = len(df)
        df = df.drop_duplicates(subset=[key_field], keep="last")
        if len(df) < initial_rows:
            logger.warning(f"Removed {initial_rows - len(df)} duplicate rows based on key field '{key_field}'")

        try:
            # Check if table exists to decide on index creation
            from sqlalchemy import inspect
            from sqlalchemy.exc import IntegrityError

            inspector = inspect(self._db_engine)
            table_exists = table_name in inspector.get_table_names()

            # Try to import data - pandas will create the table on first call
            try:
                df.to_sql(name=table_name, con=self._db_engine, if_exists="append", index=False, method="multi")
                logger.info(f"Successfully imported {len(df)} rows to table '{table_name}'")
            except IntegrityError as ie:
                # Handle duplicate key errors gracefully
                logger.warning(f"Integrity constraint violation during import: {str(ie)}")
                logger.warning("Attempting row-by-row upsert for conflicting records...")

                # Fall back to row-by-row upsert
                inserted = 0
                updated = 0
                skipped = 0

                for _idx, row in df.iterrows():
                    key_value = row[key_field]
                    try:
                        # Try insert
                        row.to_frame().T.to_sql(name=table_name, con=self._db_engine, if_exists="append", index=False)
                        inserted += 1
                    except IntegrityError:
                        # Key exists - update instead
                        try:
                            # Build UPDATE query with proper parameter names
                            # Replace spaces and special chars in column names for bind parameters
                            param_map = {}
                            set_parts = []
                            for col in df.columns:
                                if col != key_field:
                                    param_name = (
                                        col.replace(" ", "_")
                                        .replace("(", "")
                                        .replace(")", "")
                                        .replace(":", "_")
                                        .replace("-", "_")
                                    )
                                    set_parts.append(f'"{col}" = :{param_name}')
                                    param_map[param_name] = row[col]

                            # Add key field to params
                            key_param_name = (
                                key_field.replace(" ", "_")
                                .replace("(", "")
                                .replace(")", "")
                                .replace(":", "_")
                                .replace("-", "_")
                            )
                            param_map[key_param_name] = key_value

                            set_clause = ", ".join(set_parts)
                            update_sql = text(
                                f'UPDATE {table_name} SET {set_clause} WHERE "{key_field}" = :{key_param_name}'
                            )

                            with self._db_engine.connect() as conn:
                                conn.execute(update_sql, param_map)
                                conn.commit()
                            updated += 1
                        except Exception as update_err:
                            logger.warning(f"Failed to update row with key {key_value}: {update_err}")
                            skipped += 1

                logger.info(f"Upsert complete: {inserted} inserted, {updated} updated, {skipped} skipped")

            # Create unique index only if table was just created (first import)
            if not table_exists:
                with self._db_engine.connect() as conn:
                    index_sql = f'CREATE UNIQUE INDEX IF NOT EXISTS idx_{key_field.replace(" ", "_")} ON {table_name} ("{key_field}")'
                    conn.execute(text(index_sql))
                    conn.commit()
                    logger.info(f"Created unique index on '{key_field}'")

        except Exception as e:
            logger.error(f"Failed to import dataframe to database: {str(e)}")
            raise

    def get_all_keys(self) -> list[str]:
        """Get all unique keys available in the database table.

        Returns:
            List of all unique key values from the table
        """
        if self._db_engine is None:
            raise RuntimeError("Database engine not initialized")

        table_name = self.table_name
        key_field = self.get_key_field()

        logger.debug("Retrieving all unique keys from database")

        try:
            query = text(f'SELECT DISTINCT "{key_field}" FROM {table_name} ORDER BY "{key_field}"')

            with self._db_engine.connect() as conn:
                results = conn.execute(query).fetchall()

            keys = [str(row[0]) for row in results]
            logger.info(f"Found {len(keys)} unique keys in database")
            return keys

        except Exception as e:
            logger.error(f"Failed to retrieve keys from database: {str(e)}")
            raise

    def get_struct_data_by_key(self, key: str) -> BaseModel | None:
        """Load data for the given key from the SQL database."""
        if self._db_engine is None:
            raise RuntimeError("Database engine not initialized")

        table_name = self.table_name
        key_field = self.get_key_field()

        logger.debug(f"Querying database for key: {key}")

        try:
            query = text(f'SELECT * FROM {table_name} WHERE "{key_field}" = :key')

            with self._db_engine.connect() as conn:
                result = conn.execute(query, {"key": key}).fetchone()

            if result is None:
                logger.warning(f"No data found for key: {key}")
                return None

            # Convert result to dict
            row_dict = dict(result._mapping)
            logger.debug(f"Found row with {len(row_dict)} columns")

            # Use mapper function to convert to model instance
            model_instance = self.mapper_function(row_dict)

            if model_instance is None:
                logger.warning(f"Mapper function returned None for key: {key}")
                return None

            logger.info(f"Successfully retrieved and mapped data for key: {key}")
            return model_instance

        except Exception as e:
            logger.error(f"Failed to retrieve data for key {key}: {str(e)}")
            raise
