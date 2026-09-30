"""Ladybug database manager for importing converted Neo4j data.

Handles creating Ladybug (Kuzu-compatible) databases and importing JSON files using COPY FROM statements.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import ladybug
from loguru import logger
from pydantic import BaseModel, Field

from genai_graph.neo4j_import.schema_analyzer import SchemaAnalyzer


class ImportStats(BaseModel):
    """Statistics from the import process."""

    schema_statements_executed: int = 0
    nodes_imported: int = 0
    relationships_imported: int = 0
    node_tables: list[str] = Field(default_factory=list)
    rel_tables: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class KuzuImporter:
    """Imports converted Neo4j data into a Ladybug (Kuzu-compatible) database."""

    def __init__(self, db_path: str | Path) -> None:
        """Initialize with the Ladybug database path."""
        self.db_path = Path(db_path)
        self.db: ladybug.Database | None = None
        self.conn: ladybug.Connection | None = None

    def create_database(self, delete_existing: bool = False) -> None:
        """Create or open the Ladybug database.

        Args:
            delete_existing: If True, delete existing database before creating.
        """
        if delete_existing and self.db_path.exists():
            logger.warning("Deleting existing database: {}", self.db_path)
            if self.db_path.is_dir():
                shutil.rmtree(self.db_path)
            else:
                self.db_path.unlink()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Opening Ladybug database: {}", self.db_path)

        self.db = ladybug.Database(str(self.db_path))
        self.conn = ladybug.Connection(self.db)

        # Load JSON extension
        self.conn.execute("INSTALL json;")
        self.conn.execute("LOAD EXTENSION json;")
        logger.info("JSON extension loaded")

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn = None
        if self.db:
            self.db = None

    def execute_schema(self, schema_statements: list[str]) -> ImportStats:
        """Execute schema creation statements.

        Args:
            schema_statements: List of CREATE TABLE statements.

        Returns:
            ImportStats with execution results.
        """
        if not self.conn:
            raise RuntimeError("Database not open. Call create_database() first.")

        stats = ImportStats()

        for stmt in schema_statements:
            try:
                logger.debug("Executing: {}...", stmt[:100])
                self.conn.execute(stmt)
                stats.schema_statements_executed += 1

                # Track table names
                if "CREATE NODE TABLE" in stmt:
                    # Extract table name
                    parts = stmt.split()
                    idx = parts.index("TABLE") + 3  # Skip IF NOT EXISTS
                    idx = parts.index("EXISTS") + 1 if "IF" in parts and "NOT" in parts else parts.index("TABLE") + 1
                    table_name = parts[idx].split("(")[0].strip()
                    stats.node_tables.append(table_name)
                elif "CREATE REL TABLE" in stmt:
                    parts = stmt.split()
                    idx = parts.index("TABLE") + 3  # Skip IF NOT EXISTS
                    idx = parts.index("EXISTS") + 1 if "IF" in parts and "NOT" in parts else parts.index("TABLE") + 1
                    table_name = parts[idx].split("(")[0].strip()
                    stats.rel_tables.append(table_name)

            except Exception as e:
                error_msg = f"Schema error: {e} - Statement: {stmt[:200]}"
                logger.error(error_msg)
                stats.errors.append(error_msg)

        logger.info(
            f"Schema executed: {stats.schema_statements_executed} statements, "
            f"{len(stats.node_tables)} node tables, {len(stats.rel_tables)} rel tables"
        )

        return stats

    def import_from_json(
        self,
        json_dir: str | Path,
        schema_info: Any | None = None,
    ) -> ImportStats:
        """Import data from JSON files.

        Args:
            json_dir: Directory containing nodes/ and relationships/ subdirs with JSON files.
            schema_info: Optional SchemaInfo for validation.

        Returns:
            ImportStats with import results.
        """
        if not self.conn:
            raise RuntimeError("Database not open. Call create_database() first.")

        json_path = Path(json_dir)
        stats = ImportStats()

        # Import nodes
        nodes_dir = json_path / "nodes"
        if nodes_dir.exists():
            for json_file in sorted(nodes_dir.glob("*.json")):
                table_name = json_file.stem
                stats.node_tables.append(table_name)

                try:
                    stmt = f"COPY {table_name} FROM '{json_file}';"
                    logger.info("Importing nodes: {}", table_name)
                    self.conn.execute(stmt)

                    # Count imported rows
                    count_result = self.conn.execute(f"MATCH (n:{table_name}) RETURN count(n)")
                    while count_result.has_next():
                        row = count_result.get_next()
                        stats.nodes_imported += row[0]

                except Exception as e:
                    error_msg = f"Node import error for {table_name}: {e}"
                    logger.error(error_msg)
                    stats.errors.append(error_msg)

        # Import relationships
        rels_dir = json_path / "relationships"
        if rels_dir.exists():
            for json_file in sorted(rels_dir.glob("*.json")):
                # Parse relationship table name from filename
                # Filename format: REL_TYPE_FromLabel_ToLabel.json
                file_stem = json_file.stem
                stats.rel_tables.append(file_stem)

                try:
                    stmt = f"COPY {file_stem} FROM '{json_file}';"
                    logger.info("Importing relationships: {}", file_stem)
                    self.conn.execute(stmt)

                    # Count imported relationships
                    count_result = self.conn.execute(f"MATCH ()-[r:{file_stem}]->() RETURN count(r)")
                    while count_result.has_next():
                        row = count_result.get_next()
                        stats.relationships_imported += row[0]

                except Exception as e:
                    error_msg = f"Relationship import error for {file_stem}: {e}"
                    logger.error(error_msg)
                    stats.errors.append(error_msg)

        logger.info("Import complete: {} nodes, {} relationships", stats.nodes_imported, stats.relationships_imported)

        return stats

    def run_query(self, query: str) -> list[dict]:
        """Run a Cypher query and return results as list of dicts.

        Args:
            query: Cypher query string.

        Returns:
            List of result dictionaries.
        """
        if not self.conn:
            raise RuntimeError("Database not open. Call create_database() first.")

        result = self.conn.execute(query)
        columns = result.get_column_names()
        rows = []

        while result.has_next():
            row = result.get_next()
            rows.append(dict(zip(columns, row, strict=False)))

        return rows

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with database statistics.
        """
        if not self.conn:
            raise RuntimeError("Database not open. Call create_database() first.")

        stats = {
            "node_tables": {},
            "rel_tables": {},
            "total_nodes": 0,
            "total_relationships": 0,
        }

        # Get node table info
        try:
            result = self.conn.execute("CALL show_tables() RETURN *;")
            while result.has_next():
                row = result.get_next()
                # Columns: id, name, type, database name, comment
                table_name = row[1]
                table_type = row[2]

                if table_type == "NODE":
                    count_result = self.conn.execute(f"MATCH (n:{table_name}) RETURN count(n)")
                    while count_result.has_next():
                        count = count_result.get_next()[0]
                        stats["node_tables"][table_name] = count
                        stats["total_nodes"] += count

                elif table_type == "REL":
                    count_result = self.conn.execute(f"MATCH ()-[r:{table_name}]->() RETURN count(r)")
                    while count_result.has_next():
                        count = count_result.get_next()[0]
                        stats["rel_tables"][table_name] = count
                        stats["total_relationships"] += count

        except Exception as e:
            logger.warning("Error getting stats: {}", e)

        return stats


def import_neo4j_to_kuzu(
    jsonl_path: str | Path,
    db_path: str | Path,
    json_output_dir: str | Path | None = None,
    delete_existing: bool = False,
) -> dict[str, Any]:
    """Complete pipeline to import Neo4j JSONL export into Ladybug.

    Args:
        jsonl_path: Path to Neo4j JSONL export file.
        db_path: Path for the Ladybug database.
        json_output_dir: Optional directory for intermediate JSON files.
            If None, uses a temp directory next to db_path.
        delete_existing: If True, delete existing database.

    Returns:
        Dictionary with import statistics.
    """
    from genai_graph.neo4j_import.converter import Neo4jToKuzuConverter

    jsonl_path = Path(jsonl_path)
    db_path = Path(db_path)

    if json_output_dir is None:
        json_output_dir = db_path.parent / "ladybug_json_import"

    json_output_dir = Path(json_output_dir)

    logger.info("=" * 60)
    logger.info("NEO4J TO LADYBUG IMPORT PIPELINE")
    logger.info("=" * 60)

    # Step 1: Analyze and convert
    logger.info("\nStep 1: Analyzing JSONL and converting to JSON...")
    converter = Neo4jToKuzuConverter(jsonl_path)
    schema_info = converter.analyze_schema()
    conversion_stats = converter.convert(json_output_dir)

    # Step 2: Generate schema
    logger.info("\nStep 2: Generating Ladybug schema...")
    analyzer = SchemaAnalyzer(jsonl_path)
    analyzer.schema = schema_info
    schema_statements = analyzer.generate_kuzu_schema()

    # Step 3: Create database and import
    logger.info("\nStep 3: Creating Ladybug database and importing data...")
    importer = KuzuImporter(db_path)
    importer.create_database(delete_existing=delete_existing)

    schema_stats = importer.execute_schema(schema_statements)
    import_stats = importer.import_from_json(json_output_dir)

    # Get final stats
    final_stats = importer.get_stats()
    importer.close()

    result = {
        "jsonl_path": str(jsonl_path),
        "db_path": str(db_path),
        "json_output_dir": str(json_output_dir),
        "schema_info": {
            "node_tables": list(schema_info.node_tables.keys()),
            "rel_tables": list(schema_info.rel_tables.keys()),
            "total_nodes_in_source": schema_info.total_nodes,
            "total_relationships_in_source": schema_info.total_relationships,
        },
        "conversion": {
            "nodes_processed": conversion_stats.nodes_processed,
            "relationships_processed": conversion_stats.relationships_processed,
            "node_files": len(conversion_stats.node_files_created),
            "rel_files": len(conversion_stats.rel_files_created),
        },
        "import": {
            "nodes_imported": import_stats.nodes_imported,
            "relationships_imported": import_stats.relationships_imported,
            "errors": import_stats.errors + schema_stats.errors,
        },
        "database": final_stats,
    }

    logger.info("\n" + "=" * 60)
    logger.info("IMPORT COMPLETE")
    logger.info("=" * 60)
    logger.info("Database: {}", db_path)
    logger.info("Total nodes: {}", final_stats["total_nodes"])
    logger.info("Total relationships: {}", final_stats["total_relationships"])

    if import_stats.errors or schema_stats.errors:
        logger.warning("Errors encountered: {}", len(import_stats.errors + schema_stats.errors))

    return result
