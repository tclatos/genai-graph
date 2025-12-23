"""
CocoIndex Vector Store Integration Module.

This module provides reusable functions for working with CocoIndex vector stores,
including document indexing, embedding, and semantic search capabilities.

The module is designed to be used both as a standalone script and as a library
for CLI integration.
"""

import os
from typing import Any

import cocoindex
from dotenv import load_dotenv
from genai_tk.utils.config_mngr import global_config
from pgvector.psycopg import register_vector
from psycopg.sql import SQL, Identifier
from psycopg_pool import ConnectionPool

from .text_embedding_flow import get_text_embedding_flow, set_embedding_config, text_to_embedding

# Global state for connection pooling
_connection_pool: ConnectionPool | None = None


def get_connection_pool(database_url: str) -> ConnectionPool:
    """
    Get a connection pool to the database.

    Args:
        database_url: PostgreSQL connection string

    Returns:
        ConnectionPool instance
    """
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = ConnectionPool(database_url)
    return _connection_pool


def setup_database(database_url: str) -> None:
    """
    Initialize the database with pgvector extension.
    This must be called before any vector operations.

    Args:
        database_url: PostgreSQL connection string
    """
    pool = get_connection_pool(database_url)
    with pool.connection() as conn:
        with conn.cursor() as cur:
            # Create the pgvector extension if it doesn't exist
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
    print("Database setup complete: pgvector extension enabled")


def update_vector_store(config: dict[str, Any], database_url: str) -> dict[str, Any]:
    """
    Update the vector store with documents from the configured source.

    Args:
        config: Configuration dictionary
        database_url: PostgreSQL connection string

    Returns:
        Dictionary with update statistics
    """
    # Setup database with pgvector extension
    setup_database(database_url)

    # Set embedding configuration
    set_embedding_config(
        api_type=config.get("embedding", {}).get("api_type", "openai"),
        model=config.get("embedding", {}).get("model", "text-embedding-3-small"),
    )

    # Get the flow
    flow = get_text_embedding_flow(config)

    # Setup and update the flow
    print("\nSetting up indexing flow...")
    try:
        cocoindex.setup_all_flows(report_to_stdout=True)
        print("\nFlow setup complete.")

        # Perform one-time update to process all documents
        print("Processing documents...")
        stats = flow.update()
        print(f"Update complete: {stats}")
        print("Ready to search.\n")

        return {"success": True, "stats": stats}
    except Exception as e:
        print(f"Error during update: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


def search_vector_store(
    query: str,
    config: dict[str, Any],
    database_url: str,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Search the vector store for documents matching the query.

    Args:
        query: Search query text
        config: Configuration dictionary
        database_url: PostgreSQL connection string
        top_k: Number of results to return (overrides config)

    Returns:
        List of search results with filename, text, and score
    """
    # Get top_k from parameter or config
    if top_k is None:
        top_k = config.get("search", {}).get("top_k", 5)

    # Set embedding configuration
    set_embedding_config(
        api_type=config.get("embedding", {}).get("api_type", "openai"),
        model=config.get("embedding", {}).get("model", "text-embedding-3-small"),
    )

    # Get the flow to get table name
    flow = get_text_embedding_flow(config)

    # Get the table name for the export target
    table_name = cocoindex.utils.get_target_default_name(flow, "doc_embeddings")
    # CocoIndex creates tables with lowercase names, so we need to lowercase it
    table_name = table_name.lower()

    # Evaluate the transform flow with the input query to get the embedding
    query_vector = text_to_embedding.eval(query)

    # Run the query and get the results
    pool = get_connection_pool(database_url)
    with pool.connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            query_sql = SQL(
                """
                SELECT filename, text, embedding <=> %s AS distance
                FROM {} ORDER BY distance LIMIT %s
            """
            ).format(Identifier(table_name))
            cur.execute(
                query_sql,
                (query_vector, top_k),
            )
            results = [{"filename": row[0], "text": row[1], "score": 1.0 - row[2]} for row in cur.fetchall()]
            return results


def _main() -> None:
    """
    Main function to run indexing and then queries.
    This is kept for backward compatibility with standalone execution.
    """
    load_dotenv()

    # Get configuration from global config
    database_url = global_config().get_dsn("paths.postgres")
    etl_config = global_config().get_dict("etl_configs.default.cocoindex")

    # Initialize cocoindex
    os.environ["COCOINDEX_DATABASE_URL"] = database_url
    cocoindex.init()

    # Update vector store
    result = update_vector_store(etl_config, database_url)

    if result.get("success"):
        # Run a test query
        test_query = "What is machine learning?"
        print(f"\nRunning test query: '{test_query}'")
        try:
            results = search_vector_store(test_query, etl_config, database_url)
            print("\nSearch results:")
            if not results:
                print("No results found.")
            for result in results:
                print(f"[{result['score']:.3f}] {result['filename']}")
                print(f"    {result['text'][:200]}...")  # Truncate long text
                print("---")
        except Exception as e:
            print(f"Error during search: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    # _main()
    database_url = global_config().get_dsn("paths.postgres")
    etl_config = global_config().get_dict("etl_configs.default.cocoindex")

    # Initialize cocoindex
    os.environ["COCOINDEX_DATABASE_URL"] = database_url
    cocoindex.init()

    # Update vector store
    result = update_vector_store(etl_config, database_url)
