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
import numpy as np
from dotenv import load_dotenv
from genai_tk.utils.config_mngr import global_config
from numpy.typing import NDArray
from pgvector.psycopg import register_vector
from psycopg.sql import SQL, Identifier
from psycopg_pool import ConnectionPool

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


# Module-level configuration for embedding
_embedding_config = {
    "api_type": "openai",
    "model": "text-embedding-3-small",
}


def set_embedding_config(api_type: str = "openai", model: str = "text-embedding-3-small") -> None:
    """Set the embedding configuration for the module."""
    global _embedding_config
    _embedding_config = {"api_type": api_type, "model": model}


@cocoindex.transform_flow()
def text_to_embedding(
    text: cocoindex.DataSlice[str],
) -> cocoindex.DataSlice[NDArray[np.float32]]:
    """
    Embed the text using configured model.
    This is a shared logic between indexing and querying.

    Args:
        text: Text to embed

    Returns:
        Embedded vectors
    """
    # Map string api_type to enum
    api_type = _embedding_config.get("api_type", "openai")
    model = _embedding_config.get("model", "text-embedding-3-small")
    api_type_enum = cocoindex.LlmApiType.OPENAI if api_type.lower() == "openai" else cocoindex.LlmApiType.OPENAI

    return text.transform(
        cocoindex.functions.EmbedText(
            api_type=api_type_enum,
            model=model,
        )
    )


def create_text_embedding_flow(config: dict[str, Any]):
    """
    Create and configure a text embedding flow based on configuration.

    Args:
        config: Configuration dictionary with keys:
            - source.path: Root path for document indexing
            - source.included_patterns: List of file patterns to include
            - chunking.chunk_size: Size of text chunks
            - chunking.chunk_overlap: Overlap between chunks
            - chunking.language: Language for chunking (e.g., 'markdown')
            - embedding.api_type: API type for embedding
            - embedding.model: Model name

    Returns:
        Configured FlowDef object
    """

    @cocoindex.flow_def(name="TextEmbedding")
    def text_embedding_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope) -> None:
        """
        Define a flow that embeds text into a vector database.
        """
        # Extract configuration with defaults
        source_path = config.get("source", {}).get("path", ".")
        included_patterns = config.get("source", {}).get("included_patterns", ["*.md"])
        chunk_size = config.get("chunking", {}).get("chunk_size", 2000)
        chunk_overlap = config.get("chunking", {}).get("chunk_overlap", 500)
        language = config.get("chunking", {}).get("language", "markdown")

        data_scope["documents"] = flow_builder.add_source(
            cocoindex.sources.LocalFile(path=source_path, included_patterns=included_patterns),
        )

        doc_embeddings = data_scope.add_collector()

        with data_scope["documents"].row() as doc:
            doc["chunks"] = doc["content"].transform(
                cocoindex.functions.SplitRecursively(),
                language=language,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            with doc["chunks"].row() as chunk:
                chunk["embedding"] = text_to_embedding(chunk["text"])
                doc_embeddings.collect(
                    filename=doc["filename"],
                    location=chunk["location"],
                    text=chunk["text"],
                    embedding=chunk["embedding"],
                )

        doc_embeddings.export(
            "doc_embeddings",
            cocoindex.targets.Postgres(),
            primary_key_fields=["filename", "location"],
            vector_indexes=[
                cocoindex.VectorIndexDef(
                    field_name="embedding",
                    metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
                )
            ],
        )

    return text_embedding_flow


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

    # Create the flow
    flow = create_text_embedding_flow(config)

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

    # Create the flow to get table name
    flow = create_text_embedding_flow(config)

    # Get the table name for the export target
    table_name = cocoindex.utils.get_target_default_name(flow, "doc_embeddings")

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
