"""
Text Embedding Flow Module.

This module provides a reusable text embedding flow definition for CocoIndex
that can be used by CLI tools and other applications.
"""

import os
from typing import Any

import cocoindex
import numpy as np
from dotenv import load_dotenv
from genai_tk.utils.config_mngr import global_config
from numpy.typing import NDArray

# Module-level configuration for embedding
_embedding_config = {
    "api_type": "openai",
    "model": "text-embedding-3-small",
}

# Module-level configuration for the flow
_flow_config: dict[str, Any] = {
    "source": {"path": ".", "included_patterns": ["*.md"]},
    "chunking": {"chunk_size": 2000, "chunk_overlap": 500, "language": "markdown"},
}


def set_embedding_config(api_type: str = "openai", model: str = "text-embedding-3-small") -> None:
    """Set the embedding configuration for the module."""
    global _embedding_config
    _embedding_config = {"api_type": api_type, "model": model}


def set_flow_config(config: dict[str, Any]) -> None:
    """Set the flow configuration for the module."""
    global _flow_config
    _flow_config = config


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


# Define the flow at module level to ensure it's a singleton
@cocoindex.flow_def(name="TextEmbedding")
def text_embedding_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope) -> None:
    """
    Define a flow that embeds text into a vector database.
    Uses configuration from module-level _flow_config.
    """
    # Extract configuration with defaults
    source_path = _flow_config.get("source", {}).get("path", ".")
    included_patterns = _flow_config.get("source", {}).get("included_patterns", ["*.md"])
    chunk_size = _flow_config.get("chunking", {}).get("chunk_size", 2000)
    chunk_overlap = _flow_config.get("chunking", {}).get("chunk_overlap", 500)
    language = _flow_config.get("chunking", {}).get("language", "markdown")

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


def get_text_embedding_flow(config: dict[str, Any]):
    """
    Get the text embedding flow with the specified configuration.

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
        The text_embedding_flow FlowDef object
    """
    # Update the flow configuration
    set_flow_config(config)

    # Return the singleton flow
    return text_embedding_flow


def main():
    """
    Main entry point for running the text embedding flow.

    This will load configuration and execute the text embedding flow
    to index documents into the vector database.
    """
    # Load environment variables
    load_dotenv()

    # Get database URL
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is required")

    # Load configuration from global config
    try:
        config = global_config().get_dict("etl_configs.default.cocoindex")
    except (KeyError, AttributeError):
        # If config doesn't exist, use defaults
        config = {}

    # Override with environment variables if present
    if "EMBEDDING_MODEL" in os.environ:
        config.setdefault("embedding", {})["model"] = os.environ["EMBEDDING_MODEL"]
    if "EMBEDDING_API_TYPE" in os.environ:
        config.setdefault("embedding", {})["api_type"] = os.environ["EMBEDDING_API_TYPE"]
    if "SOURCE_PATH" in os.environ:
        config.setdefault("source", {})["path"] = os.environ["SOURCE_PATH"]

    # Initialize cocoindex
    os.environ["COCOINDEX_DATABASE_URL"] = database_url
    cocoindex.init()

    # Set embedding configuration
    set_embedding_config(
        api_type=config.get("embedding", {}).get("api_type", "openai"),
        model=config.get("embedding", {}).get("model", "text-embedding-3-small"),
    )

    # Get the flow
    flow = get_text_embedding_flow(config)

    # Setup and update the flow
    print("\nSetting up text embedding flow...")
    try:
        cocoindex.setup_all_flows(report_to_stdout=True)
        print("\nFlow setup complete.")

        # Perform one-time update to process all documents
        print("Processing documents...")
        stats = flow.update()
        print(f"Update complete: {stats}")
        return 0
    except Exception as e:
        print(f"Error during flow execution: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
