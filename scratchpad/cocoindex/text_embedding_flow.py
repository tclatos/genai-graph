"""
Text Embedding Flow Module.

This module provides a reusable text embedding flow definition for CocoIndex
that can be used by CLI tools and other applications.
"""

from typing import Any

import cocoindex
import numpy as np
from numpy.typing import NDArray

# Module-level configuration for the flow
_flow_config: dict[str, Any] = {
    "source": {"path": ".", "included_patterns": ["*.md"], "excluded_patterns": [".venv/**", ".git/**"]},
    "chunking": {"chunk_size": 2000, "chunk_overlap": 500, "language": "markdown"},
    "embedding": {"api_type": "openai", "model": "text-embedding-3-small"},
}


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
    # Get embedding config from module-level config
    embedding_config = _flow_config.get("embedding", {})
    api_type = embedding_config.get("api_type", "openai")
    model = embedding_config.get("model", "text-embedding-3-small")
    api_type_enum = cocoindex.LlmApiType.OPENAI if api_type.lower() == "openai" else cocoindex.LlmApiType.OPENAI

    return text.transform(
        cocoindex.functions.EmbedText(
            api_type=api_type_enum,
            model=model,
        )
    )


# Define the flow at module level as a singleton
@cocoindex.flow_def(name="TextEmbedding")
def text_embedding_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope) -> None:
    """
    Define a flow that embeds text into a vector database.
    Uses configuration from module-level _flow_config.
    """
    # Extract configuration with defaults
    source_config = _flow_config.get("source", {})
    source_path = source_config.get("path", ".")
    included_patterns = source_config.get("included_patterns", ["*.md"])
    excluded_patterns = source_config.get("excluded_patterns", [".venv/**", ".git/**"])

    chunking_config = _flow_config.get("chunking", {})
    chunk_size = chunking_config.get("chunk_size", 2000)
    chunk_overlap = chunking_config.get("chunk_overlap", 500)
    language = chunking_config.get("language", "markdown")

    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path=source_path,
            included_patterns=included_patterns,
            excluded_patterns=excluded_patterns,
        ),
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


def configure_and_get_flow(config: dict[str, Any]):
    """
    Configure and get the text embedding flow.

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
    # Update the module-level flow configuration
    global _flow_config
    _flow_config = config

    # Return the flow
    return text_embedding_flow
