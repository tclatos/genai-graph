"""Handler for computing and caching embeddings for graph node fields.

Provides a centralized interface for generating embeddings using EmbeddingsFactory
with built-in caching support for efficiency during KG builds.
"""

from typing import Any

from genai_tk.core.factories.embeddings_factory import EmbeddingsFactory, get_embeddings
from loguru import logger


class EmbeddingsHandler:
    """Compute embeddings for graph node fields using EmbeddingsFactory.

    This handler manages the computation of embeddings for fields marked in
    `index_fields` of GraphNode configurations, leveraging EmbeddingsFactory's
    built-in caching to avoid redundant computations across KG builds.

    Attributes:
        embeddings_id: Identifier of the embeddings model to use
        factory: EmbeddingsFactory instance for computing embeddings
    """

    def __init__(self, embeddings_id: str | None = None, embeddings_tag: str | None = None) -> None:
        """Initialize handler with embeddings model.

        Args:
            embeddings_id: Model identifier (e.g., 'qwen3_06b@openrouter')
            embeddings_tag: Legacy model tag from config (deprecated, use embeddings_id)

        Raises:
            ValueError: If neither embeddings_id nor embeddings_tag is provided
        """
        if not embeddings_id and not embeddings_tag:
            msg = "Either embeddings_id or embeddings_tag must be provided"
            raise ValueError(msg)

        # Resolve to embeddings_id using EmbeddingsFactory
        if embeddings_id:
            self.embeddings_id = embeddings_id
        else:
            # Legacy: resolve tag to ID
            resolved = EmbeddingsFactory.resolve_embeddings_identifier(embeddings_tag)
            self.embeddings_id = resolved

        try:
            self.factory = get_embeddings(embeddings=self.embeddings_id, cache_embeddings=True)
            logger.debug("EmbeddingsHandler initialized with model: {}", self.embeddings_id)
        except Exception as e:
            logger.error("Failed to initialize EmbeddingsFactory for {}: {}", self.embeddings_id, e)
            raise

    def compute_embeddings(self, text: str) -> list[float]:
        """Compute embedding for a text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            ValueError: If text is empty or None
        """
        if not text or not isinstance(text, str):
            msg = f"Expected non-empty string, got: {type(text).__name__}"
            raise ValueError(msg)

        try:
            # embed_query returns embedding directly as list[float]
            embeddings = self.factory.embed_query(text)
            logger.debug("Computed embedding for text ({} chars) -> {} dims", len(text), len(embeddings))
            return embeddings
        except Exception as e:
            logger.error(f"Failed to compute embedding for text: {e}")
            raise

    def compute_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for a batch of text strings.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (list of floats)
        """
        if not texts:
            return []
        try:
            embeddings = self.factory.embed_documents(texts)
            logger.debug("Computed batch embeddings for {} texts", len(texts))
            return embeddings
        except Exception as e:
            logger.error(f"Failed to compute batch embeddings for {len(texts)} texts: {e}")
            raise

    def compute_field_embeddings(self, node_data: dict[str, Any], index_fields: list[str]) -> dict[str, list[float]]:
        """Compute embeddings for specified fields in node data.

        Args:
            node_data: Node data dictionary with field values
            index_fields: List of field names to compute embeddings for

        Returns:
            Dictionary mapping field names to embedding vectors
            Fields with missing or non-string values are skipped
        """
        embeddings = {}

        for field_name in index_fields:
            if field_name not in node_data:
                logger.debug(f"Field '{field_name}' not found in node data, skipping")
                continue

            field_value = node_data[field_name]

            # Skip empty or non-string values
            if not field_value or not isinstance(field_value, str):
                logger.debug(f"Field '{field_name}' is empty or non-string, skipping")
                continue

            try:
                embedding = self.compute_embeddings(field_value)
                embeddings[field_name] = embedding
                logger.debug(f"Computed embedding for field '{field_name}'")
            except Exception as e:
                logger.warning(f"Failed to compute embedding for field '{field_name}': {e}")
                continue

        return embeddings
