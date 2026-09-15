"""Integration tests for document language detection and multi-language Ladybug FTS."""

from __future__ import annotations

from pathlib import Path

import pytest

from genai_graph.kg.backend import KuzuBackend
from genai_graph.kg.document_graph.ingest import ingest_document_graph
from genai_graph.kg.document_graph.retrieval import RetrievalConfig
from genai_graph.kg.factories.document_graph_factory import DocumentGraphFactory
from genai_graph.kg.query.document_graph_tools import search_sections

DOC_EN = """# Financial Overview

This is an English financial report detailing the annual revenue, operating expenses, and cash flow projections.

## Quarterly Earnings

The company achieved record operating margins during the fourth quarter across all European and North American segments.

## Future Projections

Long-term guidance remains strong with expected growth in enterprise software services.
"""

DOC_FR = """# Rapport Financier Annuel

Ce document est un rapport financier officiel rédigé en français présentant les résultats comptables et les bénéfices.

## Analyse des Revenus

Le chiffre d'affaires annuel a augmenté significativement grâce à nos activités internationales et technologiques.

## Perspectives Stratégiques

L'entreprise prévoit d'investir massivement dans l'intelligence artificielle et l'automatisation des processus.
"""


@pytest.fixture
def multilingual_corpus(tmp_path: Path) -> Path:
    """Create a multilingual test corpus with English and French documents."""
    (tmp_path / "financial_en.md").write_text(DOC_EN, encoding="utf-8")
    (tmp_path / "rapport_fr.md").write_text(DOC_FR, encoding="utf-8")
    return tmp_path


@pytest.mark.integration
class TestDocumentLanguageAndFts:
    def test_language_detection_and_fts_stopwords(self, graph_backend: KuzuBackend, multilingual_corpus: Path) -> None:
        factory = DocumentGraphFactory(sources=[str(multilingual_corpus)])
        cfg = RetrievalConfig(fts=True)

        result = ingest_document_graph(graph_backend, factory, retrieval_config=cfg)

        assert result.documents_processed == 2
        assert result.documents_failed == 0
        assert result.fts_index == "section_fts"

        # 1. Verify language field on Document nodes
        df_docs = graph_backend.execute_get_as_df(
            "MATCH (d:Document) RETURN d.filename AS filename, d.language AS lang ORDER BY filename",
            union=False,
        )
        doc_map = dict(zip(df_docs["filename"], df_docs["lang"], strict=False))
        assert doc_map["financial_en.md"] == "en"
        assert doc_map["rapport_fr.md"] == "fr"

        # 2. Verify _FtsStopWords node table was populated with union of en and fr stop words
        df_stops = graph_backend.execute_get_as_df(
            "MATCH (s:_FtsStopWords) RETURN count(s) AS total",
            union=False,
        )
        total_stopwords = int(df_stops["total"].iloc[0])
        assert total_stopwords > 100

        # Check specific English and French stop words in DB
        df_sample = graph_backend.execute_get_as_df(
            "MATCH (s:_FtsStopWords) WHERE s.word IN ['the', 'and', 'le', 'la', 'dans', 'des'] RETURN s.word AS word",
            union=False,
        )
        sample_words = set(df_sample["word"])
        assert "the" in sample_words
        assert "le" in sample_words or "la" in sample_words or "dans" in sample_words

        # 3. Test FTS / BM25 search over sections
        fr_hits = search_sections(graph_backend, query="bénéfices comptables", mode="bm25")
        assert len(fr_hits) > 0
        assert any(
            "Rapport Financier" in (h.get("title") or "") or "bénéfices" in (h.get("snippet") or "") for h in fr_hits
        )

        en_hits = search_sections(graph_backend, query="operating margins", mode="bm25")
        assert len(en_hits) > 0
        assert any("Financial" in (h.get("title") or "") or "Earnings" in (h.get("title") or "") for h in en_hits)
