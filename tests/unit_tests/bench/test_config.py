"""Unit tests for benchmark and docgraph configuration management."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from genai_graph.bench.config import (
    BenchConfig,
    DocGraphProfileConfig,
    load_bench_profile,
    load_docgraph_profile,
)


@pytest.fixture
def sample_config_dir(tmp_path: Path) -> Path:
    """Create a temporary project folder with config/docgraph.yaml, config/bench.yaml, and config/agents.yaml."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    docgraph_yaml = textwrap.dedent("""\
        default_profile: default

        docgraph_profiles:
          default:
            description: "Test DocGraph Profile"
            markdownize_profile: best
            paths:
              sources_dir: data/pdfs
              markdown_dir: data/markdown_multi
              kg_db: data/kg/test.db
              saved_markdown_dir: ~/OneDrive/prj/test/markdown
            llms:
              summary: deepseek-v4-flash-0731@openrouter
              image: gemini-2.5-flash@openrouter
            images:
              enabled: true
              describe_uncaptioned: true
              min_size_bytes: 10240
              max_queries_per_turn: 3
            build:
              structure_strategy: auto
              generate_summaries: true
              summary_min_tokens: 800
              context_safety_ratio: 0.9
              fts: true
              chunk_size_tokens: 1500
              workers: 4
              skip_ocr: false
              force: false
    """)
    (cfg_dir / "docgraph.yaml").write_text(docgraph_yaml, encoding="utf-8")

    bench_yaml = textwrap.dedent("""\
        default_profile: default
        dataset_adapter: test.adapter.TestAdapter

        paths:
          runs: data/test/{profile}/runs.jsonl
          scores: data/test/{profile}/scores.jsonl
          scores_summary: data/test/{profile}/scores_summary.json

        bench_profiles:
          default:
            description: "Test Bench Profile"
            docgraph_profile: default
            agent_profile: test_agent
            files:
              pathspecs:
                - "test_doc_*"
              docs: []
              limit: 5
            runner:
              concurrency: 8
              folder_id: null
              monitoring: null
            grader:
              enabled: true
              llm: DeepSeek-V4-Pro-0813@openrouter
              concurrency: 4
    """)
    (cfg_dir / "bench.yaml").write_text(bench_yaml, encoding="utf-8")

    agents_yaml = textwrap.dedent("""\
        agents:
          test_agent:
            harness: langchain
            name: "Test Agent"
            type: deep
            llm: glm_5.3_flash@openrouter
            description: "Test deep agent profile"
    """)
    (cfg_dir / "agents.yaml").write_text(agents_yaml, encoding="utf-8")

    return tmp_path


def test_load_docgraph_profile(sample_config_dir: Path) -> None:
    """Test loading a dedicated docgraph.yaml configuration."""
    dg_cfg = load_docgraph_profile(
        profile_name="default",
        project_root=sample_config_dir,
    )
    assert isinstance(dg_cfg, DocGraphProfileConfig)
    assert dg_cfg.profile_name == "default"
    assert dg_cfg.markdownize_profile == "best"
    assert dg_cfg.llms.summary == "deepseek-v4-flash-0731@openrouter"
    assert dg_cfg.llms.image == "gemini-2.5-flash@openrouter"
    assert dg_cfg.images.enabled is True
    assert dg_cfg.images.describe_uncaptioned is True
    assert dg_cfg.images.max_queries_per_turn == 3
    assert dg_cfg.build.workers == 4
    assert Path(dg_cfg.paths.kg_db).name == "test.db"


def test_load_bench_profile(sample_config_dir: Path) -> None:
    """Test loading a bench.yaml configuration that links to docgraph and agent profiles."""
    bench_cfg = load_bench_profile(
        profile_name="default",
        project_root=sample_config_dir,
    )
    assert isinstance(bench_cfg, BenchConfig)
    assert bench_cfg.profile_name == "default"
    assert bench_cfg.dataset_adapter == "test.adapter.TestAdapter"
    assert bench_cfg.docgraph_profile == "default"
    assert bench_cfg.docgraph.llms.summary == "deepseek-v4-flash-0731@openrouter"
    assert bench_cfg.agent_profile == "test_agent"
    assert bench_cfg.agent_llm == "glm_5.3_flash@openrouter"
    assert bench_cfg.grader.enabled is True
    assert bench_cfg.grader.llm == "DeepSeek-V4-Pro-0813@openrouter"
    assert bench_cfg.grader.concurrency == 4
    assert bench_cfg.runner.concurrency == 8
    assert bench_cfg.files.pathspecs == ["test_doc_*"]
    assert bench_cfg.files.limit == 5


def test_resolve_docs_with_pathspecs(sample_config_dir: Path) -> None:
    """Test pathspec matching when resolving target documents."""
    bench_cfg = load_bench_profile(project_root=sample_config_dir)
    available = ["test_doc_01.pdf", "test_doc_02.pdf", "other_doc.pdf"]
    matched = bench_cfg.resolve_docs(available)
    assert matched == ["test_doc_01.pdf", "test_doc_02.pdf"]


def test_bench_overrides(sample_config_dir: Path) -> None:
    """Test runtime CLI overrides on BenchConfig."""
    bench_cfg = load_bench_profile(
        project_root=sample_config_dir,
        limit=2,
        force_run=True,
    )
    assert bench_cfg.files.limit == 2
    assert bench_cfg.force_run is True
