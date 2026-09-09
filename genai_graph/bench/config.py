"""Configuration management for benchmarks in genai-graph.

Supports YAML-driven configuration profiles, pathspec document filtering,
dynamic parameter interpolation ({profile}), and backward-compatible aliases
like `saved_markdown_dir` (falling back to `onedrive_markdown_dir`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from genai_tk.config_mgmt.config_mngr import global_config
from loguru import logger
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

ALL_STEPS = ("fetch", "build", "run", "grade")


def load_env() -> None:
    """Load ~/.env and ensure local hosts bypass any system proxy."""
    home_env = Path.home() / ".env"
    if home_env.exists():
        load_dotenv(home_env, override=False)

    loopback_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        existing = {h.strip() for h in current.split(",") if h.strip()}
        os.environ[key] = ",".join(sorted(existing | loopback_hosts))


class BenchConfig(BaseModel):
    """Configuration model for a benchmark execution profile."""

    profile_name: str = "default"
    adapter: str | None = None  # e.g. "financebench", "officeqa", "mmlongbench"
    description: str = ""
    markdownize_profile: str = "medium"
    pdfs_dir: str = "data/pdfs"
    markdown_dir: str = "data/markdown_multi"
    kg_db: str = "data/kg/bench.db"
    saved_markdown_dir: str = "~/OneDrive/prj/bench/markdown"
    runs: str = "data/bench/{profile}/runs.jsonl"
    scores: str = "data/bench/{profile}/scores.jsonl"
    scores_summary: str = "data/bench/{profile}/scores_summary.json"
    agent_llm: str = "glm_5.2@openrouter"
    build_llm: str = "deepseek-v4-flash-0731@openrouter"
    judge_llm: str = "DeepSeek-V4-Pro-0813@openrouter"
    skip_ocr: bool = False
    build_force: bool = False
    build_llm_enabled: bool = True
    structure_strategy: str = "auto"
    generate_summaries: bool = True
    workers: int = 4
    summary_min_tokens: int = 800
    context_safety_ratio: float = 0.9
    embeddings: str | None = None
    fts: bool = True
    chunk_size_tokens: int = 1500
    pathspecs: list[str] = Field(default_factory=list)
    docs: list[str] = Field(default_factory=list)
    question_ids: list[str] = Field(default_factory=list)
    force_run: bool = False
    limit: int | None = None
    agent_profile: str = "default"
    folder_id: str | None = None
    judge_enabled: bool = True
    monitoring: str | list[str] | None = None
    question_concurrency: int = 10
    judge_concurrency: int = 5
    project_root: Path = Field(default_factory=Path.cwd)

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        """Expand '~', interpolate '{profile}', and resolve relative paths to project_root."""

        def _interpolate(p: str) -> str:
            return p.format(profile=self.profile_name)

        self.runs = _interpolate(self.runs)
        self.scores = _interpolate(self.scores)
        self.scores_summary = _interpolate(self.scores_summary)

        def _abs(p: str) -> str:
            path = Path(p).expanduser()
            return str(path if path.is_absolute() else (self.project_root / path).resolve())

        self.pdfs_dir = _abs(self.pdfs_dir)
        self.markdown_dir = _abs(self.markdown_dir)
        self.kg_db = _abs(self.kg_db)
        self.saved_markdown_dir = _abs(self.saved_markdown_dir)
        self.runs = _abs(self.runs)
        self.scores = _abs(self.scores)
        self.scores_summary = _abs(self.scores_summary)

    @property
    def onedrive_markdown_dir(self) -> str:
        """Backward-compatible alias for saved_markdown_dir."""
        return self.saved_markdown_dir

    def resolve_docs(
        self,
        available_docs: list[str],
        *,
        docs_override: list[str] | None = None,
        pathspecs_override: list[str] | None = None,
    ) -> list[str]:
        """Resolve target doc_names applying overrides and pathspecs."""
        from genai_graph.bench.adapters.base import match_docs_by_pathspecs

        if docs_override:
            return docs_override

        specs = pathspecs_override or self.pathspecs
        if specs:
            matched = match_docs_by_pathspecs(available_docs, specs)
            if matched:
                return matched
            logger.warning("No docs matched pathspecs: {}. Using configured docs or all docs.", specs)

        if self.docs:
            return self.docs

        return available_docs


def _find_bench_yaml(custom_path: Path | None = None, project_root: Path | None = None) -> Path:
    """Find the bench.yaml configuration file."""
    if custom_path and custom_path.exists():
        return custom_path

    root = project_root or Path.cwd()
    candidates = [
        root / "config" / "bench.yaml",
        root / "bench.yaml",
        Path.cwd() / "config" / "bench.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No bench.yaml configuration file found in {root} or {Path.cwd()}")


def load_raw_bench_yaml(config_path: Path | None = None) -> dict[str, Any]:
    """Read and OmegaConf-interpolate the bench YAML file."""
    path = _find_bench_yaml(config_path)
    loaded = OmegaConf.load(str(path))
    data_root = str(Path.cwd() / "data")
    paths_dict = {"data_root": data_root}
    try:
        cfg = global_config()
        if hasattr(cfg, "get"):
            c_data = cfg.get("paths.data_root", None)
            if c_data:
                paths_dict["data_root"] = str(c_data)
    except Exception:
        pass

    context = OmegaConf.create({"paths": paths_dict})
    merged = OmegaConf.merge(context, loaded)
    container = OmegaConf.to_container(merged, resolve=True)
    return container if isinstance(container, dict) else {}


def list_bench_profiles(config_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return all configured benchmark profiles."""
    try:
        raw = load_raw_bench_yaml(config_path)
        return raw.get("bench_profiles", {}) or {}
    except Exception as exc:
        logger.debug("Failed to read bench profiles: {}", exc)
        return {}


def load_bench_profile(
    profile_name: str | None = None,
    config_path: Path | None = None,
    project_root: Path | None = None,
    **overrides: Any,
) -> BenchConfig:
    """Load and instantiate a BenchConfig for a named profile."""
    root = project_root or Path.cwd()
    raw = load_raw_bench_yaml(config_path)

    active_name = profile_name or raw.get("default_profile", "default")
    profiles = raw.get("bench_profiles", {})
    if active_name not in profiles:
        available = list(profiles.keys())
        raise KeyError(f"Bench profile '{active_name}' not found in configuration. Available profiles: {available}")

    p_data = profiles[active_name] or {}
    paths = raw.get("paths", {}) or {}

    # Handle saved_markdown_dir with backward compat for onedrive_markdown_dir
    saved_md = (
        p_data.get("saved_markdown_dir")
        or paths.get("saved_markdown_dir")
        or p_data.get("onedrive_markdown_dir")
        or paths.get("onedrive_markdown_dir")
        or "~/OneDrive/prj/bench/markdown"
    )

    llms = p_data.get("llms", {}) or {}
    build_sec = p_data.get("build", {}) or {}
    agent_sec = p_data.get("agent", {}) or {}
    judge_sec = p_data.get("judge", {}) or {}
    files_sec = p_data.get("files", {}) or p_data.get("questions", {}) or {}

    cfg_kwargs: dict[str, Any] = {
        "profile_name": active_name,
        "adapter": p_data.get("adapter") or raw.get("adapter"),
        "description": p_data.get("description", ""),
        "markdownize_profile": p_data.get("markdownize_profile", "medium"),
        "pdfs_dir": str(p_data.get("pdfs_dir") or paths.get("pdfs_dir", "data/pdfs")),
        "markdown_dir": str(p_data.get("markdown_dir") or paths.get("markdown_dir", "data/markdown_multi")),
        "kg_db": str(p_data.get("kg_db") or paths.get("kg_db", "data/kg/bench.db")),
        "saved_markdown_dir": str(saved_md),
        "runs": str(p_data.get("runs") or paths.get("runs", "data/bench/{profile}/runs.jsonl")),
        "scores": str(p_data.get("scores") or paths.get("scores", "data/bench/{profile}/scores.jsonl")),
        "scores_summary": str(
            p_data.get("scores_summary") or paths.get("scores_summary", "data/bench/{profile}/scores_summary.json")
        ),
        "agent_llm": llms.get("agent", "glm_5.2@openrouter"),
        "build_llm": build_sec.get("llm") or llms.get("build", "deepseek-v4-flash-0731@openrouter"),
        "judge_llm": llms.get("judge", "DeepSeek-V4-Pro-0813@openrouter"),
        "skip_ocr": build_sec.get("skip_ocr", False),
        "build_force": build_sec.get("force", False),
        "build_llm_enabled": bool(build_sec.get("llm", True)),
        "structure_strategy": build_sec.get("structure_strategy", "auto"),
        "generate_summaries": build_sec.get("summaries", True),
        "workers": build_sec.get("workers", 4),
        "summary_min_tokens": build_sec.get("summary_min_tokens", 800),
        "context_safety_ratio": build_sec.get("context_safety_ratio", 0.9),
        "embeddings": build_sec.get("embeddings"),
        "fts": build_sec.get("fts", True),
        "chunk_size_tokens": build_sec.get("chunk_size_tokens", 1500),
        "pathspecs": files_sec.get("pathspecs", []),
        "docs": files_sec.get("docs", []),
        "limit": files_sec.get("limit"),
        "agent_profile": agent_sec.get("profile", "default"),
        "folder_id": agent_sec.get("folder_id"),
        "monitoring": p_data.get("monitoring"),
        "question_concurrency": agent_sec.get("concurrency", 10),
        "judge_enabled": judge_sec.get("enabled", True),
        "judge_concurrency": judge_sec.get("concurrency", 5),
        "project_root": root,
    }

    cfg_kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return BenchConfig(**cfg_kwargs)


def configure_bench_monitoring(monitoring: str | list[str] | None, project_name: str = "bench") -> None:
    """Configure tracing monitoring providers (langsmith, langfuse, local, otel)."""
    if not monitoring:
        return
    methods = [monitoring] if isinstance(monitoring, str) else list(monitoring)
    try:
        from genai_tk.core.monitoring import setup_monitoring

        setup_monitoring(methods, project_name=project_name)
    except Exception as exc:
        logger.warning("Failed to initialize bench monitoring ({}): {}", methods, exc)
