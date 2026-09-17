"""Configuration management for benchmarks and Document Graph profiles in genai-graph.

Separates Document Graph construction settings (config/docgraph.yaml) from benchmark
execution concerns (config/bench.yaml), with automatic LLM resolution from agent profiles
and clean Pydantic v2 data models.
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


# ---------------------------------------------------------------------------
# DocGraph Profile Models
# ---------------------------------------------------------------------------


class DocGraphPathsConfig(BaseModel):
    """File and directory paths for Document Graph construction."""

    sources_dir: str = "data/pdfs"
    markdown_dir: str = "data/markdown_multi"
    kg_db: str = "data/kg/bench.db"
    saved_markdown_dir: str = "~/OneDrive/prj/bench/markdown"

    model_config = {"arbitrary_types_allowed": True}


class DocGraphLlmsConfig(BaseModel):
    """LLM configurations for Document Graph construction."""

    summary: str = "deepseek-v4-flash-0731(none)@openrouter"
    image: str | None = None  # VLM model ID for image queries / descriptions

    model_config = {"arbitrary_types_allowed": True}


class DocGraphImagesConfig(BaseModel):
    """Multimodal image processing configuration for Document Graph."""

    enabled: bool = False
    describe_uncaptioned: bool = False
    min_size_bytes: int = 10240
    max_queries_per_turn: int = 3

    model_config = {"arbitrary_types_allowed": True}


class DocGraphBuildConfig(BaseModel):
    """Build and extraction settings for Document Graph construction."""

    structure_strategy: str = "auto"
    generate_summaries: bool = True
    summary_min_tokens: int = 800
    context_safety_ratio: float = 0.9
    fts: bool = True
    chunk_size_tokens: int = 1500
    workers: int = 4
    skip_ocr: bool = False
    force: bool = False

    model_config = {"arbitrary_types_allowed": True}


class DocGraphProfileConfig(BaseModel):
    """Complete profile definition for Document Graph creation."""

    profile_name: str = "default"
    description: str = ""
    markdownize_profile: str = "best"
    paths: DocGraphPathsConfig = Field(default_factory=DocGraphPathsConfig)
    llms: DocGraphLlmsConfig = Field(default_factory=DocGraphLlmsConfig)
    images: DocGraphImagesConfig = Field(default_factory=DocGraphImagesConfig)
    build: DocGraphBuildConfig = Field(default_factory=DocGraphBuildConfig)
    project_root: Path = Field(default_factory=Path.cwd)

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        """Resolve all relative paths against project_root and expand '~'."""

        def _abs(p: str) -> str:
            path = Path(p).expanduser()
            return str(path if path.is_absolute() else (self.project_root / path).resolve())

        self.paths.sources_dir = _abs(self.paths.sources_dir)
        self.paths.markdown_dir = _abs(self.paths.markdown_dir)
        self.paths.kg_db = _abs(self.paths.kg_db)
        self.paths.saved_markdown_dir = _abs(self.paths.saved_markdown_dir)


# ---------------------------------------------------------------------------
# Benchmark Profile Models
# ---------------------------------------------------------------------------


class GraderConfig(BaseModel):
    """LLM-as-judge evaluation configuration."""

    enabled: bool = True
    llm: str = "DeepSeek-V4-Pro-0813@openrouter"
    concurrency: int = 5

    model_config = {"arbitrary_types_allowed": True}


class RunnerConfig(BaseModel):
    """Question execution runtime configuration."""

    concurrency: int = 10
    folder_id: str | None = None
    monitoring: str | list[str] | None = None

    model_config = {"arbitrary_types_allowed": True}


class BenchFilesConfig(BaseModel):
    """Document filtering and slicing options."""

    pathspecs: list[str] = Field(default_factory=list)
    docs: list[str] = Field(default_factory=list)
    limit: int | None = None

    model_config = {"arbitrary_types_allowed": True}


class BenchConfig(BaseModel):
    """Configuration model for a benchmark execution profile."""

    profile_name: str = "default"
    dataset_adapter: str = ""
    description: str = ""
    docgraph_profile: str = "default"
    docgraph: DocGraphProfileConfig = Field(default_factory=DocGraphProfileConfig)
    agent_profile: str = "default"
    agent_llm: str = "default"
    grader: GraderConfig = Field(default_factory=GraderConfig)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    files: BenchFilesConfig = Field(default_factory=BenchFilesConfig)
    question_ids: list[str] = Field(default_factory=list)
    force_run: bool = False
    runs: str = "data/bench/{profile}/runs.jsonl"
    scores: str = "data/bench/{profile}/scores.jsonl"
    scores_summary: str = "data/bench/{profile}/scores_summary.json"
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

        self.runs = _abs(self.runs)
        self.scores = _abs(self.scores)
        self.scores_summary = _abs(self.scores_summary)

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

        specs = pathspecs_override or self.files.pathspecs
        if specs:
            matched = match_docs_by_pathspecs(available_docs, specs)
            if matched:
                return matched
            logger.warning("No docs matched pathspecs: {}. Using configured docs or all docs.", specs)

        if self.files.docs:
            return self.files.docs

        return available_docs


# ---------------------------------------------------------------------------
# YAML Loading & Profile Resolution
# ---------------------------------------------------------------------------


def _find_config_file(filename: str, custom_path: Path | None = None, project_root: Path | None = None) -> Path:
    """Find a configuration file by name in standard locations."""
    if custom_path and custom_path.exists():
        return custom_path

    root = project_root or Path.cwd()
    candidates = [
        root / "config" / filename,
        root / filename,
        Path.cwd() / "config" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No {filename} configuration file found in {root} or {Path.cwd()}")


def _load_interpolated_yaml(path: Path) -> dict[str, Any]:
    """Read and OmegaConf-interpolate a YAML file with paths.data_root context."""
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


def load_raw_docgraph_yaml(
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Read and interpolate config/docgraph.yaml."""
    path = _find_config_file("docgraph.yaml", config_path, project_root)
    return _load_interpolated_yaml(path)


def list_docgraph_profiles(
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Return all configured DocGraph profiles."""
    try:
        raw = load_raw_docgraph_yaml(config_path, project_root)
        return raw.get("docgraph_profiles", {}) or {}
    except Exception as exc:
        logger.debug("Failed to read docgraph profiles: {}", exc)
        return {}


def load_docgraph_profile(
    profile_name: str | None = None,
    config_path: Path | None = None,
    project_root: Path | None = None,
    **overrides: Any,
) -> DocGraphProfileConfig:
    """Load and instantiate a DocGraphProfileConfig."""
    root = project_root or Path.cwd()
    raw = load_raw_docgraph_yaml(config_path, project_root=root)

    active_name = profile_name or raw.get("default_profile", "default")
    profiles = raw.get("docgraph_profiles", {})
    if active_name not in profiles:
        available = list(profiles.keys())
        raise KeyError(f"DocGraph profile '{active_name}' not found in configuration. Available profiles: {available}")

    p_data = profiles[active_name] or {}
    paths_raw = p_data.get("paths", {}) or {}
    llms_raw = p_data.get("llms", {}) or {}
    images_raw = p_data.get("images", {}) or {}
    build_raw = p_data.get("build", {}) or {}

    docgraph_cfg = DocGraphProfileConfig(
        profile_name=active_name,
        description=p_data.get("description", ""),
        markdownize_profile=p_data.get("markdownize_profile", "best"),
        paths=DocGraphPathsConfig(
            sources_dir=str(paths_raw.get("sources_dir") or paths_raw.get("pdfs_dir", "data/pdfs")),
            markdown_dir=str(paths_raw.get("markdown_dir", "data/markdown_multi")),
            kg_db=str(paths_raw.get("kg_db", "data/kg/bench.db")),
            saved_markdown_dir=str(paths_raw.get("saved_markdown_dir", "~/OneDrive/prj/bench/markdown")),
        ),
        llms=DocGraphLlmsConfig(
            summary=str(llms_raw.get("summary", "deepseek-v4-flash-0731@openrouter")),
            image=llms_raw.get("image"),
        ),
        images=DocGraphImagesConfig(
            enabled=bool(images_raw.get("enabled", False)),
            describe_uncaptioned=bool(images_raw.get("describe_uncaptioned", False)),
            min_size_bytes=int(images_raw.get("min_size_bytes", 10240)),
            max_queries_per_turn=int(images_raw.get("max_queries_per_turn", 3)),
        ),
        build=DocGraphBuildConfig(
            structure_strategy=str(build_raw.get("structure_strategy", "auto")),
            generate_summaries=bool(build_raw.get("generate_summaries", True)),
            summary_min_tokens=int(build_raw.get("summary_min_tokens", 800)),
            context_safety_ratio=float(build_raw.get("context_safety_ratio", 0.9)),
            fts=bool(build_raw.get("fts", True)),
            chunk_size_tokens=int(build_raw.get("chunk_size_tokens", 1500)),
            workers=int(build_raw.get("workers", 4)),
            skip_ocr=bool(build_raw.get("skip_ocr", False)),
            force=bool(build_raw.get("force", False)),
        ),
        project_root=root,
    )

    for k, v in overrides.items():
        if v is not None and hasattr(docgraph_cfg, k):
            setattr(docgraph_cfg, k, v)

    return docgraph_cfg


def load_raw_bench_yaml(
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Read and interpolate config/bench.yaml."""
    path = _find_config_file("bench.yaml", config_path, project_root)
    return _load_interpolated_yaml(path)


def list_bench_profiles(
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Return all configured benchmark profiles."""
    try:
        raw = load_raw_bench_yaml(config_path, project_root)
        return raw.get("bench_profiles", {}) or {}
    except Exception as exc:
        logger.debug("Failed to read bench profiles: {}", exc)
        return {}


def _resolve_agent_llm(agent_profile_name: str, project_root: Path) -> str:
    """Resolve the agent LLM model identifier directly from config/agents.yaml."""
    try:
        from genai_tk.agents.harness.profiles import load_langchain_profiles

        profiles = load_langchain_profiles(str(project_root / "config" / "agents.yaml"))
        if agent_profile_name in profiles:
            return profiles[agent_profile_name].llm
    except Exception as exc:
        logger.debug("Could not resolve agent LLM for '{}': {}", agent_profile_name, exc)
    return "default"


def load_bench_profile(
    profile_name: str | None = None,
    config_path: Path | None = None,
    docgraph_config_path: Path | None = None,
    project_root: Path | None = None,
    **overrides: Any,
) -> BenchConfig:
    """Load and instantiate a BenchConfig for a named benchmark profile.

    Merges the benchmark profile with its referenced DocGraph profile from docgraph.yaml
    and resolves the agent LLM from agents.yaml.
    """
    root = project_root or Path.cwd()
    raw = load_raw_bench_yaml(config_path, project_root=root)

    active_name = profile_name or raw.get("default_profile", "default")
    profiles = raw.get("bench_profiles", {})
    if active_name not in profiles:
        available = list(profiles.keys())
        raise KeyError(f"Bench profile '{active_name}' not found in configuration. Available profiles: {available}")

    p_data = profiles[active_name] or {}
    paths = raw.get("paths", {}) or {}

    docgraph_prof_name = p_data.get("docgraph_profile", "default")
    docgraph_cfg = load_docgraph_profile(
        profile_name=docgraph_prof_name,
        config_path=docgraph_config_path,
        project_root=root,
    )

    agent_prof_name = p_data.get("agent_profile") or p_data.get("agent", {}).get("profile", "default")
    agent_llm = p_data.get("agent_llm") or _resolve_agent_llm(agent_prof_name, root)

    grader_sec = p_data.get("grader", {}) or {}
    runner_sec = p_data.get("runner", {}) or {}
    files_sec = p_data.get("files", {}) or {}

    bench_cfg = BenchConfig(
        profile_name=active_name,
        dataset_adapter=str(p_data.get("dataset_adapter") or raw.get("dataset_adapter", "")),
        description=p_data.get("description", ""),
        docgraph_profile=docgraph_prof_name,
        docgraph=docgraph_cfg,
        agent_profile=agent_prof_name,
        agent_llm=agent_llm,
        grader=GraderConfig(
            enabled=bool(grader_sec.get("enabled", True)),
            llm=str(grader_sec.get("llm", "DeepSeek-V4-Pro-0813@openrouter")),
            concurrency=int(grader_sec.get("concurrency", 5)),
        ),
        runner=RunnerConfig(
            concurrency=int(runner_sec.get("concurrency", 10)),
            folder_id=runner_sec.get("folder_id"),
            monitoring=runner_sec.get("monitoring"),
        ),
        files=BenchFilesConfig(
            pathspecs=list(files_sec.get("pathspecs", [])),
            docs=list(files_sec.get("docs", [])),
            limit=files_sec.get("limit"),
        ),
        runs=str(p_data.get("runs") or paths.get("runs", "data/bench/{profile}/runs.jsonl")),
        scores=str(p_data.get("scores") or paths.get("scores", "data/bench/{profile}/scores.jsonl")),
        scores_summary=str(
            p_data.get("scores_summary") or paths.get("scores_summary", "data/bench/{profile}/scores_summary.json")
        ),
        project_root=root,
    )

    # Apply any runtime overrides
    for k, v in overrides.items():
        if v is not None:
            if k == "limit":
                bench_cfg.files.limit = v
            elif k == "docs":
                bench_cfg.files.docs = v if isinstance(v, list) else [v]
            elif k == "pathspecs":
                bench_cfg.files.pathspecs = v if isinstance(v, list) else [v]
            elif k == "force_run":
                bench_cfg.force_run = bool(v)
            elif k == "question_ids":
                bench_cfg.question_ids = v if isinstance(v, list) else [v]
            elif hasattr(bench_cfg, k):
                setattr(bench_cfg, k, v)

    return bench_cfg


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
