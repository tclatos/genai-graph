"""Unit tests for DocGraph Access Control and Security Trimming."""

from pathlib import Path

import pytest
import yaml

from genai_graph.kg.access.base import DefaultPublicAccessControlProvider
from genai_graph.kg.access.context import UserContext, set_active_user_context
from genai_graph.kg.access.factory import create_access_control_provider
from genai_graph.kg.access.yaml_provider import YamlAccessControlProvider
from genai_graph.kg.backend import KuzuBackend, create_in_memory_backend
from genai_graph.kg.document_graph.ingest import ingest_document_graph
from genai_graph.kg.factories.document_graph_factory import DocumentGraphFactory
from genai_graph.kg.query.document_graph_tools import (
    _get_authorized_markdown_hashes,
    _is_document_authorized,
    create_document_graph_tools,
    document_toc_yaml,
    list_documents,
)


@pytest.mark.unit
def test_user_context_creation_and_matching():
    # Anonymous / public user
    anon = UserContext.create()
    assert anon.user_id == "anonymous"
    assert anon.has_access(["public"])
    assert not anon.has_access(["group:finance"])
    assert not anon.has_access(["user:alice"])

    # HR User
    alice = UserContext.create(
        user_id="alice",
        groups=["hr_team", "all_employees"],
        roles=["viewer"],
    )
    assert alice.has_access(["public"])
    assert alice.has_access(["user:alice"])
    assert alice.has_access(["group:hr_team"])
    assert alice.has_access(["role:viewer"])
    assert not alice.has_access(["group:finance"])
    assert not alice.has_access(["role:cfo"])

    # Admin user
    admin = UserContext.create(user_id="root", is_admin=True)
    assert admin.has_access(["group:classified"])
    assert admin.has_access([])


@pytest.mark.unit
def test_yaml_access_control_provider(tmp_path: Path):
    rules_file = tmp_path / "rules.yaml"
    rules_data = {
        "default": ["public"],
        "rules": [
            {
                "path": "finance/**",
                "allowed_principals": ["group:finance", "role:cfo"],
                "inheritance": "override",
            },
            {
                "path": "hr/confidential/*.md",
                "allowed_principals": ["group:hr_exec", "user:alice"],
                "inheritance": "intersection",
            },
        ],
    }
    rules_file.write_text(yaml.safe_dump(rules_data), encoding="utf-8")

    provider = YamlAccessControlProvider(yaml_path=rules_file)

    # Public document
    res_pub = provider.get_document_acl_sync(Path("/docs/intro.md"), "intro.md")
    assert res_pub.allowed_principals == ["public"]

    # Finance document
    res_fin = provider.get_document_acl_sync(Path("/docs/finance/q3/report.md"), "finance/q3/report.md")
    assert "group:finance" in res_fin.allowed_principals
    assert res_fin.inheritance == "override"

    # HR confidential
    res_hr = provider.get_document_acl_sync(Path("/docs/hr/confidential/payroll.md"), "hr/confidential/payroll.md")
    assert "user:alice" in res_hr.allowed_principals
    assert res_hr.inheritance == "intersection"


@pytest.mark.unit
def test_access_control_factory():
    default_p = create_access_control_provider("default")
    assert isinstance(default_p, DefaultPublicAccessControlProvider)

    yaml_p = create_access_control_provider("yaml", {"rules": [{"path": "*.md", "allowed_principals": ["public"]}]})
    assert isinstance(yaml_p, YamlAccessControlProvider)


@pytest.mark.unit
def test_document_graph_security_trimming_e2e(tmp_path: Path):
    # Setup test documents:
    # 1. public/doc_public.md
    # 2. finance/doc_finance.md
    # 3. executive/doc_exec.md
    source_dir = tmp_path / "source"
    pub_dir = source_dir / "public"
    fin_dir = source_dir / "finance"
    exec_dir = source_dir / "executive"

    pub_dir.mkdir(parents=True)
    fin_dir.mkdir(parents=True)
    exec_dir.mkdir(parents=True)

    (pub_dir / "doc_public.md").write_text("# Public Overview\nWelcome to public info.", encoding="utf-8")
    (fin_dir / "doc_finance.md").write_text("# Finance 2026\nConfidential revenue numbers.", encoding="utf-8")
    (exec_dir / "doc_exec.md").write_text("# Executive Board\nTop secret strategic plans.", encoding="utf-8")

    rules = [
        {"path": "public/**", "allowed_principals": ["public"], "inheritance": "override"},
        {"path": "finance/**", "allowed_principals": ["group:finance"], "inheritance": "override"},
        {"path": "executive/**", "allowed_principals": ["group:board"], "inheritance": "override"},
    ]

    acl_provider = YamlAccessControlProvider(rules=rules)
    factory = DocumentGraphFactory(
        sources=[str(source_dir)],
        access_control_provider=acl_provider,
        inheritance_mode="override",
    )

    db_path = tmp_path / "test_acl.db"
    backend = KuzuBackend()
    backend.connect(str(db_path))

    ingest_document_graph(backend, factory)

    # 1. Verify Public User
    public_ctx = UserContext.create()
    docs = list_documents(backend, user_context=public_ctx)
    doc_names = [d["filename"] for d in docs]
    assert doc_names == ["doc_public.md"]

    # 2. Verify Finance User
    finance_ctx = UserContext.create(user_id="bob", groups=["finance"])
    fin_docs = list_documents(backend, user_context=finance_ctx)
    fin_names = sorted(d["filename"] for d in fin_docs)
    assert fin_names == ["doc_finance.md", "doc_public.md"]

    # 3. Verify Executive User
    exec_ctx = UserContext.create(user_id="ceo", groups=["board"])
    exec_docs = list_documents(backend, user_context=exec_ctx)
    exec_names = sorted(d["filename"] for d in exec_docs)
    assert exec_names == ["doc_exec.md", "doc_public.md"]

    # 4. Direct Document TOC access control
    # Attempting to fetch executive doc TOC as public user
    set_active_user_context(public_ctx)
    try:
        toc_res = document_toc_yaml(backend, "doc_exec.md", user_context=public_ctx)
        assert "Access Denied" in toc_res

        # Authorized TOC fetch
        toc_ceo = document_toc_yaml(backend, "doc_exec.md", user_context=exec_ctx)
        assert "Executive Board" in toc_ceo
    finally:
        set_active_user_context(None)

    # 5. Tool suite verification with trimming feedback
    tools = {t.name: t for t in create_document_graph_tools(str(db_path), trimming_feedback="aggregate_notice")}

    set_active_user_context(public_ctx)
    try:
        list_tool_res = tools["list_documents"].invoke({})
        assert "doc_public.md" in list_tool_res
        assert "omitted due to access permissions" in list_tool_res

        # Search sections test
        search_res = tools["search_sections"].invoke({"query": "revenue"})
        # Public user cannot see finance revenue
        assert "Finance 2026" not in search_res
    finally:
        set_active_user_context(None)

    set_active_user_context(finance_ctx)
    try:
        search_res_fin = tools["search_sections"].invoke({"query": "revenue", "mode": "cypher"})
        assert "Finance 2026" in search_res_fin
    finally:
        set_active_user_context(None)


@pytest.mark.unit
def test_missing_allowed_principals_column_fails_open_by_default(monkeypatch: pytest.MonkeyPatch):
    """A pre-ACL database (no allowed_principals column) is unfiltered by default."""
    monkeypatch.delenv("GENAI_GRAPH_ACL_REQUIRED", raising=False)
    backend = create_in_memory_backend()
    backend.execute(
        "CREATE NODE TABLE Document(content_hash STRING PRIMARY KEY, markdown_hash STRING, filename STRING)"
    )
    backend.execute("CREATE (:Document {content_hash: 'h1', markdown_hash: 'mh1', filename: 'doc.md'})")

    ctx = UserContext.create(user_id="bob")
    assert _get_authorized_markdown_hashes(backend, ctx) is None
    assert _is_document_authorized(backend, "doc.md", ctx) is True


@pytest.mark.unit
def test_missing_allowed_principals_column_fails_closed_when_required(monkeypatch: pytest.MonkeyPatch):
    """Setting GENAI_GRAPH_ACL_REQUIRED=1 denies access when the ACL column is absent."""
    monkeypatch.setenv("GENAI_GRAPH_ACL_REQUIRED", "1")
    backend = create_in_memory_backend()
    backend.execute(
        "CREATE NODE TABLE Document(content_hash STRING PRIMARY KEY, markdown_hash STRING, filename STRING)"
    )
    backend.execute("CREATE (:Document {content_hash: 'h1', markdown_hash: 'mh1', filename: 'doc.md'})")

    ctx = UserContext.create(user_id="bob")
    assert _get_authorized_markdown_hashes(backend, ctx) == set()
    assert _is_document_authorized(backend, "doc.md", ctx) is False

    # Admins always bypass, even when the column is missing and enforcement is required.
    admin_ctx = UserContext.create(user_id="root", is_admin=True)
    assert _get_authorized_markdown_hashes(backend, admin_ctx) is None
    assert _is_document_authorized(backend, "doc.md", admin_ctx) is True
