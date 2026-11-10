"""Schema and table introspection helpers for Kùzu graph databases."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from genai_graph.kg.backend import KgBackend


def get_node_types(backend: KgBackend) -> list[str]:
    """Get all node table names (types) in the graph."""
    try:
        df = backend.execute_get_as_df("CALL show_tables() RETURN name, type", union=False)
        node_tables = df[df["type"] == "NODE"]["name"].tolist()
        return sorted(node_tables)
    except Exception as exc:
        logger.warning("Could not fetch node types: {}", exc)
        return []


def get_relationship_types(backend: KgBackend) -> list[str]:
    """Get all relationship table names in the graph."""
    try:
        df = backend.execute_get_as_df("CALL show_tables() RETURN name, type", union=False)
        rel_tables = df[df["type"] == "REL"]["name"].tolist()
        return sorted(rel_tables)
    except Exception as exc:
        logger.warning("Could not fetch relationship types: {}", exc)
        return []


def get_primary_key(backend: KgBackend, node_type: str) -> str:
    """Return the primary key property name for a node table in Kùzu."""
    if not node_type or node_type in ("(NA)", "NA"):
        return "id"
    try:
        df = backend.execute_get_as_df(f"CALL table_info('{node_type}') RETURN name, `primary key`", union=False)
        pk = df[df["primary key"].astype(bool)]["name"].tolist()
        return pk[0] if pk else "id"
    except Exception:
        # Fallback heuristic for common table schemas
        if node_type == "Opportunity":
            return "opportunity_id"
        if node_type == "Folder":
            return "folder_id"
        if node_type == "Document":
            return "content_hash"
        if node_type == "MarkdownSection":
            return "section_id"
        if node_type == "Requirement":
            return "requirement_id"
        return "id"


def get_top_node_types(backend: KgBackend, configured_default: str = "Opportunity") -> list[str]:
    """Get candidate node types that can act as top/root nodes in hierarchy visualization."""
    all_types = get_node_types(backend)
    preferred_roots = [
        configured_default,
        "Opportunity",
        "Folder",
        "Customer",
        "PortfolioPillar",
    ]
    ordered: list[str] = ["(NA)"]
    for pr in preferred_roots:
        if pr in all_types and pr not in ordered:
            ordered.append(pr)
    for t in all_types:
        if t not in ordered and t not in (
            "Image",
            "MarkdownSection",
            "SectionChunk",
            "NA",
            "(NA)",
        ):
            ordered.append(t)
    return ordered


def get_top_nodes_of_type(backend: KgBackend, node_type: str, limit: int = 250) -> list[dict[str, str]]:
    """Get all nodes of a top node type with display labels, names, and IDs."""
    if not node_type or node_type in ("(NA)", "NA"):
        return []

    pk = get_primary_key(backend, node_type)
    try:
        query = f"MATCH (n:{node_type}) RETURN n LIMIT {limit}"
        df = backend.execute_get_as_df(query, union=False)
        results = []
        for _, row in df.iterrows():
            d = row["n"]
            if not isinstance(d, dict):
                continue
            node_id = str(d.get(pk) or d.get("id") or d.get("name") or "")
            if not node_id:
                continue

            name = str(d.get("name") or d.get("title") or d.get("filename") or node_id)
            label = f"{name} ({node_id})" if name != node_id else name

            results.append({"id": node_id, "label": label, "name": name})
        return sorted(results, key=lambda x: x["label"])
    except Exception as exc:
        logger.warning("Could not fetch top nodes for {}: {}", node_type, exc)
        return []


def get_nodes_of_type(backend: KgBackend, node_type: str, limit: int = 250) -> list[str]:
    """Get list of node display names for a specific node type."""
    try:
        query = f"MATCH (n:{node_type}) RETURN n LIMIT {limit}"
        df = backend.execute_get_as_df(query, union=False)
        names = []
        for _, row in df.iterrows():
            d = row["n"]
            if not isinstance(d, dict):
                continue
            val = str(d.get("name") or d.get("title") or d.get("id") or "")
            if val and val not in names:
                names.append(val)
        return sorted(names)
    except Exception as exc:
        logger.warning("Could not fetch nodes for {}: {}", node_type, exc)
        return []


def get_opportunity_folders(backend: KgBackend) -> list[dict[str, str]]:
    """Get all ingested Opportunity anchors or Folder nodes with human-readable labels."""
    import re

    results: list[dict[str, str]] = []
    # 1. First check Opportunity anchor nodes
    try:
        df_opp = backend.execute_get_as_df(
            "MATCH (o:Opportunity) RETURN o.opportunity_id AS opportunity_id, o.name AS name",
            union=False,
        )
        if not df_opp.empty:
            for _, row in df_opp.iterrows():
                oid = str(row["opportunity_id"])
                name = str(row.get("name") or oid)
                results.append({"folder_id": oid, "label": f"🎯 Opportunity: {name}", "uri": ""})
    except Exception as exc:
        logger.debug("Could not fetch Opportunity nodes: {}", exc)

    # 2. Check Folder nodes
    try:
        df = backend.execute_get_as_df(
            "MATCH (f:Folder) RETURN f.folder_id AS folder_id, f.name AS name, f.uri AS uri",
            union=False,
        )
        if not df.empty:
            for _, row in df.iterrows():
                fid = str(row["folder_id"])
                uri = str(row.get("uri", ""))
                name = str(row.get("name", ""))

                m_ex = re.search(r"examples/([^/]+)/([^/]+)", uri)
                m_up = re.search(r"uploaded_rfq/([^/]+)", uri)
                if m_ex:
                    label = f"📁 {m_ex.group(1)} ({m_ex.group(2)})"
                elif m_up:
                    label = f"📁 {m_up.group(1)}"
                elif name:
                    label = f"📁 {name}"
                else:
                    label = f"📁 {fid}"

                results.append({"folder_id": fid, "label": label, "uri": uri})
    except Exception as exc:
        logger.warning("Could not fetch opportunity folders: {}", exc)

    return sorted(results, key=lambda x: x["label"])


def introspect_schema_data(backend: KgBackend, graph_name: str = "") -> dict[str, Any]:
    """Extract full schema in D3-compatible format directly from a live graph database.

    Args:
        backend: Connected KgBackend instance.
        graph_name: Optional graph identifier to include in metadata.

    Returns:
        Dictionary conforming to the D3 schema visualization format
        with keys 'meta', 'nodes', 'links', and 'vector_indexes'.
    """
    from datetime import UTC, datetime

    node_types = get_node_types(backend)
    rel_types = get_relationship_types(backend)

    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []

    for node_type in node_types:
        pk = get_primary_key(backend, node_type)
        fields: list[dict[str, Any]] = []
        try:
            info_df = backend.execute_get_as_df(f"CALL table_info('{node_type}') RETURN *", union=False)
            for _, f_row in info_df.iterrows():
                col_name = str(f_row["name"])
                if col_name.startswith("_") or col_name in ("_ID", "_SRC", "_DST"):
                    continue
                col_type = str(f_row.get("type", "STRING"))
                is_pk = bool(f_row.get("primary key", False)) or (col_name == pk)
                fields.append(
                    {
                        "name": col_name,
                        "type_human": col_type,
                        "type_kuzu": col_type,
                        "description": "",
                        "indexed": is_pk,
                        "embedded": False,
                    }
                )
        except Exception as exc:
            logger.debug("Failed to get table_info for node {}: {}", node_type, exc)

        nodes.append(
            {
                "id": node_type,
                "label": node_type,
                "description": "",
                "primary_key": pk,
                "name_from": pk,
                "index_fields": [pk] if pk else [],
                "fields": fields,
            }
        )

    for rel_type in rel_types:
        conn_df = None
        try:
            conn_df = backend.execute_get_as_df(f"CALL show_connection('{rel_type}') RETURN *", union=False)
        except Exception as exc:
            logger.debug("Failed to get show_connection for rel {}: {}", rel_type, exc)

        props: list[dict[str, Any]] = []
        try:
            info_df = backend.execute_get_as_df(f"CALL table_info('{rel_type}') RETURN *", union=False)
            for _, p_row in info_df.iterrows():
                prop_name = str(p_row["name"])
                if prop_name.startswith("_") or prop_name in ("_ID", "_SRC", "_DST"):
                    continue
                prop_type = str(p_row.get("type", "STRING"))
                props.append(
                    {
                        "name": prop_name,
                        "type_human": prop_type,
                        "type_kuzu": prop_type,
                        "description": "",
                    }
                )
        except Exception as exc:
            logger.debug("Failed to get table_info for rel {}: {}", rel_type, exc)

        if conn_df is not None and not conn_df.empty:
            for _, c_row in conn_df.iterrows():
                src = str(c_row.get("source table name", ""))
                dst = str(c_row.get("destination table name", ""))
                links.append(
                    {
                        "id": f"{src}::{rel_type}::{dst}",
                        "source": src,
                        "target": dst,
                        "label": rel_type,
                        "description": "",
                        "field_paths": [],
                        "properties": props,
                    }
                )

    vector_indexes: list[dict[str, Any]] = []
    try:
        idx_df = backend.execute_get_as_df("CALL show_indexes() RETURN *", union=False)
        if not idx_df.empty and "index_name" in idx_df.columns:
            for _, i_row in idx_df.iterrows():
                vector_indexes.append(
                    {
                        "table": str(i_row.get("table_name", "")),
                        "index_name": str(i_row.get("index_name", "")),
                        "embedding_column": "",
                        "source_field": str(i_row.get("property_names", "")),
                    }
                )
    except Exception as exc:
        logger.debug("Failed to get indexes: {}", exc)

    return {
        "meta": {
            "format": "genai_graph.resolved_schema",
            "format_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "graphs": [graph_name] if graph_name else [],
            "root_model": nodes[0]["id"] if nodes else None,
        },
        "nodes": nodes,
        "links": links,
        "vector_indexes": vector_indexes,
    }
