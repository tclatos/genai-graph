"""Cypher query composition for filtered graph visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from genai_graph.ui.graph_html.schema_inspect import get_primary_key

if TYPE_CHECKING:
    from genai_graph.kg.backend import KgBackend


def build_filtered_cypher_query(
    backend: KgBackend | None = None,
    node_types: list[str] | None = None,
    relationship_types: list[str] | None = None,
    limit: int = 500,
    node_name: str | None = None,
    folder_id: str | None = None,
    excluded_node_types: list[str] | None = None,
    top_node_type: str | None = None,
    top_node_id: str | None = None,
    max_hops: int = 6,
) -> tuple[str, dict[str, Any]]:
    """Build a parameterized Cypher query with relationship and exclusion filters.

    Supports:
    - Root-scoped multi-hop graph queries (anchor models)
    - Folder and document hierarchy queries
    - Node-type and single-node focal search
    - Node-type exclusion (e.g. Image, MarkdownSection)

    Returns:
        Tuple of (cypher_query_string, parameters_dict)
    """
    parameters: dict[str, Any] = {"limit": limit}

    # Normalize top_node_type and top_node_id ('(NA)' or 'NA' means no top-node filter / normal display)
    if top_node_type in ("(NA)", "NA"):
        top_node_type = None
    if top_node_id in ("(NA)", "NA"):
        top_node_id = None

    # Build relationship pattern
    if relationship_types:
        rel_pattern = "|".join(relationship_types)
        rel_clause = f"[r:{rel_pattern}]"
        multi_hop_rel = f"[r:{rel_pattern}*1..{max_hops}]"
    else:
        rel_clause = "[r]"
        multi_hop_rel = f"[r*1..{max_hops}]"

    # Build exclusion conditions
    exclusion_conditions = []
    if excluded_node_types:
        for exc in excluded_node_types:
            exclusion_conditions.append(f"label(n) <> '{exc}'")
            exclusion_conditions.append(f"label(m) <> '{exc}'")
    exclusion_filter = " AND ".join(exclusion_conditions)

    # Case 1: Specific node focus
    if node_types and len(node_types) == 1 and node_name:
        target_type = node_types[0]
        parameters["name"] = node_name
        half_limit = max(25, limit // 2)

        # Check by name or title
        prop_field = "title" if target_type == "MarkdownSection" else "name"
        if target_type == "Folder":
            prop_check = "n.name = $name OR n.folder_id = $name"
            prop_check_m = "m.name = $name OR m.folder_id = $name"
        else:
            prop_check = f"n.{prop_field} = $name"
            prop_check_m = f"m.{prop_field} = $name"

        if exclusion_filter:
            q1 = f"MATCH (n:{target_type})-{rel_clause}->(m) WHERE ({prop_check}) AND {exclusion_filter} RETURN n, r, m LIMIT {half_limit}"
            q2 = f"MATCH (n)-{rel_clause}->(m:{target_type}) WHERE ({prop_check_m}) AND {exclusion_filter} RETURN n, r, m LIMIT {half_limit}"
        else:
            q1 = f"MATCH (n:{target_type})-{rel_clause}->(m) WHERE ({prop_check}) RETURN n, r, m LIMIT {half_limit}"
            q2 = f"MATCH (n)-{rel_clause}->(m:{target_type}) WHERE ({prop_check_m}) RETURN n, r, m LIMIT {half_limit}"
        return f"{q1}; {q2}", parameters

    # Case 2A: Top Node selected: display all child connected nodes
    if top_node_id:
        target_top_type = top_node_type or "Opportunity"
        parameters["top_id"] = top_node_id
        parameters["fid"] = top_node_id

        pk = (
            get_primary_key(backend, target_top_type)
            if backend
            else (
                "opportunity_id"
                if target_top_type == "Opportunity"
                else "folder_id"
                if target_top_type == "Folder"
                else "id"
            )
        )

        where_conds = [f"(root.{pk} = $top_id OR root.name = $top_id)"]
        if excluded_node_types:
            for exc in excluded_node_types:
                where_conds.append(f"label(m) <> '{exc}'")

        where_sql = " AND ".join(where_conds)
        query = (
            f"MATCH (root:{target_top_type})-{multi_hop_rel}->(m) "
            f"WHERE {where_sql} RETURN root AS n, r, m LIMIT {limit}"
        )
        return query, parameters

    # Case 2B: Legacy Folder scoped query
    if folder_id:
        parameters["fid"] = folder_id
        queries = [
            "MATCH (o:Opportunity)-[r:HAS_RFQ]->(f:Folder) WHERE o.opportunity_id = $fid RETURN o AS n, r, f AS m",
            "MATCH (o:Opportunity)-[r:HAS_PROPOSAL]->(f:Folder) WHERE o.opportunity_id = $fid RETURN o AS n, r, f AS m",
            "MATCH (o:Opportunity)-[r:HUMAN_REF]->(f:Folder) WHERE o.opportunity_id = $fid RETURN o AS n, r, f AS m",
            "MATCH (o:Opportunity)-[r:HAS_ANALYSIS]->(a:RfqAnalysis) WHERE o.opportunity_id = $fid RETURN o AS n, r, a AS m",
            "MATCH (f:Folder)-[r:CONTAINS]->(d:Document) WHERE f.folder_id = $fid RETURN f AS n, r, d AS m",
            "MATCH (f:Folder)-[r:HAS_ANALYSIS]->(a:RfqAnalysis) WHERE f.folder_id = $fid RETURN f AS n, r, a AS m",
            "MATCH (d:Document)-[r:HAS_SECTION]->(s:MarkdownSection) WHERE d.folder_id = $fid RETURN d AS n, r, s AS m",
            "MATCH (d:Document)-[:HAS_SECTION]->(s:MarkdownSection)-[r:HAS_SUBSECTION]->(sub:MarkdownSection) WHERE d.folder_id = $fid RETURN s AS n, r, sub AS m",
        ]
        if relationship_types:
            queries = [q for q in queries if any(f"[r:{rt}]" in q for rt in relationship_types)]
            if not queries:
                queries = [
                    f"MATCH (n)-{rel_clause}->(m) WHERE (n.folder_id = $fid OR m.folder_id = $fid) RETURN n, r, m LIMIT {limit}"
                ]

        if excluded_node_types:
            fixed_queries = []
            for q in queries:
                if " RETURN " in q:
                    match_part, ret_part = q.split(" RETURN ", 1)
                    exc_conds = " AND ".join(f"label(m) <> '{exc}'" for exc in excluded_node_types)
                    sep = " AND " if " WHERE " in match_part else " WHERE "
                    fixed_queries.append(f"{match_part}{sep}{exc_conds} RETURN {ret_part}")
                else:
                    fixed_queries.append(q)
            queries = fixed_queries

        return "; ".join(queries), parameters

    # Case 3: Top node type provided without specific top_node_id
    if top_node_type and not node_types:
        where_conds = []
        if excluded_node_types:
            for exc in excluded_node_types:
                where_conds.append(f"label(m) <> '{exc}'")
        where_sql = f" WHERE {' AND '.join(where_conds)}" if where_conds else ""
        query = f"MATCH (root:{top_node_type})-{multi_hop_rel}->(m){where_sql} RETURN root AS n, r, m LIMIT {limit}"
        return query, parameters

    # Case 4: Node types filter
    if node_types:
        queries = []
        per_type_limit = max(50, limit // len(node_types))
        for nt in node_types:
            if exclusion_filter:
                queries.append(
                    f"MATCH (n:{nt})-{rel_clause}->(m) WHERE {exclusion_filter} RETURN n, r, m LIMIT {per_type_limit}"
                )
                queries.append(
                    f"MATCH (n)-{rel_clause}->(m:{nt}) WHERE {exclusion_filter} RETURN n, r, m LIMIT {per_type_limit}"
                )
            else:
                queries.append(f"MATCH (n:{nt})-{rel_clause}->(m) RETURN n, r, m LIMIT {per_type_limit}")
                queries.append(f"MATCH (n)-{rel_clause}->(m:{nt}) RETURN n, r, m LIMIT {per_type_limit}")
        return "; ".join(queries), parameters

    # Case 5: General query
    if exclusion_filter:
        return (
            f"MATCH (n)-{rel_clause}->(m) WHERE {exclusion_filter} RETURN n, r, m LIMIT {limit}",
            parameters,
        )
    return f"MATCH (n)-{rel_clause}->(m) RETURN n, r, m LIMIT {limit}", parameters
