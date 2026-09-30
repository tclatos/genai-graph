"""Streamlit page for Knowledge Graph HTML visualization.

Provides an interface to visualize the Knowledge Graph with:
- Interactive HTML graph visualization using D3.js
- Filter by node types, specific nodes, and relationships
- Configurable LIMIT parameter
- KG configuration selector

Usage:
    Navigate to this page in the Streamlit app to visualize the KG.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st
from loguru import logger
from streamlit import session_state as sss

from genai_graph.kg.backend import create_backend_from_config
from genai_graph.kg.export import generate_html
from genai_graph.webapp.ui_components.kg_config_selector import (
    init_kg_config_session_state,
    render_kg_config_selector,
)

if TYPE_CHECKING:
    from genai_graph.kg.backend import KgBackend

DEFAULT_VIZ_LIMIT = 2000


def get_node_types(backend: KgBackend) -> list[str]:
    """Get all node types (labels) in the graph.

    Args:
        backend: KgBackend instance

    Returns:
        List of node type names
    """
    try:
        df = backend.execute_get_as_df("CALL show_tables() RETURN name, type", union=False)
        # Filter for NODE tables only
        node_tables = df[df["type"] == "NODE"]["name"].tolist()
        return sorted(node_tables)
    except Exception as e:
        logger.warning(f"Could not fetch node types: {e}")
        return []


def get_relationship_types(backend: KgBackend) -> list[str]:
    """Get all relationship types in the graph.

    Args:
        backend: KgBackend instance

    Returns:
        List of relationship type names
    """
    try:
        df = backend.execute_get_as_df("CALL show_tables() RETURN name, type", union=False)
        # Filter for REL tables only
        rel_tables = df[df["type"] == "REL"]["name"].tolist()
        return sorted(rel_tables)
    except Exception as e:
        logger.warning(f"Could not fetch relationship types: {e}")
        return []


def get_nodes_of_type(backend: KgBackend, node_type: str, limit: int = 200) -> list[str]:
    """Get list of nodes of a specific type with their display names.

    Args:
        backend: KgBackend instance
        node_type: Node type to query
        limit: Maximum number of nodes to return

    Returns:
        List of node display names
    """
    try:
        # Try to get a name field - common names are 'name', '_name', or the first non-internal field
        query = f"MATCH (n:{node_type}) RETURN n LIMIT {limit}"
        df = backend.execute_get_as_df(query, union=False)

        node_names = []
        for _, row in df.iterrows():
            node_data = row["n"]
            if isinstance(node_data, dict):
                # Try common name fields
                name = (
                    node_data.get("name")
                    or node_data.get("_name")
                    or node_data.get("title")
                    or node_data.get("id")
                    or str(node_data.get("_id", ""))[:30]
                )
                if name:
                    node_names.append(str(name))

        return sorted(set(node_names))
    except Exception as e:
        logger.warning(f"Could not fetch nodes of type {node_type}: {e}")
        return []


def build_filtered_cypher_query(
    node_types: list[str],
    node_name: str | None,
    relationship_types: list[str],
    limit: int,
    excluded_node_types: list[str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Build a Cypher query based on selected filters.

    Args:
        node_types: Selected node types (empty means all)
        node_name: Selected specific node name (None means all of that type)
        relationship_types: Selected relationship types (empty means all)
        limit: Maximum number of results
        excluded_node_types: Node types to exclude from results (None/empty means no exclusion)

    Returns:
        Tuple of (Cypher query string with ``$name``-style placeholders, parameters dict).
        Node/relationship type names are still interpolated (Cypher doesn't allow
        parameterizing labels), but ``node_name`` — which can originate from
        document-derived data ingested into the graph, not just fixed schema
        labels — is always passed as a bound parameter to avoid Cypher injection.
    """
    # Use multi-hop only when a specific node is selected (explore its neighbourhood).
    # For node-type-only filters, multi-hop causes a path-count explosion that makes
    # the LIMIT hit after just 2-3 source nodes, hiding all others of the same type.
    use_multi_hop = bool(node_types and node_name)
    HOPS = 5

    # Build relationship filter
    if relationship_types:
        rel_patterns = "|".join(relationship_types)
        rel_filter = f"[r:{rel_patterns}*1..{HOPS}]" if use_multi_hop else f"[r:{rel_patterns}]"
    else:
        rel_filter = f"[r*1..{HOPS}]" if use_multi_hop else "[r]"

    # Build exclusion filter for WHERE clause
    exclusion_filter = ""
    if excluded_node_types:
        exclusion_conditions = []
        for exc_type in excluded_node_types:
            exclusion_conditions.append(f"label(n) <> '{exc_type}'")
            exclusion_conditions.append(f"label(m) <> '{exc_type}'")
        exclusion_filter = " AND ".join(exclusion_conditions)

    # Build query based on filters
    parameters: dict[str, str] = {}
    if node_types and node_name:
        # Specific node and type - filter by name (use first selected type for node name filter).
        # node_name is bound as a parameter (never string-interpolated) since it can
        # originate from document-derived data ingested into the graph.
        parameters["name"] = node_name
        node_type = node_types[0]  # Use first type for specific node selection
        half_limit = limit // 2

        if exclusion_filter:
            query = (
                f"MATCH (n:{node_type})-{rel_filter}->(m) WHERE n.name = $name AND {exclusion_filter} RETURN n, r, m LIMIT {half_limit}; "
                f"MATCH (n)-{rel_filter}->(m:{node_type}) WHERE m.name = $name AND {exclusion_filter} RETURN n, r, m LIMIT {half_limit}"
            )
        else:
            query = (
                f"MATCH (n:{node_type})-{rel_filter}->(m) WHERE n.name = $name RETURN n, r, m LIMIT {half_limit}; "
                f"MATCH (n)-{rel_filter}->(m:{node_type}) WHERE m.name = $name RETURN n, r, m LIMIT {half_limit}"
            )
    elif node_types:
        # Node type filter - query both directions, allowing connections to any node type
        queries = []
        # Split limit across types (not across directions) – single-hop deduplication
        # handles any overlap between the two direction queries.
        per_type_limit = max(50, limit // len(node_types))

        for node_type in node_types:
            if exclusion_filter:
                queries.append(
                    f"MATCH (n:{node_type})-{rel_filter}->(m) WHERE {exclusion_filter} RETURN n, r, m LIMIT {per_type_limit}"
                )
                queries.append(
                    f"MATCH (n)-{rel_filter}->(m:{node_type}) WHERE {exclusion_filter} RETURN n, r, m LIMIT {per_type_limit}"
                )
            else:
                queries.append(f"MATCH (n:{node_type})-{rel_filter}->(m) RETURN n, r, m LIMIT {per_type_limit}")
                queries.append(f"MATCH (n)-{rel_filter}->(m:{node_type}) RETURN n, r, m LIMIT {per_type_limit}")
        query = "; ".join(queries)
    else:
        # No node type filter - use single hop
        if exclusion_filter:
            query = f"MATCH (n)-{rel_filter}->(m) WHERE {exclusion_filter} RETURN n, r, m LIMIT {limit}"
        else:
            query = f"MATCH (n)-{rel_filter}->(m) RETURN n, r, m LIMIT {limit}"

    return query, parameters


def initialize_session_state() -> None:
    """Initialize session state variables."""
    init_kg_config_session_state()
    if "graph_html" not in sss:
        sss.graph_html = None
    if "viz_limit" not in sss:
        sss.viz_limit = DEFAULT_VIZ_LIMIT
    if "selected_node_types" not in sss:
        sss.selected_node_types = []
    if "selected_node_name" not in sss:
        sss.selected_node_name = None
    if "selected_rel_types" not in sss:
        sss.selected_rel_types = []
    if "excluded_node_types" not in sss:
        sss.excluded_node_types = []
    if "current_query" not in sss:
        sss.current_query = None


def show_empty_kg_message(config_name: str) -> None:
    """Show message when KG has no data, with CLI command to create it.

    Args:
        config_name: Name of the KG configuration
    """
    st.warning(
        f"""
        ### 📊 Knowledge Graph '{config_name}' is Empty

        The Knowledge Graph configuration **{config_name}** has not been generated yet,
        or the database is empty.

        To create the Knowledge Graph, run the following command in your terminal:

        ```bash
        cli kg create --profile={config_name}
        ```

        Then refresh this page to visualize the graph.
        """
    )


def main() -> None:
    """Main Streamlit app for KG visualization."""
    st.set_page_config(
        page_title="Knowledge Graph Visualization",
        page_icon="🕸️",
        layout="wide",
    )

    initialize_session_state()

    st.title("🕸️ Knowledge Graph Visualization")

    # Sidebar - Configuration and Filters
    with st.sidebar:
        render_kg_config_selector(
            help="Select the Knowledge Graph configuration to visualize",
            on_change=lambda: setattr(sss, "graph_html", None),
        )
        selected_config = sss.kg_config_selected

        # Get database connection for filters
        try:
            backend = create_backend_from_config("default", selected_config)
            if not backend:
                st.error("❌ No Knowledge Graph database found")
                return
        except Exception as e:
            st.error(f"Failed to connect to database: {e}")
            return

        # Check if KG has any data - get node and relationship types early
        available_node_types = get_node_types(backend)
        available_rel_types = get_relationship_types(backend)

        # If no data exists, show message with CLI command
        if not available_node_types and not available_rel_types:
            show_empty_kg_message(selected_config)
            return

        st.markdown("### 🎯 Filters")

        # Track previous filter values to detect changes
        if "prev_node_types" not in sss:
            sss.prev_node_types = []
        if "prev_node_name" not in sss:
            sss.prev_node_name = None
        if "prev_rel_types" not in sss:
            sss.prev_rel_types = []
        if "prev_viz_limit" not in sss:
            sss.prev_viz_limit = DEFAULT_VIZ_LIMIT
        if "prev_excluded_node_types" not in sss:
            sss.prev_excluded_node_types = []

        # Node type filter (multiselect)
        if available_node_types:
            selected_node_types = st.multiselect(
                "Node Types",
                options=available_node_types,
                default=[],
                help="Select node types to include (empty = all types)",
            )
            sss.selected_node_types = selected_node_types
        else:
            st.info("No node types available")
            sss.selected_node_types = []

        # Exclude node type filter (multiselect)
        if available_node_types:
            # Filter out the selected include node types to prevent conflict
            exclude_options = [t for t in available_node_types if t not in sss.selected_node_types]
            selected_exclude_types = st.multiselect(
                "Exclude Node Types",
                options=exclude_options,
                default=[],
                help="Select node types to exclude from the visualization",
            )
            sss.excluded_node_types = selected_exclude_types
        else:
            sss.excluded_node_types = []

        # Specific node selector (only if exactly one node type is selected)
        if len(sss.selected_node_types) == 1:
            node_type = sss.selected_node_types[0]
            available_nodes = get_nodes_of_type(backend, node_type)
            if available_nodes:
                node_options = ["(All)", *available_nodes]
                selected_node_display = st.selectbox(
                    f"{node_type} Nodes",
                    options=node_options,
                    index=0,
                    help=f"Select a specific {node_type} node to focus on",
                )
                sss.selected_node_name = None if selected_node_display == "(All)" else selected_node_display
            else:
                st.info(f"No {node_type} nodes found")
                sss.selected_node_name = None
        else:
            sss.selected_node_name = None

        # Relationship type filter (available_rel_types already fetched above)
        if available_rel_types:
            selected_rel_types = st.multiselect(
                "Relationship Types",
                options=available_rel_types,
                default=[],
                help="Select relationship types to display (empty = all types)",
            )
            sss.selected_rel_types = selected_rel_types
        else:
            st.info("No relationship types available")
            sss.selected_rel_types = []

        # Limit parameter
        limit = st.number_input(
            "Result Limit",
            min_value=10,
            max_value=10000,
            value=DEFAULT_VIZ_LIMIT,
            step=50,
            help="Maximum number of relationships to display",
        )
        sss.viz_limit = limit

        # Detect filter changes and clear cached visualization
        if (
            sss.selected_node_types != sss.prev_node_types
            or sss.selected_node_name != sss.prev_node_name
            or sss.selected_rel_types != sss.prev_rel_types
            or sss.viz_limit != sss.prev_viz_limit
            or sss.excluded_node_types != sss.prev_excluded_node_types
        ):
            sss.graph_html = None  # Clear cached graph when filters change
            sss.prev_node_types = sss.selected_node_types.copy() if sss.selected_node_types else []
            sss.prev_node_name = sss.selected_node_name
            sss.prev_rel_types = sss.selected_rel_types.copy() if sss.selected_rel_types else []
            sss.prev_viz_limit = sss.viz_limit
            sss.prev_excluded_node_types = sss.excluded_node_types.copy() if sss.excluded_node_types else []

        st.markdown("---")

        # Generate button
        if st.button("🎨 Generate Visualization", type="primary", width="stretch"):
            with st.spinner("Generating visualization..."):
                try:
                    # Build query based on filters
                    query, query_params = build_filtered_cypher_query(
                        sss.selected_node_types,
                        sss.selected_node_name,
                        sss.selected_rel_types,
                        sss.viz_limit,
                        sss.excluded_node_types,
                    )

                    # Store the query in session state
                    sss.current_query = query

                    # Generate HTML visualization
                    # filter_orphan_nodes: removes unconnected intermediates from multi-hop paths.
                    # selected_node_types: ensures ALL instances of the chosen types are fetched
                    # via a supplemental query, and exempts them from orphan pruning.
                    html_content = generate_html(
                        backend,
                        query=query,
                        parameters=query_params or None,
                        filter_orphan_nodes=bool(sss.selected_node_types),
                        selected_node_types=sss.selected_node_types or None,
                    )
                    sss.graph_html = html_content
                    st.success("✅ Visualization generated!")
                except Exception as e:
                    st.error(f"Failed to generate visualization: {e}")
                    logger.exception("Visualization generation failed")

    if sss.graph_html:
        # Display current filters and query side by side
        filter_info = []
        if sss.selected_node_types:
            if sss.selected_node_name:
                filter_info.append(f"Node: {sss.selected_node_types[0]}/{sss.selected_node_name}")
            else:
                filter_info.append(f"Node Types: {', '.join(sss.selected_node_types)}")
        if sss.excluded_node_types:
            filter_info.append(f"Excluding: {', '.join(sss.excluded_node_types)}")
        if sss.selected_rel_types:
            filter_info.append(f"Relationships: {', '.join(sss.selected_rel_types)}")
        filter_info.append(f"Limit: {sss.viz_limit}")

        _col1, _col2 = st.columns([1, 2])
        with st.sidebar:
            st.caption("Filters: " + " | ".join(filter_info))
            if sss.current_query:
                st.caption(f"Query: `{sss.current_query}`")

        # Display the graph
        with st.container():
            st.iframe(sss.graph_html, height=700)
    else:
        st.info("👈 Click 'Generate Visualization' in the sidebar to display the graph with your selected filters.")


if __name__ == "__main__":
    main()
