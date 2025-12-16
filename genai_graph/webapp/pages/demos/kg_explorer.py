"""Streamlit page for interactive Knowledge Graph visualization.

Provides an interactive interface to visualize and query the Knowledge Graph with:
- Interactive HTML graph visualization using D3.js
- Cypher query selector with predefined examples
- Editable Cypher query input
- Results displayed as both graph and table
- Text-to-Cypher natural language query interface
- Execute generated queries and display results

Usage:
    Navigate to this page in the Streamlit app to interactively explore the KG.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yaml
from loguru import logger
from streamlit import session_state as sss

from genai_graph.core.graph_backend import create_backend_from_config
from genai_graph.core.graph_html import generate_html_from_cypher
from genai_graph.core.text2cypher import text2cypher_chain

if TYPE_CHECKING:
    from genai_graph.core.graph_backend import GraphBackend

# Configuration
CYPHER_EXAMPLES_CONFIG = "config/cypher_examples.yaml"
GRAPH_DB_CONFIG = "default"


def load_cypher_examples() -> list[dict]:
    """Load example Cypher queries from YAML config file.

    Returns:
        List of query dictionaries with name, description, and cypher fields
    """
    config_path = Path(CYPHER_EXAMPLES_CONFIG)
    if not config_path.exists():
        logger.warning(f"Cypher examples config not found: {config_path}")
        return [
            {
                "name": "All Graph",
                "description": "Show all nodes and relationships",
                "cypher": "MATCH (n)-[r]->(m) RETURN *",
            }
        ]

    with open(config_path) as f:
        config = yaml.safe_load(f)
        return config.get("queries", [])


def initialize_session_state() -> None:
    """Initialize session state variables."""
    if "cypher_query" not in sss:
        sss.cypher_query = "MATCH (n)-[r]->(m) RETURN * LIMIT 200"
    if "query_result" not in sss:
        sss.query_result = None
    if "generated_cypher" not in sss:
        sss.generated_cypher = None
    if "graph_html" not in sss:
        sss.graph_html = None
    if "selected_example" not in sss:
        sss.selected_example = "All Graph"
    if "display_mode" not in sss:
        sss.display_mode = "Both"
    if "selected_example" not in sss:
        sss.selected_example = "All Graph"
    if "display_mode" not in sss:
        sss.display_mode = "Both"


def execute_cypher_query(cypher: str, backend: "GraphBackend") -> tuple[pd.DataFrame | None, str | None]:
    """Execute a Cypher query and return results.

    Args:
        cypher: The Cypher query to execute
        backend: GraphBackend instance

    Returns:
        Tuple of (DataFrame result, error message)
    """
    try:
        result = backend.execute(cypher)
        df = result.get_as_df()
        return df, None
    except Exception as e:
        error_msg = f"Query execution error: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


def main() -> None:
    """Main Streamlit app for KG visualization."""
    st.set_page_config(
        page_title="Knowledge Graph Explorer",
        page_icon="🕸️",
        layout="wide",
    )

    initialize_session_state()

    st.title("🕸️ Knowledge Graph Explorer")

    # Get database connection
    try:
        backend = create_backend_from_config(GRAPH_DB_CONFIG)
        if not backend:
            st.error("❌ No Knowledge Graph database found")
            return
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        return

    # Load example queries
    examples = load_cypher_examples()

    # Create tabs for different query methods
    tab1, tab2 = st.tabs(["📝 Cypher Query", "💬 Natural Language"])

    with tab1:
        # Compact header with example query button
        col1, col2, col3 = st.columns([1, 4, 1])
        with col1:
            with st.popover("📋 Examples"):
                st.markdown("**Example Queries**")
                st.markdown("*Click to copy the query*")
                for example in examples:
                    st.markdown(f"**{example['name']}**")
                    st.caption(example["description"])
                    st.code(example["cypher"], language="cypher")
                    st.markdown("---")

        with col2:
            st.markdown("**Cypher Query**")

        with col3:
            execute_btn = st.button("▶️ Execute", type="primary", use_container_width=True)

        # Editable Cypher query input
        cypher_query = st.text_area(
            "Query",
            value=sss.cypher_query,
            height=80,
            help="Enter Cypher query (see Examples for templates)",
            key="cypher_input",
            label_visibility="collapsed",
        )

        # Update session state when user types
        if cypher_query != sss.cypher_query:
            sss.cypher_query = cypher_query

        # Display mode selector
        display_mode_options = ["Graph", "Table", "Both"]
        current_index = display_mode_options.index(sss.display_mode)
        sss.display_mode = st.radio(
            "Display:",
            options=display_mode_options,
            horizontal=True,
            index=current_index,
        )

        if execute_btn and cypher_query:
            with st.spinner("Executing query..."):
                # Execute the query
                df, error = execute_cypher_query(cypher_query, backend)

                if error:
                    st.error(error)
                else:
                    sss.query_result = df

                    # Generate HTML visualization if graph display is requested
                    if sss.display_mode in ["Graph", "Both"]:
                        try:
                            html_content = generate_html_from_cypher(
                                backend,
                                cypher_query,
                            )
                            sss.graph_html = html_content
                        except Exception as e:
                            st.warning(f"Could not generate graph visualization: {e}")
                            sss.graph_html = None

    with tab2:
        st.markdown("### Text-to-Cypher: Natural Language Query")
        st.markdown("Enter your question in natural language and it will be converted to Cypher")

        # Get available subgraphs
        from genai_graph.core.graph_registry import GraphRegistry

        try:
            registry = GraphRegistry()
            available_subgraphs = registry.listsubgraphs()
        except Exception:
            available_subgraphs = []

        # Subgraph selection
        selected_subgraphs = st.multiselect(
            "Select subgraphs to query",
            options=available_subgraphs,
            default=available_subgraphs[:1] if available_subgraphs else [],
            help="Select which subgraphs' schemas to include in the query generation",
        )

        # Natural language input
        nl_query = st.text_input(
            "Your question",
            placeholder="e.g., List all opportunities with their customers",
            help="Ask a question about the knowledge graph in natural language",
        )

        # LLM selection (optional)
        col1, col2 = st.columns([3, 1])
        with col1:
            llm_id = st.text_input(
                "LLM model (optional)",
                value="",
                placeholder="Leave empty for default",
                help="Specify a custom LLM model ID",
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            generate_btn = st.button("🤖 Generate Cypher", type="primary", width="stretch")

        if generate_btn and nl_query:
            if not selected_subgraphs:
                st.warning("Please select at least one subgraph")
            else:
                with st.spinner("Generating Cypher query..."):
                    try:
                        # Generate Cypher from natural language
                        chain = text2cypher_chain(
                            nl_query,
                            selected_subgraphs,
                            llm_id=llm_id if llm_id else None,
                        )
                        generated_cypher = chain.invoke({})
                        sss.generated_cypher = generated_cypher

                        st.success("✅ Cypher query generated!")
                        st.code(generated_cypher, language="cypher")

                        # Ask if user wants to execute
                        if st.button("▶️ Execute Generated Query"):
                            with st.spinner("Executing query..."):
                                df, error = execute_cypher_query(generated_cypher, backend)

                                if error:
                                    st.error(error)
                                else:
                                    sss.query_result = df
                                    sss.cypher_query = generated_cypher

                                    # Generate HTML visualization
                                    try:
                                        html_content = generate_html_from_cypher(
                                            backend,
                                            generated_cypher,
                                        )
                                        sss.graph_html = html_content
                                    except Exception as e:
                                        st.warning(f"Could not generate graph visualization: {e}")
                                        sss.graph_html = None

                                    # Display results below
                                    st.rerun()

                    except Exception as e:
                        st.error(f"Failed to generate Cypher query: {e}")
                        logger.exception("Text-to-Cypher generation failed")

    # Display results section
    st.markdown("---")

    if sss.query_result is not None:
        # Display graph visualization
        if sss.graph_html and sss.display_mode in ["Graph", "Both"]:
            st.markdown("### 📊 Graph Visualization")
            with st.container():
                components.html(sss.graph_html, height=600, scrolling=True)

        # Display table results
        if sss.display_mode in ["Table", "Both"]:
            st.markdown("### 📋 Query Results Table")

            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"**{len(sss.query_result)} rows returned**")
            with col2:
                if st.button("📥 Download CSV"):
                    csv = sss.query_result.to_csv(index=False)
                    st.download_button(
                        "Download",
                        csv,
                        "query_results.csv",
                        "text/csv",
                        key="download-csv",
                    )

            # Display dataframe
            st.dataframe(
                sss.query_result,
                height=400,
            )
    else:
        st.info("👆 Execute a query above to see results")

    # Sidebar with resources
    with st.sidebar:
        st.markdown("### 🔗 Resources")
        st.markdown("""
        - [Cypher Query Language](https://neo4j.com/docs/cypher-manual/current/)
        - [Graph Patterns](https://neo4j.com/docs/cypher-manual/current/patterns/)
        """)


if __name__ == "__main__":
    main()
