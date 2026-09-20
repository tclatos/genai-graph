"""Streamlit page for Knowledge Graph query interface.

Provides an interface to query the Knowledge Graph with:
- Cypher query editor with predefined examples
- Text-to-Cypher natural language query interface
- Results displayed as DataFrame with CSV export
- KG configuration selector

Usage:
    Navigate to this page in the Streamlit app to query the KG.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st
import yaml
from loguru import logger
from streamlit import session_state as sss
from streamlit_monaco import st_monaco

from genai_graph.kg.backend import create_backend_from_config
from genai_graph.kg.query import text2cypher_chain
from genai_graph.kg.schema import GraphRegistry
from genai_graph.webapp.ui_components.kg_config_selector import (
    init_kg_config_session_state,
    render_kg_config_selector,
)

if TYPE_CHECKING:
    from genai_graph.kg.backend import KgBackend

# Configuration
CYPHER_EXAMPLES_CONFIG = "config/cypher_examples.yaml"


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
    init_kg_config_session_state()
    if "cypher_query" not in sss:
        sss.cypher_query = "MATCH (n)-[r]->(m) RETURN * LIMIT 200"
    if "query_result" not in sss:
        sss.query_result = None
    if "generated_cypher" not in sss:
        sss.generated_cypher = None


def execute_cypher_query(cypher: str, backend: "KgBackend") -> tuple[pd.DataFrame | None, str | None]:
    """Execute a Cypher query and return results.

    Args:
        cypher: The Cypher query to execute
        backend: KgBackend instance

    Returns:
        Tuple of (DataFrame result, error message)
    """
    try:
        df = backend.execute_get_as_df(cypher, union=True)
        return df, None
    except Exception as e:
        error_msg = f"Query execution error: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


def main() -> None:
    """Main Streamlit app for KG querying."""
    st.set_page_config(
        page_title="Knowledge Graph Query",
        page_icon="🔍",
        layout="wide",
    )

    initialize_session_state()

    st.title("🔍 Knowledge Graph Query")

    # Sidebar - KG Configuration Selector
    with st.sidebar:
        render_kg_config_selector(help="Select the Knowledge Graph configuration to query")

        st.markdown("---")
        st.markdown("### 🔗 Resources")
        st.markdown("""
        - [Cypher Query Language (neo4j)](https://neo4j.com/docs/cypher-manual/current/)
        - [Cypher Query Language (kuzu/ladybug)](https://docs.ladybugdb.com/cypher/)       
        - [Graph Patterns](https://neo4j.com/docs/cypher-manual/current/patterns/)
        """)

    # Get database connection
    try:
        # Use selected config to create backend
        backend = create_backend_from_config("default", sss.kg_config_selected)
        if not backend:
            st.error("❌ No Knowledge Graph database found")
            st.info(f"Selected configuration: {sss.kg_config_selected}")
            return
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        st.info(f"Selected configuration: {sss.kg_config_selected}")
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
                for example in examples:
                    st.markdown(f"**{example['name']}**")
                    st.caption(example["description"])
                    st.code(example["cypher"], language="cypher")
                    if st.button("⬆️ Load", key=f"load_{example['name']}"):
                        sss.cypher_query = example["cypher"]
                        st.rerun()
                    st.markdown("---")

        with col2:
            st.markdown("**Cypher Query**")

        # Editable Cypher query input — value is read on Execute, not on every keystroke
        edited = st_monaco(
            value=sss.cypher_query,
            height="200px",
            language="cypher",
            theme="vs-dark",
            minimap=False,
            lineNumbers=True,
        )

        with col3:
            execute_btn = st.button("▶️ Execute", type="primary", key="execute_cypher", width="stretch")

        if execute_btn:
            # Capture whatever is currently in the editor (may be None if unchanged)
            query = edited if edited is not None else sss.cypher_query
            sss.cypher_query = query
            with st.spinner("Executing query..."):
                df, error = execute_cypher_query(query, backend)

                if error:
                    st.error(error)
                else:
                    sss.query_result = df
                    st.success(f"✅ Query executed successfully! {len(df)} rows returned.")

    with tab2:
        st.markdown("### Text-to-Cypher: Natural Language Query")
        st.markdown("Enter your question in natural language and it will be converted to Cypher")

        # Get available subgraphs

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
            col2a, col2b = st.columns(2)
            with col2a:
                generate_btn = st.button("🤖 Generate", type="primary", width="stretch")
            with col2b:
                execute_nl_btn = st.button(
                    "▶️ Execute", type="secondary", width="stretch", disabled=not sss.generated_cypher
                )

        # Execute previously generated query
        if execute_nl_btn and sss.generated_cypher:
            with st.spinner("Executing query..."):
                df, error = execute_cypher_query(sss.generated_cypher, backend)
                if error:
                    st.error(error)
                else:
                    sss.query_result = df
                    st.success(f"✅ Query executed! {len(df)} rows returned.")

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
                            llm_id=llm_id or None,
                        )
                        generated_cypher = chain.invoke({})
                        sss.generated_cypher = generated_cypher
                        sss.cypher_query = generated_cypher

                        st.success("✅ Cypher query generated!")
                        st.code(generated_cypher, language="cypher")

                        # Auto-execute the generated query
                        with st.spinner("Executing query..."):
                            df, error = execute_cypher_query(generated_cypher, backend)

                            if error:
                                st.error(error)
                            else:
                                sss.query_result = df
                                st.success(f"✅ Query executed! {len(df)} rows returned.")

                    except Exception as e:
                        st.error(f"Failed to generate Cypher query: {e}")
                        logger.exception("Text-to-Cypher generation failed")

    # Display results section
    st.markdown("---")

    if sss.query_result is not None:
        st.markdown("### 📋 Query Results")

        col1, col2 = st.columns([2, 1])
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

        # Display dataframe with pagination
        st.dataframe(
            sss.query_result,
            height=500,
            width="stretch",
        )
    else:
        st.info("👆 Execute a query above to see results")


if __name__ == "__main__":
    main()
