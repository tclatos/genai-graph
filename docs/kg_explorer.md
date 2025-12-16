# KG Explorer - Interactive Knowledge Graph Visualization

## Overview

The KG Explorer is a Streamlit page that provides an interactive interface for exploring and querying the Knowledge Graph. It offers multiple ways to visualize and interact with the graph data.

## Features

### 1. Cypher Query Interface
- **Predefined Query Examples**: Select from a dropdown of common Cypher queries
- **Editable Query Input**: Modify queries or write custom Cypher
- **Flexible Display**: View results as graph, table, or both
- **Query Examples Include**:
  - All nodes and relationships
  - Node/relationship type counts
  - Opportunities with customers/competitors
  - Technology stacks
  - High-degree nodes
  - Shortest paths between types

### 2. Natural Language Query (Text-to-Cypher)
- **Natural Language Input**: Ask questions in plain English
- **Automatic Cypher Generation**: AI-powered conversion to Cypher queries
- **Subgraph Selection**: Choose which schemas to include
- **Custom LLM Support**: Optionally specify a different LLM model
- **Review & Execute**: See generated Cypher before executing

### 3. Visualization Options
- **Interactive D3.js Graph**:
  - Zoom and pan
  - Drag nodes
  - Hover tooltips with full node properties
  - Color-coded by node type
  
- **Tabular Results**:
  - Sortable columns
  - Full data display
  - CSV export

## Usage

### Running the KG Explorer

1. Make sure you have a Knowledge Graph database set up:
   ```bash
   uv run cli kg create
   uv run cli kg add-doc --key <your-key>
   ```

2. Launch the Streamlit app:
   ```bash
   cd /home/tcl/prj/genai-graph
   streamlit run genai_graph/webapp/main.py
   ```

3. Navigate to "KG Explorer" in the sidebar

### Using Cypher Queries

1. **Select an Example**:
   - Choose from the dropdown (e.g., "All Graph", "Opportunities with Customers")
   - The query will auto-populate in the text area

2. **Edit the Query** (optional):
   - Modify the Cypher query to fit your needs
   - Add LIMIT clauses to control result size
   - Use WHERE clauses for filtering

3. **Execute**:
   - Select display mode (Graph/Table/Both)
   - Click "🚀 Execute Query"
   - Results appear below

### Using Natural Language Queries

1. **Select Subgraphs**:
   - Choose which schemas to include (e.g., "ReviewedOpportunity", "ArchitectureDocument")

2. **Ask a Question**:
   - Enter your question: "Show all opportunities created after 2020"
   - Optionally specify a custom LLM model

3. **Generate & Execute**:
   - Click "🤖 Generate Cypher"
   - Review the generated query
   - Click "▶️ Execute Generated Query"

## Configuration

### Cypher Query Examples

The example queries are defined in `/home/tcl/prj/genai-graph/config/cypher_examples.yaml`:

```yaml
queries:
  - name: "All Graph"
    description: "Show all nodes and relationships"
    cypher: "MATCH (n)-[r]->(m) RETURN *"
  
  - name: "Node Types Count"
    description: "Count nodes by type"
    cypher: "MATCH (n) RETURN labels(n)[0] AS NodeType, count(n) AS Count ORDER BY Count DESC"
```

You can add your own examples by editing this file.

### Database Configuration

The KG Explorer uses the default graph database configuration. To change it, modify the `GRAPH_DB_CONFIG` constant in `kg_explorer.py`.

## Technical Implementation

### New Functions

#### `generate_html_from_cypher()`
Located in `genai_graph/core/graph_html.py`, this function executes a custom Cypher query and generates an HTML visualization:

```python
from genai_graph.core.graph_backend import create_backend_from_config
from genai_graph.core.graph_html import generate_html_from_cypher

backend = create_backend_from_config("default")
html = generate_html_from_cypher(
    backend,
    "MATCH (n:Person)-[r]->(m) RETURN n, r, m LIMIT 50"
)
```

### Integration with Existing Code

The KG Explorer integrates with:
- **graph_html.py**: Uses existing HTML generation functions
- **text2cypher.py**: Leverages text-to-Cypher conversion
- **graph_backend.py**: Connects to the Knowledge Graph database
- **graph_registry.py**: Accesses schema information

The existing `export-html` CLI command continues to work as before and is unaffected by these changes.

## Tips & Best Practices

### Performance
- Use `LIMIT` clauses to control result size
- Graph visualization works best with < 500 nodes
- For large datasets, use table view
- Filter early in the query with `WHERE` clauses

### Query Writing
- Start with example queries and modify them
- Use `RETURN DISTINCT` to avoid duplicates
- Test queries with small limits first
- Use node labels for better performance: `MATCH (o:Opportunity)` not `MATCH (o)`

### Visualization
- Use "Graph" mode for exploring relationships
- Use "Table" mode for detailed data inspection
- Use "Both" mode to see both perspectives
- Zoom out to see the full graph structure

## Troubleshooting

### No results displayed
- Check if the database has data: `uv run cli kg info`
- Verify the query syntax is valid
- Try a simpler query like `MATCH (n) RETURN n LIMIT 10`

### Graph visualization not showing
- Results might be too large (use LIMIT)
- Query might not return graph elements (nodes/relationships)
- Check browser console for JavaScript errors

### Text-to-Cypher not working
- Ensure subgraphs are selected
- Check LLM configuration in `config/app_conf.yaml`
- Review generated Cypher for syntax errors

## Future Enhancements

Potential improvements:
- Save favorite queries
- Query history
- Export graph as PNG/SVG
- Advanced filtering UI
- Real-time schema inspection
- Query performance metrics

## Related Commands

- `uv run cli kg create`: Create the KG database
- `uv run cli kg add-doc --key <key>`: Add documents to KG
- `uv run cli kg schema`: View the schema
- `uv run cli kg export-html`: Export full graph as HTML
- `uv run cli kg query`: Interactive Cypher shell (CLI)
