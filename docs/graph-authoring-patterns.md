# Graph Authoring Patterns

A pattern catalog for common data source types in genai-graph.

## Pattern 1 — JSON Files

The most common pattern. Each JSON file contains a list of objects matching the root model.

```python
# factories.py
from genai_graph.kg.factories import JsonFileBackedFactory
from genai_graph.kg.schema import GraphNode, GraphRelation, GraphSchema
from pydantic import BaseModel


class Customer(BaseModel):
    id: str
    name: str


class Opportunity(BaseModel):
    id: str
    title: str
    customer: Customer
    value: float | None = None


customer_node = GraphNode(node_class=Customer, name_from="name", key_from="id")
opportunity_node = GraphNode(node_class=Opportunity, name_from="title", key_from="id")


class OpportunityGraph(JsonFileBackedFactory):
    schema = GraphSchema(
        root_model_class=Opportunity,
        nodes=[opportunity_node, customer_node],
        relations=[GraphRelation(from_node=opportunity_node, to_node=customer_node, name="FOR_CUSTOMER")],
    )
    source_model = Opportunity
    source_dir = "data/opportunities"  # directory of *.json files
```

**When to use**: structured JSON exports, API responses saved to disk, pipeline outputs.

## Pattern 2 — CRM / Spreadsheet Tables

Use a DataFrame loader to convert tabular data (xlsx, csv) into Pydantic models before ingestion.

```python
from pydantic import BaseModel
import pandas as pd


class CrmRow(BaseModel):
    opportunity_id: str
    customer_name: str
    stage: str
    amount: float | None = None


def load_crm_data(path: str) -> list[CrmRow]:
    df = pd.read_excel(path)
    # normalise column names, handle NaN, etc.
    df = df.rename(columns={"OppID": "opportunity_id", "Client": "customer_name", "Stage": "stage", "Amount": "amount"})
    return [CrmRow(**row) for row in df.to_dict(orient="records")]
```

Then ingest as usual:
```python
rows = load_crm_data("crm_export.xlsx")
graph.ingest(rows)
```

**When to use**: CRM exports (Salesforce, Dynamics), spreadsheet data, legacy system dumps.

## Pattern 3 — Neo4j Exports

Import Neo4j JSON exports using the built-in CLI command.

```bash
# Convert Neo4j export to Ladybug format
uv run cli neo4j import --source neo4j_export/ --dest ladybug_db/
```

For programmatic access:
```python
from genai_graph.kg.backend import create_backend_from_config

backend = create_backend_from_config("my_graph")
backend.import_neo4j_json("neo4j_nodes.json", "neo4j_rels.json")
```

**When to use**: migrating from an existing Neo4j instance, loading community datasets.

## Pattern 4 — Document / Markdown Ingestion

Use `DocumentGraphFactory` to ingest a Markdown corpus as a navigable
`Folder → Document → MarkdownSection` graph (heading hierarchy, no chunking or
embeddings):

```python
from genai_graph.kg.backend import KuzuBackend
from genai_graph.kg.document_graph.ingest import ingest_document_graph
from genai_graph.kg.factories.document_graph_factory import DocumentGraphFactory

backend = KuzuBackend()
backend.connect("./data/kg/tree.db")

factory = DocumentGraphFactory(sources=["data/reports"], include=["*.md"])
ingest_document_graph(backend, factory)
```

Each file becomes a `Document` node (keyed by content hash) with one `MarkdownSection`
per heading. See [Document Graph](document-graph.md) for the full schema and CLI
(`cli docgraph build`/`run`).

If you only need file-level provenance (no sections), use
`DocumentDirectoryFactory` instead — it produces plain `Document` nodes:

```python
from genai_graph.kg.factories import DocumentDirectoryFactory
from genai_graph.kg.nodes.document import DocumentNode


class ReportGraph(DocumentDirectoryFactory):
    def build_schema(self) -> GraphSchema:
        return GraphSchema(root_model_class=None, nodes=[DocumentNode], relations=[])
```

**When to use**: technical documentation, research papers, internal wiki pages;
`DocumentGraphFactory` when you want table-of-contents navigation, `DocumentDirectoryFactory`
for simple file-provenance tracking.

## Pattern 4b — Inline BAML Entity Extraction over Markdown

Use `MarkdownBamlFactory` to extract structured entities from Markdown files via a
BAML function, without a separate `cli baml extract` step — the extraction result is
cached as JSON automatically:

```python
from genai_graph.kg.factories import MarkdownBamlFactory
from genai_graph.kg.schema import GraphSchema
from pydantic import BaseModel


class ReviewedOpportunityGraph(MarkdownBamlFactory):
    def build_schema(self) -> GraphSchema:
        nodes, relations = self.get_document_schema_elements(ReviewedOpportunityNode)
        return GraphSchema(
            root_model_class=ReviewedOpportunity,
            nodes=[ReviewedOpportunityNode, *nodes],
            relations=relations,
        )

    def extract_from_markdown(self, md_text: str) -> BaseModel:
        from genai_tk.extra.structured.baml_processor import BamlStructuredProcessor

        processor = BamlStructuredProcessor(
            model_cls=ReviewedOpportunity, function_name="ExtractRainbow", kvstore_id=""
        )
        return processor.analyze_document("doc", md_text)
```

Because `get_document_schema_elements()` adds the same `Document` node type as
`DocumentGraphFactory`, running both factories against the same Markdown produces one
shared `Document` node carrying provenance *and* extracted entities — see
`genai_graph.orchestration.workflow_steps.docgraph_build_step`.

**When to use**: LLM-extracted entities (opportunities, risks, requirements, …) that
should live in the same KG as the document provenance graph.

## Pattern 5 — Similarity / Embedding Nodes

Mark a field with `EmbeddingField` to enable vector similarity search.

```python
from pydantic import BaseModel, Field
from genai_graph.kg.schema import GraphNode
from genai_graph.kg.embeddings import EmbeddingField


class Service(BaseModel):
    name: str
    description: str = Field(default="", json_schema_extra={"embedding": True})


service_node = GraphNode(
    node_class=Service,
    name_from="name",
    key_from="name",
    embedding_field="description",  # field to embed
)
```

Query by similarity:
```python
from genai_graph.kg.query import SimilarityFactory

matcher = SimilarityFactory(backend=backend, node=service_node)
results = matcher.find_similar("cloud-native platform", top_k=5)
```

**When to use**: semantic search over services, products, skills, or documents.

## Pattern 6 — Canonical Node Reuse

Share node types across multiple factory graphs using a central canonical schema file.

```python
# schema/canonical_nodes.py
from pydantic import BaseModel
from genai_graph.kg.schema import GraphNode


class Customer(BaseModel):
    id: str
    name: str


class Product(BaseModel):
    sku: str
    name: str


# Canonical definitions — import these in all factories
customer_node = GraphNode(node_class=Customer, name_from="name", key_from="id")
product_node = GraphNode(node_class=Product, name_from="name", key_from="sku")
```

```python
# schema/order_graph.py
from schema.canonical_nodes import customer_node, product_node


class OrderGraph(JsonFileBackedFactory):
    schema = GraphSchema(
        root_model_class=Order,
        nodes=[order_node, customer_node, product_node],
        relations=[...],
    )
```

The `KgRegistry` automatically deduplicates nodes by `label` when building a combined schema across
multiple factories. Canonical nodes are only created once in the database regardless of how many
factories reference them.

**When to use**: multi-source pipelines (CRM + product catalogue + documents) that share entity types.

## Pattern 7 — Multi-Source Merge via KgRegistry

Register multiple factories and let the registry build a unified schema.

```python
from genai_graph.kg.schema.registry import KgRegistry

registry = KgRegistry()
registry.register(OpportunityGraph)
registry.register(ReportGraph)
registry.register(ServiceGraph)

combined_schema = registry.build_combined_schema()
print(combined_schema.to_markdown())
```

Each factory's nodes and relations are merged. Nodes with the same `label` are deduplicated.

## Choosing a Pattern

| Data source | Recommended pattern |
|-------------|---------------------|
| JSON export / API output | JSON Files (1) |
| Excel / CSV spreadsheet | CRM Tables (2) |
| Existing Neo4j graph | Neo4j Import (3) |
| Markdown / text files (navigable TOC) | Document Ingestion (4) |
| LLM entity extraction from Markdown (inline) | Inline BAML Extraction (4b) |
| Semantic search use case | Similarity Nodes (5) |
| Multiple sources, shared entities | Canonical Reuse (6) + Registry (7) |
