# Graph Definition Guide

Define a knowledge graph in genai-graph: from Python models to a queryable Ladybug database in five minutes.

## 1. Define Your Domain Models

Use standard Pydantic models. Nested fields become relations.

```python
from pydantic import BaseModel


class Company(BaseModel):
    name: str
    sector: str | None = None


class Person(BaseModel):
    name: str
    title: str | None = None


class Project(BaseModel):
    title: str
    client: Company  # → FOR_CLIENT relation auto-detected
    lead: Person | None = None  # → HAS_LEAD relation auto-detected
```

## 2. Declare Graph Nodes

Wrap each model in a `GraphNode` and specify identity fields.

```python
from genai_graph.kg.schema import GraphNode

company_node = GraphNode(node_class=Company, name_from="name", key_from="name")
person_node = GraphNode(node_class=Person, name_from="name", key_from="name")
project_node = GraphNode(node_class=Project, name_from="title", key_from="title")
```

| Parameter | Purpose |
|-----------|---------|
| `name_from` | Field whose value is used as the human-readable display name |
| `key_from` | Field used as the unique identifier (primary key) in Ladybug |
| `description` | Optional free-text description for documentation and LLM prompts |
| `table_name` | Override the Ladybug table/label name (defaults to class name) |

## 3. Declare Relations

Relations can be explicit or auto-detected from field paths.

```python
from genai_graph.kg.schema import GraphRelation

client_rel = GraphRelation(
    from_node=project_node,
    to_node=company_node,
    name="FOR_CLIENT",  # Cypher relation type
    field_paths=[{"from": "", "to": "client"}],  # Optional: explicit path
)
lead_rel = GraphRelation(from_node=project_node, to_node=person_node, name="HAS_LEAD")
```

> **Auto-deduction**: If you omit `field_paths`, genai-graph traverses the root model's fields
> to find where each node type appears. This covers most cases. Use explicit `field_paths` when
> the same node type appears in multiple locations or in nested lists.

## 4. Build a GraphSchema

```python
from genai_graph.kg.schema import GraphSchema

schema = GraphSchema(
    root_model_class=Project,  # Entry point for field-path traversal
    nodes=[project_node, company_node, person_node],
    relations=[client_rel, lead_rel],
)
```

`GraphSchema` validates the schema at construction time. Any label collisions or orphaned nodes
produce `UserWarning` messages so you can catch problems early.

## 5. Create a Factory (for persistent, file-backed ingestion)

For a one-off in-memory model, skip straight to [step 6](#6-ingest-data). To ingest a
directory of JSON files (e.g. produced by `cli baml extract`), subclass
`JsonFileBackedFactory` and implement `build_schema()`:

```python
from genai_graph.kg.factories import JsonFileBackedFactory
from pydantic import BaseModel


class ProjectGraph(JsonFileBackedFactory, BaseModel):
    data_root: str = "data/projects"  # directory of {ModelName}/*.json files

    def build_schema(self) -> GraphSchema:
        return GraphSchema(
            root_model_class=Project,
            nodes=[project_node, company_node, person_node],
            relations=[client_rel, lead_rel],
        )
```

`get_keys()` (file discovery) and `get_struct_data_by_key()` (JSON → `Project`) are
provided by the base class. See [Graph Authoring Patterns](graph-authoring-patterns.md)
for the other factory types (tables, Neo4j exports, Markdown documents, inline BAML
extraction).

## 6. Ingest Data

For a single in-memory model, call `create_graph()` directly — it creates the schema
tables and MERGEs the data in one call:

```python
from genai_graph.kg.backend import create_backend_from_config
from genai_graph.kg.ingest import create_graph

backend = create_backend_from_config("my_graph")

project = Project(title="Alpha", client=Company(name="Acme", sector="Tech"), team=[Person(name="Alice", title="Lead")])
create_graph(backend, project, schema)
```

For a factory that reads many files (Step 5), wire it into a workflow instead of
calling it directly — the workflow engine handles schema creation, caching, and the
HTML/warnings exports:

```yaml
# config/workflows/my_graph.yaml
workflows:
  my_project_kg:
    run: genai_graph.orchestration.workflow_steps.kg_build_step
    defaults:
      kg_name: my_project_kg
    params:
      graph: {required: true}
```

```bash
cli workflow run my_project_kg --set graph='{factory: myapp.schema.ProjectGraph, data_root: data/projects}'
# or, once the profile is registered:
cli kg create my_project_kg
```

## 7. Query the Graph

```python
results = backend.execute_cypher("""
    MATCH (p:Project)-[:FOR_CLIENT]->(c:Company)
    RETURN p.title, c.name
""")
for row in results:
    print(row)
```

## Visualise the Schema

```python
from genai_graph.kg.schema import ResolvedSchema

resolved = ResolvedSchema.from_graph_schema(schema)
print(resolved.to_markdown())  # markdown table
resolved.to_html_file("schema.html")  # interactive D3 graph
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Two models with the same class name | Set `table_name="UniqueName"` on the `GraphNode` |
| Relations not auto-detected | Check that `root_model_class` is the model that contains the nested reference |
| Duplicate keys on ingest | Ensure `key_from` resolves to a unique value per entity |
| Missing `p_` prefix for edge properties | Prefix relation-specific properties with `p_` to separate them from node properties |

## Next Steps

- [Graph Authoring Patterns](graph-authoring-patterns.md) — JSON files, CRM tables, Neo4j exports, document ingestion
- [Document Graph](document-graph.md) — the `Folder`/`Document`/`MarkdownSection` schema, inline BAML extraction, `cli docgraph`
- [Workflows](workflows.md) — running factories through the workflow engine, `cli kg create`
- [Schema Compilation Reference](schema-compilation.md) — field-path deduction rules, `table_name`, exclusion mechanics
