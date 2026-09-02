# Schema Compilation Reference

Advanced reference for genai-graph's schema compilation — how field paths are deduced, how
identity works, and how to override defaults.

## Overview

When you construct a `GraphSchema`, the schema automatically compiles itself:

1. **Field-path deduction** — traverses the root model's field tree to find where each node type lives
2. **Exclusion computation** — marks fields that are relation endpoints (not stored on the node itself)
3. **Coherence validation** — emits `UserWarning` for label collisions and orphaned nodes

You can also call any of these steps independently via `genai_graph.kg.schema.compiler`:

```python
from genai_graph.kg.schema.compiler import (
    build_model_field_map,
    compile_schema,
    compute_excluded_fields,
    deduce_node_field_paths,
    deduce_relation_field_paths,
    validate_schema_coherence,
)
```

## Field-Path Deduction

### How It Works

`deduce_node_field_paths(schema)` performs a breadth-first traversal of `root_model_class`'s
field tree. At each step it records the dotted path from the root to each `GraphNode`'s model class.

```python
class Company(BaseModel):
    name: str


class Project(BaseModel):
    title: str
    client: Company
    partner: Company | None = None
```

For `Company`, two paths are found: `"client"` and `"partner"`. Both are recorded on `company_node.field_paths`.

### Explicit Override

When auto-deduction finds multiple paths (or misses a path), you can set `field_paths` explicitly:

```python
company_node = GraphNode(
    node_class=Company,
    name_from="name",
    key_from="name",
    field_paths=[{"from": "", "to": "client"}],  # only use the primary relationship
)
```

Explicit `field_paths` are never overridden by the compiler.

### Relation Field Paths

`deduce_relation_field_paths(schema)` works similarly but uses the
`(from_node, to_node)` types declared on each `GraphRelation`. The result is stored on
`relation.field_paths`.

## Identity and Labels

### Default Label

A node's Ladybug table name (label) defaults to `node_class.__name__`:

```python
class MyService(BaseModel): ...
node = GraphNode(node_class=MyService, ...)
assert node.label == "MyService"
```

### Overriding with `table_name`

Use `table_name` when:
- Two classes from different modules share the same class name
- You want a shorter or versioned label in the database

```python
class v1.Service(BaseModel): ...
class v2.Service(BaseModel): ...

node_v1 = GraphNode(node_class=v1.Service, table_name="ServiceV1", ...)
node_v2 = GraphNode(node_class=v2.Service, table_name="ServiceV2", ...)
```

### Collision Detection

At schema construction time, `_validate_coherence()` checks for two different node classes that
resolve to the same label. A `UserWarning` is emitted:

```
UserWarning: Two different node classes share the label 'Service': 'v1.Service' and 'v2.Service'.
Set `table_name` on one of them to resolve the collision.
```

### Registry Deduplication

`KgRegistry.build_combined_schema()` uses `node.label` (not class identity) for deduplication.
Two factories registering the same canonical `customer_node` will result in exactly one `Customer`
node in the combined schema, regardless of whether they are literally the same Python object.

## Exclusion Mechanics

### The `p_` Convention

Fields prefixed with `p_` on a relation endpoint are treated as **edge properties** — they belong
to the relation, not the node. They are added to `excluded_fields` on the target node and stored
on the relation instead.

```python
class Contract(BaseModel):
    supplier: Supplier
    p_start_date_: str     # ← edge property: stored on the relation
    p_value_: float        # ← edge property: stored on the relation
    internal_id: str       # ← regular node property

contract_node = GraphNode(node_class=Contract, ...)
# After compilation:
# contract_node.excluded_fields == {"p_start_date_", "p_value_"}
```

### Manual Exclusion

You can also exclude fields explicitly via `excluded_fields` on the `GraphNode`:

```python
person_node = GraphNode(
    node_class=Person,
    name_from="name",
    key_from="id",
    excluded_fields={"internal_notes", "raw_source"},
)
```

### `compute_excluded_fields(schema)`

Runs the full exclusion computation and populates `excluded_fields` on every node. Called
automatically during `GraphSchema.__init__`. Call it manually if you reconstruct a schema dict
and need to recompute:

```python
from genai_graph.kg.schema.compiler import compute_excluded_fields

compute_excluded_fields(schema)
```

## Coherence Validation

`validate_schema_coherence(schema, context=None)` returns a list of warning strings. These are
also emitted as `UserWarning` during construction.

```python
from genai_graph.kg.schema.compiler import validate_schema_coherence

warnings = validate_schema_coherence(schema)
for w in warnings:
    print(w)
```

Or use the convenience accessor on the schema object:

```python
warnings = schema.get_warnings()
```

Currently detected issues:

| Warning | Meaning |
|---------|---------|
| `"No field paths found for X"` | Node type not reachable from the root model |
| `"Two different node classes share the label 'X'"` | Label collision — set `table_name` |

## Compiler Functions Reference

All functions live in `genai_graph.kg.schema.compiler` and are also exported from
`genai_graph.kg.schema`:

| Function | What it does |
|----------|-------------|
| `build_model_field_map(schema)` | Returns `{model_class: GraphNode}` mapping |
| `deduce_node_field_paths(schema)` | Fills `node.field_paths` by BFS traversal |
| `deduce_relation_field_paths(schema)` | Fills `relation.field_paths` |
| `compute_excluded_fields(schema)` | Fills `node.excluded_fields` from `p_*_` fields |
| `validate_schema_coherence(schema)` | Returns list of warning strings |
| `compile_schema(schema)` | Runs all of the above in order |

## ResolvedSchema and Descriptions

`ResolvedSchema` is the canonical enriched representation used for rendering and LLM prompts.

```python
from genai_graph.kg.schema import ResolvedSchema

# Basic (descriptions from GraphNode.description)
resolved = ResolvedSchema.from_graph_schema(schema)

# With BAML descriptions injected (from your project's baml client)
from ekg_atos.baml_client.inlinedbaml import file_map
from genai_graph.kg.schema._helpers import _parse_baml_descriptions

descriptions = _parse_baml_descriptions(file_map)
resolved = ResolvedSchema.from_graph_schema(schema, descriptions=descriptions)
```

The `descriptions` dict has the shape `{"classes": {...}, "fields": {...}, "enums": {...}}`.
Pass `None` or `{}` to use only `GraphNode.description` fallbacks.

## Full Pipeline Example

```python
from genai_graph.kg.schema import (
    GraphNode,
    GraphRelation,
    GraphSchema,
    ResolvedSchema,
    build_model_field_map,
    validate_schema_coherence,
)

# 1. Define
schema = GraphSchema(root_model_class=Project, nodes=[...], relations=[...])

# 2. Inspect
print(build_model_field_map(schema))
print(validate_schema_coherence(schema))

# 3. Render
resolved = ResolvedSchema.from_graph_schema(schema)
print(resolved.to_markdown())

# 4. Export for LLM tool
resolved.to_json_str()  # D3 JSON
resolved.to_html_file(...)  # Interactive D3 graph

# 5. Ingest + query (see graph-definition-guide.md)
```
