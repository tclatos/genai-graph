# BAML Extraction and Graph Factory Creation Guide

This guide covers genai-graph–specific patterns for creating graph factories that consume
BAML-extracted data. For BAML setup fundamentals (writing `.baml` files, generating Python
types, configuring the LLM client, and running `cli baml extract`), see the
[genai-tk BAML documentation](../../genai-tk/docs/baml.md).

## Quick Reference

| Task | Files to Modify | Command |
|------|----------------|---------|
| Add/modify extracted fields | `ekg_atos/ekg_atos/baml_src/schema/*.baml` | `cd ekg_atos/ && baml-cli generate` |
| Update graph schema | `ekg_atos/ekg_atos/schema/*_review.py` | N/A |
| Configure KG | `config/workflows/*.yaml` (a `kg_build` pipeline step) | N/A |
| Extract from documents | N/A | `cli baml extract <root_dir> <output_dir> --function ExtractRainbow` |
| Build knowledge graph | N/A | `cli kg create <name>` |

## Architecture Overview

```
Text Documents (MD/PDF)
    ↓  cli tools markdownize
Markdown files
    ↓  cli baml extract
JSON files  (BAML-extracted, one per document)
    ↓
JsonFileBackedFactory → Pydantic Models
    ↓
GraphSchema → Node/Relationship Configuration
    ↓
Ladybug Graph Database
```

## Step 1: Define BAML Schema

**Location**: `ekg_atos/ekg_atos/baml_src/schema/*.baml`

Add or modify class definitions using standard BAML syntax. For BAML language reference
(types, annotations, enums, etc.) see the [genai-tk BAML docs](../../genai-tk/docs/baml.md).

**Example** — adding a field to an existing class:

```baml
// In rainbow_review.baml
class ReviewedOpportunity {
  opportunity Opportunity
  start_date string? @description("Planned start date")
  statement_of_work KeyStatementOfWorkElement? @description("Key requirements")
  team Person[] @description("Team members")
}

class KeyStatementOfWorkElement {
  objectives string[]? @description("Key business objectives")
  scope string? @description("Scope of work")
  requirements string[]? @description("Key requirements")
}
```

After modifying `.baml` files, regenerate Python types:

```bash
cd ekg_atos/ekg_atos
baml-cli generate
```

This generates `baml_client/types.py` with Pydantic models.

## Step 2: Create or Update Graph Factory

Graph factories define how extracted data becomes nodes and relationships in the graph.

**Location**: `ekg_atos/ekg_atos/schema/*.py`

### Basic Factory Structure

Define `GraphNode` singletons at **module scope** — once per class. These are
reused in `build_schema()` and passed directly to `GraphRelation`.

```python
from pydantic import BaseModel
from ekg_atos.baml_client.types import Opportunity, Person, ReviewedOpportunity
from ekg_atos.schema.canonical_nodes import CustomerNode  # canonical shared node
from ekg_atos.schema.common_nodes import Geo
from genai_graph.kg.factories import JsonFileBackedFactory
from genai_graph.kg.schema import GraphNode, GraphRelation, GraphSchema

# ---------------------------------------------------------------------------
# Module-scope node singletons — defined once, referenced in GraphRelation
# ---------------------------------------------------------------------------
OpportunityNode: GraphNode = GraphNode(
    node_class=Opportunity,
    name_from="name",
    key_from="opportunity_id",
    description="Core opportunity information",
    index_fields=["name", "status"],
)
GeoNode: GraphNode = GraphNode(
    node_class=Geo,
    name_from="name",
    key_from="name",
    explicitly_defined=True,
)
ReviewedOpportunityNode: GraphNode = GraphNode(
    node_class=ReviewedOpportunity,
    name_from=lambda data, _: "Review:" + str(data.get("start_date")),
    key_from="AUTO_ID",
)


class ReviewedOpportunityGraph(JsonFileBackedFactory, BaseModel):
    """Factory for processing reviewed opportunity documents."""

    def get_model_class(self) -> type[BaseModel]:
        """Return the root Pydantic model class."""
        return ReviewedOpportunity

    def build_schema(self) -> GraphSchema:
        """Define nodes and relationships."""
        nodes = [
            ReviewedOpportunityNode,
            OpportunityNode,
            CustomerNode,  # canonical shared node (from canonical_nodes)
            GeoNode,
            # ... more nodes
        ]

        relations = [
            GraphRelation(
                from_node=ReviewedOpportunityNode,  # GraphNode instance
                to_node=OpportunityNode,  # GraphNode instance
                name="REVIEWS",
                description="Links review to opportunity",
            ),
            # ... more relationships
        ]

        return GraphSchema(
            root_model_class=ReviewedOpportunity,
            nodes=nodes,
            relations=relations,
        )
```

> **Why module-scope singletons?** `GraphRelation` validates that `from_node`
> and `to_node` are `GraphNode` instances (not raw classes). Defining them at
> module scope makes them easy to reference and ensures the same object is
> used in both `nodes=[...]` and `GraphRelation(from_node=..., to_node=...)`.

### Node Configuration

```python
GraphNode(
    node_class=Person,  # Pydantic class
    name_from="name",  # Field for display name (or callable)
    key_from="name",  # Primary key for dedup (or "AUTO_ID" or callable)
    description="Team member",  # Documentation
    index_fields=["name"],  # Fields to index for search
)
```

#### `key_from` Options

The `key_from` parameter is **critical** for deduplication. It determines the
PRIMARY KEY in the Kuzu table and controls when a MERGE creates a new node vs
updates an existing one.

| Value | Behaviour | Dedup? | Use when |
|-------|-----------|--------|----------|
| `"field_name"` | Use a Pydantic field as PK | ✅ Yes | Entity has a stable identifier (name, ID code, …) |
| `"AUTO_ID"` | Generate a UUID per extraction | ❌ No | Each occurrence must be kept separate |
| `lambda data, _: …` | Compute a key dynamically | ✅ Yes | Key requires logic (combination, normalisation) |

> **Deduplication tip:** When the same entity appears in multiple documents
> (e.g. two JSON files about the same opportunity), use a **field-based** or
> **lambda** key to ensure cross-document deduplication.  `"AUTO_ID"` creates
> a new node for every extraction — use it only for inherently unique objects
> like `TechnicalApproach`.

```python
# ✅ Good: opportunity_id is unique per opportunity
GraphNode(node_class=Opportunity, key_from="opportunity_id", …)

# ✅ Good: person name used for merging across documents
GraphNode(node_class=Person, key_from="name", …)

# ✅ Good: computed key for conditional dedup
GraphNode(
    node_class=RiskAnalysis,
    key_from=lambda data, _: (
        getattr(data.get("risk_category"), "name", None)
        or None   # Return None to skip this item entirely
    ),
)

# ❌ Avoid: AUTO_ID prevents dedup across documents
GraphNode(node_class=Customer, key_from="AUTO_ID", …)
```

**Returning `None` from a callable `key_from`** signals "skip this item" —
no node is created. This is useful for filtering out incomplete data (e.g.
risk entries without a valid category).

#### Unicode normalisation

All key values and display names pass through `_normalize_key()`, which
applies NFKC Unicode normalisation and replaces variant
dash/hyphen characters (U+2011 NON-BREAKING HYPHEN, U+2013 EN DASH, etc.)
with a plain ASCII hyphen-minus. This ensures that `"Gérard Lassalle‑Valier"`
(non-breaking hyphen) and `"Gérard Lassalle-Valier"` (regular hyphen)
merge as the same node.

#### Enum values in `name_from` / `key_from`

When working with enums, `model_dump()` keeps enum objects — it does **not**
serialise them to strings. Use `getattr(…, "name", …)` to get the member
name:

```python
GraphNode(
    node_class=RiskAnalysis,
    # ✅ Correct: getattr extracts the enum member name
    name_from=lambda data, _: (
        getattr(data.get("risk_category"), "name", None) or str(data.get("risk_category", "other_risk"))
    ),
    # ❌ Wrong: str(enum) gives "SWProjectRisks.ScheduleRisk"
    #   name_from=lambda data, _: str(data.get("risk_category")),
)
```

#### Embedding Structs in Parent Nodes

To avoid creating separate nodes for simple structs, use `extra_classes`:

```python
GraphNode(
    node_class=ReviewedOpportunity,
    extra_classes=[FinancialMetrics, KeyStatementOfWorkElement],
    name_from=lambda data, _: "Review:" + str(data.get("start_date")),
    key_from=lambda data, _: str(data.get("opportunity", {}).get("opportunity_id", "unknown")),
)
```

Properties from `FinancialMetrics` and `KeyStatementOfWorkElement` will be embedded as columns in the `ReviewedOpportunity` table.

### Relationship Configuration

`GraphRelation` takes `GraphNode` **instances** (not class objects) for
`from_node` and `to_node`. Use the module-scope singletons defined above:

```python
GraphRelation(
    from_node=opportunity_node,  # GraphNode instance
    to_node=customer_node,  # GraphNode instance
    name="HAS_CUSTOMER",  # Relationship type name
    description="Opportunity belongs to customer",
)
```

#### Automatic Path Deduction

The system auto-deduces relationship paths by traversing the Pydantic model
structure from the root class. The algorithm:

1. Collects all field paths where each node class appears in the model tree.
2. For each relationship, finds `(from_path, to_path)` candidate pairs.
3. Selects the **best** pair using a scoring heuristic that **prefers
   containment** (target nested inside source) over lateral/sibling paths.

For example, given this model structure:

```
ReviewedOpportunity          # root
  ├─ opportunity             # Opportunity
  │   └─ customer            # Customer
  │       └─ employees[]     # list[Person]
  └─ team[]                  # list[Person]
```

The deduction produces:

| Relationship | from_path | to_path | Why |
|---|---|---|---|
| `REVIEWS` (Root → Opportunity) | `""` | `opportunity` | Direct child of root |
| `HAS_CUSTOMER` (Opportunity → Customer) | `opportunity` | `opportunity.customer` | Customer is nested inside Opportunity |
| `HAS_CONTACT` (Customer → Person) | `opportunity.customer` | `opportunity.customer.employees` | `employees` is nested inside `customer` (containment wins) |
| `HAS_TEAM_MEMBER` (Root → Person) | `""` | `team` | Explicit `field_paths` overrides |

#### When to Use Explicit `field_paths`

Use `field_paths` when:
- The target type appears under **multiple** parents (e.g. `Person` appears
  as both `team` and `customer.employees`).
- The auto-deduced path is wrong (check the "Multiple valid paths" warning).
- The path goes through a non-standard structure (e.g. `delivery_info.locations`).

```python
GraphRelation(
    from_node=reviewed_opp_node,
    to_node=person_node,
    name="HAS_TEAM_MEMBER",
    # Tuple: (from_path, to_path) — "" means root
    field_paths=[("", "team")],
)

GraphRelation(
    from_node=reviewed_opp_node,
    to_node=geo_node,
    name="DELIVERED_IN",
    field_paths=[("", "delivery_info.locations")],
)
```

#### `explicitly_defined` Nodes

Some nodes are reachable only via multi-hop paths (e.g.
Root → Opportunity → Customer). If the auto-deduction can't
find a field path (because the node class differs from the one in the model),
set `explicitly_defined=True` to suppress the "orphan node" warning:

```python
GraphNode(
    node_class=Customer,
    name_from="name",
    key_from="name",
    explicitly_defined=True,  # Reached via Opportunity → Customer
)
```

> **Note:** `explicitly_defined` is set automatically for nodes coming from
> Neo4j import mappings.

### Relationship Properties (Edge Properties)

Properties stored on relationships rather than nodes use the `p_*_` naming
convention:

```python
class Partner(BaseModel):
    name: str
    p_role_: str | None = None  # → becomes "role" edge property
```

Fields prefixed with `p_` and suffixed with `_` are:
1. **Excluded** from the node table (not stored on the Partner node).
2. **Extracted** as properties on the relationship that connects to this node.
3. **Stripped** of the `p_` prefix and `_` suffix in the graph (e.g. `p_role_` → `role`).

```python
# In the graph, the Partner node has only 'name'.
# The HAS_PARTNER relationship has a 'role' property.
# MATCH (ro:ReviewedOpportunity)-[r:HAS_PARTNER]->(p:Partner)
# RETURN p.name, r.role
```

## Step 3: Configure KG Creation

**Location**: `config/workflows/*.yaml` (a `kg_build` pipeline step — `kg_configs` is
derived automatically by scanning every workflow's `with.kg_name` + `with.graph`,
there is no separate KG-config file; see [docs/workflows.md](workflows.md))

```yaml
workflows:
  my_kg:
    pipeline:
      - id: build
        run: kg_build
        with:
          kg_name: my_kg
          graph:
            factory: ekg_atos.schema.rainbow_review.ReviewedOpportunityGraph
            md_root: ${paths.rainbow_md}
            json_cache_root: ${paths.rainbow_json}
            include:
              - "*CNES*TMA*VENUS*"  # File glob pattern
            exclude:
              - "fake/*"            # Exclude test/fake data
            recursive: true         # Search subdirectories
```

### File Pattern Matching

The `include` patterns are **glob patterns** matched against filenames
(not full paths). Common pitfalls:

```yaml
# ✅ Correct: matches "03.RESM-SOL-9000559500 CNES TMA VENUS…"
include: ["*CNES*"]

# ❌ Wrong: underscores don't match spaces in filenames
include: ["*_CNES_*"]

# ✅ Match everything
include: ["*.json"]
```

Always add `exclude: ["fake/*"]` when test/fake data exists in the data
directory. Set `recursive: true` when files are in subdirectories.

### Import from Other KGs

Chain another `kg_build` pipeline step targeting the same `kg_name`, or set
`KgGraphConfig`'s `import`/`imports` field on the `graph:` block, to reuse another
KG's parquet cache:

```yaml
workflows:
  combined_kg:
    pipeline:
      - id: crm
        run: kg_build
        with:
          kg_name: combined_kg
          graph:
            factory: ekg_atos.schema.crm_export.CrmExtractGraph
            files: ['${paths.ekg_data}/crm_export/report.xlsx']
      - id: my_graph
        run: kg_build
        after: [crm]
        with:
          kg_name: combined_kg
          delete_first: false
          graph:
            factory: ekg_atos.schema.my_factory.MyGraph
            data_root: ${paths.my_data}
            include: ["*CNES*"]
            exclude: ["fake/*"]
            recursive: true
```

Imported KG data is loaded from parquet cache — the imported KG must have
been created first (either earlier in `--all-graphs` or manually).

## Step 4: Extract Data from Documents

```bash
cli baml extract \
  '${paths.rainbow_md}/real' \
  '${paths.rainbow_json}' \
  --function ExtractRainbow \
  --include "*CNES*.md" \
  --force
```

This runs LLM extraction on markdown files and generates one JSON file per document.

## Step 5: Build Knowledge Graph

```bash
# Build a specific KG
cli kg create my_kg

# Build all KGs defined as kg_* workflow profiles
cli kg create --all

# Open HTML visualization in browser
cli kg view
```

## Common Patterns

### Pattern 1: Simple Embedded Properties

For simple structs that should be embedded (not separate nodes):

**BAML**:
```baml
class ReviewedOpportunity {
  financials FinancialMetrics
}

class FinancialMetrics {
  tcv float?
  annual_revenue float?
}
```

**Factory**:
```python
GraphNode(
    node_class=ReviewedOpportunity,
    extra_classes=[FinancialMetrics],  # Embed into parent
    key_from=lambda data, _: str(data.get("opportunity", {}).get("opportunity_id", "unknown")),
)
```

**Result**: `ReviewedOpportunity` table has columns `tcv` and `annual_revenue`.

### Pattern 2: Separate Nodes with Relationships

For complex entities that should be separate nodes:

**BAML**:
```baml
class ReviewedOpportunity {
  partners Partner[]  // List of partners
}

class Partner {
  name string
  p_role_ string?    // Edge property (not on the node)
}
```

**Factory**:
```python
nodes = [
    GraphNode(node_class=ReviewedOpportunity, …),
    GraphNode(
        node_class=Partner,
        name_from="name",
        key_from="name",  # Dedup by name
    ),
]

relations = [
    GraphRelation(
        from_node=ReviewedOpportunity,
        to_node=Partner,
        name="HAS_PARTNER",
    ),
]
```

**Result**: Separate `Partner` nodes with `HAS_PARTNER` relationships. The
`p_role_` field becomes a `role` property on the `HAS_PARTNER` edge.

### Pattern 3: Relationship Properties

For storing properties on relationships (edge properties):

**BAML**:
```baml
class Competitor {
  name KnownCompetitor
  p_name_ string    // "p_" prefix + "_" suffix → edge property
  p_comment_ string?
}
```

**Factory**: Properties matching `p_*_` are automatically excluded from the
node and stored as properties on the inbound relationship.

**Result**:
```cypher
-- The Competitor node has only 'name'.
-- HAS_COMPETITOR relationship has 'name_' and 'comment' properties.
MATCH (ro)-[r:HAS_COMPETITOR]->(c:Competitor)
RETURN c.name, r.name_, r.comment
```

### Pattern 4: Cross-Factory Node Unification (Canonical Types)

To ensure nodes from different factories merge into the same Kuzu table, use
canonical types from `common_nodes.py`.

**When to use**: 
- Entity types that appear in multiple data sources (Customer, Geo, Person, …)
- Need deduplication across factories (e.g. same customer from BAML and CRM)

**Important**: The canonical wrapper class must have the **same `__name__`** as
the BAML type. The system matches node classes by `__name__`, not by Python
class identity — this allows `common_nodes.Customer` (which extends
`BamlCustomer`) and `BamlCustomer` to map to the same `Customer` table.

**Step 1**: Define canonical type in `common_nodes.py`:
```python
from ekg_atos.baml_client.types import Customer as BamlCustomer
from ekg_atos.baml_client.types import Partner as BamlPartner


class Customer(BamlCustomer):
    """Extended Customer with fields from multiple sources."""

    iris_code: str | None = None  # From Neo4j / CRM
    country: str | None = None


class Partner(BamlPartner):
    """Partner organization (canonical type for deduplication)."""

    # Unifies Neo4j TechnologyPartner and BAML Partner


class Geo(BamlGeo):
    """Canonical geographic location type."""

    # __name__ == "Geo" matches BamlGeo.__name__ — same table
```

**❌ WRONG**: Naming it differently breaks deduplication:
```python
class GeoLocation(BamlGeo):  # ❌ __name__ = "GeoLocation" ≠ "Geo"
    pass
```

**Step 2**: Import canonical types in factories that need cross-source dedup:
```python
# In rainbow_review.py
from ekg_atos.schema.common_nodes import Customer, Geo, Partner

# In stratnav.py (Neo4j TechnologyPartner → canonical Partner)
from ekg_atos.schema.common_nodes import Customer, Geo, Partner

# In crm_export.py
from ekg_atos.schema.common_nodes import Customer
```

**For Neo4j factories**: The `neo4j_label` parameter preserves the original
label while mapping to the canonical class:
```python
Neo4jNodeMapping(
    neo4j_label="TechnologyPartner",  # Label in the Neo4j export
    node_class=Partner,  # Canonical type (table = "Partner")
    key_field="name",
)
```

**Result**: All factories create nodes in the same table; Kuzu MERGE
deduplicates by primary key.

### Pattern 5: Filtering Items via `key_from` Returning None

When source data contains items that should not become nodes (e.g. incomplete
or invalid entries), return `None` from a callable `key_from`:

```python
GraphNode(
    node_class=RiskAnalysis,
    key_from=lambda data, _: (
        getattr(data.get("risk_category"), "name", None)
        if data.get("risk_category")
        else None  # ← Skip: no valid risk category
    ),
)
```

Items for which `key_from` returns `None` are silently omitted — no node is
created and no relationship endpoints reference them.

## Workflow Summary

### Initial Setup
1. Define BAML schema (`*.baml`)
2. Run `baml-cli generate`
3. Create factory class in `ekg/schema/`
4. Configure in `config/workflows/*.yaml` (a `kg_build` pipeline step)

### Iteration
1. **Modify extraction**: Edit BAML → regenerate → re-extract with `--force`
2. **Modify graph schema**: Edit factory → `cli kg create`
3. **Add fields**: Update BAML → regenerate → update factory → re-extract → rebuild

### Testing
```bash
# Build and view a test KG
cli kg create my_kg
cli kg view

# Regression: build all KGs
cli kg create --all
```

## Troubleshooting

### Issue: "Called Option::unwrap() on a None value"
**Cause**: BAML fingerprinting error after schema changes  
**Solution**: Sometimes happens with complex schemas. The JSON files are generated successfully despite the error.

### Issue: "Expression X has data type STRING but expected STRUCT"
**Cause**: Old JSON files don't have new fields  
**Solution**: Re-run extraction with `--force` to regenerate all JSON files

### Issue: "Cannot import name 'GEO' from baml_client.types"
**Cause**: Case mismatch — BAML generates `Geo` not `GEO`  
**Solution**: Fix imports to match generated class names (check `baml_client/types.py`)

### Issue: "Cannot find property X for n"
**Cause**: Schema evolved but old parquet caches exist  
**Solution**: Clear parquet caches and rebuild:
```bash
cli kg create --all --clear-all-caches
```
See [Cache Management](cache_management.md) for details.

### Issue: "No graphs are registered in the GraphRegistry"
**Cause**: Factory import failed (syntax error, missing dependency, etc.)  
**Solution**: The error message now shows which factories failed and why. Check the listed module paths and fix the import errors.

### Issue: "No files matched include patterns in …"
**Cause**: Glob pattern doesn't match any files in the data directory.  
**Solution**: Check actual filenames — common mistakes:
- Using underscores `*_CNES_*` when filenames have spaces `* CNES *`
- Missing `recursive: true` when files are in subdirectories
- Missing or wrong `data_root` path

### Issue: "Multiple valid paths found for RELATION_NAME"
**Cause**: A relationship can be inferred from multiple field paths.  
**Solution**: The system now prefers **containment** paths (target nested inside
source) over lateral/sibling paths. If the auto-chosen path is still wrong,
specify `field_paths` explicitly:
```python
GraphRelation(
    from_node=Customer,
    to_node=Person,
    name="HAS_CONTACT",
    field_paths=[("customer", "customer.employees")],
)
```

### Issue: Duplicate nodes despite same name
**Cause**: Unicode variants of the same character (e.g. regular hyphen `-` vs
non-breaking hyphen `‑`) produce different keys.  
**Solution**: This is now handled automatically — all keys pass through
Unicode normalisation. If you still see duplicates, check for other whitespace
or encoding differences in the source data.

### Issue: Orphan nodes (no relationships)
**Cause**: Typically a `key_from` mismatch between how the node was created
and how the relationship looks it up. For example, node created with
`key_from="AUTO_ID"` but the relationship tries to match by `name`.  
**Solution**: Use the same key strategy consistently. For entities that should
deduplicate, use `key_from="name"` or `key_from="some_id_field"`.

### Issue: `p_*_` fields not appearing as edge properties
**Cause**: The `merge_relationships_batch` function must handle edge properties
via row-by-row MERGE + SET (not batch LOAD FROM).  
**Solution**: This is now implemented correctly. Verify that the `p_*_` fields
are defined on the **target** node class of the relationship (the `to_node`).

## Best Practices

1. **Choose `key_from` carefully** — it determines deduplication. Use a stable
   business identifier (`opportunity_id`, `name`, `iris_code`) rather than
   `AUTO_ID` whenever possible.
2. **Use canonical types** from `common_nodes.py` for entities shared across
   factories (Customer, Opportunity, Person, Geo).
3. **Use `p_*_` prefix** for fields that belong on the relationship, not the node.
4. **Add `exclude: ["fake/*"]`** to the `graph:` block in your workflow YAML to skip test data.
5. **Start small**: Extract a few fields, verify, then expand.
6. **Use descriptions**: Help LLM understand what to extract.
7. **Specify `field_paths`** on relationships when the same target type appears
   under multiple parents.
8. **Run `cli kg create --all`** after any schema change as a regression
   test.
9. **Check warnings**: The warnings report (displayed in CLI output and saved
   to `*-warnings.md`) highlights schema issues.

## Reference Commands

```bash
# BAML generation
cd ekg_atos/ && baml-cli generate

# Extract data
cli baml extract SOURCE DEST --function ExtractRainbow --force

# Build graphs
cli kg create my_kg
cli kg create --all

# View graph
cli kg view

# Query graphs
cli kg cypher "MATCH (n:Customer) RETURN n.name LIMIT 10"

# Rebuild from scratch
cli kg create my_kg --delete-first
```
