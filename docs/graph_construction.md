# Knowledge Graph Construction Process

This document describes how knowledge graphs are constructed from multiple heterogeneous data sources.

## Related Documentation

- **[BAML Extraction Guide](baml_extraction_guide.md)** - Extract data from text documents using BAML
- **[Primary Key Implementation](primary_key_implementation.md)** - Node deduplication strategy
- **[Cache Management](cache_management.md)** - Parquet cache invalidation and rebuilds

## Overview

The KG construction pipeline combines data from multiple sources (Neo4j exports, databases, BAML extractions) into a unified Ladybug graph database. The process handles:

- **Type unification**: Different sources may define the same entity (e.g., `Account` vs `Customer`)
- **Data deduplication**: MERGE operations prevent duplicate nodes and relationships
- **Schema evolution**: Adding columns when importing from different schema versions
- **Batch processing**: Efficient import of large datasets via parquet files

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Neo4j Export   │     │  Database/Excel │     │  BAML Extract   │
│    (JSONL)      │     │   (Tables)      │     │    (JSON)       │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Neo4jFactory    │     │TableBackedFactory│    │JsonFileFactory  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   GraphSchema Merge    │
                    │  (dedupe by class name)│
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Ladybug Database      │
                    │  (MERGE from DataFrame)│
                    └────────────────────────┘
```

## Graph Factories

Factories convert source data into graph nodes and relationships. Four main types:

### 1. JsonFileBackedFactory

**Use case**: Process BAML-extracted JSON files from text documents (a prior `cli baml extract` run already produced the JSON)
**Documentation**: See [BAML Extraction Guide](baml_extraction_guide.md) for complete details

```python
class ReviewedOpportunityGraph(JsonFileBackedFactory, BaseModel):
    def get_model_class(self) -> type[BaseModel]:
        return ReviewedOpportunity
```

### 1b. MarkdownBamlFactory

**Use case**: Extract entities from Markdown via a BAML function **inline** (no separate JSON-extraction step) — the JSON result is cached automatically.
**Documentation**: See [Document Graph](document-graph.md#markdownbamlfactory-inline-baml-entity-extraction)

```python
class ReviewedOpportunityGraph(MarkdownBamlFactory):
    def extract_from_markdown(self, md_text: str) -> BaseModel: ...  # call your BAML function
```

### 2. Neo4jImportFactory

**Use case**: Import from Neo4j JSONL exports  
**Key feature**: Schema-based mappings with automatic type conversion

```python
class StratnavGraph(Neo4jImportFactory):
    def get_node_mappings(self) -> list[Neo4jNodeMapping]:
        return [
            Neo4jNodeMapping(
                neo4j_label="Account",  # Neo4j label
                node_class=Customer,  # Target Pydantic class
                key_field="name",  # Primary key
                property_mappings={  # Field mapping
                    "irisCode": "iris_code",
                    "subMarket": "segment",
                },
            ),
        ]
```

### 3. TableBackedFactory

**Use case**: Import from database tables or Excel files via pandas  
**Key feature**: Row-by-row transformation with custom mapper function

```python
class CrmExtractGraph(TableBackedFactory):
    def get_model_class(self) -> type[BaseModel]:
        return Opportunity
    
    def mapper_function(self, row: dict) -> Opportunity:
        return Opportunity(
            opportunity_id=row["Atos Opportunity ID"],
            customer=Customer(name=row["Account Name"]),
        )
```

## Common Nodes for Type Unification

Two modules work together to ensure entities are shared across factories:

- **`common_nodes.py`** — defines the canonical Pydantic **classes** (data schema)
- **`canonical_nodes.py`** — defines the canonical `GraphNode` **singletons** (graph configuration: primary key, name field, index fields)

**Key principle**: Nodes with the same class name write to the same Ladybug table, regardless of which factory imports them. The `canonical_nodes` singletons are the single source of truth for how each shared entity is stored.

### Currently defined canonical types

| GraphNode singleton | Class | Used by factories | Notes |
|---------------------|-------|-------------------|-------|
| `CustomerNode` | `Customer` | Stratnav (Account→Customer), RainbowReview, CRM | Extended with iris_code, country, etc. |
| `GeoNode` | `Geo` | Stratnav (GEO→Geo), RainbowReview | Geographic location |
| `PartnerNode` | `Partner` | Stratnav (TechnologyPartner→Partner), RainbowReview | Technology vendors, subcontractors |
| `OpportunityNode` | `Opportunity` | RainbowReview, CRM | Extended with lead, win_loss |
| `PersonNode` | `Person` | RainbowReview, CRM, Stratnav | Individual contacts and team members |
| `L3Node` | `L3` | Stratnav, StratnavSubset | Service catalog; ada-002 embedding pinned |

### Adding a new canonical type

When an entity appears in multiple data sources under different names
(e.g. Neo4j `TechnologyPartner` and BAML `Partner`), unify them:

1. **Create a canonical class** in `common_nodes.py` extending the BAML type:
```python
from ekg_atos.baml_client.types import Partner as BamlPartner


class Partner(BamlPartner):
    """Partner organization (canonical type for deduplication)."""
```

2. **Create a `GraphNode` singleton** in `canonical_nodes.py`:
```python
from ekg_atos.schema.common_nodes import Partner
from genai_graph.kg.schema import GraphNode

PartnerNode: GraphNode = GraphNode(
    node_class=Partner,
    name_from="name",
    key_from="name",
    description="Partner organization (technology vendor, subcontractor, etc.)",
)
```

3. **Import the singleton** in all factories (not the raw class):
```python
from ekg_atos.schema.canonical_nodes import CustomerNode, PartnerNode
```

4. **Map the Neo4j label** to the canonical class (Neo4j factories only):
```python
Neo4jNodeMapping(
    neo4j_label="TechnologyPartner",  # Original Neo4j label
    node_class=Partner,               # Canonical class (table name = "Partner")
    ...
)
```

**Critical**: The canonical class `__name__` must match the BAML type's `__name__` (e.g. both are `"Partner"`), so all factories write to the same Ladybug table.

### Example: Customer

```python
# common_nodes.py
from ekg_atos.baml_client.types import Customer as BamlCustomer


class Customer(BamlCustomer):
    """Extended Customer with fields from multiple sources."""

    # Fields from Neo4j/Stratnav import (Account)
    iris_code: str | None = Field(default=None)
    country: str | None = Field(default=None)
    business_line: str | None = Field(default=None)

    # Fields from BAML extraction
    location: Geo | None = None
    services: list[L3] = Field(default_factory=list)


# canonical_nodes.py
CustomerNode: GraphNode = GraphNode(
    node_class=Customer,
    name_from="name",
    key_from="name",
    description="Customer organization details",
    index_fields=["name"],
    explicitly_defined=True,
)
```

**Usage in factories**:
```python
# Import the GraphNode singleton, not the raw class
from ekg_atos.schema.canonical_nodes import CustomerNode, GeoNode, PartnerNode, OpportunityNode

# Use in build_schema()
nodes = [OpportunityNode, CustomerNode, PartnerNode]
relations = [
    GraphRelation(from_node=OpportunityNode, to_node=CustomerNode, name="HAS_CUSTOMER"),
]
```

**Benefits**:
- **Deduplication**: Same customer from different sources → single node
- **Schema evolution**: New fields added without breaking existing data
- **Type safety**: Pydantic validation across all sources
- **Consistency**: Primary key, name field, and indexes defined once, shared everywhere

## Schema Merging

When combining multiple graph factories, the `GraphRegistry.build_combined_schema()` method:

1. **Deduplicates nodes by class name** (not class identity)
2. **Deduplicates relationships by (from_name, to_name, rel_name)**
3. **Preserves metadata** from the first-seen definition (descriptions, index fields)
4. **Validates consistency** across merged schemas

```python
# Merge example - stratnav_subset_rainbow_crm
#   - StratnavGraph defines: Customer (from Neo4j Account)
#   - ReviewedOpportunityGraph defines: Customer (from BAML extraction)
#   - Result: Single Customer node type with combined properties
```

Nodes and relationships are identified by name, not Python class identity. This allows different factories to contribute to the same graph structure.

## Data Merging with Ladybug

The ingest layer uses Ladybug's `MERGE` operations for deduplication. Primary keys determine when to create vs. update nodes.

### Nodes
```cypher
LOAD FROM df
MERGE (n:Customer {name: name})
ON CREATE SET n.country = country, n.segment = segment
ON MATCH SET n.country = country, n.segment = segment
```

- **First import**: Creates node with all properties
- **Subsequent imports**: Updates properties on existing node
- **Key selection**: See [Primary Key Implementation](primary_key_implementation.md)

### Relationships

Relationships use row-by-row MERGE + SET to support edge properties:

```cypher
-- Without edge properties
MATCH (a:Customer {name: $from_id}), (b:Person {name: $to_id})
MERGE (a)-[:HAS_CONTACT]->(b)

-- With edge properties (from p_*_ fields)
MATCH (a:ReviewedOpportunity {id: $from_id}), (b:Partner {name: $to_id})
MERGE (a)-[r:HAS_PARTNER]->(b)
SET r.role = $role
```

Relationships are unique by (from_node, to_node, rel_type). Edge properties
(from `p_*_` fields on the target node class) are stored on the relationship.

## Import/Export via Parquet

KG configurations can import from other configurations via parquet cache, enabling incremental builds. `KgProfileConfig.imports` (a list of KG names, aliased `import` in YAML) is the underlying field; a project populates it inside a `graph:`/`graphs:` block of one of its own workflow pipeline steps (see [Configuration](#configuration) below) rather than a separate config file.

**Import process**:
1. **Recursively creates schemas** from imported KG configurations
2. **Detects schema changes** and adds missing columns via `ALTER TABLE ADD`
3. **Converts array types** (numpy → Python lists) for Ladybug compatibility
4. **Imports data** using MERGE from parquet files (nodes first, then relationships)

**Benefits**:
- **Faster iteration**: Avoid re-processing unchanged data sources
- **Modular composition**: Combine pre-built graphs
- **Schema evolution**: Handles backward compatibility automatically

**Cache location**: `~/kg_outputs/{kg_name}/parquet/`

## Configuration

There is no standalone KG-config file. `KgManager.from_global_config()` derives
`kg_configs` (a `dict[str, KgProfileConfig]`) by scanning **every workflow's**
`pipeline`/`steps` for a `with.kg_name` + `with.graph`/`with.graphs` entry — see
[docs/workflows.md](workflows.md). A project defines its KGs as workflow pipeline
steps in its own `config/workflows/*.yaml`:

```yaml
# a project's config/workflows/graph_construction.yaml
workflows:
  one_rainbow_with_db:
    pipeline:
      - id: rainbow
        run: kg_build
        with:
          kg_name: one_rainbow_with_db
          graph:
            factory: ekg_atos.schema.rainbow_review.ReviewedOpportunityGraph
            md_root: '${paths.rainbow_md}'
            json_cache_root: '${paths.rainbow_json}'
            include: ['*CNES*TMA*VENUS*']
            exclude: ['fake/*']
      - id: crm
        run: kg_build
        after: [rainbow]
        with:
          kg_name: one_rainbow_with_db
          delete_first: false
          graph:
            factory: ekg_atos.schema.crm_export.CrmExtractGraph
            files: ['${paths.ekg_data}/crm_export/report.xlsx']
```

**`graph:` fields** (factory-specific; the ones shown are common to the Markdown-backed
factories):
- `factory`: dotted path to the factory class
- `md_root` / `data_root`: base directory the factory reads from
- `include` / `exclude`: glob patterns
- `recursive`: search subdirectories

See [docs/workflows.md](workflows.md) for the full DSL (`pipeline:`, `run:`, `with:`,
`after:`, presets, `--set` overrides).

## CLI Commands

```bash
# Create default KG (uses kg_config from config)
cli kg create

# Create one or more specific KGs
cli kg create stratnav_subset_rainbow_crm
cli kg create rainbow_add_crm
cli kg create stratnav_subset_rainbow_crm

# Create all KGs defined as kg_* workflow profiles
cli kg create --all

# force: re-ingest even if parquet fingerprints match
cli kg create my_kg --force parquet

# Clear all parquet caches (fixes struct field-order mismatches)
cli kg create my_kg --clear-all-caches

# View schema (reads the auto-generated schema artifact)
cli kg schema

# Execute Cypher queries (against the active KG profile)
cli kg cypher "MATCH (c:Customer) RETURN c.name LIMIT 10"
cli kg cypher "MATCH ()-[r]->() RETURN type(r), count(r)"

# Natural-language query (Text-to-Cypher)
cli kg query "Which customers have the most opportunities?"

# Open HTML visualization in browser
cli kg view

# Display database info and statistics
cli kg info
```

## Data Source Priority

Sources are processed in order defined in the configuration. Data merging behavior:

- **MERGE operations**: Update existing nodes/relationships with new properties
- **Primary keys**: Determine when to create vs. update (see [Primary Key Implementation](primary_key_implementation.md))
- **Property updates**: Later sources update properties on existing nodes
- **No overwrites**: Existing non-null values are preserved unless explicitly configured

**Typical order**:
1. **Neo4j imports** - Curated enterprise data (e.g., service catalog, customer master)
2. **Database/Excel imports** - CRM exports, operational data
3. **BAML extractions** - LLM-extracted data from documents (fills gaps, adds context)

**Example**: If Neo4j defines `Customer.country` and BAML extraction doesn't, the Neo4j value persists.

## Warnings and How to Handle Them

### Structured Warnings Report

As of the latest version, KG creation generates a **comprehensive warnings report** in Markdown format. This report provides better visibility into cross-graph issues and categorizes warnings with actionable suggestions.

**Report Location**: `{kg_outputs}/{profile}-{tag}-warnings.md`

The report includes:
- **Categorized warnings**: Duplicate relationships, missing nodes, orphaned nodes, schema failures
- **Structured tables**: Easy-to-scan summary of issues
- **Actionable suggestions**: Specific recommendations for each category
- **Cross-graph detection**: Spots issues spanning multiple subgraph definitions

**Access methods**:
1. View the report file directly
2. Follow the link in `{profile}-{tag}-info.md`
3. Check CLI output at the end of KG creation

Warnings are printed at the end of `cli kg create` and saved to `{kg_outputs}/{profile}-{tag}-warnings.md`.

### Common Warning Types

During KG creation, various warnings may appear. Here's a reference guide:

### Schema Warnings

#### "No graphs are registered in the GraphRegistry"
```
No graphs are registered in the GraphRegistry.
The following factories failed to load:
  - ekg_atos.schema.my_factory.MyGraph: ImportError: cannot import name ...
```
**Cause**: Factory import failed due to syntax error, missing dependency, or wrong module path.  
**Solution**: Check the listed module paths and fix the import errors shown in the message.

#### "Multiple valid paths found for RELATION_NAME"
```
Multiple valid paths found for HAS_CONTACT (Customer → Person). 
Using: customer → customer.employees. Alternatives: customer → lead.
```
**Cause**: A relationship can be inferred from multiple field paths in the Pydantic model.  
**Solution**: The system auto-selects the best path using a **containment-first** heuristic:
1. **Containment preferred**: If the target path starts with the source path (e.g. `customer.employees` starts with `customer`), this path is preferred over lateral paths.
2. **Shallow depth as tiebreaker**: Among containment-equivalent paths, shallower ones are preferred.

If the auto-chosen path is still wrong, specify `field_paths` explicitly as a list of `(from_path, to_path)` tuples:
```python
GraphRelation(
    from_node=customer_node,  # GraphNode instance
    to_node=person_node,  # GraphNode instance
    name="HAS_CONTACT",
    field_paths=[("customer", "customer.employees")],  # Explicit path
)
```

#### "Multiple relationships defined between X and Y"
```
Multiple relationships defined between L3 and L3: SIMILAR_TO, CROSS_SELL
```
**Cause**: Multiple relationship types exist between the same node types.  
**Solution**: This is expected for rich schemas. No action needed unless relationships should be consolidated.

#### "No field paths found for X in the root model structure"
```
No field paths found for Customer in the root model structure; this node may be orphaned.
```
**Cause**: A node type cannot be reached by traversing fields from the root Pydantic model.  
**Solution**: 
- For **BAML/JSON factories**: Ensure the node type is reachable from the root model
- For **Neo4j factories**: Set `explicitly_defined=True` on `GraphNode` (done automatically for Neo4j imports)
- For **combined schemas**: This warning is suppressed for explicitly-defined nodes

#### "Class X is referenced in relationships but has no GraphNode"
```
Class Partner is referenced in relationships but has no GraphNode
```
**Cause**: A relationship references a node class that wasn't configured.  
**Solution**: Add a `GraphNode` configuration for the missing class.

### Embedded Struct Warnings

#### "Embedded class X is not referenced on Y"
```
Embedded class Financials is not referenced on ReviewedOpportunity
```
**Cause**: A class listed in `extra_classes` isn't actually a field on the parent node.  
**Solution**: Verify the class is referenced as a field, or remove it from `extra_classes`.

#### "Embedded field 'X' on class Y has incompatible type"
```
Embedded field 'financials' on class ReviewedOpportunity has incompatible type list[Financials]
```
**Cause**: Embedded structs must be single objects, not lists.  
**Solution**: Use `Optional[Financials]` instead of `list[Financials]` for embedded fields.

### Import Warnings

#### "Failed to import X nodes/relationships: Cannot find property Y"
**Cause**: Schema mismatch between parquet data and current schema definition.  
**Solution**: Clear parquet caches and rebuild source graphs:
```bash
cli kg create target_graph --clear-all-caches
```
See [Cache Management](cache_management.md) for details.

#### "Failed to import X relationships: Binder exception"
**Cause**: Missing relationship properties in parquet due to schema evolution.  
**Solution**: Same as above - rebuild source graphs with current schema.

#### "Scanning of type <class 'numpy.ndarray'> has not been implemented"
**Cause**: Parquet contains numpy arrays that need conversion.  
**Solution**: System auto-converts numpy arrays to Python lists. If persists, clear parquet caches.

### Document Processing Warnings

#### "Subgraph root model must expose a 'metadata' map field"
```
Subgraph root model 'MyModel' must expose a 'metadata' map field
```
**Cause**: The root Pydantic model doesn't have a `metadata: dict` field.  
**Solution**: Add `metadata: dict[str, str] | None = None` to your root model.

## Suppressing Warnings

Schema validation warnings are informational and don't block execution. To suppress during automated builds:

```python
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Graph schema validation:")
```

Critical errors (schema mismatches, import failures) raise exceptions and must be fixed.

## Development Workflow

### Initial Setup
1. Define data sources (Neo4j exports, databases, BAML extractions)
2. Create factory classes for each source
3. Define canonical types in `common_nodes.py`
4. Wire the factory into a `kg_build` pipeline step in `config/workflows/*.yaml`
5. Build and validate

### Adding New Data Source
1. Create factory class (extends appropriate base factory)
2. Implement required methods (`build_schema()`, `get_struct_data_by_key()`/`get_keys()`)
3. Import canonical types from `common_nodes.py`
4. Add a pipeline step for it in `config/workflows/*.yaml`
5. Test with sample data
6. Build full graph

### Iteration
1. Modify source data → rebuild affected KG
2. Modify schema → rebuild from scratch (`--delete-first`)
3. Add new factory → add to config → rebuild

### Debugging
```bash
# Activate the profile, then inspect it
cli kg create my_kg --dry-run   # or a prior 'cli kg create my_kg' run

# Check schema
cli kg schema

# Query node counts
cli kg cypher "MATCH (n) RETURN labels(n)[0], count(n)"

# Inspect relationships
cli kg cypher "MATCH ()-[r]->() RETURN type(r), count(r)"

# View in browser
cli kg view
```

## Performance Considerations

- **Batch processing**: Data loaded via pandas DataFrames (not row-by-row)
- **Parquet caching**: Avoids re-processing unchanged sources
- **Index fields**: Specified in `GraphNode.index_fields` for faster queries
- **MERGE efficiency**: Primary keys should be indexed fields

For large datasets (>100K nodes), consider:
- Breaking into smaller KG configurations
- Using parquet import/export extensively
- Optimizing primary key selection
