# Architecture Document Knowledge Graph Schema

## Overview

This document describes the graph schema for Software Architecture documents extracted using BAML from `genai_graph/ekg/baml_src/architecture_doc.baml`.

The schema models a Software Architecture as a knowledge graph with:
- **Nodes**: Representing architecture concepts (documents, projects, technologies, solutions)
- **Edges**: Representing relationships between these concepts
- **Edge Properties**: Enriching relationships with contextual information (e.g., purpose, role)

## Nodes

### 1. **SWArchitectureDocument** (Root Node)
Represents the complete architecture document.

**Properties:**
- `document_date`: Publication or last-update date (ISO 8601 format)
- `keywords`: List of extracted keywords/tags for search and similarity

**Purpose:** Serves as the root node aggregating all technical and solution information from the architecture document.

### 2. **Opportunity**
Represents the project or opportunity that the architecture addresses.

**Properties:**
- `name`: Project/opportunity identifier

**Purpose:** Links the architecture to its business context and project definition.

### 3. **TechnicalComponent**
Represents individual technologies, frameworks, platforms, tools, or infrastructure components.

**Properties:**
- `name`: Component name (e.g., "Python", "Docker", "PostgreSQL", "Kubernetes")
- `type`: Component category from `TechnicalComponentType` enum:
  - Languages
  - Frameworks
  - Frontend
  - Backend
  - Databases
  - Messaging
  - Cloud_providers
  - Containers
  - Orchestration
  - Serverless
  - Devops_tools
  - Iac_tools
  - Observability
  - Security_tools
  - Ai_ml
  - Authentication
  - Authorization
  - Networking
  - Other

**Purpose:** Represents the individual technology building blocks used in the architecture.

### 4. **Solution**
Represents specific products, managed services, or OSS solutions used in the architecture.

**Properties:**
- `name`: Product/solution name (e.g., "Azure API Management", "HashiCorp Vault")
- `vendor`: Organization providing the solution (e.g., "Microsoft", "HashiCorp")
- `type`: Category (e.g., "PaaS", "SaaS", "Database", "Message Broker")

**Purpose:** Represents higher-level solutions that may be composed of multiple technical components.

## Relationships (Edges)

### 1. **DOCUMENTS** (SWArchitectureDocument → Opportunity)
Architecture document describes an opportunity/project.

**Purpose:** Links the document to the project it describes.

---

### 2. **USES_TECHNOLOGY** (SWArchitectureDocument → TechnicalComponent)
Architecture includes this technology component.

**Edge Properties:**
- `p_purpose_`: The role of the component in the architecture (e.g., "message queue for async processing", "data persistence layer")

**Purpose:** Directly connects the architecture to technologies used.

---

### 3. **USES_SOLUTION** (SWArchitectureDocument → Solution)
Architecture leverages this solution.

**Edge Properties:**
- `p_purpose_`: The solution's role in the architecture (e.g., "API gateway and management", "secrets management")

**Purpose:** Directly connects the architecture to solutions used.

---

### 4. **REQUIRES_TECHNOLOGY** (Opportunity → TechnicalComponent)
Project/opportunity uses this technology.

**Edge Properties:**
- `p_purpose_`: Why and how the component is used (e.g., "backend runtime", "containerization platform")

**Purpose:** Links project requirements directly to technology choices.

---

### 5. **IMPLEMENTS_WITH** (Opportunity → Solution)
Project/opportunity is implemented using this solution.

**Edge Properties:**
- `p_purpose_`: The role of this solution (e.g., "managed database service", "CI/CD pipeline")

**Purpose:** Links project implementation to solutions.

---

### 6. **INTEGRATES_WITH** (TechnicalComponent → TechnicalComponent)
Technology integrates with or depends on another technology.

**Edge Properties:**
- `p_purpose_`: The nature of the integration (e.g., "data persistence for session storage", "client for message queue")

**Purpose:** Captures technology dependencies and integration patterns.

---

### 7. **BUILT_ON** (Solution → TechnicalComponent)
Solution is built on or uses this technology.

**Edge Properties:**
- `p_purpose_`: The relationship details (e.g., "runs on", "leverages", "built with")

**Purpose:** Shows composition of solutions from underlying technologies.

---

### 8. **INTEGRATES_WITH** (Solution → Solution)
Solution integrates with another solution.

**Edge Properties:**
- `p_purpose_`: The integration pattern (e.g., "message broker integration", "authentication provider")

**Purpose:** Captures solution-to-solution interactions.

---

## Edge Property Convention: `p_purpose_`

All BAML properties matching the pattern `p_*_` are automatically converted to edge properties in the knowledge graph. This allows rich semantic relationships.

**Example:**
```baml
class TechnicalComponent {
  name string 
  type TechnicalComponentType
  p_purpose_ string @description("role of the component in the architecture")
}
```

The `p_purpose_` field becomes a property on the edge, not a node property, enabling:
- Rich descriptions of relationships
- Semantic queries based on relationship purpose
- Graph traversal with contextual filters

## Query Examples

### Find all technologies in a project stack with their purposes:
```cypher
MATCH (doc:SWArchitectureDocument)-[r:USES_TECHNOLOGY]->(tech:TechnicalComponent)
RETURN tech.name, tech.type, r.p_purpose_
```

### Find technology integrations:
```cypher
MATCH (tech1:TechnicalComponent)-[r:INTEGRATES_WITH]->(tech2:TechnicalComponent)
RETURN tech1.name, tech2.name, r.p_purpose_
```

### Find all database solutions:
```cypher
MATCH (sol:Solution {type: 'Database'})
RETURN sol.name, sol.vendor
```

### Find what technologies a specific solution is built on:
```cypher
MATCH (sol:Solution {name: 'Azure App Service'})-[r:BUILT_ON]->(tech:TechnicalComponent)
RETURN tech.name, r.p_purpose_
```

### Find all opportunities and their technical requirements:
```cypher
MATCH (opp:Opportunity)-[r:REQUIRES_TECHNOLOGY]->(tech:TechnicalComponent)
RETURN opp.name, tech.name, tech.type, r.p_purpose_
```

## Graph Patterns

### Pattern 1: Complete Architecture Overview
```
SWArchitectureDocument 
  ├─ DOCUMENTS → Opportunity
  ├─ USES_TECHNOLOGY → TechnicalComponent(s)
  └─ USES_SOLUTION → Solution(s)
```

### Pattern 2: Technology Stack with Integration
```
TechnicalComponent (Language/Runtime)
  ├─ INTEGRATES_WITH → TechnicalComponent (Framework)
  ├─ INTEGRATES_WITH → TechnicalComponent (Database)
  └─ INTEGRATES_WITH → TechnicalComponent (Message Broker)
```

### Pattern 3: Solution Composition
```
Solution (Managed Service)
  └─ BUILT_ON → TechnicalComponent (underlying technologies)
```

### Pattern 4: Cross-Solution Integration
```
Solution (API Gateway)
  └─ INTEGRATES_WITH → Solution (Authentication Service)
```

## Deduplication Strategy

- **TechnicalComponent**: Deduplicated by `name` (e.g., "Python" is a single node regardless of mentions)
- **Solution**: Deduplicated by `name` (e.g., "Azure SQL Database" is a single node)
- **Opportunity**: Deduplicated by `name` (one project per name)

This prevents duplicate nodes while allowing multiple relationships with different purposes.

## Integration with BAML

The BAML schema (`architecture_doc.baml`) defines the extraction structure:

```baml
class TechnicalComponent {
  name string 
  type TechnicalComponentType
  p_purpose_ string @description("role of the component in the architecture")
}

class Solution {
  name string?
  vendor string?
  type string?
  p_purpose_ string? @description("Role/purpose within the architecture.")
}

class SWArchitectureDocument {
  opportunity Opportunity
  document_date string?
  technical_stack TechnicalComponent[]?
  used_solutions Solution[]?
  keywords string[]?
}
```

The Python subgraph implementation (`architecture_subgraph.py`) automatically:
1. Converts BAML classes to graph nodes
2. Deduces relationship paths from the object model
3. Extracts `p_purpose_` fields as edge properties
4. Provides sample Cypher queries for common use cases

## Implementation File

See `genai_graph/ekg/architecture_subgraph.py` for the complete Python implementation of this schema.
