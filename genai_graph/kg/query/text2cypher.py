import pandas as pd
from genai_tk.core.factories.llm_factory import get_llm
from genai_tk.core.prompts import def_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from loguru import logger

from genai_graph.kg.backend import create_backend_from_config
from genai_graph.kg.manager import get_kg_manager

# taken from https://kuzudb.github.io/blog/post/improving-text2cypher-for-graphrag-via-schema-pruning/

SYSTEM_PROMPT = """  
Translate the given question into a single, valid Cypher statement that respects the provided graph schema.

- You MUST use ONLY the node labels, relationship types and properties that are literally listed in the schema above; inventing new ones is forbidden.  

- For every keyword in the question map it to the **real** labels that contain that keyword, then build the **shortest valid path(s)** (≤ 4 hops) from the anchor node that owns the filter property; if several paths exist return them with `UNION`.

- Start EVERY query with MATCH (or OPTIONAL MATCH) and finish with RETURN; no leading/trailing text.  
  **Exception**: when using vector similarity search, start with CALL QUERY_VECTOR_INDEX (see the vector-search section below).

- Relationship directions are VERY important. If the relationship HAS_CREATOR is documented “from A to B”, it means B created A.  
  For clarity: (a)-[:R]->(b) always reads “a → b”, so (ro)-[:HAS_COMPETITOR]->(comp) means “the ReviewedOpportunity lists comp as a competitor”.

- Relationship syntax is **always**  
  `(a)-[:TYPE]->(b)` or `(a)<-[:TYPE]-(b)`;  
  the arrow is **outside** the brackets.  
  Illegal forms: `[:TYPE<]` `[:TYPE>]` `[:TYPE<>]`.  
  Memorise this cheat-sheet fragment:  
  MATCH (c:Customer)<-[:HAS_CUSTOMER]-(o:Opportunity)

- Use short, meaningful, alphanumeric variable names (2-4 chars) that hint at the entity.  

- When comparing string properties ALWAYS:  
  – lower-case both sides with toLower()  
  – use the WHERE clause  
  – use CONTAINS (not =)  

- DO NOT use APOC; the database does not support it.  

- For datetime queries use the DATE or TIMESTAMP type.  
  When the user asks for “after <date>”, translate to  
  date(o.start_date) > date('YYYY-MM-DD’)  
  (or ro.document_date, whichever field is present).  
  Never compare an opportunity_id string to a date literal.

- Ensure all node labels, relationship types and properties exist in the schema.
- If you need a value that belongs to a related node, always traverse the relationship first and read the property from the target node variable. Never use dot-notation on the anchor node to reach “through” the relationship (e.g. avoid anchor.rel.field); it will fail.
</SYNTAX>
<VECTOR_SEARCH>
When the user's question is about **semantic similarity** — asking for items
"similar to", "related to", "about", or matching a natural-language description
(e.g. "offerings around web security", "approaches similar to cloud migration") —
use the Ladybug vector search function instead of string CONTAINS filters.

Available vector indexes are listed in the '### Vector-Indexed Fields' section of the schema.
Pick the index whose source field best matches the user intent.

Syntax:
  CALL QUERY_VECTOR_INDEX('<Table>', '<index_name>', $query_vector, <k>)
  WITH node AS <var>, distance
  [MATCH (<var>)-[:REL]->(...)]
  RETURN ... ORDER BY distance LIMIT <n>

Rules:
- `$query_vector` is a **runtime parameter** — write it literally as `$query_vector`.
  The system will automatically embed the user's question and inject it.
- `<k>` is the number of nearest neighbours to retrieve from the index (use 10–20
  for exploration, or a smaller number when the user asks for a specific count).
- The CALL returns two variables: `node` (the matched node object) and `distance`
  (cosine distance, lower = more similar).
- Always alias `node` with `WITH node AS <var>` before continuing with MATCH
  clauses so you can traverse the graph from the matched nodes.
- Always `ORDER BY distance` and add a `LIMIT`.
- You can combine vector search with graph traversal to find related entities,
  e.g. find similar L3 offerings then follow relationships to their opportunities.
- If the '### Vector-Indexed Fields' section is absent or empty, fall back to standard
  MATCH + CONTAINS queries.

Example — find offerings semantically similar to the user's question:
  CALL QUERY_VECTOR_INDEX('L3', 'description_index', $query_vector, 10)
  WITH node AS l3, distance
  RETURN DISTINCT l3.name, l3.description, distance
  ORDER BY distance LIMIT 10

Example — vector search followed by graph traversal:
  CALL QUERY_VECTOR_INDEX('L3', 'description_index', $query_vector, 10)
  WITH node AS l3, distance
  MATCH (l3)<-[:HAS_L3]-(l2)
  RETURN DISTINCT l3.name, l2.name, distance
  ORDER BY distance LIMIT 10
</VECTOR_SEARCH>
<RETURN_RESULTS>
- If the result is an integer, return it as an integer (not a string).
- When returning results, return property values rather than the entire node or relationship.
- Do not attempt to coerce data types to number formats (e.g., integer, float) in your results.
- NO Cypher keywords should be returned by your query.
- Reply with the raw Cypher statement only; do not wrap it in ```cypher … ``` or any markdown.
- When you need a field that lives inside an embedded object (e.g. `financials.tcv`, `competition.comment`)  
  or on a relationship property, return it with dot-notation **without back-ticks**:  
  `ro.financials.tcv` or `hc.comment` if the relationship is bound as `hc`.
- Append `LIMIT 30` to every query unless the user explicitly asks for a different number.
- Return only distinct rows: start the RETURN clause with `RETURN DISTINCT` unless the user explicitly asks for duplicates.
"""

USER_PROMPT = """

    <SCHEMA>
{schema}
    </SCHEMA>

    <QUESTION>
    {question}
    </QUESTION>

    <OUTPUT>: 
    Valid Cypher query, conform to schema, with no newlines: 
"""


def _get_schema_from_file(kg_config_name: str | None = None) -> str:
    """Read the schema description from the stored schema file.

    The schema file is automatically created during graph creation.
    """
    manager = get_kg_manager()
    profile = kg_config_name if kg_config_name is not None else manager.profile
    schema_path = manager.get_schema_path_for(profile)
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema file not found at {schema_path}. Run 'cli kg create' or 'cli kg schema --regen --kg {profile}' to generate the schema."
        )
    return schema_path.read_text(encoding="utf-8")


_QUERY_VECTOR_PARAM = "$query_vector"


def _ensure_vector_indexes(kg_config_name: str | None, backend: object) -> None:
    """Create all vector indexes declared in the schema JSON for this KG profile.

    Loads ``VectorIndexInfo`` entries from the saved canonical schema JSON so that
    import-only profiles (e.g. ``stratnav_subset_rainbow_crm``) that never ran
    ``create_vector_indexes_task`` get their indexes created on first query.

    Falls back silently if the JSON file is absent or the backend doesn't support
    vector indexing.
    """
    if not hasattr(backend, "create_vector_index"):
        return
    manager = get_kg_manager()
    profile = kg_config_name if kg_config_name is not None else manager.profile
    json_path = manager.get_schema_json_path_for(profile)
    if not json_path.exists():
        return
    try:
        from genai_graph.kg.schema.resolved import ResolvedSchema

        resolved = ResolvedSchema.from_json_file(str(json_path))
        for vi in resolved.vector_indexes:
            try:
                backend.create_vector_index(vi.table, vi.embedding_column, vi.index_name, metric="cosine")
            except Exception as exc:
                logger.debug("Vector index {}.{}: {}", vi.table, vi.index_name, exc)
    except Exception as exc:
        logger.debug("Could not load vector indexes from schema JSON: {}", exc)


def _embed_query_vector(cypher_query: str, question: str) -> dict[str, list[float]] | None:
    """If the generated Cypher contains $query_vector, compute the embedding.

    Returns a parameter dict ``{"query_vector": [...]}`` when the placeholder
    is present, or *None* when no vector search is used.
    """
    if _QUERY_VECTOR_PARAM not in cypher_query:
        return None

    from genai_tk.config_mgmt.config_mngr import global_config

    from genai_graph.kg.embeddings_handler import EmbeddingsHandler

    try:
        embeddings_id: str | None = global_config().get_str("kg_build.embeddings.default")
    except Exception:
        embeddings_id = None

    if not embeddings_id:
        raise RuntimeError(
            "Cannot compute query embedding: no default embeddings model configured in kg_build.embeddings.default"
        )

    handler = EmbeddingsHandler(embeddings_id=embeddings_id)
    vector = handler.compute_embeddings(question)
    logger.info("Computed query embedding ({} dims) for vector search", len(vector))
    return {"query_vector": vector}


def text2cypher_chain(question: str, llm: str | None = None, kg_config_name: str | None = None) -> Runnable:
    """Generate system and user prompts for text to Cypher conversion.

    Args:
        question: The user's question in natural language.
        llm: Optional LLM identifier to use for generation.
        kg_config_name: Optional KG config name to load schema from.
    """
    prompt = {
        "question": RunnableLambda(lambda _: question),
        "schema": RunnableLambda(lambda _: _get_schema_from_file(kg_config_name)),
    } | def_prompt(system=SYSTEM_PROMPT, user=USER_PROMPT)
    # Only pass llm to get_llm() if explicitly provided; otherwise use default
    if llm is not None:
        return prompt | get_llm(llm=llm) | StrOutputParser()
    else:
        return prompt | get_llm() | StrOutputParser()


def query_kg(query: str, llm: str | None = None, kg_config_name: str | None = None) -> pd.DataFrame:
    """Generate a Cypher query from a natural language query and execute it against the knowledge graph.

    Args:
        query: The user's question in natural language.
        llm: Optional LLM identifier to use for generation.
        kg_config_name: Optional KG config name to query. Defaults to the active manager profile.
    """
    manager = get_kg_manager()
    profile = kg_config_name if kg_config_name is not None else manager.profile
    backend = create_backend_from_config("default", profile)
    if not backend:
        raise Exception("EKG database not found")

    # Ensure vector extension is loaded (needed if LLM generates vector search queries)
    if hasattr(backend, "ensure_vector_extension"):
        backend.ensure_vector_extension()

    cypher_query = text2cypher_chain(query, llm=llm, kg_config_name=kg_config_name).invoke({})
    logger.info("Generated Cypher query: {}", cypher_query)

    # Create any vector indexes declared in the schema JSON (structured, not regex-based)
    _ensure_vector_indexes(kg_config_name, backend)

    # Detect $query_vector and compute embedding if needed
    params = _embed_query_vector(cypher_query, query)

    try:
        result = backend.execute(cypher_query, parameters=params) if params else backend.execute(cypher_query)
        df = result.get_as_df()
    except Exception as e:
        raise RuntimeError(f"Error in Cypher command execution: {cypher_query}\nException:{e}") from e
    return df


if __name__ == "__main__":
    # Quick test
    query = "List the names of all competitors for opportunities created after January 1, 2012."
    df = query_kg(query, llm=None)
    print(df)
