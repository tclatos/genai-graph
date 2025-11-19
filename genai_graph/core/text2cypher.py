import pandas as pd
from genai_tk.core.llm_factory import get_llm
from genai_tk.core.prompts import def_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from loguru import logger

from genai_graph.core.graph_backend import create_backend_from_config
from genai_graph.core.schema_doc_generator import (
    generate_combined_schema_markdown,
    generate_schema_markdown,
)

# taken from https://kuzudb.github.io/blog/post/improving-text2cypher-for-graphrag-via-schema-pruning/

SYSTEM_PROMPT = """  
Translate the given question into a single, valid Cypher statement that respects the provided graph schema.

<SYNTAX>
- Reply with the raw Cypher statement only; do not wrap it in ```cypher … ``` or any markdown.  
- Start EVERY query with MATCH (or OPTIONAL MATCH) and finish with RETURN; no leading/trailing text.  

- Relationship directions are VERY important. If the relationship HAS_CREATOR is documented “from A to B”, it means B created A.  
  For clarity: (a)-[:R]->(b) always reads “a → b”, so (ro)-[:HAS_COMPETITOR]->(comp) means “the ReviewedOpportunity lists comp as a competitor”.

- Use short, concise, alphanumeric variable names (e.g.  a1, r1, hc).  

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
</SYNTAX>

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


def _schema_markdown_for_subgraphs(subgraphs: list[str]) -> str:
    """Return schema Markdown for one or more subgraphs.

    If multiple (or zero) subgraphs are provided, a combined schema is
    generated. When exactly one subgraph name is given, the single-subgraph
    documentation is used for backwards-compatible behavior.
    """
    if not subgraphs or len(subgraphs) > 1:
        # Empty list means "all registered" for the combined generator
        return generate_combined_schema_markdown(subgraphs)
    return generate_schema_markdown(subgraphs[0])


def text2cypher_chain(question: str, subgraphs: list[str], llm_id: str | None = None) -> Runnable:
    """Generate system and user prompts for text to Cypher conversion.

    Args:
        question: The user's question in natural language.
        subgraphs: Names of the subgraphs whose combined schema should be used.
    """
    prompt = {
        "question": RunnableLambda(lambda _: question),
        "schema": RunnableLambda(lambda _: _schema_markdown_for_subgraphs(subgraphs)),
    } | def_prompt(system=SYSTEM_PROMPT, user=USER_PROMPT)
    return prompt | get_llm(llm_id=llm_id) | StrOutputParser()


def query_kg(query: str, subgraphs: list[str], llm_id: str | None = None) -> pd.DataFrame:
    """Generate a Cypher query from a natural language query and execute it against the knowledge graph.

    Args:
        query: The user's question in natural language.
        subgraphs: Names of the subgraphs whose combined schema should be used.
    """
    backend = create_backend_from_config("default")
    if not backend:
        raise Exception("EKG database not found")
    cypher_query = text2cypher_chain(query, subgraphs, llm_id=llm_id).invoke({})
    logger.info(f"Generated Cypher query: {cypher_query}")
    try:
        result = backend.execute(cypher_query)
        df = result.get_as_df()
    except Exception as e:
        raise RuntimeError(f"Error in Cypher command execution: {cypher_query}\nException:{e}") from e
    return df


if __name__ == "__main__":
    # Quick test
    query = "List the names of all competitors for opportunities created after January 1, 2012."
    df = query_kg(query, subgraphs=["ReviewedOpportunity"], llm_id=None)
    print(df)
