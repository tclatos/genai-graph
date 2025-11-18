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
    Translate the given question into a valid Cypher query that respects the given graph schema.
    <SYNTAX>
    - Relationship directions are VERY important to the success of a query. Here's an example: If
    the relationship `HAS_CREATOR` is marked as `from` A `to` B, it means that B created A.
    - Use short, concise alphanumeric strings as names of variable bindings (e.g., `a1`, `r1`, etc.)    
    - When comparing string properties, ALWAYS do the following:
      - Lowercase the property values before comparison
      - Use the WHERE clause
      - Use the CONTAINS operator to check for presence of one substring in the other
    - DO NOT use APOC as the database does not support it.
    - For datetime queries, use the TIMESTAMP type, which combines the date and time.
    - Ensure all nodes, relationships and properties are conform to the given schema.
    </SYNTAX>

    <RETURN_RESULTS>
    - If the result is an integer, return it as an integer (not a string).
    - When returning results, return property values rather than the entire node or relationship.
    - Do not attempt to coerce data types to number formats (e.g., integer, float) in your results.
    - NO Cypher keywords should be returned by your query.
    </RETURN_RESULTS> """

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
