# Ideas around evolution of the Tk and Bleuprint


## Agents 



## Better React

Refactor Langchain agent creation and run code to use the new Agent Midleware feature
to simplify code. 
Typically refactor genai_tk/extra/agents/langgraph_agent_shell.py, genai_tk/utils/langgraph.py and genai_tk/extra/agents/commands_agents.py to simplify the trace of agents and tools execution.
To to so, create custom langchain middlewares (AgentMiddleware) to print (usinf Rich) the tool call and the agent execution output, and add them to the  agent.
You might change the user experience.

See https://docs.langchain.com/oss/python/langchain/middleware/custom and https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/types.py for the doc.
And/or call MCP tools "langchain-doc" or "Context7" for details.

To test, you can use "uv run cli agents react --chat" and enter a simple prompt.


- Use LangChain Midlewares to print tool calls, either in CLI or Streamlit




## Better HTML visualisation
- User can 
  - select the types of nodes and relationsips

- Use G.V()

## Hybrid search extension to genai_tk/core/embeddings_store.py
- use BM25S + Spacy (but configurable)
- call it RAG store ? 

## Optimize Markdown chunking
- A tester


# Import tables
- new command add-table

LOAD CSV WITH HEADERS FROM 'file:///data.csv' AS row
MATCH (existingNode {opportunity: toInteger(row.opportunity)})
CREATE (newNode:Person {name: row.name, age: toInteger(row.age)})
CREATE (existingNode)-[:RELATED_TO]->(newNode)


   - 
- new command relink


# Text2Cypher
- possibly Prune the schema with https://kuzudb.github.io/blog/post/improving-text2cypher-for-graphrag-via-schema-pruning/#pruned-graph-schema-results

## ReAct agents
- tools: 
    - graph_search()  (or cypher_run()  so the schema is known by agent)
    - doc_search()  (from Chonkie)
    - node_search()

## better llm / embeddings naming


## Better 'rag' commands
- pass a configurable chunker
https://docs.chonkie.ai/oss/pipelines 

##  Better KG


# To Test :
- ```uv run cli kg delete -f ; uv run cli kg add-doc --key cnes-venus-tma --subgraph ReviewedOpportunity ; uv run cli kg export-html -g ReviewedOpportunity```

- ```uv run cli baml extract $ONEDRIVE/prj/atos-kg/rainbow-md/cnes-venus-tma.md --function ExtractRainbow --force```

- ```uv run cli baml run FakeRainbow -i "Project for CNES; Marc Ferrer as sales lead in Atos team" --kvstore-key fake_cnes_1 --force```


- ```cli baml run FakeArchitectureJson -i "IT platform for CNES with 3-tier, Java based"  --kvstore-key fake-cnes-1```

- ```uv run cli kg delete -f ; uv run cli kg add-doc --key cnes-venus-tma --key fake_cnes_1 -g ReviewedOpportunity ; uv run cli kg add-doc --key fake-cnes-1 -g ArchitectureDocument; uv run cli kg export-html```

- ```cli kg schema```


uv run cli kg delete -f ; uv run cli kg add-doc --key fake-cnes-1 --subgraph ArchitectureDocument


uv run cli kg delete -f ; uv run cli kg add-doc --key cnes-venus-tma --g ReviewedOpportunity ; uv run cli kg add-doc --key fake-cnes-1 -g ArchitectureDocument; uv run cli kg export-html``

# Misc

Use https://github.com/GrahamDumpleton/wrapt for @once


# Doc to add in KG
- Sales presentations describing references (case studies)
- L1/L2 Offerings (from Nessie code ? GRD code ? ) 
- GTM conversations / BL Offerings ? 
- Win / Loss review
- RFQ
- Architecture document
- ....

