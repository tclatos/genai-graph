# Ideas around evolution of the Tk and Bleuprint

## Better HTML visualisation
- User can 
  - select the types of nodes and relationsips

- Use G.V()

## Hybrid search extension to genai_tk/core/embeddings_store.py
- use BM25S + Spacy (but configurable)
- call it RAG store ? 

## Optimize Markdown chunking
- use https://docs.chonkie.ai/oss/chefs/markdownchef ,  https://pypi.org/project/chonkie/
- custom Loader ? 
- "write a Markdown file chunker, inheriting LangChain Loader.  It takes a list of Markdown file as input, and chunk them using the Chonkie package https://docs.chonkie.ai/oss/chefs/markdownchef ,  https://pypi.org/project/chonkie/, https://docs.chonkie.ai/oss/chunkers/table-chunker, ... . Set filename in metadata.  Use Context7 to get Chonkie usage. Makes parameters such as cheun siez configurable, but provide common default values for that kind of file" 


# Import tables
- new command add-table

LOAD CSV WITH HEADERS FROM 'file:///data.csv' AS row
MATCH (existingNode {opportunity: toInteger(row.opportunity)})
CREATE (newNode:Person {name: row.name, age: toInteger(row.age)})
CREATE (existingNode)-[:RELATED_TO]->(newNode)


   - 
- new command relink


# Text2Cypher
- Add enum descriptions in schema

- possibly Prune the schema with https://kuzudb.github.io/blog/post/improving-text2cypher-for-graphrag-via-schema-pruning/#pruned-graph-schema-results

## ReAct agents
- tools: 
    - graph_search()  (or cypher_run()  so the schema is known by agent)
    - doc_search()  (from Chonkie)
    - node_search()

Create a CLI commmand 'agent' in genai_graph/core/commands_ekg.py that launch an LangChain agent, possibly interactive (chat mode).  
Get inpiration from command 'react'  in /home/tcl/prj/genai-tk/genai_tk/extra/agents/commands_agents.py, but simpler: 
- tools, MCP servers and system prompt are harcoded
- create a langchain tool that execute a Cypher query (Take it from genai_graph/core/commands_ekg.py). Add that tools to the Agent 

For the system prompt, explain that the role of the agent is (for now) to answer questions on enterprise data, and that it can use the provided tool to query a Cypher graph database.  
Explain how to use the tool, and give it, as in function query_kg in genai_graph/core/text2cypher.py, the schema of the database and the the same SYSTEM_PROMPT.
Take into account that, later, the agent will have other tools (notably to query a vector store and the web).
Use best practices to write that system prompt.

You can put some support core (such as system prompt) in another file.  Try to mutualize code and avoid duplication (you can modify genai_graph/core/text2cypher.py and other commands ).

You can test using for ex: 'uv run cli kg agent -i "List the names of all competitors"


## Better 'rag' commands
- pass a configurable chunker
https://docs.chonkie.ai/oss/pipelines 

##  Better KG



# To Test :
- ```uv run cli kg delete -f ; uv run cli kg add-doc --key cnes-venus-tma --subgraph ReviewedOpportunity ; uv run cli kg export-html -g ReviewedOpportunity```

- ```uv run cli baml extract $ONEDRIVE/prj/atos-kg/rainbow-md/cnes-venus-tma.md --function ExtractRainbow --force```

- ```uv run cli baml run FakeRainbow -i "Project for CNES; Marc Ferrer as sales lead in Atos team" --kvstore-key fake_cnes_1 --force```


- ```cli baml run FakeArchitectureJson -i "IT platform for CNES with 3-tier, Java based"  --kvstore-key fake-cnes-1```

- ```export GRAPH=ArchitectureDocument;  uv run cli kg delete -f ; uv run cli kg add-doc --key fake-cnes-1 -g $GRAPH ; uv run cli kg export-html -g $GRAPH```

- ```cli kg schema```


uv run cli kg delete -f ; uv run cli kg add-doc --key fake-cnes-1 --subgraph ArchitectureDocument


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

