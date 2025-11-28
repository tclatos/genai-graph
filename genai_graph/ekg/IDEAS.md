# Ideas around evolution of the Tk and Bleuprint


## Fix Typing
There a quit a lot of 'Any' types in function parameters in current project (/home/tcl/prj/genai-graph)).

Try to fix that (ie find the correct type of the parameter, add the import, etc). Check with Pylance or ruff that there's no error or warning




## Documents nodes

Modify (deeply) add-doc commmand in /home/tcl/prj/genai-graph/genai_graph/core/commands_ekg.py. For each doc added
- create a node (if not exists) in the Knowledge Graph (of type Document) with properties 'uuid' (set to the key name) and 'metadata' (an empty map). 
- Create a relationship "SOURCE" between the node associated to the top class (such as 'ReviewedOpportunity' - but keep code independant of the schema) and the new node.

Modify create_graph accordingly
Ensure these nodes are displayed when calling 'uv run cli kg info' and uv run cli kg schema .






## Better React with Agent Midleware


- Use LangChain Midlewares to print tool calls, either in CLI or Streamlit


# UI for Graph Query
Create a Streamlit page to  visualize the KG, and run queries on it (either in natural language, or in Cypher)
Look at CLI commmands in /home/tcl/prj/genai-graph/genai_graph/core/commands_ekg.py (query_ekg, cypher, export_html).
Vizualize HTML as in /home/tcl/prj/genai-blueprint/genai_blueprint/webapp/pages/demos/cognee_KG.py

# Doc  Manager
Create a repository "doc_manager" with files to manage docs (import, export, index, ...).
The backend is a relational database, handled by SQLAlchemy .
There are 2 tables: 
   - One for documents, with title, path, hash-code of the document, language (english by default), date, the content itself (in Markdown), metadata (JSON) 
   - One for Chunks; with fields for the chunk, the embeddings (a vector)
 and metadata. The table name encore the name of the embeddings (as the size of vector depends of it). 
 Use the pg_vectorstore langchain library, and possibly code from /home/tcl/prj/genai-tk/genai_tk/extra/pgvector_factory.py
Commands to load ....


 ...


 ## better LLM support

 Allow LiteLLM defined LLM to be created in genai_tk.core.llm_factory  by LlmFactory.
 If the pattern contains / and is the form azure_ai/mistral-document-ai-2505, or openrouter/google/palm-2-chat-bison, then return a langchain object of class 'ChatLiteLLM' (from package langchain-litellm).
 Call get_llm_provider to check that the model is correct (or better API if you know). Try to hace a nice code
 structure for maintainability. 
 Check with : uv run cli core llm -i 'tell me a jole' -m openrouter/google/openai/gpt-4.1-mini

LiteLLM









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

- ```uv run cli kg delete -f ; uv run cli kg add-doc --key rainbow-cnes-venus-tma --key rainbow-fake-cnes-1 -g ReviewedOpportunity ; uv run cli kg add-doc --key add-fake-cnes-1 -g ArchitectureDocument; uv run cli kg export-html```

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




# OLD OLD #


## Extra Data

Today the graph is built from BAML genererated Pydantic objects, except for a special case 'metadata' added to top classes. 
We want to generalise the mecanism of adding fields to a node that are not in BAML (that is used for document parsing). Our idea is to have a Pydantic class that will contain the extra fields.

For that, I've added a field "extra_classes" in class GraphNode. I's  a list of Pydantic classes that inherit an abstract class 'ExtraFields' with a abstract method 'get_data' that you should refine and implement in some descendant.

The content returned by get_data should be stored as a field of type Struct in the node, whose name is the name of the class (in snakecase).

I think that 'get_data' should be called in merge_node_in_graph or related functions, to get the extended list of fields to be inserted in the node.
Adapt parameters and typing. Idealy 'get_data' should return an instance of the concrete class to enfore type safety, but you can return a dict. 
Adap callers. Possibly refactor code  to make it clean, generic and maintenable. 

Fist use case is to change the hard-coded 'metadata' field by new class 'FileMetadata". (I've added extra_classes=[FileMetadata] in the top classes). You shoud generate a node struc field 'file_metadata" with a field 'source'  (similar to old 'metadata').  Remove old hard-coded stuff and refactor using the mechanism described above (notably implement the 'get_data' method). 

Second use case is to implement a WinLoss struct in the 'Opportunity' nodes. It will later be coded by getting information from the database.  Today, just return fake data (but ensure we have access to the opportunity id and other fields)

you can test using 'uv run cli kg delete -f ; uv run cli kg add-doc --key rainbow-cnes-venus-tma  ; uv run cli kg add-doc --key add-fake-cnes-1 -g ArchitectureDocument'

Use 'uv run cli kg schema' to check the new fields are there (file_metadata.source, win_loss.result, win_loss.reason,..), with their description

Enforce typing whenever possible (avoid Any...)