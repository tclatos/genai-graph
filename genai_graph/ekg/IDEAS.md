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

## better llm / embeddings naming


## Better 'rag' commands
- pass a configurable chunker
https://docs.chonkie.ai/oss/pipelines 

##  Better KG

## Better deduplication
Correct and possibly improve node deduplication for merging.  The current key 'deduplication_key' does not seems to work: in the given example, the Opportunity nodes named 'CNES IT Platform – 3‑Tier Java Solution' and 'CNES Satellite Data Platform Modernisation' should be merged because thet have the same deduplication_key ("opportunity_id"). Fix that, and improve it: 1/ IF possible, put the name of the merged node in a field 'alternate names' (of type list) IF its different of the nominal one  2/ allow to have a lambda to define the deduplication criteria (similar to the one to generate names).  To test, you can use: ``` uv run cli kg delete -f ; uv run cli kg add-doc --key cnes-venus-tma --key fake_cnes_1 -g ReviewedOpportunity ; uv run cli kg add-doc --key fake-cnes-1 -g ArchitectureDocument; uv run cli kg export-html``` 

# To Test :
- ```uv run cli kg delete -f ; uv run cli kg add-doc --key cnes-venus-tma --subgraph ReviewedOpportunity ; uv run cli kg export-html -g ReviewedOpportunity```

- ```uv run cli baml extract $ONEDRIVE/prj/atos-kg/rainbow-md/cnes-venus-tma.md --function ExtractRainbow --force```

- ```uv run cli baml run FakeRainbow -i "Project for CNES; Marc Ferrer as sales lead in Atos team" --kvstore-key fake_cnes_1 --force```


- ```cli baml run FakeArchitectureJson -i "IT platform for CNES with 3-tier, Java based"  --kvstore-key fake-cnes-1```

- ```export GRAPH=ArchitectureDocument;  uv run cli kg delete -f ; uv run cli kg add-doc --key fake-cnes-1 -g $GRAPH ; uv run cli kg export-html -g $GRAPH```

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

