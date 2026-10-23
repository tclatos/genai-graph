# Neo4j 
Implement the Neo4j backend studied in /home/tcl/prj/genai-graph/docs/neo4j_backend_integration_study.md.

We will use au Aura instance. It should be the latest version (check...). The  URI and credential are in the .env. 

Convert Ladybug Maps as JSON. 
Focus on the doc-graph first. That's easier.

Try to male something extensible.  We might need to supprt other Cypher compatibe graph DB in the future.


The env viariables are: 
NEO4J_URI
NEO4J_USERNAME
NEO4J_PASSWORD
NEO4J_DATABASE
AURA_INSTANCEID
AURA_INSTANCENAME



# Security
Study https://github.com/openfga/openfga  - Benefit ?  

# Use BAML

We saw when running some error on handlig bad LLM formated JSO?
We have already BAML in gnai-graph, that as a bettter parser. 
TRy to use in in place of langchain "with_structured_output"
/home/tcl/prj/genai-graph/genai_graph/bench/models.py




# replace D3.js with echarts  (graphs, trees, ) ? 
https://echarts.apache.org/examples/en/index.html#chart-type-graph



# Next ? 
https://arxiv.org/abs/2607.11192






# Generalize skills on doc navigation
ex: skills/custom/officeqa-qa/SKILL.md "Targeted Search with `search_sections(query="<query>", document_id="<id>")`**:






# better genai-graph

- Update in genai-graph  the  cli docgraph commands so that they  take into account the new feature related to chunks



- more tests, notably semantic  search. Create test graph in memory ;   Use real LLM

- update doc and skills


Update doc 

# CLI
 ? Merge cli docgraph ls and cli docgraph folders 

 

- need better 'docgraph cat' commmand -> section range, section separator 


# Benchmarks
- Analyse https://github.com/NanoNets/nanoindex and see what can be taken
- Test on:
  FinanceBench (84 SEC filings, avg 143 pagesn ~53,900 pages total) 
  DocBench Legal (51 court filings, avg 54 pages => ~2800 pages )
  OfficeQA Pro (696 Treasury Bulletins, ~89,000-page corpus. 133 questions) 
- Compare with NanoNet, PageIndex, Mistral Agentic Search https://mistral.ai/news/agentic-search/


# Tests
Add https://github.com/cbornet/blockbuster  to detect blocking