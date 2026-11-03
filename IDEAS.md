# Decision Models — DONE (closed 2026-11-03)

**Status**: implemented and benchmarked. Deliverables:
- Design: [docs/design/06_decision_model_navigation.md](docs/design/06_decision_model_navigation.md)
- Study + campaign record: [docs/studies/decision_model_navigation.md](docs/studies/decision_model_navigation.md) (§6)
- Benchmark report: [../officeqa/report/DECISION_NAVIGATION_33Q_REPORT.md](../officeqa/report/DECISION_NAVIGATION_33Q_REPORT.md)

**Outcome — decision-model routing vs LLM routing** (paired: same 16 questions, same
`glm_5.3_Flash` synthesizer, both on a healthy database):

| Routing | Strict | Weighted | Input tokens | Tool calls |
|---|---|---|---|---|
| LLM routing (baseline) | 69% | 75.0% | 12.28M | 373 |
| Decision-model routing | 50% | 62.5% | 8.23M (−33%) | 263 (−29%) |

Navigation itself costs 352 joint-scoring calls, 815k tokens, $0.029 total (~$0.001 and 10 s per
question) and generates zero tokens.

**Verdict**: keep it — it is a **cost** instrument today, not yet a quality one. The 12.5-point
weighted gap is *not* a navigation failure: the navigator selects the right document and table, and
the losses are downstream reading failures (wrong column/series inside a correct table). Two open
items remain: the agent-side memory growth (~370 MB per question, not reclaimed by `gc.collect()`)
that blocks full-subset measurement in one process (use `officeqa/run_subset_chunked.py`), and the
table-column reading discipline. The `deepseek4.1flash-1017` synthesizer swap was not an improvement
(same strict, worse weighted, +41% tokens).

Original request, for the record:

In genai-graph : 
We want to use Decision Model to reduce  calls to LLM  when an agent navigate in a document graph - it's typically a process where such model can evaluate Folders / Focuments in prder to pick the most relevant Sections.

Get inspiration from: 

https://www.researchgate.net/publication/416921648_Retriever-Free_Retrieval_Navigating_Documents_with_Decision_Models


Use our Decision model class and factory (se) get_decision_model in genai-tk . Read that paper.

Our idea is to have a new Agent Profile with different tools; skills etc. Make decision model configurable, as well as other important paramaters. 

We will compare the new approach with the LLm based one on subset of the benchmarks officequa. Have a look at that project to ensure will at least equivalent results.
Think, analyse alternatives, propose a plan and write it in docs/studies



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