
# Middleware
Consider ToolCallLimitMiddleware  


# Marjdownization
- Analyse embedded diagrams

# Multi-write
Analyse how KG building van be speed uo wuth new 
 (kg/backend.py): KuzuBackend.connect(..., enable_multi_writes=) forwards to ladybug.Database; new KuzuBackend.attach(db) reuses an already-open Database with a fresh Connection (no vector extension) — the shape Ladybug requires for shared-DB workers.
and async call

Move genai_graph/kg/parallel.py  to utils/ladybug (or better)
Make SharedKuzuParallel a Pydantic class
Refactor KuzuBackend ? 



# better genai-graph

- Update in genai-graph  the  cli docgraph commands so that they  take into account the new feature related to chunks

- Use Chonkie instead of genai_graph/kg/document_graph/chunker.py 


- more tests, notably semantic  search. Create test graph in memory ;   Use real LLM

- update doc and skills


Update doc 

# CLI
 ? Merge cli docgraph folder-toc and cli docgraph folders 

# Benchmarks
- Analyse https://github.com/NanoNets/nanoindex and see what can be taken
- Test on:
  FinanceBench (84 SEC filings, avg 143 pagesn ~53,900 pages total) 
  DocBench Legal (51 court filings, avg 54 pages => ~2800 pages )
  OfficeQA Pro (696 Treasury Bulletins, ~89,000-page corpus. 133 questions) 
- Compare with NanoNet, PageIndex, Mistral Agentic Search https://mistral.ai/news/agentic-search/


# Tests
Add https://github.com/cbornet/blockbuster  to detect blocking