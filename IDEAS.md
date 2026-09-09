
# Mutialize Bench

We have implemented 2 benchmarks projects,  financebench and officeqa.  The second was a clone of the first. They worked well.
We wan now to mutualize code in genai-graph, to ease maintenance and, more important, facilitate development to test  our graph based framework against benchmarks.
Typically, the commmands 'cli bench '  should be the same for all bechnmarks, with shared code to load datasets, run, evaluate, judge, etc.  Top level woroflow, if it is benchmark specific, could be written using our Prefect YAML DSL, with reusable or modifiabke Prefect tasls / workflow behind.  ABC , YAML config with factory pattern could be used. Code could be placed in genai-graph/bench/ dir.  

Try also to make configuration more generic to ease test reproduction.  For example the config field 'onedrive_markdown_dir' could be renamed to  more  general "saved_markdown_dir" or "saved_markdown_dir" or else (I let you choose).

Constraint: keep compatility with existing bench files results and outcome.

Example of new benchmark we could tun is : https://github.com/mayubo2333/MMLongBench-Doc

Do quick run to test financebench and officeqa commands after refactoring. Don't break generated files (especially the built graph ! ) 

Think about that, ask questions, suggest/evaluare alternatives, propose a plan.




# Generalize skills on doc navigation
ex: skills/custom/officeqa-qa/SKILL.md "Targeted Search with `search_sections(query="<query>", document_id="<id>")`**:

# Language detection
 Detect the language of the inhected document with 
https://github.com/pemistahl/lingua-py 

(restrict to common language  for business : European languages, Chinese, ...)

Also get the stopwords.  I sugest using Spacy:  from spacy.lang.fr import French, ...
stopwords = French.Defaults.stop_words

Add the language code in the Document field. 
Use code and stopwords to configure Ladybug BM25 : https://docs.ladybugdb.com/extensions/full-text-search/ 

Put generic code in /home/tcl/prj/genai-tk/genai_tk/extra/nlp 


# refactor genai_tk/utils
- move trace, monitoring and trajectories files in a genai_tk/extra/monitoring



# Middleware
Consider ToolCallLimitMiddleware  


# Marjdownization
- Analyse embedded diagrams

# Multi-write
Analyse how KG building van be speed uo wuth new 
 (kg/backend.py): KuzuBackend.connect(..., enable_multi_writes=) forwards to ladybug.Database; new KuzuBackend.attach(db) reuses an already-open Database with a fresh Connection (no vector extension) — the shape Ladybug requires for shared-DB workers.
and async call




# better genai-graph

- Update in genai-graph  the  cli docgraph commands so that they  take into account the new feature related to chunks

- Use Chonkie instead of genai_graph/kg/document_graph/chunker.py 


- more tests, notably semantic  search. Create test graph in memory ;   Use real LLM

- update doc and skills


Update doc 

# CLI
 ? Merge cli docgraph folder-toc and cli docgraph folders 

 

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