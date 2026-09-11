# better search

In GraphDoc tool search_ ... 

# LightRAG
We have a well working docgraph construction process, efficient (SOA on benchmarks).
On the other side, we have graph fabrics to extract entitoes and relationship from docs through a BAML schema. That's  fine.

But my feeling is that we could without too much effort extract entities and relationship using technique inspired by LightRag (https://github.com/hkuds/lightrag) : we have very good text chunking capabilities, LLM to summerize documents or sections (they could also extract entities..), hybrid search, etc

https://github.com/NanoNets/nanoindex/ also extract entities fro a table of content strucrire similar to ours, but they use GLiner.  That's look unneccessary if we aleady pass  docs to LLM.

Mu idea is that we could have a new graph factory to create these entities / relationsih using a light-rag inspired approach without too much complexity.
 
Investigate this idea, and write a report in genai-graph/design.  You can be critical

Think about that, ask questions, suggest/evaluare alternatives, propose a plan.


# Faster processing

Building the doc graph is very slow.  See how we could increase the injection speed.
Ideas : 
- add more paralelism in summerization and embeddings calculation
- process more documents at a time
- Check there's no semaphore or lock somewhere
- Use non-thinking, fast model, fast provider (easy: we can now select llm that wau deepseekv4flash(none)@openai:speed )
- ...
- Analyse Prefect logs ? 



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