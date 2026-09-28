# Access Control 
We want to add access control to the Doc Graph, taking the security triming approach used by many search engine. 
Here a draft: 
Build time : 
- Add a "allowed_principals" field in every "Document" or 'Folder" node. Could be a list of strings = ["group:03d8e370", "group:9e2f7b44", "user:5d8a7ca1"]"
- Have in the doc graph configuration the qualified name of a Python class that get access control information from the file path. So far it would be a simple one for test, taking information from a YAML file, but later it will be SharePoint access lists (using  Microsoft Graph SDK  or office365-rest-python-client or equivalent) , or simililar for AWS, CGP etc.  => Should be flexible, async, and efficient.

Run-time : 
- Propagate  a "user_principal" from the CLI commands or other UI down the the tools accessing the graph. In these tools, check that  the user has read access to the doc or folder
- Consider using LangGraph's `Runtime[ContextT]` (added v0.6.0) and LangChain's
`create_agent(context_schema=...)` provide first-class support for typed per-invocation
context. However, check it can be used with Deerflow harness. 
-  Update the tool result structure so it can inform the caller that some documments were not read due to authentication.  Possibly modify skill accordingly. 

More generaly : 
- get inspiration from well established practices used by search engines
- Be critical and investigate weakness in the design that could lead to security issue

Think (critically), suggest alterntaives, ask question, propose a plan

 




# Use BAML

We saw when running some error on handlig bad LLM formated JSO?
We have already BAML in gnai-graph, that as a bettter parser. 
TRy to use in in place of langchain "with_structured_output"
/home/tcl/prj/genai-graph/genai_graph/bench/models.py




# replace D3.js with echarts  (graphs, trees, ) ? 
https://echarts.apache.org/examples/en/index.html#chart-type-graph



# Next ? 
https://arxiv.org/abs/2607.11192




# Refactor


/home/tcl/prj/genai-graph/genai_graph/agent/middleware/wrap_up.py => tcl-tk





# Generalize skills on doc navigation
ex: skills/custom/officeqa-qa/SKILL.md "Targeted Search with `search_sections(query="<query>", document_id="<id>")`**:

# Language detection
In genai-graph, detect the language of the injected document into the graphdoxwith 
https://github.com/pemistahl/lingua-py 

(restrict to common language  for business : European languages, Chinese, ...)

Also get the stopwords.  I sugest using Spacy:  from spacy.lang.fr import French, ...
stopwords = French.Defaults.stop_words

Add the language code in the Document field. 
Use code and stopwords to configure Ladybug BM25 : https://docs.ladybugdb.com/extensions/full-text-search/ 

Put generic code in /home/tcl/prj/genai-tk/genai_tk/extra/nlp 


# refactor genai_tk/utils
- move trace, monitoring and trajectories files in a genai_tk/extra/monitoring




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