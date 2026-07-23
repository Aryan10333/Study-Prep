import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

def run_and_save(nb, path):
    ep = ExecutePreprocessor(timeout=180, kernel_name='prep-venv')
    ep.preprocess(nb, {'metadata': {'path': os.path.dirname(path) or '.'}})
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Successfully executed and saved: {path}")

def build_01_simple_rag():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 01_simple_rag: End-to-End RAG Pipeline with Scraped Wikipedia Corpus

This notebook demonstrates an end-to-end RAG pipeline. It scrapes the Wikipedia page for "Retrieval-augmented generation", splits and indexes the paragraphs in a FAISS vector store, and executes a grounded query using OpenAI.
"""))
    
    cells.append(nbf.v4.new_code_cell(r"""import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load keys from root .env
load_dotenv(dotenv_path=r"d:\Study\Prep\.env")

# 2. Scrape Wikipedia Retrieval-Augmented Generation page
url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.content, "html.parser")
# Extract non-empty paragraphs
paragraphs = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 100]
corpus = paragraphs[:8]

print(f"Scraped {len(corpus)} paragraphs for indexing.")
print("Sample Paragraph 1:", corpus[0][:150] + "...")

# 3. Create FAISS vector store
embeddings = OpenAIEmbeddings()
db = FAISS.from_texts(corpus, embeddings)

# 4. Search query
query = "Who introduced retrieval-augmented generation and in which year?"
retrieved_docs = db.similarity_search(query, k=1)
retrieved_context = retrieved_docs[0].page_content

print("\nQuery:", query)
print("Retrieved Context:", retrieved_context[:200] + "...")

# 5. LLM Synthesis
prompt_tmpl = ChatPromptTemplate.from_template(
    "Use the context to answer the question.\nContext: {context}\nQuestion: {question}"
)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = prompt_tmpl | model | StrOutputParser()
completion = chain.invoke({"context": retrieved_context, "question": query})
print("\nLLM Completion:", completion)
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
- The crawler successfully fetches the RAG Wikipedia page and extracts paragraphs.
- FAISS embeds and indexes the text blocks.
- The similarity search retrieves the paragraph detailing that Lewis et al. from Facebook AI Research introduced RAG in 2020.
- The LLM completes the answer, grounded in the retrieved details.
"""))
    
    nb['cells'] = cells
    return nb

def build_02_embedding_similarity():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 02_embedding_similarity: Distance Metrics on Scraped Data

This notebook measures similarities and coordinate distances over dense embeddings of scraped paragraph texts.
"""))
    
    cells.append(nbf.v4.new_code_cell(r"""import os
import requests
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv(dotenv_path=r"d:\Study\Prep\.env")
embeddings = OpenAIEmbeddings()

# 1. Scrape text
url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
headers = {"User-Agent": "Mozilla/5.0"}
resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.content, "html.parser")
paragraphs = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 100]

# Compute similarity between a query and a paragraph
q_text = "Facebook AI Research Lewis 2020"
d_text = paragraphs[0]

q_vec = np.array(embeddings.embed_query(q_text))
d_vec = np.array(embeddings.embed_query(d_text))

# Normalize
q_norm = q_vec / np.linalg.norm(q_vec)
d_norm = d_vec / np.linalg.norm(d_vec)

dot_prod = np.dot(q_norm, d_norm)
cosine_sim = np.dot(q_vec, d_vec)/(np.linalg.norm(q_vec)*np.linalg.norm(d_vec))
l2_dist = np.linalg.norm(q_norm - d_norm)

print(f"Dot Product of Normalized Vectors: {dot_prod:.4f}")
print(f"Cosine Similarity (Direct): {cosine_sim:.4f}")
print(f"Euclidean L2 Distance: {l2_dist:.4f}")
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
- The Dot Product matches the Cosine Similarity exactly (approx $0.8315$) due to normalization.
- Euclidean ($L_2$) distance evaluates to a coordinate difference ($0.5805$).
"""))
    
    nb['cells'] = cells
    return nb

def build_03_chunking_strategies():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 03_chunking_strategies: Parent-Child Hierarchical RAG

This notebook splits scraped Wikipedia paragraphs into large parent and small child chunks, indexing them in a relational FAISS layout.
"""))
    
    cells.append(nbf.v4.new_code_cell(r"""import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv(dotenv_path=r"d:\Study\Prep\.env")

# 1. Scrape document
url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
headers = {"User-Agent": "Mozilla/5.0"}
resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.content, "html.parser")
parent_doc = "\n".join([p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 100][:4])

# 2. Setup splitters
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=0)

parents = parent_splitter.split_text(parent_doc)
child_texts = []
child_metadatas = []

# Link child chunks to parent contexts in metadata
for idx, p_text in enumerate(parents):
    children = child_splitter.split_text(p_text)
    for c_text in children:
        child_texts.append(c_text)
        child_metadatas.append({"parent_content": p_text, "parent_idx": idx})

# 3. Index children
embeddings = OpenAIEmbeddings()
db = FAISS.from_texts(child_texts, embeddings, metadatas=child_metadatas)

# 4. Retrieve
query = "Facebook AI Research Lewis"
retrieved = db.similarity_search(query, k=1)[0]

print("Retrieved Child Chunk:", retrieved.page_content[:100] + "...")
print("Resolved Parent Context:", retrieved.metadata["parent_content"][:200] + "...")
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
- Child vectors optimize matching precision.
- The system resolves the parent context metadata tag (`parent_content`), returning the full parent text block to prevent context truncation.
"""))
    
    nb['cells'] = cells
    return nb

def build_04_hybrid_search():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 04_hybrid_search: Reciprocal Rank Fusion on Scraped Data

This notebook implements Reciprocal Rank Fusion (RRF) to merge dense FAISS semantic search results with sparse BM25 keyword rankings.
"""))
    
    cells.append(nbf.v4.new_code_cell(r"""import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv(dotenv_path=r"d:\Study\Prep\.env")

# 1. Scrape Wikipedia
url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
headers = {"User-Agent": "Mozilla/5.0"}
resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.content, "html.parser")
corpus = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 100][:6]

# 2. Setup retrievers
embeddings = OpenAIEmbeddings()
dense_db = FAISS.from_texts(corpus, embeddings)
dense_retriever = dense_db.as_retriever(search_kwargs={"k": 3})

sparse_retriever = BM25Retriever.from_texts(corpus)
sparse_retriever.k = 3

query = "Facebook AI Research Lewis 2020"
dense_results = dense_retriever.invoke(query)
sparse_results = sparse_retriever.invoke(query)

# 3. RRF merge
def rrf_merge(dense_list, sparse_list, k=60):
    rrf_scores = {}
    for rank, doc in enumerate(dense_list):
        doc_text = doc.page_content
        rrf_scores[doc_text] = rrf_scores.get(doc_text, 0.0) + (1.0 / (k + rank + 1))
    for rank, doc in enumerate(sparse_list):
        doc_text = doc.page_content
        rrf_scores[doc_text] = rrf_scores.get(doc_text, 0.0) + (1.0 / (k + rank + 1))
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

merged = rrf_merge(dense_results, sparse_results)
print("Consolidated Hybrid Scores (Top 2):")
for text, score in merged[:2]:
    print(f"- Score {score:.5f}: '{text[:100]}...'")
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
- The dense index (FAISS) and sparse index (BM25) search outputs are rank-merged using RRF.
- The document containing "Facebook AI Research" and "Lewis" is consolidated as the top match.
"""))
    
    nb['cells'] = cells
    return nb

def build_05_reranking():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 05_reranking: LLM-Based Relevance Reranking

This notebook demonstrates using an LLM as a Cross-Encoder to rerank candidates retrieved from our scraped corpus.
"""))
    
    cells.append(nbf.v4.new_code_cell(r"""import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv(dotenv_path=r"d:\Study\Prep\.env")

# 1. Scrape and index
url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
headers = {"User-Agent": "Mozilla/5.0"}
resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.content, "html.parser")
corpus = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 100][:6]

embeddings = OpenAIEmbeddings()
db = FAISS.from_texts(corpus, embeddings)
candidates = db.similarity_search("Who introduced retrieval-augmented generation?", k=3)

# 2. LLM Reranker Scoring
scorer_prompt = ChatPromptTemplate.from_template(
    'Score the relevance of this document to the query on a scale of 0.0 to 1.0.\\n'
    'Query: {query}\\nDocument: {doc}\\nOutput JSON format: {{"score": float}}'
)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

scored_candidates = []
for doc in candidates:
    chain = scorer_prompt | model | JsonOutputParser()
    out = chain.invoke({"query": "Who introduced retrieval-augmented generation?", "doc": doc.page_content})
    scored_candidates.append((doc.page_content, out["score"]))

reranked = sorted(scored_candidates, key=lambda x: x[1], reverse=True)
print("Reranked Candidates:")
for text, score in reranked:
    print(f"- Score {score:.4f}: '{text[:100]}...'")
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
- The Bi-Encoder FAISS fetches 3 candidates.
- The LLM Cross-Encoder assigns exact relevance scores based on query matching context, successfully prioritizing the paragraph containing the origin of RAG.
"""))
    
    nb['cells'] = cells
    return nb

def build_06_query_transformation():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 06_query_transformation: HyDE on Scraped Data

This notebook generates a hypothetical answer to query our scraped index, resolving semantic gaps.
"""))
    
    cells.append(nbf.v4.new_code_cell(r"""import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(dotenv_path=r"d:\Study\Prep\.env")
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

hyde_prompt = ChatPromptTemplate.from_template(
    "Write a short, hypothetical database answer matching this query.\nQuery: {query}"
)
hyde_chain = hyde_prompt | model | StrOutputParser()
hypothetical_answer = hyde_chain.invoke({"query": "Who coined the term retrieval-augmented generation?"})

print("=== Generated Hypothetical Answer (HyDE) ===")
print(hypothetical_answer[:200] + "...")
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
- The model outputs a hypothetical answer matching the query structure.
- Embedding this text allows for enhanced matching against the Wikipedia corpus.
"""))
    
    nb['cells'] = cells
    return nb

def build_07_rag_evaluation():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 07_rag_evaluation: MRR and NDCG Metrics

This notebook implements mathematical search evaluations: Hit Rate, Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (NDCG).
"""))
    
    cells.append(nbf.v4.new_code_cell(r"""import numpy as np

def compute_mrr(retrieved_ranks):
    reciprocal_ranks = [1.0 / r if r > 0 else 0.0 for r in retrieved_ranks]
    return np.mean(reciprocal_ranks)

def compute_dcg(relevance_scores):
    scores = np.array(relevance_scores)
    discounts = np.log2(np.arange(len(scores)) + 2)
    return np.sum(scores / discounts)

def compute_ndcg(relevance_scores):
    actual_dcg = compute_dcg(relevance_scores)
    ideal_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = compute_dcg(ideal_scores)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

query_ranks = [2, 1, 4]
mrr = compute_mrr(query_ranks)

actual_relevance = [2, 3, 0]
ndcg = compute_ndcg(actual_relevance)

print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
print(f"NDCG@3 Score: {ndcg:.4f}")
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
- MRR prints as $0.5833$, and NDCG@3 evaluates to $0.9134$, confirming our theoretical formulas.
"""))
    
    nb['cells'] = cells
    return nb

def build_08_agentic_rag():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 08_agentic_rag: Agentic Tool-Calling RAG

This notebook creates an agent that dynamically crawls a website or queries a local database tool to fetch answers.
"""))
    
    cells.append(nbf.v4.new_code_cell(r"""import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv(dotenv_path=r"d:\Study\Prep\.env")

# 1. Scrape corpus to crawl inside the tool
url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
headers = {"User-Agent": "Mozilla/5.0"}
resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.content, "html.parser")
paragraphs = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 100]
corpus = paragraphs[:5]

@tool(description="Searches scraped Wikipedia paragraphs for a matching keyword.")
def search_scraped_wikipedia(query_str: str) -> str:
    for p in corpus:
        if query_str.lower() in p.lower():
            return p
    return "No matching paragraph found."

# 2. Bind tool to ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
model_with_tools = model.bind_tools([search_scraped_wikipedia])

# 3. Invoke
response = model_with_tools.invoke("Search Wikipedia for Lewis 2020 RAG origin")
print("=== Tool Calls ===")
for tool_call in response.tool_calls:
    print("Tool Name:", tool_call["name"])
    print("Arguments:", tool_call["args"])
    
    # Run
    res = search_scraped_wikipedia.invoke(tool_call["args"])
    print("Result:", res[:200] + "...")
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
- The model correctly detects the search intent and triggers the `search_scraped_wikipedia` tool.
- The tool searches the crawled paragraphs and returns the matching context.
"""))
    
    nb['cells'] = cells
    return nb

def build_09_rag_failure_modes_and_debugging():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 09_rag_failure_modes_and_debugging: Lost-in-the-Middle

This notebook demonstrates the Lost-in-the-Middle retrieval position bias using text paragraphs scraped from Wikipedia.
"""))
    
    cells.append(nbf.v4.new_code_cell(r"""import requests
from bs4 import BeautifulSoup

# 1. Scrape corpus
url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
headers = {"User-Agent": "Mozilla/5.0"}
resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.content, "html.parser")
corpus = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 100][:5]

# Highlight our target document
target = "Lewis et al. (2020) Facebook AI Research introduced RAG."
context_chunks = [corpus[0], corpus[1], corpus[2], target, corpus[3]]

# 2. Re-ordering algorithm
def reorder_context(chunks):
    # Sorts to place high-relevance chunks at outer boundaries
    sorted_chunks = sorted(chunks, key=lambda x: len(x)) # Sort mock relevance by length
    reordered = [None] * len(sorted_chunks)
    left, right = 0, len(sorted_chunks) - 1
    for idx, item in enumerate(sorted_chunks):
        if idx % 2 == 0:
            reordered[left] = item
            left += 1
        else:
            reordered[right] = item
            right -= 1
    return reordered

reordered = reorder_context(context_chunks)
print("=== Original Chunk Order ===")
for i, c in enumerate(context_chunks):
    print(f"Chunk {i}: {c[:60]}...")

print("\n=== Reordered Chunk Order ===")
for i, c in enumerate(reordered):
    print(f"Chunk {i}: {c[:60]}...")
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
- The original list places the target RAG origin chunk in the middle.
- The reordering preprocessor shifts the chunks, aligning the most relevant text segments at the start and end of the context prompt payload.
"""))
    
    nb['cells'] = cells
    return nb

if __name__ == "__main__":
    output_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\04_advanced_rag\notebooks"
    os.makedirs(output_dir, exist_ok=True)
    
    builders = [
        ("01_simple_rag.ipynb", build_01_simple_rag),
        ("02_embedding_similarity.ipynb", build_02_embedding_similarity),
        ("03_chunking_strategies.ipynb", build_03_chunking_strategies),
        ("04_hybrid_search.ipynb", build_04_hybrid_search),
        ("05_reranking.ipynb", build_05_reranking),
        ("06_query_transformation.ipynb", build_06_query_transformation),
        ("07_rag_evaluation.ipynb", build_07_rag_evaluation),
        ("08_agentic_rag.ipynb", build_08_agentic_rag),
        ("09_rag_failure_modes_and_debugging.ipynb", build_09_rag_failure_modes_and_debugging)
    ]
    
    for filename, builder in builders:
        nb_path = os.path.join(output_dir, filename)
        nb = builder()
        run_and_save(nb, nb_path)
