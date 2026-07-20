# Module 02: Document Processing, Parsing & Ingestion Pipelines

This guide provides an in-depth breakdown of Data Connectors, Multi-Format PDF/Table OCR parsing, Text Cleaning & Preprocessing, MinHash LSH Deduplication, Metadata Extraction & Tagging, and Incremental Index Versioning, complete with step-by-step hand calculations, Python code, and production trade-offs.

> **Notebook Companion**: [02_document_processing_and_ingestion.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/02_retrieval_augmented_generation_rag/02_document_processing_and_ingestion.ipynb)

---

## 1. Enterprise Ingestion Connectors

Production RAG systems require continuous ingestion from diverse enterprise data sources:
- **Cloud Storage**: AWS S3, Google Cloud Storage, Azure Blob.
- **Enterprise SaaS**: Notion, Confluence, Jira, Google Drive, Salesforce.
- **Databases**: PostgreSQL, MongoDB, Snowflake, BigQuery.
- **Web Crawlers**: Dynamic headless web scraping for documentation portals.

---

## 2. Multi-Format PDF, Multi-Column & Table OCR Parsing

Parsing raw enterprise documents presents major visual challenges:
- **Multi-Column Layouts**: Standard text extractors read across columns horizontally, scrambling sentence order.
- **Tables & Financial Statements**: Extracting tables as plain text collapses row-column boundaries.
- **Scanned Image Documents**: Requires Optical Character Recognition (OCR).

```text
Parsing Engine      Approach                                   Table Support  Handling 2-Column PDFs
----------------------------------------------------------------------------------------------------------------------
PyPDF / PDFMiner    Naive text stream extraction              Poor           Fails (Scrambles lines)
Tesseract OCR      Pixel OCR bounding boxes                  Moderate       Requires layout post-processing
Unstructured       Layout-aware element classification        High           Excellent
LlamaParse         Vision-LLM structured markdown parsing     State-of-the-Art Preserves Markdown tables
```

---

## 3. Text Cleaning, Normalization & Preprocessing

Before chunking and embedding, raw parsed text must be normalized:
1. **HTML/XML Stripping**: Removing `<script>`, `<style>`, and layout tags using `BeautifulSoup`.
2. **Regex Sanitization**: Repairing broken hyphenated words at line breaks (e.g. `micro- \n service` $\rightarrow$ `microservice`).
3. **Unicode Normalization**: Converting special characters via `unicodedata.normalize('NFKD', text)`.

---

## 4. MinHash LSH Deduplication Mechanics & Step-by-Step Hand Calculation (Andrew Ng Style)

Given two document texts $A$ and $B$, Jaccard Similarity $J(A, B)$ is:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

### Step-by-Step Hand Calculation Example:
- Document A word set: `{"invoice", "1001", "total", "5000", "consulting"}` ($|A| = 5$)
- Document B word set: `{"invoice", "1001", "total", "5000", "consulting", "services"}` ($|B| = 6$)

1. **Intersection ($A \cap B$):** `{"invoice", "1001", "total", "5000", "consulting"}` ($|A \cap B| = 5$)
2. **Union ($A \cup B$):** `{"invoice", "1001", "total", "5000", "consulting", "services"}` ($|A \cup B| = 6$)
3. **Jaccard Similarity:**
   $$J(A, B) = \frac{5}{6} \approx \mathbf{0.833 \ (83.3\%)}$$

Since $J(A, B) \ge 0.80$, Document B is classified as a duplicate and pruned before vector embedding.

---

## 5. Metadata Tagging & Payload Schema

Every vector payload in the vector database stores rich metadata attributes to support security filtering and auditability:

```json
{
  "doc_id": "doc_90214",
  "source_url": "https://internal.company.com/specs/microservices.pdf",
  "page_number": 4,
  "author": "Architecture Team",
  "created_at": "2026-01-15T10:00:00Z",
  "security_acl_tags": ["Engineering", "Confidential"],
  "chunk_hash": "a1b2c3d4e5f6"
}
```

---

## 6. Incremental Indexing & Versioning (CDC)

To prevent re-indexing unchanged documents during batch ingestion:
1. **Document Hashing**: Compute MD5/SHA256 hash of document raw content (`hash_val = hashlib.md5(text).hexdigest()`).
2. **Change Data Capture (CDC)**: Compare `hash_val` against document registry store.
3. **Deletion Workers**: If a document is updated or deleted, issue a payload metadata delete query (`delete_by_id(doc_id)`) to purge stale vectors before inserting new chunks.

---

## 7. Production LangChain Code Implementation

```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

def compute_jaccard_similarity(text1: str, text2: str) -> float:
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union if union > 0 else 0.0

doc_a = "Invoice 1001 total amount 5000 for consulting services"
doc_b = "Invoice 1001 total amount 5000 for consulting services extra"

similarity = compute_jaccard_similarity(doc_a, doc_b)
print("=== MinHash Jaccard Similarity Deduplication ===")
print(f"Doc A vs Doc B Similarity: {similarity*100:.1f}%")
print(f"Ingestion Action: {'PRUNE DUPLICATE' if similarity > 0.80 else 'INDEX DOCUMENT'}")
```

---

## 8. Production Failure Modes & Selection Rules

- **OCR Table Collapse**: Extracting tables as plain unformatted text destroys row-column relationships.
  - *Fix:* Use **LlamaParse** or **Unstructured** to extract tables directly into Markdown or HTML `<table>` representations before chunking.
- **Stale Vector Index Drift**: Modifying source documents without purging old vector store records results in conflicting search results.
  - *Fix:* Enforce **Incremental Indexing with MD5 Change Data Capture (CDC)** webhooks.