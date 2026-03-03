<div align="center">

# 🧠 Corrective RAG — Production-Grade Hybrid Search Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-green?logo=langchain)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-Cloud%20%7C%20Local-purple?logo=qdrant)](https://qdrant.tech/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-orange?logo=openai)](https://platform.openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A state-of-the-art Corrective RAG (CRAG) system built on LangGraph with Hybrid Search, Maximal Marginal Relevance (MMR), Cross-Encoder Reranking, and intelligent web fallback — designed for production deployments.**

</div>

---

## 📐 Architecture Overview

```
                          ┌─────────────────────────────────────────────────┐
                          │              User Query + UserContext            │
                          └──────────────────────┬──────────────────────────┘
                                                 │
                                    ┌────────────▼────────────┐
                                    │     fetch_candidates    │
                                    │  Hybrid Search (Dense + │
                                    │  Sparse BM25) → MMR     │
                                    │  → Cross-Encoder Rerank │
                                    └────────────┬────────────┘
                                                 │
                                    ┌────────────▼────────────┐
                                    │      grade_chunks       │
                                    │  GPT-4o-mini scores     │
                                    │  each chunk [0.0–1.0]   │
                                    └────────────┬────────────┘
                                                 │
                           ┌─────────────────────┼──────────────────────┐
                           │  verdict < 0.35     │   0.35 ≤ v ≤ 0.80   │ verdict > 0.80
                           │   (INCORRECT)       │    (AMBIGUOUS)       │ (CORRECT)
                           ▼                     ▼                      │
                  ┌─────────────────┐   ┌────────────────┐             │
                  │ expand_with_web │   │ expand_with_web│             │
                  │  Query Rewrite  │   │  Web augment   │             │
                  │  + Tavily Search│   │  + local keep  │             │
                  └────────┬────────┘   └───────┬────────┘             │
                           │                    │                       │
                           └──────────┬─────────┘                      │
                                      ▼                                 │
                          ┌───────────────────────┐                    │
                          │    distill_context    │◄───────────────────┘
                          │  Sentence-level filter│
                          │  + PII redaction      │
                          │  + deduplication      │
                          └───────────┬───────────┘
                                      │
                          ┌───────────▼───────────┐
                          │   synthesize_answer   │
                          │  GPT-4o grounded gen  │
                          │  with citations       │
                          └───────────────────────┘
```

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| 🔀 **Hybrid Search** | Dense (BGE-small) + Sparse (BM25) via Qdrant `RetrievalMode.HYBRID` |
| 📊 **MMR Retrieval** | `max_marginal_relevance_search` — fetch 50 candidates, keep 20 diverse |
| 🎯 **Cross-Encoder Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` precision reranks the MMR pool |
| ⚖️ **Adaptive CRAG Routing** | 3-way verdict (`CORRECT` / `AMBIGUOUS` / `INCORRECT`) with tunable thresholds |
| 🌐 **Web Fallback** | Tavily-powered search with LLM query rewriting and domain allowlist |
| 📝 **Query Rewriting** | Structured LLM decomposition into subqueries + entity extraction |
| 🔒 **PII Redaction** | Automatic scrubbing of emails, phones, SSNs before context is sent to LLM |
| 🧾 **Grounded Generation** | GPT-4o answers strictly from context; abstains with `"I don't know."` when information is missing |
| 🗃️ **Metadata Filtering** | Per-request Qdrant payload filters (tenant, doc type, date range, roles) |
| 📈 **Structured Outputs** | Pydantic-validated LLM outputs for grading, rewriting, filtering, and generation |
| 🔭 **Observability** | Structured JSON logs per node with trace IDs and latency metrics |

---

## 🗂️ Project Structure

```
RAG_git/
├── .env.example                        # Template — copy to .env and fill in keys
├── .gitignore
├── corrective_rag_new/
│   ├── medium_app_qdrant.ipynb         # ← Main implementation notebook
│   ├── requirements.txt
│   └── documents/                      # Drop your PDF / TXT / MD corpus here
│       ├── book1.pdf
│       └── book3.pdf
└── output.png
```

---

## ⚙️ Pipeline Deep-Dive

### 1 · Hybrid Retrieval (Dense + Sparse)

The vector store is built using **`BAAI/bge-small-en-v1.5`** for dense semantic embeddings and **Qdrant's BM25 sparse model** (`Qdrant/bm25` via FastEmbed). Both vector types are stored in the same Qdrant collection, and retrieval uses `RetrievalMode.HYBRID` which fuses scores via **Reciprocal Rank Fusion (RRF)** internally.

```python
vs = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=HuggingFaceEmbeddings("BAAI/bge-small-en-v1.5"),
    sparse_embedding=FastEmbedSparse("Qdrant/bm25"),
    retrieval_mode=RetrievalMode.HYBRID,
    ...
)
```

### 2 · MMR → Cross-Encoder Reranking

```
50 candidates (fetch_k)
    │
    ▼ MMR (k=20): maximises coverage, penalises near-duplicate chunks
    │
    ▼ Cross-Encoder (ms-marco-MiniLM-L-6-v2): re-scores each (query, chunk) pair
    │
    ▼ Top-K (default 6): final precision-ranked chunks passed to grader
```

### 3 · CRAG Routing

Each chunk is independently graded `[0.0–1.0]` by `gpt-4o-mini`. An aggregate verdict is derived:

| Verdict | Condition | Action |
|---|---|---|
| `CORRECT` | avg score ≥ 0.80 | Skip web, go straight to distillation |
| `AMBIGUOUS` | 0.35 ≤ avg < 0.80 | Augment local chunks with web results |
| `INCORRECT` | avg < 0.35 | Rewrite query, fetch entirely from web |

### 4 · Context Distillation

A sentence-level LLM filter (`SENTENCE_FILTER_PROMPT`) retains only high-signal factual sentences. When contradictions exist across sources, the prompt instructs the LLM to **favour the most recent or most authoritative source**. PII is redacted before sending to the generator.

### 5 · Grounded Generation

`gpt-4o` answers using **only** the distilled context. It returns structured `GroundedAnswer` with:
- `answer` — the response
- `citations_used` — source references
- `abstained: true` — if context is insufficient

---

## 🛠️ Setup

### Prerequisites

- Python 3.10+
- A [Qdrant Cloud](https://cloud.qdrant.io/) account **or** local Qdrant server
- OpenAI API key
- Tavily API key (for web fallback)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/KaranHC/RAG.git
cd RAG

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r corrective_rag_new/requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your keys (see Configuration section below)
```

### Running

Open and run `corrective_rag_new/medium_app_qdrant.ipynb` in Jupyter or VS Code.

On first run, set `already_ingested = False` to build the Qdrant collection. On subsequent runs, set it back to `True` to skip re-ingestion.

---

## 🔧 Configuration

All settings are driven by environment variables. Copy `.env.example` → `.env` and fill in your values:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key (**required**) |
| `TAVILY_API_KEY` | — | Tavily search API key (**required** for web fallback) |
| `CRAG_EVALUATOR_MODEL` | `gpt-4o-mini` | LLM used for grading / rewriting / filtering |
| `CRAG_GENERATOR_MODEL` | `gpt-4o` | LLM used for final answer generation |
| `CRAG_TEMPERATURE_JUDGE` | `0.0` | Temperature for deterministic grading |
| `CRAG_TEMPERATURE_GEN` | `0.1` | Temperature for answer generation |
| `CRAG_UPPER_THRESHOLD` | `0.80` | Score above which retrieval is `CORRECT` |
| `CRAG_LOWER_THRESHOLD` | `0.35` | Score below which retrieval is `INCORRECT` |
| `CRAG_TOP_K_FINAL` | `6` | Number of chunks passed to the grader after reranking |
| `CRAG_CORPUS_DIR` | `./documents` | Path to your document corpus (PDF / TXT / MD) |
| `QDRANT_URL` | — | Qdrant Cloud or self-hosted URL |
| `QDRANT_API_KEY` | — | Qdrant Cloud API key |
| `QDRANT_PREFER_GRPC` | `true` | Use gRPC for faster Qdrant communication |
| `QDRANT_PATH` | — | Local embedded Qdrant path (alternative to URL) |
| `QDRANT_COLLECTION` | `crag_docs` | Qdrant collection name |
| `CRAG_WEB_MAX_RESULTS` | `5` | Max Tavily results per web search |
| `CRAG_WEB_ALLOWLIST` | `` | Comma-separated allowed domains for web fallback |

### Qdrant Modes

**Cloud / Self-hosted (recommended):**
```env
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
```

**Local embedded (no server needed):**
```env
QDRANT_PATH=/tmp/my_qdrant_db
# Leave QDRANT_URL blank
```

---

## 🔒 Security Notes

- **Never commit `.env`** — it is listed in `.gitignore`
- The grading, filtering, and generation prompts include **prompt injection guards** — context is treated as untrusted input
- PII (emails, phone numbers, SSNs) is automatically redacted before the context reaches the LLM

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `langgraph` | Stateful multi-node RAG workflow orchestration |
| `langchain`, `langchain-openai` | LLM chains, prompt templates, OpenAI integration |
| `langchain-qdrant` | Qdrant vector store integration with LangChain |
| `qdrant-client` | Native Qdrant Python client |
| `langchain-huggingface` | HuggingFace embedding models (BGE-small) |
| `fastembed` | Qdrant's BM25 sparse embeddings |
| `sentence-transformers` | Cross-encoder reranker |
| `pypdf` | PDF corpus loading |
| `tiktoken` | Token counting for OpenAI models |
| `pydantic` | Structured LLM output validation |

---

## 🗺️ Roadmap

- [ ] Streaming response support via LangGraph streaming
- [ ] Multi-tenant vector namespace isolation
- [ ] FastAPI / Gradio demo interface
- [ ] Evaluation harness with RAGAS metrics
- [ ] Docker Compose setup with local Qdrant

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

Built with 🔥 using [LangGraph](https://github.com/langchain-ai/langgraph) · [Qdrant](https://qdrant.tech/) · [OpenAI](https://openai.com/)

</div>
