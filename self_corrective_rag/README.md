# Self-Corrective RAG (SC-RAG)

## Overview

SC-RAG is a **hybrid Self-RAG + Corrective RAG pipeline** built as a single LangGraph state machine. It fuses two seminal retrieval-augmented generation papers into one system that **decides when to retrieve**, **corrects bad retrieval**, and **verifies its own outputs** through self-correction loops.

| Paper | Contribution |
|-------|-------------|
| [Self-RAG](https://arxiv.org/abs/2310.11511v1) (Asai et al., 2023) | Adaptive retrieval gate, grounding verification, utility assessment, self-correction loop |
| [CRAG](https://arxiv.org/abs/2401.15884) (Yan et al., 2024) | Retrieval quality evaluation, 3-way verdict routing, web augmentation, context distillation |

**Target users:** ML engineers and researchers building production-grade RAG systems that need to go beyond naive retrieve-and-generate.

## Features

- **Adaptive retrieval gate** — skips retrieval entirely for simple parametric-knowledge queries (Self-RAG `RETRIEVE` token)
- **Hybrid search** — Qdrant dense (BGE-small-en-v1.5) + sparse (BM25) + MMR diversity + cross-encoder reranking
- **3-way retrieval verdict** — CORRECT / AMBIGUOUS / INCORRECT with configurable thresholds
- **Web augmentation** — Tavily multi-query search fallback when local retrieval is insufficient
- **Context distillation** — sentence-level LLM filtering + PII redaction (email, phone, SSN)
- **Grounding verification** — post-generation check that the answer is supported by context (Self-RAG `ISSUP` token)
- **Utility assessment** — checks if the answer is complete and useful (Self-RAG `ISUSE` token)
- **Self-correction loop** — claim-focused query rewrite targeting unsupported claims, with configurable max iterations
- **Docling PDF parsing** — structured section extraction preserving document hierarchy (titles, headings, tables)
- **Section-aware chunking** — word-based chunking that respects document section boundaries
- **Structured outputs** — all LLM judges use Pydantic v2 models with OpenAI strict JSON mode
- **Fail-safe defaults** — every LLM node degrades gracefully on failure (e.g., retrieval gate defaults to `True`)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) (StateGraph) |
| LLM — Judge | OpenAI `gpt-4o-mini` (all reflection/grading nodes) |
| LLM — Generator | OpenAI `gpt-4o` (final answer synthesis) |
| Vector Store | [Qdrant Cloud](https://qdrant.tech/) (hybrid dense + sparse) |
| Embeddings | [BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) via HuggingFace |
| Sparse Embeddings | FastEmbed BM25 via `langchain-qdrant` |
| Reranker | [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) |
| PDF Parsing | [Docling](https://github.com/DS4SD/docling) + pypdfium2 |
| Web Search | [Tavily](https://tavily.com/) |
| Structured Outputs | Pydantic v2 |
| Runtime | Jupyter Notebook (Python 3.10+) |

## Project Structure

```
self_corrective_rag/
├── sc_rag.ipynb                 # Main implementation notebook (60 cells)
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (not committed)
├── data/                        # PDF corpus for ingestion
│   ├── 10-Q4-2024-As-Filed.pdf  # Apple 10-K FY2024
│   └── tsla-20231231-gen.pdf    # Tesla 10-K FY2023
└── docs/
    ├── SOTA_ARCHITECTURE.md     # Full architecture design (10 nodes, routing, schemas)
    ├── brainstorms/
    │   └── 2026-03-11-self-crag-architecture-brainstorm.md
    └── plans/
        └── 2026-03-11-feat-self-corrective-rag-pipeline-plan.md
```

### Notebook sections

| Cells | Section | Description |
|-------|---------|-------------|
| 0–2 | Setup | dotenv, imports |
| 3–6 | Config | Logging, `SCRagConfig` frozen dataclass |
| 7–10 | Types | Enums, TypedDicts, 7 Pydantic structured output models |
| 11–14 | Prompts & Utils | 9 prompt templates, PII redaction, sentence splitting |
| 15–20 | Ingestion | DoclingParser, HybridSectionChunker, corpus loader |
| 21–23 | Vector Store | Qdrant build/load, ingest toggle |
| 24–29 | Nodes | `init_state`, 5 CRAG nodes, 5 Self-RAG nodes |
| 30–34 | Graph | 3 routing functions, graph assembly, compilation |
| 35–42 | Tests (basic) | Direct path, happy path, web fallback |
| 43–56 | Tests (data) | Chunk quality inspection, Apple/Tesla 10-K questions, cross-document |
| 57–59 | Summary | Execution summary printer |

## Prerequisites

- **Python** 3.10+
- **OpenAI API key** with access to `gpt-4o` and `gpt-4o-mini`
- **Qdrant Cloud** account (or local Qdrant instance)
- **Tavily API key** for web search fallback
- **Jupyter** (Notebook or Lab)
- ~2 GB disk space for model downloads (embeddings + cross-encoder)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd self_corrective_rag

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Copy the example below into a `.env` file at the project root:

```bash
cp .env.example .env  # if .env.example exists, otherwise create manually
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key for GPT-4o / GPT-4o-mini |
| `QDRANT_URL` | Yes | — | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Yes | — | Qdrant API key |
| `QDRANT_COLLECTION` | No | `scrag_docs` | Qdrant collection name |
| `QDRANT_PREFER_GRPC` | No | `true` | Use gRPC for Qdrant connections |
| `TAVILY_API_KEY` | Yes | — | Tavily web search API key |
| `CRAG_EVALUATOR_MODEL` | No | `gpt-4o-mini` | Model for all judge/grading nodes |
| `CRAG_GENERATOR_MODEL` | No | `gpt-4o` | Model for answer synthesis |
| `CRAG_TEMPERATURE_JUDGE` | No | `0.0` | Temperature for judge LLM |
| `CRAG_TEMPERATURE_GEN` | No | `0.1` | Temperature for generator LLM |
| `CRAG_UPPER_THRESHOLD` | No | `0.80` | Score above this → CORRECT verdict |
| `CRAG_LOWER_THRESHOLD` | No | `0.35` | Score below this → INCORRECT verdict |
| `CRAG_TOP_K_FINAL` | No | `6` | Number of chunks after reranking |
| `CRAG_CORPUS_DIR` | No | `./data` | Directory containing PDF files |
| `CRAG_WEB_MAX_RESULTS` | No | `5` | Max Tavily search results |
| `CRAG_WEB_ALLOWLIST` | No | `""` | Comma-separated allowed domains for web search |
| `SCRAG_MAX_LOOPS` | No | `3` | Max self-correction iterations |
| `SCRAG_UTILITY_THRESHOLD` | No | `0.7` | Utility score threshold (0–1) for answer acceptance |
| `SCRAG_PDF_MAX_PAGES` | No | `500` | Max pages per PDF for Docling |
| `SCRAG_PDF_MAX_SIZE_MB` | No | `50` | Max PDF file size in MB |
| `SCRAG_PDF_DO_OCR` | No | `false` | Enable OCR for scanned PDFs |
| `SCRAG_PDF_TABLE_STRUCTURE` | No | `true` | Enable table structure extraction |
| `SCRAG_CHUNK_SIZE_WORDS` | No | `600` | Target chunk size in words |
| `SCRAG_CHUNK_OVERLAP_WORDS` | No | `100` | Overlap between chunks in words |
| `SCRAG_MIN_CHUNK_WORDS` | No | `100` | Minimum chunk size (smaller sections get merged) |
| `LOG_LEVEL` | No | `INFO` | Logging level |

## Running Locally

1. **Start Jupyter:**

```bash
jupyter notebook
# or
jupyter lab
```

2. **Open `sc_rag.ipynb`** and run cells sequentially.

3. **First run — ingest PDFs:**

   In the Vector Store section (cell 23), set the toggle:

   ```python
   already_ingested = False
   ```

   This will parse PDFs with Docling, chunk them, and upload to Qdrant. On subsequent runs, set it back to `True` to skip ingestion and load the existing collection.

4. **Run the pipeline:**

   After ingestion, the remaining cells compile the LangGraph workflow and execute test queries.

## Usage

### Run a single query through the pipeline

```python
user_ctx: UserContext = {
    "user_id": "u-123",
    "tenant_id": "tenant_demo",
    "roles": ["analyst"],
}

query = "What was Apple's total net sales in 2024?"
state = init_state(query=query, user=user_ctx, query_id="my-query-001")
final_state = app.invoke(state, config={"recursion_limit": 50})

print(final_state["final_response"])
```

### Inspect pipeline decisions

```python
print(f"Need retrieval : {final_state['need_retrieval']}")
print(f"Verdict        : {final_state['verdict']}")
print(f"Support        : {final_state['support_verdict']}")
print(f"Utility score  : {final_state['utility_score']}")
print(f"Loops used     : {final_state['loop_count']}/{final_state['max_loops']}")
print(f"Chunks retrieved: {len(final_state['retrieved_chunks'])}")
print(f"Web chunks     : {len(final_state['web_chunks'])}")
```

### Add your own PDFs

Place PDF files in the `data/` directory and re-run with `already_ingested = False`. Docling extracts structured sections automatically.

## Architecture

```
START → decide_retrieval
          ├── [no retrieval] → direct_generate ──────────────────┐
          └── [need retrieval] → fetch_candidates → grade_chunks │
                                    ├── [CORRECT] → distill ─────┤
                                    └── [AMBIGUOUS/INCORRECT] ───┤
                                         → expand_with_web ──────┤
                                                                 ▼
                                                        synthesize_answer
                                                                 │
                                                         check_support
                                                   ┌─────────┴──────────┐
                                              [supported]        [hallucinated]
                                                   │              → rewrite_and_retry
                                                   ▼                → fetch_candidates (loop)
                                            check_utility
                                          ┌──────┴──────┐
                                     [≥ 0.7]        [< 0.7]
                                        END       → expand_with_web (loop)
```

The pipeline supports up to 3 self-correction loops (configurable). Each loop refines the query based on specific unsupported claims identified by the grounding checker.

For the full architecture specification, see [`docs/SOTA_ARCHITECTURE.md`](docs/SOTA_ARCHITECTURE.md).

## Pipeline Nodes

| Node | Origin | Model | Purpose |
|------|--------|-------|---------|
| `decide_retrieval` | Self-RAG | gpt-4o-mini | Gate: does the query need document retrieval? |
| `direct_generate` | Self-RAG | gpt-4o | Answer from parametric knowledge only |
| `fetch_candidates` | CRAG | — | Hybrid Qdrant search (MMR k=20, fetch_k=50) + cross-encoder rerank |
| `grade_chunks` | CRAG + Self-RAG | gpt-4o-mini | Score each chunk [0,1] → 3-way verdict |
| `expand_with_web` | CRAG | gpt-4o-mini + Tavily | Query rewrite + web search fallback |
| `distill_context` | CRAG | gpt-4o-mini | Sentence-level relevance filtering + PII redaction |
| `synthesize_answer` | CRAG | gpt-4o | Grounded answer generation with citations |
| `check_support` | Self-RAG | gpt-4o-mini | Is the answer grounded in the context? |
| `check_utility` | Self-RAG | gpt-4o-mini | Is the answer complete and useful? |
| `rewrite_and_retry` | Self-RAG | gpt-4o-mini | Claim-focused query refinement for re-retrieval |

## Sample Data

The repository includes two SEC 10-K filings for testing:

| File | Company | Period | Pages |
|------|---------|--------|-------|
| `data/10-Q4-2024-As-Filed.pdf` | Apple Inc. | FY ended Sept 28, 2024 | 121 |
| `data/tsla-20231231-gen.pdf` | Tesla Inc. | FY ended Dec 31, 2023 | 130 |

## Limitations

- **No streaming** — responses are returned after full pipeline completion; streaming support is deferred
- **No authentication/multi-tenancy** — `UserContext` fields exist in state but are not enforced
- **LLM-as-judge, not fine-tuned** — uses prompted GPT-4o-mini for all reflection checks rather than fine-tuned Self-RAG tokens
- **Cost** — a single query can make 5–10+ LLM calls depending on the correction path; each self-correction loop adds ~4 additional calls
- **Notebook format** — not packaged as an installable module; intended for experimentation and research

## References

- Asai, A., et al. (2023). [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511v1)
- Yan, S.-Q., et al. (2024). [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)
- LangChain. [Agentic RAG with LangGraph](https://blog.langchain.com/agentic-rag-with-langgraph/)

## License

Needs confirmation — no LICENSE file found in the repository.
