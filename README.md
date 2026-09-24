# Enterprise RAG Assistant

A production-style Retrieval-Augmented Generation (RAG) backend built with Python and FastAPI.

The project demonstrates how an initial semantic-search RAG pipeline can evolve into a more realistic system with hybrid retrieval, reranking, relevance filtering, evaluation, observability, configuration management, API boundaries, and Docker support.

The focus of this project is not only generating answers with an LLM, but engineering the retrieval pipeline around it.

---

## Architecture

```text
                         ┌─────────────────────┐
                         │      Documents      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Ingestion      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Chunking + Metadata │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
          ┌───────────────────┐           ┌───────────────────┐
          │    Embeddings     │           │    BM25 Index     │
          └─────────┬─────────┘           └─────────┬─────────┘
                    │                               │
                    ▼                               │
          ┌───────────────────┐                     │
          │      Qdrant       │                     │
          │   Vector Store    │                     │
          └─────────┬─────────┘                     │
                    │                               │
                    ▼                               ▼
          ┌───────────────────┐           ┌───────────────────┐
          │  Vector Retrieval │           │  Lexical Retrieval│
          └─────────┬─────────┘           └─────────┬─────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Reciprocal Rank     │
                         │ Fusion (RRF)        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Cross-Encoder       │
                         │ Reranker            │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Relevance Filtering │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Grounded Generation │
                         │   Qwen via Ollama   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       FastAPI       │
                         └─────────────────────┘
```

---

## Features

- Local `.txt` and `.md` document ingestion
- Recursive document chunking with metadata
- Sentence Transformer embeddings
- Persistent Qdrant vector storage
- BM25 lexical retrieval
- Hybrid semantic + lexical search
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Reranker-based relevance filtering
- Grounded LLM generation using Ollama
- Fallback response when relevant context is unavailable
- FastAPI REST endpoints
- Centralized application configuration
- Application-specific exception handling
- Dependency lifecycle management
- Retrieval evaluation
- Rejection evaluation for irrelevant questions
- Pipeline logging and latency measurements
- Unit tests with pytest
- Dockerized application

---

## Tech Stack

| Area | Technology |
|---|---|
| Language | Python 3.12 |
| API | FastAPI |
| LLM | Qwen via Ollama |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database | Qdrant |
| Lexical Search | BM25 (`rank-bm25`) |
| Fusion | Reciprocal Rank Fusion |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Testing | pytest |
| Configuration | pydantic-settings |
| Containerization | Docker |

---

## Retrieval Pipeline

The retrieval pipeline combines semantic and lexical retrieval instead of relying on vector similarity alone.

```text
Question
   │
   ├──► Embedding ──► Qdrant Vector Search
   │
   └───────────────► BM25 Search
                           │
              ┌────────────┘
              ▼
      Reciprocal Rank Fusion
              │
              ▼
      Cross-Encoder Reranking
              │
              ▼
       Relevance Filtering
              │
              ▼
        Relevant Context
```

### Why hybrid retrieval?

Vector search is useful for semantic similarity, while BM25 performs well when exact terminology, identifiers, or domain-specific keywords matter.

Their raw scores are not directly comparable, so Reciprocal Rank Fusion combines the ranked result lists based on rank rather than raw score.

A cross-encoder then reranks the fused candidates using the question and chunk together.

---

## Grounded Generation

Only chunks that survive retrieval, fusion, reranking, and relevance filtering are passed to the LLM.

If no sufficiently relevant context remains, the application returns:

```text
I don't have enough information to answer that question.
```

This prevents unnecessary LLM calls and reduces unsupported answers when the knowledge base does not contain useful context.

---

## API

### Health

```http
GET /health
```

Example response:

```json
{
  "status": "healthy"
}
```

### Ingest Document

```http
POST /ingest
```

Request:

```json
{
  "file_path": "data/kafka.txt"
}
```

Example response:

```json
{
  "status": "success",
  "chunks_ingested": 15
}
```

### Ask Question

```http
POST /ask
```

Request:

```json
{
  "question": "How do Kafka consumers receive messages?"
}
```

Example response:

```json
{
  "answer": "Consumers read messages from Kafka topics."
}
```

---

## Evaluation

The project includes a lightweight evaluation framework rather than relying only on manual testing.

### Retrieval Evaluation

Retrieval is evaluated using **Hit Rate@K**.

A retrieval is considered a hit when the expected document source appears in the top K retrieved results.

Example:

```text
Retrieval Evaluation
--------------------
Questions evaluated: 2
Hits@3: 2
Hit Rate@3: 100.00%
```

These numbers represent the included evaluation dataset and should not be interpreted as general RAG accuracy.

### Rejection Evaluation

Irrelevant questions are evaluated separately.

A rejection is considered correct when the retrieval pipeline determines that no sufficiently relevant context is available.

Example:

```text
Rejection Evaluation
--------------------
Irrelevant questions: 3
Correctly rejected: 3
Rejection Rate: 100.00%
```

Run evaluation with:

```bash
python -m evaluation.run_evaluation
```

---

## Observability

The application logs important RAG pipeline events and timings.

Example retrieval log:

```text
retrieval_completed vector_results=3 bm25_results=2 fused_results=4 final_results=2 duration_ms=112.43
```

Example generation log:

```text
generation_completed context_chunks=2 duration_ms=1387.52
```

When no relevant context survives retrieval:

```text
retrieval_no_relevant_context
generation_skipped reason=no_relevant_context
```

Questions and retrieved document contents are intentionally not written to application logs.

---

## Running Locally

### Prerequisites

- Python 3.12
- Ollama
- Qwen model available through Ollama

Create a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure the configured Ollama model is available:

```bash
ollama list
```

Start the API:

```bash
uvicorn app.api.main:app --reload
```

The API is available at:

```text
http://localhost:8000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

---

## Running with Docker

Build the image:

```bash
docker build -t enterprise-rag-assistant .
```

When Ollama is running on the macOS host, start the application with:

```bash
docker run --rm \
  -p 8000:8000 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  enterprise-rag-assistant
```

`host.docker.internal` allows the Docker container to communicate with Ollama running on the host machine.

Verify:

```bash
curl http://localhost:8000/health
```

Ingest the sample document:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path":"data/kafka.txt"}'
```

Ask a question:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How do Kafka consumers receive messages?"}'
```

---

## Testing

Run all tests:

```bash
pytest
```

Format the project:

```bash
black .
```

The test suite covers core components including ingestion, chunking, retrieval, reranking, fusion, relevance filtering, API behavior, and evaluation logic.

---

## Configuration

Runtime configuration is managed using `pydantic-settings`.

Configuration includes values such as:

```text
Embedding model
Reranker model
LLM model
Ollama base URL
Qdrant path
Qdrant collection
Reranker relevance threshold
```

Defaults support local development and values can be overridden using environment variables.

For example:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
```

---

## Project Structure

```text
enterprise-rag-assistant/
│
├── app/
│   ├── api/
│   ├── chunking/
│   ├── core/
│   ├── embeddings/
│   ├── exceptions/
│   ├── generation/
│   ├── ingestion/
│   ├── lexical_search/
│   ├── llm/
│   ├── models/
│   ├── reranking/
│   ├── retrieval/
│   ├── services/
│   └── vector_store/
│
├── data/
│   └── kafka.txt
│
├── evaluation/
│   ├── dataset.py
│   ├── retrieval_evaluator.py
│   └── run_evaluation.py
│
├── tests/
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## Engineering Decisions

### Hybrid retrieval over vector-only retrieval

Semantic similarity alone can miss exact technical terminology. BM25 complements vector retrieval with lexical matching.

### Reciprocal Rank Fusion over raw-score combination

Vector similarity and BM25 scores exist on different scales. RRF combines rankings without requiring score normalization.

### Cross-encoder reranking

Initial retrieval favors recall. The cross-encoder provides a more expensive but more precise relevance estimate over a smaller candidate set.

### Relevance filtering after reranking

After RRF, the retrieval score represents fusion rank rather than semantic similarity. Final relevance decisions therefore use the reranker score.

### LLM abstraction

Generation depends on an `llm_client` abstraction rather than coupling the generation service directly to Ollama-specific behavior.

### Dependency lifecycle

Heavy components such as embedding and reranking models are created during application startup and reused across requests instead of being recreated for every API call.

---

## Current Limitations

This project intentionally keeps several infrastructure concerns outside its scope.

- BM25 is currently maintained in memory and must be rebuilt through ingestion when the application process restarts.
- Re-ingesting the same document can create duplicate vector entries because document lifecycle/versioning is not implemented.
- The included evaluation dataset is intentionally small and demonstrates the evaluation approach rather than providing statistically meaningful quality benchmarks.
- Authentication and authorization are not implemented.
- Document ingestion currently operates on local files rather than production object storage.
- Ollama is expected to run separately from the application container.
- Distributed tracing and metrics infrastructure such as OpenTelemetry, Prometheus, or Grafana are not included.

These would be natural extensions for a larger production deployment.

---

## What I Learned

Building this project highlighted that production-style RAG involves significantly more than connecting a vector database to an LLM.

Key lessons included:

- separating ingestion and query pipelines
- understanding semantic versus lexical retrieval
- combining heterogeneous retrievers using rank fusion
- using reranking to improve precision
- separating retrieval relevance from generation
- refusing questions when useful context is unavailable
- evaluating retrieval independently from generation
- managing expensive model lifecycles
- designing configurable service boundaries
- instrumenting AI pipelines for latency and behavior
- containerizing applications that depend on locally hosted models

The project evolved incrementally, with each capability implemented and tested as a separate engineering step.

---

## Future Improvements

Potential extensions include persistent/rebuildable lexical indexes, deterministic document IDs and deduplication, document update/delete workflows, larger evaluation datasets, automated answer-quality evaluation, citations in API responses, authentication, external document storage, and production metrics/tracing.

---

## Status

**Capstone complete.**

The current version demonstrates an end-to-end production-style RAG backend with:

```text
Ingestion
→ Chunking
→ Embeddings
→ Qdrant
→ BM25
→ Hybrid Retrieval
→ RRF
→ Reranking
→ Relevance Filtering
→ Grounded Generation
→ FastAPI
→ Evaluation
→ Observability
→ Docker
```