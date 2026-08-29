# Enterprise RAG Assistant

A production-style Retrieval-Augmented Generation (RAG) application built as a hands-on AI engineering capstone.

The project simulates an enterprise knowledge assistant capable of ingesting technical documentation and answering user questions using grounded information from the indexed knowledge base.

## Goals

The project will progressively implement:

- Document ingestion
- Chunking and metadata
- Embeddings
- Persistent vector storage
- Semantic retrieval
- Metadata filtering
- Hybrid search
- Reranking
- Relevance filtering
- LLM-based answer generation
- Source citations
- FastAPI
- Evaluation
- Observability
- Tests
- Docker

## Architecture

```text
Documents
    ↓
Ingestion
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Database
    ↓
Retrieval
    ↓
Reranking
    ↓
Relevant Context
    ↓
LLM
    ↓
Answer + Citations