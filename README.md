# RAG Pipeline From Scratch

A Retrieval-Augmented Generation (RAG) pipeline built entirely from scratch in Python — **no LangChain, no LlamaIndex**. Every stage of the pipeline (document loading, chunking, embedding, vector storage, semantic search, prompt construction, and generation) is implemented manually to understand exactly how a RAG system works under the hood.

## Overview

This project ingests documents in multiple formats, converts them into searchable vector embeddings, and answers natural-language questions about their content using an LLM — grounded strictly in the retrieved context, with multi-turn conversation memory.

It's split into two independent pipelines:

- **`ingestion_pipeline.py`** — reads documents, chunks them, generates embeddings, and stores everything in a persistent vector database. Run once (or whenever new documents are added).
- **`retrieval_pipeline.py`** — takes a user's question, retrieves the most relevant chunks, and generates a grounded answer using an LLM. Run every time a question is asked.

## Tech Stack

| Component | Tool |
|---|---|
| Document parsing | `PyPDF2` (PDF), `python-docx` (DOCX), `pandas` (CSV), built-in file I/O (TXT) |
| Chunking | Custom sentence-preserving chunking function (no external library) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector database | `ChromaDB` (persistent local storage) |
| LLM generation | OpenRouter API (OpenAI-compatible client), using free-tier models |
| Environment management | `python-dotenv` |

## How It Works

### Ingestion Pipeline
1. **Read** — documents are loaded based on file extension (`.txt`, `.pdf`, `.docx`, `.csv`). CSV files auto-detect which column contains the relevant text, using common column-name matching with a length-based fallback.
2. **Chunk** — raw text is split into ~500-character chunks while preserving full sentences, so no chunk cuts a sentence in half.
3. **Embed & Store** — each chunk is embedded using `sentence-transformers` and stored in ChromaDB along with metadata (source filename, chunk index), batched for efficiency.

### Retrieval Pipeline
1. **Session management** — each conversation gets a unique session ID, with its own isolated message history (supports multiple independent conversations at once).
2. **Query contextualization** — follow-up questions (e.g. "what about for remote workers?") are rewritten by the LLM into standalone questions using conversation history, before retrieval happens — this significantly improves retrieval accuracy on multi-turn conversations.
3. **Semantic search** — the (contextualized) query is embedded and matched against stored chunks in ChromaDB to find the most relevant results.
4. **Context assembly** — retrieved chunks are combined into a single context block, alongside their source citations.
5. **Generation** — the LLM generates an answer strictly grounded in the retrieved context and conversation history, explicitly instructed to say "I don't know" rather than hallucinate when the answer isn't present.

## Data Used

This project was built and tested using publicly available data related to enterprise knowledge / HR documentation — including a freely distributed employee handbook PDF and Kaggle datasets related to HR/company policy. **No dataset is included in this repository.**

## Roadmap

This project is actively being extended. Planned additions include:
- A proper UI (likely Streamlit) instead of terminal-based interaction
- Document upload support directly through the UI
- Persistent conversation history (currently in-memory only, resets on restart)
- Re-ranking of retrieved chunks before generation

## Notes

- Free-tier LLM models via OpenRouter are rate-limited and may occasionally return a `429` error under high load — this is expected behavior on shared free infrastructure, not a bug in the pipeline.
