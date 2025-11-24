# go-vect

**A lightweight, local embedding and vector search engine written from scratch in Go.**

## Motivations

In an effort to learn Go and how vector databases work under the hood, I've been building `go-vect` as a fun side project in my spare time.

My aim with this project isn't to build the next Pinecone. Instead, it is to:

- Build a configurable semantic search playground
- Implement a complete embedding pipeline (from scratch) with local inference
- Use as few external dependencies as possible

## Features

- Local tokenisation with [SugarMe/tokenizer](https://github.com/sugarme/tokenizer), configured to use MiniLM-L6-v2's tokeniser requirements.
- Pure-Go cosine similarity search reconstructs vectors from disk, uses a custom O(n log k) min-heap for top-k similarity, and maps heap positions back to original texts via metadata.
- Thread-safe MiniLM ONNX sessions through a [Go ONNX Runtime](https://github.com/yalue/onnxruntime_go), with support for both single and batched inference with custom attention-aware mean pooling and lazy session reuse.
- Pluggable chunker and model interfaces if you want to use a different model or chunking strategy.
- A binary vector store writes normalised embeddings to an append-only .bin, journals offsets + raw text in JSONL for search and rolls back on metadata failures.
- Atomic writes to both files, reverts to last good state on failure.

This project is an active work in progress, with plans for optimisation, improvements and new additions.
