# RAG-Based Intelligent Product Recommendation Agent

A Python application that interprets a shopping request, enforces explicit
constraints in code, retrieves relevant products with embeddings and FAISS, and
produces grounded, evidence-backed explanations.

> **Status:** work in progress. Built milestone by milestone; this README is
> expanded as each stage is implemented and verified.

## Scope (version 1)

- Domain: laptops only, using a small **synthetic** demo catalog.
- Hard constraints (budget, RAM, weight, stock) are enforced by code, never by an LLM.
- Semantic retrieval over the eligible subset; explanations validated against catalog data.

## Roadmap

| Milestone | Focus |
|---|---|
| M0 | Scope and environment - done |
| M1 | Catalog schema, validation, eligibility filters - done |
| M2 | Embeddings and FAISS retrieval - done (tested with a fake embedder; real model pending) |
| M3 | Query parsing, LLM adapter, grounding - done (tested with a mocked LLM; real model pending) |
| M4 | FastAPI and Streamlit |
| M5 | Evaluation and hardening |
| M6 | CI, documentation, release |
