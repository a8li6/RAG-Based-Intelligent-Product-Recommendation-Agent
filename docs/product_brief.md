# Product Brief

## Problem statement

A catalog filter can enforce a price ceiling, but it does not understand a
request such as "a portable laptop for coding and commuting." An unconstrained
chatbot can explain tradeoffs, but it may invent products or specifications.
This project combines semantic retrieval with deterministic checks and
evidence-backed explanations.

## Version 1 scope

- **Domain:** laptops only. Start with eight fixed fictional products for tests,
  expand to 40-100 reviewed synthetic records for a demonstration, and use a
  larger synthetic catalog only for explicitly labeled scale experiments.
- **Inputs:** free-text needs plus optional structured filters: maximum budget,
  currency, minimum RAM, maximum weight, and stock availability.
- **Outputs:** up to three distinct products, verified catalog facts, short
  reasons with source fields, tradeoffs, and a question when the request cannot
  be interpreted safely.
- **Interaction:** a small session state, including follow-ups such as
  "under $700 instead." A reset action and visible active filters.

## Out of scope for version 1

Purchases, payment, web scraping, live prices, user tracking, collaborative
filtering, model training, and an autonomous tool loop.

## What "agent" means here

A bounded recommendation workflow: understand, validate, retrieve, explain,
verify, and respond. It has explicit state and decisions. Autonomous planning
and free-form tool use are optional later extensions.

## Example requests and observable behavior

| # | Request | Expected behavior |
|---|---|---|
| 1 | "Laptop for Python development and travel, under $900, at least 16 GB RAM, at most 1.5 kg." | Apply the exact constraints, retrieve among eligible items, return up to three grounded options (two if only two qualify). |
| 2 | "16 GB RAM, under $300." | `no_match`: empty result, explain which explicit constraints excluded products, offer a relaxation as a question; never silently change the budget. |
| 3 | "Something cheap." | `needs_clarification`: ask for a budget; never invent a numeric ceiling. |
| 4 | "Can this run my exact CAD workload at 120 fps?" | State that workload benchmarks are unavailable; do not infer performance from product prose. |
| 5 | Any request while the LLM is unavailable (timeout or invalid output). | Still show eligible products and verified facts using deterministic templates, labeled as a basic explanation. |

## Definition of a successful interaction

The user can see why each option was selected, what facts support the
explanation, what is unknown, and which constraints are active. The system never
substitutes a similar but ineligible item merely to fill three slots.

## Feature-to-milestone mapping

| Feature | Milestone |
|---|---|
| Typed catalog, validation, hard-constraint filters | M1 |
| Embeddings, FAISS index, eligible-subset retrieval | M2 |
| Query parsing, session follow-ups, LLM explanation plan, grounding, fallback | M3 |
| FastAPI endpoints and Streamlit interface | M4 |
| Golden evaluation set and metrics | M5 |
| CI, documentation, release | M6 |

> All sample products, prices, and URLs are fictional snapshot values. No example
> in this project is a real purchase recommendation.
