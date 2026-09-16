# AGENTS.md

Learnings and patterns for future agents working on this project.

## Feedback Instructions

TEST COMMANDS: `uv run pytest tests/unit` (no network needed); `uv run pytest tests/integration` (hits live Neo4j Companies KG + TypeSafe API via `.env`); `uv run pytest` for both.
BUILD COMMANDS: `uv sync`
LINT COMMANDS: none configured
FORMAT COMMANDS: none configured

## Project Overview

PRIMARY LANGUAGES: Python (>=3.12)

Graph-navigation demo combining TypeSafe's `system_one` structured-decision API (`Choice` + `Noul`) with Neo4j to beam-search a path through a graph one hop at a time, driven by a free-text goal, an explicit target node, or a natural-language path-intent pattern. Targets the public Neo4j "Companies KG" demo instance by default but stays schema-agnostic (labels/properties are discovered live, never hardcoded).

## Build System

BUILD SYSTEMS: `uv` (hatchling backend), `src/` layout, package `neo4jev` under `src/neo4jev/`.

Dependencies: `neo4j-rust-ext`, `neo4j-viz[neo4j,streamlit]`, `typesafe-sdk`, `python-dotenv`, `streamlit`. Dev deps: `pytest`, `pytest-asyncio`.

## Testing Framework

TESTING FRAMEWORKS: `pytest` + `pytest-asyncio`

`tests/unit/` mocks both the Neo4j driver and the TypeSafe client — no network or credentials required. `tests/integration/` runs against the live public Companies KG (`neo4j+s://demo.neo4jlabs.com:7687`, db `companies`, creds `companies`/`companies`) and a real TypeSafe call, using the same `.env` (credentials are public/non-secret, so no separate `integration.env`).

## Notebooks

Notebooks under `notebooks/` are executed with their outputs saved — the saved render is the deliverable, not scratch. Executing one needs an ipykernel in the kernel environment; `uv run --with nbclient --with ipykernel --with nbformat` plus a kernelspec whose `argv` points at the running interpreter works, and `JUPYTER_PATH=<dir>` must contain `kernels/<name>/kernel.json` (the entries are Jupyter *data* dirs, so `kernels/` is part of the path).

Careful with output size: `VisualizationGraph.render()` inlines neo4j-viz's ~8 MB JS template into **every** HTML output, so `03_full_traversal.ipynb` (three inline graph renders) is ~25 MB on disk and ~5 MB gzipped in git. Re-executing one of these notebooks rewrites that whole blob, so re-save outputs only when the notebook genuinely changed, and do not strip the renders to save space — an inline rendered graph is an acceptance criterion for `03`.

## Architecture

ARCHITECTURE PATTERN: Layered library + delivery surfaces (notebooks, Streamlit app) on top.

- `config.py` — env loading (`NEO4J_URL`/`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`, `TYPESAFE_API_KEY`).
- `types.py` — shared contract: `NavCandidate`, `NavStep`, `NavResult`, `BeamState`, `GoalSpec` (tagged union: `free_text` / `target_node` / `path_intent`).
- `neo4j_access.py` — driver lifecycle, label/index introspection, start-node search (exact/fulltext/vector, auto-detected per label), neighborhood fetch with candidate capping.
- `navigator.py` — one `system_one` call per hop (`Choice` over synthetic edge keys `e0,e1,…` + `Noul` "goal reached?"), beam search (visited-node guard, max depth, call budget, sum-of-log-prob scoring), all three goal modes, top-N distinct paths for `path_intent`.
- `viz.py` — `NavResult` → styled `neo4j-viz` `VisualizationGraph` (path vs. neighborhood, multi-path styling), both static `render()` and `render_widget()`-ready builders.

Library code (`src/neo4jev/`) has zero Streamlit/notebook-specific dependencies; `app/` and `notebooks/` only consume it.

## Deployment

DEPLOYMENT STRATEGY: none — local dev only (`streamlit run app/streamlit_app.py`, or run notebooks under `notebooks/`). No hosting/CI configured.

---

_This AGENTS.md was generated using agent-based project discovery._
