# PRD: Graph Navigation Demo — TypeSafe Jev + Neo4j

## Overview

A demo application that navigates a Neo4j graph one hop at a time using TypeSafe's `system_one` structured-decision API instead of free-text LLM generation. At each visited node, outgoing relationships are presented to the API as `Choice` options (each carrying relationship type, relationship properties, and target node label/properties), and the API returns a full probability distribution over which relationship(s) to follow next. A `Noul` ("has the goal been reached?") question rides in the same `system_one` call as the `Choice`, so each hop costs exactly one round-trip regardless of how many questions are asked. Top-k/cutoff selection over the returned `probabilities` implements a beam search: at each step the search may branch into several candidate next-hops; the best-scoring chain(s) (ranked by sum of log-probabilities, to avoid float underflow and length bias) become "the path(s) taken", rendered in an interactive `neo4j-viz` graph alongside the surrounding neighborhood.

The demo targets the public Neo4j Companies KG (`neo4j+s://demo.neo4jlabs.com:7687`, db `companies`) but the application itself is schema-agnostic: labels, relationship types, and property names are discovered live via introspection, never hardcoded, so it can point at any other Neo4j instance via `.env`.

Delivery is in two layers: first a set of Jupyter notebooks that prove out the approach end-to-end against the real graph, then a Streamlit app that wraps the same library code in an interactive UI (start-node search, goal specification, run, visualize).

## Goals

- Prove that an LLM "structured decision" API (TypeSafe `Choice`/`Noul`) can drive graph traversal without any free-text parsing, using the API's native probability output for top-k/cutoff branching.
- Support three distinct ways of specifying navigation intent, selectable per run:
  1. **Free-text goal** — a natural-language description of what's being looked for (e.g. "find a company that competed with Apple").
  2. **Explicit target node** — a specific node picked via lookup (exact/fulltext/vector), navigate until that exact node is reached.
  3. **Path-intent / relationship-pattern goal** — a natural-language description of the *kind* of path to follow (e.g. "find me the paths from the patient to the gene expressions for their diseases"), which describes a semantic multi-hop pattern without a single fixed target node. This mode reasons about a *class* of destination and, since "paths" is plural, returns multiple distinct top-scoring paths rather than one.
- Let the user pick a start node (and, in target-node mode, a goal node) via whichever lookup mechanisms actually exist for the chosen label on the connected graph (exact property match always; fulltext/vector only if a matching index is detected).
- Visualize the start node, the path(s) taken, and the neighborhood of path nodes in an interactive graph widget, with the path(s) visually distinguished from context.
- Keep the core library (`src/neo4jev/`) fully reusable between the notebooks and the Streamlit app — no logic duplicated between them.

## Non-Goals

- Building a general-purpose Cypher query planner or a rigid path-pattern-matching DSL. Path-intent mode uses natural-language guidance fed into the per-hop `Choice`/`Noul` state, not a formal grammar compiled to Cypher.
- Persisting past run history/results across sessions (e.g. a database of prior traversals) — out of scope for this demo.
- Supporting write operations against Neo4j — the app is strictly read-only.
- Building a generic multi-tenant/multi-user web service — this is a single-user local demo (notebook + `streamlit run`).
- Authentication/authorization beyond the Neo4j and TypeSafe credentials already in `.env`.

## Requirements

### Functional Requirements

- REQ-F-001: `README.md` documents project purpose, architecture, setup (`uv sync`, `.env` contents), and how to run the notebooks and the Streamlit app.
- REQ-F-002: `src/neo4jev/config.py` loads settings from `.env` via `python-dotenv`, accepting either `NEO4J_URL` or `NEO4J_URI`, plus `NEO4J_USERNAME`/`NEO4J_PASSWORD`/`NEO4J_DATABASE` and `TYPESAFE_API_KEY`.
- REQ-F-003: `src/neo4jev/types.py` defines the shared data contract: `NavCandidate` (edge_key, rel_element_id, rel_type, rel_props, target_element_id, target_label, target_props), `NavStep` (current node, candidates considered, chosen candidate(s), probabilities, log_prob delta), `NavResult` (one or more `NavStep` sequences ["paths"], visited-neighborhood nodes/rels, termination reason(s)), `BeamState` (frontier of in-progress partial paths with cumulative log-prob and visited-set), and `GoalSpec` (a tagged union of the three goal modes: `free_text`, `target_node`, `path_intent`).
- REQ-F-004: `src/neo4jev/neo4j_access.py` provides: driver lifecycle (context manager, using `neo4j` — installed via `neo4j-rust-ext` — driven by config); `list_labels()`; `detect_indexes(label)` returning which of exact/fulltext/vector are usable for that label (fulltext/vector via `SHOW INDEXES`, filtered to the label/property, exact always available); `search_start_nodes(label, query, mode)` implementing exact substring/equality match, fulltext query, and vector query (embedding the query text) as applicable; `get_outgoing_relationships(node_id, rel_type_cap, total_cap)` returning capped `NavCandidate` list with truncation info when capped; `get_node_neighborhood(node_ids)` for viz context.
- REQ-F-005: `src/neo4jev/navigator.py` wraps `typesafe_sdk.TypeSafeClient` (or `AsyncTypeSafeClient`) to run one hop: build a `Choice` (candidate edges keyed by synthetic opaque ids `e0, e1, …` with a side mapping table back to `NavCandidate`) plus a `Noul` ("has the goal been reached at this node?") in a single `system_one` call, using `state` built from the current node, the `GoalSpec`, and (for path-intent mode) the current hop index within the described pattern.
- REQ-F-006: The beam search in `navigator.py` supports: visited-node cycle guard, configurable max depth, configurable max API-call budget, configurable top-k and/or probability-cutoff branching per hop, and path scoring by sum of log-probabilities.
- REQ-F-007: For `free_text` and `target_node` goal modes, the navigator returns the single best-scoring path (by cumulative log-probability) as `NavResult`, terminating early when the `Noul` "goal reached" answer crosses a configurable threshold or (for `target_node`) the exact target node id is reached.
- REQ-F-008: For `path_intent` goal mode, the navigator: (a) treats the natural-language pattern description as ordered stage guidance fed into each hop's `state`/`Choice.instructions`/`Noul.criteria`, biasing edge selection toward the semantics of the next expected stage; (b) tests the `Noul` "goal reached" criterion against the *class* described by the final stage (e.g. "this node is a gene-expression-like result for the patient's disease"), not a fixed node id; (c) runs multiple beam branches concurrently and returns the top-N distinct scoring paths (N configurable) rather than a single path.
- REQ-F-009: `src/neo4jev/viz.py` builds a `neo4j_viz.VisualizationGraph` from a `NavResult`: nodes/relationships on the returned path(s) are styled/colored distinctly from neighborhood-only nodes/relationships; when multiple paths are returned (path-intent mode), each path is distinguishable (e.g. by color or caption). Exposes both a static-render helper (`.render()`, for notebook/HTML use) and a widget-ready builder (`.render_widget()` / usable via `neo4j_viz.streamlit.display_widget()`).
- REQ-F-010: Notebooks in `notebooks/` demonstrate, against the live Companies KG: graph/index introspection (`01_explore_graph.ipynb`), a single dry-run hop with inspected probabilities (`02_navigator_dry_run.ipynb`), and full end-to-end traversals exercising all three goal modes with rendered visualization (`03_full_traversal.ipynb`).
- REQ-F-011: `app/streamlit_app.py` provides: label picker → start-node search using whichever lookup modes are detected for that label → goal-mode picker (free text / target node / path intent) with the corresponding input widget(s) → run controls (top-k, cutoff, max depth, max calls, N for path-intent) → interactive `neo4j-viz` widget render of the result, plus a per-hop probability/trace view.

### Non-Functional Requirements

- REQ-NF-001: Unit tests (`tests/unit/`) run with no network access, mocking both the Neo4j driver and the TypeSafe client; they cover edge-key mapping correctness, top-k/cutoff selection, candidate capping/truncation reporting, log-probability accumulation, cycle/depth/budget guards, and path-intent's multi-path return.
- REQ-NF-002: Integration tests (`tests/integration/`) run against the live public Companies KG using the same `.env` (public read-only credentials — no separate `integration.env` needed, as an explicit, documented deviation from the usual convention since these credentials are not secret).
- REQ-NF-003: All label/relationship-type/property names used at runtime are discovered from the live database (via introspection or user-provided search terms) — never hardcoded in application code.
- REQ-NF-004: The library code in `src/neo4jev/` has no Streamlit-specific or notebook-specific dependencies; both consumers import the same modules.
- REQ-NF-005: `system_one` calls stay within the 255-option `Choice` cap at all times via candidate capping (configurable, default ≤10 per relationship type and ≤60 total per hop), surfacing "N of M edges considered" whenever truncation occurs.

## Technical Considerations

- **TypeSafe SDK surface** (confirmed by reading the installed `typesafe_sdk` package source, not just docs): `TypeSafeClient`/`AsyncTypeSafeClient`; `Choice(criteria: Mapping[str, JSONContent | None], instructions: JSONContent | None = None)` returning `ChoiceAnswer(choice, probabilities: dict[str, float], confidence)`; `Noul(...)` returning `NoulAnswer(noul: float)` (0..1 probability of "yes"); `Score` (not used in this feature, but part of the same `Questions` mapping type). `system_one(state, questions, *, model=None, retry=None, timeout=None, extra_headers=None, extra_body=None) -> SystemOneResponse`, with `.answers/.nouls/.choices/.scores` accessors; `api_key`/`model` fall back to `TYPESAFE_API_KEY`/`TYPESAFE_DEFAULT_MODEL` env vars when not passed explicitly.
- **Edge-key collision avoidance**: `Choice` options are keyed by synthetic ids (`e0`, `e1`, …), never by relationship type, since multiple relationships of the same type to different targets would otherwise collapse into one option. A side mapping table (`edge_key -> NavCandidate`) is maintained per hop.
- **neo4j-viz surface** (confirmed by reading the installed package source): `Node`/`Relationship` pydantic models with `caption`, `caption_align`, `caption_size`, `size`, `color`, `properties`, `pinned`, `x`/`y`; `VisualizationGraph(nodes, relationships)` with `.color_nodes()`/`.color_relationships()`/`.set_legend()`/`.resize_nodes()`/`.resize_relationships()` for styling, `.render()` returning an `IPython.display.HTML` object, and `.render_widget()` returning a `GraphWidget` for interactive use (`neo4j_viz.streamlit.display_widget()` bridges it into Streamlit with two-way state sync via `widget.selected`). `from_neo4j()` is a convenience constructor that preserves Neo4j element ids as viz node/relationship ids and auto-styles by label/type — used as a starting point in `viz.py` where convenient, then restyled to distinguish path vs. neighborhood.
- **neo4j-rust-ext**: confirmed empirically to be a transparent drop-in accelerator for `import neo4j` — same top-level package namespace, no code changes needed anywhere that imports `neo4j`.
- **Path-intent mode design**: rather than compiling the pattern description into a rigid Cypher path expression, the description is decomposed (by the calling code, using simple heuristics or an upfront TypeSafe call) into an ordered list of stage-guidance strings (e.g. `["find the patient's diseases", "find genes associated with those diseases", "find expression data for those genes"]`), and each stage's guidance text is included in the `state`/`Choice.instructions` for hops expected to be in that stage, advancing the stage pointer each time the `Noul` for "have we advanced to the next stage" (or the top-ranked edge's semantics) indicates progress. The exact stage-advancement heuristic is an implementation detail to be finalized during the `navigator.py` task; the requirement is behavioral (biases edge choice by expected stage, tests "done" against the final stage's class description, returns top-N distinct paths).
- **Beam search correctness**: path scoring must use sum of log-probabilities (not naive multiplication) to avoid floating-point underflow on long paths and to avoid unfairly penalizing longer-but-more-confident paths relative to shorter-but-less-confident ones.
- **Already-bootstrapped state**: `pyproject.toml` already declares `neo4j-rust-ext`, `neo4j-viz[neo4j,streamlit]`, `typesafe-sdk`, `python-dotenv`, `streamlit` as dependencies and `pytest`/`pytest-asyncio` as dev dependencies (managed via `uv`); `src/neo4jev/__init__.py` exists (empty); `.gitignore` already correctly excludes `.env`, `.venv`, `.ipynb_checkpoints`, `.streamlit/secrets.toml`; the git repository already exists with one prior commit (`.gitignore` + `LICENSE`), tracking `origin` = `git@github.com:jexp/neo4jev.git`.

## Acceptance Criteria

- [ ] `README.md` exists and accurately documents setup and usage for both notebooks and the Streamlit app.
- [ ] `uv sync && pytest tests/unit` passes with zero network access.
- [ ] `pytest tests/integration` passes against the live Companies KG using `.env`.
- [ ] All three notebooks execute top-to-bottom without error against the live Companies KG, and `03_full_traversal.ipynb` visibly demonstrates all three goal modes with rendered graphs.
- [ ] `streamlit run app/streamlit_app.py` allows: picking a label, searching for a start node via each detected lookup mode, selecting each of the three goal modes with mode-appropriate inputs, running a traversal, and seeing an interactive graph with the path(s) visually distinct from neighborhood context.
- [ ] Path-intent mode, given a multi-hop natural-language pattern description, returns more than one distinct path when multiple qualifying paths exist in the graph.
- [ ] No relationship type, label, or property name is hardcoded in `src/neo4jev/` application logic (aside from illustrative examples in docs/comments).

## Out of Scope

- Formal path-pattern grammar/DSL compiled to Cypher.
- Persistent run history/database.
- Write access to Neo4j.
- Multi-user/auth beyond existing `.env` credentials.
- Deployment/hosting of the Streamlit app.

## Open Questions

- Exact heuristic for stage-advancement in path-intent mode (upfront LLM decomposition of the pattern vs. simple keyword-based stage list) — left as an implementation decision for the `navigator.py` task, to be resolved empirically against the Companies KG (which lacks a patient/gene/disease schema, so path-intent mode will be demonstrated with a structurally analogous multi-hop pattern native to the Companies KG, e.g. "find the news articles that mention companies competing with a given company's suppliers").
- Whether `Score` (the third TypeSafe question primitive) has any use in this feature — currently unused; left available for future refinement (e.g. ranking path quality) but not required.
