# neo4jev — Graph Navigation Demo (TypeSafe Jev + Neo4j)

A demo application that navigates a Neo4j graph one hop at a time using
TypeSafe's (`typesafe-sdk`) `system_one` structured-decision API instead of
free-text LLM generation. At each visited node, the outgoing relationships are
presented as `Choice` options (relationship type, relationship properties, target
node label/properties), and the API returns a full probability distribution over
which relationship to follow next. A `Noul` ("has the goal been reached?") question
rides in the same `system_one` call as the `Choice`, so each hop costs exactly one
round-trip regardless of how many questions are asked.

Top-k/cutoff selection over the returned probabilities implements a beam search: at
each step the search may branch into several candidate next-hops, and the
best-scoring chain(s) — ranked by sum of log-probabilities, to avoid float underflow
and length bias — become "the path(s) taken". Results are rendered in an interactive
[`neo4j-viz`](https://pypi.org/project/neo4j-viz/) graph alongside the surrounding
neighborhood.

> **Status**: the library, the three notebooks and the Streamlit app are built and run
> against the live `companies2` graph. The notebooks and the app execute end to end; the
> TypeSafe `system_one` calls inside them need `TYPESAFE_API_KEY` in `.env` — without it
> each call is attempted, its failure is displayed verbatim, and the surrounding pipeline
> is exercised on explicitly labelled stand-in answers. Nothing is ever presented as
> TypeSafe output that did not come from TypeSafe. See
> [`.plans/prd-graph-navigation-demo.md`](.plans/prd-graph-navigation-demo.md) for the
> authoritative requirements and design rationale, and
> [`.plans/tasks-graph-navigation-demo.yml`](.plans/tasks-graph-navigation-demo.yml) for
> the build status of each piece.

## Architecture overview

The demo targets the public Neo4j "Companies KG" instance
(`neo4j+s://demo.neo4jlabs.com:7687`, database `companies2`) by default, but the
application itself is **schema-agnostic**: labels, relationship types, and property
names are discovered live via introspection and are never hardcoded, so it can be
pointed at any other Neo4j instance via `.env`. The `companies2` database is the target
here; the older `companies` database on the same server has a different index layout (see
[`AGENTS.md`](AGENTS.md)).

Delivery is in two layers built on a single shared library:

- **`src/neo4jev/`** — the reusable core: Neo4j access/introspection, the
  `system_one`-driven single-hop navigator and beam search, shared data types, and
  `neo4j-viz` rendering helpers. No Streamlit- or notebook-specific code lives here.
- **`notebooks/`** — Jupyter notebooks that prove out the approach end-to-end
  against the real graph.
- **`app/streamlit_app.py`** — a Streamlit UI wrapping the same library code
  (start-node search, goal specification, run controls, interactive visualization).

Modules under `src/neo4jev/`:

| Module | Responsibility |
| --- | --- |
| `config.py` | Loads settings from `.env` via `python-dotenv` (`NEO4J_URL`/`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`, `TYPESAFE_API_KEY`). |
| `types.py` | Shared data contract: `NavCandidate`, `NavStep`, `NavResult`, `BeamState`, `GoalSpec`. |
| `neo4j_access.py` | Driver lifecycle, label/index discovery, start-node search (exact/fulltext/vector), capped outgoing-relationship listing, neighborhood fetch for visualization. |
| `navigator.py` | Wraps the TypeSafe client to run one hop (`Choice` + `Noul` in a single `system_one` call) and drives the beam search across hops. |
| `viz.py` | Builds a `neo4j_viz.VisualizationGraph` from a `NavResult`, styling path(s) distinctly from neighborhood context; exposes both a static `.render()` helper and a `render_widget()`/Streamlit-ready helper. |

### Goal modes

Every run is driven by a `GoalSpec`, selectable per run, in one of three modes:

1. **Free-text goal** — a natural-language description of what's being looked for
   (e.g. "find a company that competed with Apple"). The navigator runs a beam
   search and returns the single best-scoring path, terminating early once the
   `Noul` "goal reached" answer crosses a configurable threshold.
2. **Explicit target node** — a specific node picked via lookup (exact, fulltext,
   or vector, depending on what indexes are detected for the chosen label).
   Navigation continues until that exact node is reached.
3. **Path-intent / relationship-pattern goal** — a natural-language description of
   the *kind* of path to follow (e.g. "find me the paths from the patient to the
   gene expressions for their diseases"), describing a semantic multi-hop pattern
   rather than one fixed destination. Since "paths" is plural, this mode runs
   multiple beam branches and returns the top-N distinct scoring paths rather than
   a single path.

## Setup

Requires Python >=3.12 and [`uv`](https://docs.astral.sh/uv/).

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Copy the environment template and fill in your credentials:

   ```bash
   cp .env.example .env
   ```

   Set the following in `.env`:

   | Variable | Description |
   | --- | --- |
   | `NEO4J_URL` (or `NEO4J_URI`) | Bolt connection URI, e.g. `neo4j+s://demo.neo4jlabs.com:7687` for the public Companies KG. If both `NEO4J_URI` and `NEO4J_URL` are set, `NEO4J_URI` takes precedence. |
   | `NEO4J_USERNAME` | Database username (`companies2` for the public demo instance). |
   | `NEO4J_PASSWORD` | Database password (`companies2` for the public demo instance). |
   | `NEO4J_DATABASE` | Database name (`companies2` for the public demo instance). |
   | `TYPESAFE_API_KEY` | API key for the TypeSafe `system_one` structured-decision API. Required for the navigator: without it a run is attempted, fails with the API's own error, and the notebooks and app fall back to explicitly labelled stand-in answers rather than fabricating model output. |

   The public Companies KG credentials above are not secret and are used directly
   by the integration tests — no separate `integration.env` is needed for this
   project.

### What the demo graph offers

Everything schema-shaped is read from the live database, so these are facts about the
default target (`companies2`), not assumptions in the code — `notebooks/01_explore_graph.ipynb`
prints them for whatever instance you point at:

| | |
| --- | --- |
| Labels | 15 labels: `Organization`, `Person`, `Article`, `Chunk`, `Patent`, `SECFiling`, `City`, `Country`, `Region`, `IndustryCategory`, `Technology`, `Investment`, `NAICSCode`, `IPCClass`, `CPCClass` |
| Fulltext indexes | `organization_fullName` (on `Organization.fullName`), `person_name` (on `Person.name`) |
| Vector indexes | `news_openai_small` (on `Chunk.embedding_3_small`, 1536 dims, cosine) |
| Embeddings | the server's `genai` plugin is unconfigured and no provider key is supplied, so `neo4j_access.default_embedder` is a deterministic hash-seeded placeholder — vector search is dimensionally real but not semantically meaningful. Pass a real `embedder` to `open_access()` to change that. |

**One more property worth knowing before reading a traversal:** each hop is capped at ≤10
relationships per type and ≤60 in total (REQ-NF-005), and relationship types are visited in
*name* order, so the total cap fills alphabetically. On a supernode such as Apple Inc. (1354
outgoing edges) the 60-edge budget is spent on patent and classification edges before the
alphabet reaches `HAS_COMPETITOR`, `HAS_SUPPLIER` or `USES_TECHNOLOGY` — those contribute
nothing at all to that hop. Pick a start node whose capped neighbourhood can express the
goal, or raise `total_cap` (it is a parameter, not a constant).

For API requests, each node or relationship property map is summarized to at most
1,200 characters. Short values are retained first; long text, aliases and vectors
may be previewed or omitted, with an explicit truncation marker. Full properties
remain in navigation results and visualization. Goals requiring omitted details
may need more focused graph data. This is a per-map character budget, not a
request-wide token limit: large candidate sets, goals or path histories can still
exceed the API's input limit.

## Usage

### Notebooks

Jupyter is not yet a declared project dependency, so run notebooks under
`notebooks/` via an ad hoc environment, e.g.:

```bash
uv run --with jupyterlab jupyter lab
```

(or use your editor's built-in notebook support against the `uv`-managed
`.venv`, adding `ipykernel` as needed). Run the notebooks in order, against the
live Companies KG:

- `01_explore_graph.ipynb` — introspects the connected graph: lists labels,
  detects available indexes (fulltext/vector) per label, and demonstrates
  start-node search.
- `02_navigator_dry_run.ipynb` — runs a single navigator hop against a real node,
  inspecting the returned `Choice` probabilities, confidence, and `Noul` value,
  and the top-k/cutoff selection applied to them. (The full beam search is
  notebook 03.)
- `03_full_traversal.ipynb` — runs the full beam search end-to-end for all three
  goal modes (free text, target node, path intent) and renders each `NavResult`
  via `viz.py`.

`02` and `03` issue real `system_one` calls, so they need `TYPESAFE_API_KEY`. Without it they
still run top-to-bottom: each call is attempted, the API's error is shown verbatim, and the
surrounding pipeline is exercised on explicitly labelled stand-in answers (never presented as
TypeSafe output). As a consequence, nothing in those two notebooks terminates on
`goal_reached` until a key is present, and `target_node` mode cannot be demonstrated by a
stand-in at all — both limitations are stated in the notebooks themselves.

### Streamlit app

```bash
uv run streamlit run app/streamlit_app.py
```

The app provides:

- A label picker for the connected graph.
- Start-node search using whichever lookup modes (exact/fulltext/vector) are
  detected for the chosen label.
- A goal-mode picker (free text / target node / path intent) with mode-specific
  input widgets.
- Run controls (top-k, probability cutoff, max depth, max API-call budget, and N
  for path-intent mode).
- An interactive `neo4j-viz` graph of the result, with the path(s) visually
  distinguished from neighborhood context, plus a per-hop probability/trace view.

## Testing

See [`AGENTS.md`](AGENTS.md) for the authoritative feedback-loop commands. In
summary:

```bash
uv run pytest tests/unit          # no network required
uv run pytest tests/integration   # hits the live Companies KG + TypeSafe API via .env
uv run pytest                     # everything
```

## Further reading

- [`.plans/prd-graph-navigation-demo.md`](.plans/prd-graph-navigation-demo.md) —
  full PRD with detailed functional/non-functional requirements and technical
  considerations.
- [`.plans/tasks-graph-navigation-demo.yml`](.plans/tasks-graph-navigation-demo.yml)
  — task breakdown and current implementation status.
