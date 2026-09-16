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

> **Status**: this repository is being built out from an approved PRD/task plan.
> The layout below (`src/neo4jev/`, `notebooks/`, `app/streamlit_app.py`) describes
> the *target* structure; most of these files do not exist yet. See
> [`.plans/prd-graph-navigation-demo.md`](.plans/prd-graph-navigation-demo.md) for
> the full, authoritative requirements and design rationale, and
> [`.plans/tasks-graph-navigation-demo.yml`](.plans/tasks-graph-navigation-demo.yml)
> for the current build status of each piece.

## Architecture overview

The demo targets the public Neo4j "Companies KG" instance
(`neo4j+s://demo.neo4jlabs.com:7687`, database `companies`) by default, but the
application itself is **schema-agnostic**: labels, relationship types, and property
names are discovered live via introspection and are never hardcoded, so it can be
pointed at any other Neo4j instance via `.env`.

Delivery is in two layers built on a single shared library:

- **`src/neo4jev/`** — the reusable core: Neo4j access/introspection, the
  `system_one`-driven single-hop navigator and beam search, shared data types, and
  `neo4j-viz` rendering helpers. No Streamlit- or notebook-specific code lives here.
- **`notebooks/`** — Jupyter notebooks that prove out the approach end-to-end
  against the real graph.
- **`app/streamlit_app.py`** — a Streamlit UI wrapping the same library code
  (start-node search, goal specification, run controls, interactive visualization).

Planned modules under `src/neo4jev/`:

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
   | `NEO4J_URL` (or `NEO4J_URI`) | Bolt connection URI, e.g. `neo4j+s://demo.neo4jlabs.com:7687` for the public Companies KG. If both `NEO4J_URI` and `NEO4J_URL` are set, `NEO4J_URI` takes precedence (subject to change while `config.py` is still being built out — see the task file). |
   | `NEO4J_USERNAME` | Database username (`companies` for the public demo instance). |
   | `NEO4J_PASSWORD` | Database password (`companies` for the public demo instance). |
   | `NEO4J_DATABASE` | Database name (`companies` for the public demo instance). |
   | `TYPESAFE_API_KEY` | API key for the TypeSafe `system_one` structured-decision API. |

   The public Companies KG credentials above are not secret and are used directly
   by the integration tests — no separate `integration.env` is needed for this
   project.

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
  inspecting the returned `Choice` probabilities, confidence, and `Noul` value
  before running a full beam search.
- `03_full_traversal.ipynb` — runs the full beam search end-to-end for all three
  goal modes (free text, target node, path intent) and renders each `NavResult`
  via `viz.py`.

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
