# Decentralized Intelligence

Decentralized Intelligence is a desktop research graph engine that turns a two-target research prompt into a live, source-grounded knowledge graph. It combines web search, linguistic extraction, native graph physics, graph-walking thought agents, and local synthesis into an Electron desktop app with a Three.js visualization.

The app is designed around a simple question shape:

```text
How does Target A relate to Target B?
```

From there, the runtime gathers source material, extracts relationship candidates, simulates the graph into a navigable 3D structure, launches thought agents through the stabilized graph, and writes a concise cited synthesis.

## Visual Preview

Use this section for GIFs and screenshots of the physics simulation, graph settling behavior, and final research interface.

### Physics Simulation

<!-- Replace these placeholder paths after recording the simulation. -->

| Graph Forming | Graph Settling |
| --- | --- |
| ![Physics simulation forming a research graph](docs/media/physics-simulation-forming.gif) | ![Physics simulation settling into a stable graph](docs/media/physics-simulation-settling.gif) |

### Dashboard Screenshots

| Research Input | Stabilized Graph | Synthesized Report |
| --- | --- | --- |
| ![Target input view](docs/media/dashboard-input.png) | ![Stable 3D knowledge graph](docs/media/dashboard-stable-graph.png) | ![Generated synthesis report](docs/media/dashboard-report.png) |

Suggested media names:

- `docs/media/physics-simulation-forming.gif`
- `docs/media/physics-simulation-settling.gif`
- `docs/media/dashboard-input.png`
- `docs/media/dashboard-stable-graph.png`
- `docs/media/dashboard-report.png`

## What It Does

- Searches for source material with Serper.
- Scrapes and cleans web pages into extractable text blocks.
- Extracts semantic relationship candidates from source text.
- Stores information agents and graph connections in shared memory.
- Runs a native C++ physics engine to settle the graph in 3D space.
- Streams the graph into a Three.js dashboard at `http://localhost:8000`.
- Launches native thought processes that route through graph connections.
- Synthesizes successful relationship chains into a local report.
- Wraps the runtime in Electron for a desktop-first experience.

## Runtime Phases

The application moves through a small set of explicit phases:

| Phase | Description |
| --- | --- |
| Idle | The dashboard waits for Target A and Target B. |
| Research | Scraper and extraction workers gather source-backed relationship data. |
| Physics | The C++ physics engine settles the graph layout through shared memory. |
| Stable | The graph is pruned into an active connector subgraph. |
| Thinking | Thought agents traverse candidate paths between the two targets. |
| Synthesis | The local synthesizer writes the final relationship report. |

## Architecture

```text
Electron shell
  |
  |-- Python runtime
  |     |-- FastAPI / WebSocket dashboard server
  |     |-- scraper workers
  |     |-- extraction workers
  |     |-- graph and route management
  |     |-- synthesis and reporting
  |
  |-- Native C++ engines
  |     |-- physics-engine
  |     |-- thought-engine
  |
  |-- Frontend
        |-- Three.js graph visualization
        |-- research chat/report sidebar
        |-- query history sidebar
```

Key files:

- `electron/main.cjs` starts the desktop app, runtime setup, API-key prompt, backend process, and dashboard window.
- `main.py` coordinates shared memory, workers, physics, thought processes, and shutdown.
- `frontend/ui.py` serves the dashboard and streams graph state.
- `frontend/main.js` renders the live Three.js graph.
- `p_physics.cpp` runs the native graph physics simulation.
- `p_thought.cpp` runs native graph traversal for thought agents.
- `m_runtime.py` manages prompt execution and runtime state.
- `s_synthesis.py` turns successful paths into a source-cited report.

## Requirements

For development on macOS:

- Node.js and npm
- Python 3.11
- Homebrew
- `g++`
- raylib
- A Serper API key
- Ollama running locally for synthesis

The bootstrap script can install Python 3.11 and raylib through Homebrew when they are missing.

## API Key

The app needs:

```text
SERPER_API_KEY
```

Do not commit `.env`. Use `.env.example` as the template. On first launch, the Electron app asks for your Serper API key and stores it in a private runtime `.env` file.

## First Run From Git

1. Install Homebrew, Node.js, and npm if needed.
2. Clone this repository.
3. Install dependencies and start the Electron app:

```bash
npm install
npm start
```

On first launch, the app prompts for `SERPER_API_KEY`, creates `venv/`, installs Python dependencies, and builds the native `physics-engine` and `thought-engine` binaries.

You can also bootstrap manually:

```bash
npm run setup
```

## Development

Start the desktop app:

```bash
npm start
```

Run the runtime setup manually:

```bash
npm run setup
```

Package the Electron app:

```bash
npm run package
```

Build distributable artifacts:

```bash
npm run make
```

Artifacts are written to `out/make/`.

## Using the App

1. Launch the desktop app with `npm start`.
2. Enter a concept, entity, or topic in `Target A`.
3. Enter a second concept, entity, or topic in `Target B`.
4. Click `Launch Research`.
5. Watch the dashboard move from research to physics simulation, stabilized graph, thought traversal, and synthesis.
6. Review the generated report in the research chat sidebar.

Example prompts:

- `Urban Sprawl` and `Housing Affordability`
- `Microplastics` and `Marine Food Chains`
- `Remote Work` and `Downtown Retail`

## Capturing Physics Media

Recommended captures for the README:

- A short GIF of nodes appearing during research.
- A short GIF of the physics simulation settling into a stable structure.
- A screenshot of the empty dashboard before launch.
- A screenshot of the final stable graph.
- A screenshot of the generated report.

Place exported media in:

```text
docs/media/
```

Then replace or keep the placeholder filenames already referenced in the Visual Preview section.

## Runtime Data

During local development, runtime data is written inside the project directory:

- `venv/` for the Python environment and native binaries.
- `logs/` for startup and runtime logs.
- `sql/` for local caches.
- `results/` for generated reports and run output.

When packaged, Electron uses the app's private user-data runtime folder instead of writing runtime data into the read-only app bundle.

## Notes

- The dashboard server runs on `http://localhost:8000`.
- The frontend imports Three.js from `unpkg.com`.
- The native physics engine reads and writes graph positions through shared memory.
- The native thought engine samples routes through the stabilized graph and writes paths back for Python to load.
- Synthesis expects a local Ollama server at `http://localhost:11434`.

## License

ISC
