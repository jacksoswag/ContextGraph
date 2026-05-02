# Brain

Desktop wrapper and local runtime for the Brain research graph engine.

## First Run From Git

1. Install Homebrew, Node.js, and npm if needed.
2. Clone this repo.
3. Run:

```bash
npm install
npm start
```

On first launch, the Electron app asks for your Serper API key and stores it in a private runtime `.env`. It then creates `venv/`, installs Python dependencies, and builds `physics-engine` and `thought-engine`.

You can also bootstrap manually:

```bash
npm run setup
```

## API Key

The app needs:

```text
SERPER_API_KEY
```

Do not commit `.env`. Use `.env.example` as the template. Electron stores your real key in app runtime storage.

## Build App

```bash
npm run make
```

Artifacts are written to `out/make/`.
