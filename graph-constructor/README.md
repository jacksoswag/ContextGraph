# Relation extractor

Terminal tool: give it some topics, it fetches articles, extracts
(subject, relation, predicate) relations with spaCy dependency parsing plus
rule-based cleanup, embeds each one with a sentence-transformer
(`all-MiniLM-L6-v2`), and writes a SQLite file with the triples, the article
link, and the publish date.

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run

```bash
# Wikipedia + arXiv (default), 20 articles per topic per source, hyperedge output
python main.py photosynthesis "neural networks" --count 20

# add Google web search + Scholar (needs SERPER_API_KEY), plain flat triples
python main.py "gene editing" --count 30 --google --scholar --flat -o crispr.sqlite3
```

| Flag | Meaning |
| --- | --- |
| `domains` | one or more topics (positional) |
| `-n, --count` | max articles per topic per source (default 10) |
| `--wikipedia / --no-wikipedia` | Wikipedia source (default on) |
| `--arxiv / --no-arxiv` | arXiv source (default on) |
| `--google` | Google web search via serper.dev (default off) |
| `--scholar` | Google Scholar via serper.dev (default off) |
| `--flat` | plain binary triples instead of hyperedges |
| `-o, --out` | output SQLite path (default `graph.sqlite3`) |

`--google` / `--scholar` need a serper.dev key in `SERPER_API_KEY` (read from the
environment or a local `.env`).

## Output

One SQLite file. Every row has the triple (`subject`, `relation`, `predicate`),
the article `url`, the `published` date, the `source`, the `topic`, and an
`embedding` (float32 blob). Hyperedge mode (default) also keeps
`subject_modifiers`, `predicate_modifiers`, `quantifier`, `tense`, `truth`, and
`specifics` (dates/numbers as JSON). `--flat` writes a `triples` table; hyperedge
mode writes a `hyperedges` table.

Rows are de-duplicated on `(subject, relation, predicate, url)`, so re-running
against an existing output file adds only new relations. A companion
`vocab.sqlite3` store (path set in `config.py`) is created alongside the output
to map concept/literal surface text to stable integer ids.

## Files

| File | Role |
| --- | --- |
| `main.py` | CLI, pipeline orchestration, SQLite output |
| `sources.py` | fetch articles from Wikipedia / arXiv / Google / Scholar |
| `extract.py` | sentence/clause → relation extraction (`find_connections`) |
| `nlp.py` | spaCy roles + text cleanup + coordinated-phrase expansion |
| `vocab.py` | concept/literal text→id store, embeddings, `ConnectionEndpoint` |
| `config.py` | configuration |
