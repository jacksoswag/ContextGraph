# Sweep 2 — a second 500k that does not overlap the first

Everything for a second corpus pass plus its own graph corpus, guaranteed
disjoint from sweep 1. All commands run from the repo root.

## What "no overlap" rests on

`data/raw/corpus_v2/index.sqlite3` is pre-seeded with every id sweep 1 collected
(501,358 of them), tagged as channel `_prior`. The scraper dedups with
`INSERT OR IGNORE` on `id`, so any prior article is dropped before it is counted
or written. `_prior` is not one of the four real channels, so the scraper's
count-resume starts `wiki_taxonomy / arxiv_main / wiki_random / arxiv_niche` back
at 0 and stops only after 500k brand-new articles. Sweep 1's `data/raw/corpus/`
and `data/graph/` are never touched.

Verified by construction and by test: a known prior id is rejected by dedup, all
four channels resume at 0, and the graph orchestrator counts `seen WHERE channel
!= '_prior'` so the sentinels never satisfy the target prematurely.

## One-time prep (already done)

```
scripts/run_sweep2.sh prepare        # builds corpus_v2 index + seeds.txt + manifest
```

This copies prior ids as `_prior`, carries the 19,084 walked categories, and
recovers sweep 1's unwalked category frontier into `seeds.txt` (the ~108k
subcategories sweep 1 discovered but never reached). The scraper reads those via
`DI_SEEDS_FILE` so the taxonomy walk continues into new ground instead of
re-treading sweep 1.

Re-run with `--force` to rebuild. `--no-network` skips seed discovery (taxonomy
then leans on wiki_random + arxiv for new articles).

## Disk — read before building the graph

The graph build needs roughly 40-45 GB free. Right now the volume has ~20 GB
free because sweep 1's graph artifacts hold ~44 GB:

- `data/graph/graph.corpus.sql` — 27 GB text dump, regenerable from the `.sqlite`
- `data/graph/graph.corpus.sqlite` — 18 GB binary store

Free space before `graph` (the launcher refuses below `DI_MIN_FREE_GB`, default
45). The cheapest win is deleting the 27 GB `.sql` dump. It is redundant with the
binary store and can be recreated any time with `sqlite3 data/graph/graph.corpus.sqlite .dump > graph.corpus.sql`.
The scrape itself is tiny (~200 MB) and has no disk concern.

## Run it

```
scripts/run_sweep2.sh calibrate 180   # optional: 180s probe -> projected wall-clock
scripts/run_sweep2.sh scrape          # full 500k NEW scrape, backgrounded under caffeinate
scripts/run_sweep2.sh status          # scrape + graph progress
# ... wait for the scrape to reach 500k (hours; see progress.json eta) ...
scripts/run_sweep2.sh graph           # build the graph corpus from corpus_v2
scripts/run_sweep2.sh stop            # stop scrape and/or graph
```

`scrape` and `graph` detach and keep the Mac awake. Both are resumable: re-run
the same command and they pick up from `seen` / `visited_cat` and the part
stores. You can start `graph` while `scrape` is still running; it waits for the
scrape to finish (or for 500k new articles) before extracting.

## Outputs

| Sweep 1 | Sweep 2 |
| --- | --- |
| `data/raw/corpus/` | `data/raw/corpus_v2/` |
| `data/graph/graph.corpus.sqlite` | `data/graph_v2/graph.corpus_v2.sqlite` |

The v2 shards under `data/raw/corpus_v2/<channel>/` contain only the new
articles, so the graph build reads exactly the 500k new set.

## Knobs

- `DI_CORPUS_DIR` — scrape output dir (launcher sets it to `corpus_v2`)
- `DI_SEEDS_FILE` — fresh taxonomy seeds (launcher sets it to `corpus_v2/seeds.txt`)
- `DI_TARGET` — total article target (default 500000)
- `DI_MIN_FREE_GB` — disk guard for `graph` (default 45)
- `INGEST_WORKERS` — extract parallelism for the graph build (default 6)

## Pipeline dependency

Graph construction uses the ingest code in the `ingest-hyperedge` worktree at
`/tmp/di-hyperedge`. That path is on volatile `/tmp`; if it is gone (reboot),
`ingest_run_sweep2.py` recreates it with `git worktree add /tmp/di-hyperedge
ingest-hyperedge` automatically.
