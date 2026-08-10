#!/usr/bin/env python3
"""Topic -> articles -> spaCy relations -> embeddings -> SQLite.

    python main.py photosynthesis "neural networks" --count 20
    python main.py "gene editing" --count 30 --google --scholar --flat -o crispr.sqlite3

Wikipedia and arXiv are on by default (no key). Google web search and Google
Scholar are opt-in (--google / --scholar) and need SERPER_API_KEY (read from the
environment or a local .env). Output is one SQLite file of relation triples with
the article link, publish date, and a sentence embedding per row. Hyperedge mode
(default) also keeps modifiers, quantifier, tense, truth, and date/number
specifics; --flat keeps plain binary triples.
"""
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import sources
from config import MAX_CONNECTIONS_PER_QUERY
from extract import find_connections
from nlp import expand_clean_connection_record
from vocab import ConnectionEndpoint, embed_texts, literal_from_index

FLAT_COLUMNS = ["subject", "relation", "predicate", "source", "url", "published", "topic"]
HYPEREDGE_COLUMNS = [
    "subject", "relation", "predicate", "subject_modifiers", "predicate_modifiers",
    "quantifier", "tense", "truth", "specifics", "source", "url", "published", "topic",
]


def _load_env(path=".env"):
    env = Path(path)
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _literal(index):
    try:
        index = int(index)
    except (TypeError, ValueError):
        return ""
    return literal_from_index(index) if index >= 0 else ""


def _triple(connection):
    subject = connection.get("subject")
    predicate = connection.get("predicate")
    if not isinstance(subject, ConnectionEndpoint) or not isinstance(predicate, ConnectionEndpoint):
        return None
    s, p = subject.asu_value(), predicate.asu_value()
    r = _literal(connection.get("connection"))
    if not (s and r and p):
        return None
    return s, r, p


def _flat_rows(connection, article):
    for clean in expand_clean_connection_record(connection) or [connection]:
        triple = _triple(clean)
        if triple is None:
            continue
        s, r, p = triple
        yield {
            "subject": s, "relation": r, "predicate": p,
            "source": article.source, "url": article.url,
            "published": article.published, "topic": article.topic,
        }


def _hyperedge_row(connection, article):
    triple = _triple(connection)
    if triple is None:
        return None
    s, r, p = triple
    subject = connection["subject"]
    predicate = connection["predicate"]
    specifics = {
        "subject": connection.get("subject_specifics", []),
        "predicate": connection.get("predicate_specifics", []),
        "connection": connection.get("connection_specifics", []),
    }
    return {
        "subject": s, "relation": r, "predicate": p,
        "subject_modifiers": "; ".join(subject.modifier_value()),
        "predicate_modifiers": "; ".join(predicate.modifier_value()),
        "quantifier": _literal(getattr(subject, "quantifier", -1)),
        "tense": _literal(getattr(subject, "tense", -1)),
        "truth": int(getattr(subject, "truth", -1)),
        "specifics": json.dumps(specifics),
        "source": article.source, "url": article.url,
        "published": article.published, "topic": article.topic,
    }


def extract_rows(articles, hyperedge):
    rows = []
    for article in articles:
        block = {
            "content": article.text,
            "tag": f"{article.title}|{article.url}",
            "url": article.url,
        }
        connections = find_connections(
            [block], query=article.topic, connection_limit=MAX_CONNECTIONS_PER_QUERY
        )
        for connection in connections:
            if hyperedge:
                row = _hyperedge_row(connection, article)
                if row:
                    rows.append(row)
            else:
                rows.extend(_flat_rows(connection, article))
    return rows


def embed_rows(rows):
    if not rows:
        return
    texts = [f"{r['subject']} {r['relation']} {r['predicate']}" for r in rows]
    vectors = embed_texts(texts)
    for row, vector in zip(rows, vectors):
        row["embedding"] = vector.tobytes()


def write_sqlite(rows, out_path, hyperedge):
    table = "hyperedges" if hyperedge else "triples"
    columns = (HYPEREDGE_COLUMNS if hyperedge else FLAT_COLUMNS) + ["embedding"]
    coldefs = ",\n    ".join(
        f"{c} BLOB" if c == "embedding" else (f"{c} INTEGER" if c == "truth" else f"{c} TEXT")
        for c in columns
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(out)
    conn.executescript(
        f"CREATE TABLE IF NOT EXISTS {table} (id INTEGER PRIMARY KEY,\n    {coldefs});\n"
        f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_unique "
        f"ON {table}(subject, relation, predicate, url);"
    )
    placeholders = ",".join("?" for _ in columns)
    with conn:
        conn.executemany(
            f"INSERT OR IGNORE INTO {table}({','.join(columns)}) VALUES({placeholders})",
            [tuple(row.get(c) for c in columns) for row in rows],
        )
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.close()
    return table, count


def main():
    parser = argparse.ArgumentParser(
        description="Extract relation triples from articles into SQLite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("domains", nargs="+", help="topics to search for")
    parser.add_argument("-n", "--count", type=int, default=10,
                        help="max articles per topic per source (default 10)")
    parser.add_argument("--wikipedia", action=argparse.BooleanOptionalAction, default=True,
                        help="Wikipedia source (default on; --no-wikipedia to disable)")
    parser.add_argument("--arxiv", action=argparse.BooleanOptionalAction, default=True,
                        help="arXiv source (default on; --no-arxiv to disable)")
    parser.add_argument("--google", action="store_true",
                        help="Google web search via serper.dev (needs SERPER_API_KEY)")
    parser.add_argument("--scholar", action="store_true",
                        help="Google Scholar via serper.dev (needs SERPER_API_KEY)")
    parser.add_argument("--flat", action="store_true",
                        help="emit plain binary triples instead of hyperedges")
    parser.add_argument("-o", "--out", default="graph.sqlite3", help="output SQLite path")
    args = parser.parse_args()

    _load_env()
    hyperedge = not args.flat

    articles = sources.fetch(
        args.domains, args.count,
        wikipedia=args.wikipedia, arxiv=args.arxiv,
        google=args.google, scholar=args.scholar,
    )
    if not articles:
        print("[error] no articles fetched", file=sys.stderr)
        return 1
    print(f"[extract] {len(articles)} articles -> "
          f"{'hyperedges' if hyperedge else 'flat triples'}")

    rows = extract_rows(articles, hyperedge)
    print(f"[extract] {len(rows)} relations")
    embed_rows(rows)
    table, count = write_sqlite(rows, args.out, hyperedge)
    print(f"[done] {count} rows in '{table}' -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
