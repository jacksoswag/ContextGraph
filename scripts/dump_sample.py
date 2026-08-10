#!/usr/bin/env python3
"""Dump N random rendered triples/hyperedges from the corpus graph to a text file
for manual inspection, plus a sample of node-merge folds to spot-check merge quality.

Rendering: subject --relation--> object. A reified hyperedge endpoint (an edge whose
source/target is itself an edge) is shown in parentheses, recursively. Node text that
contains '|' is a set of merged synonyms (e.g. "united states|usa").
"""
import argparse
import random
import sqlite3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    con = sqlite3.connect(args.store)
    ncache, ecache = {}, {}

    def get_node(nid):
        if nid not in ncache:
            r = con.execute("SELECT text FROM nodes WHERE id=?", (nid,)).fetchone()
            ncache[nid] = (r[0] if r and r[0] else nid)
        return ncache[nid]

    def get_edge(eid):
        if eid not in ecache:
            ecache[eid] = con.execute(
                "SELECT source_id,rel_type,target_id FROM edges WHERE id=?", (eid,)).fetchone()
        return ecache[eid]

    def render(eid, depth=0):
        r = get_edge(eid)
        if not r or depth > 6:
            return eid
        s, rel, t = r
        sp = get_node(s) if s.startswith("n_") else "(" + render(s, depth + 1) + ")"
        tp = get_node(t) if t.startswith("n_") else "(" + render(t, depth + 1) + ")"
        return f"{sp} --{rel}--> {tp}"

    tot_n = con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    tot_e = con.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    sample = con.execute(
        "SELECT id FROM edges ORDER BY RANDOM() LIMIT ?", (args.n,)).fetchall()

    with open(args.out, "w") as f:
        f.write(f"# {len(sample)} random entries from {args.store}\n")
        f.write(f"# graph: {tot_n:,} nodes / {tot_e:,} edges\n")
        f.write("# format:  subject --relation--> object\n")
        f.write("#   (parentheses) = nested reified hyperedge;  a|b = merged synonyms\n")
        f.write("#" + "=" * 70 + "\n\n")
        for i, (eid,) in enumerate(sample, 1):
            f.write(f"{i:4d}.  {render(eid)}\n")

        try:
            folds = con.execute(
                "SELECT victim_text, canonical_text FROM sleep_log ORDER BY RANDOM() LIMIT ?",
                (args.folds,)).fetchall()
        except sqlite3.Error:
            folds = []
        if folds:
            f.write("\n\n" + "=" * 72 + "\n")
            f.write(f"SAMPLE OF {len(folds)} NODE MERGES  (victim  =>  canonical)\n")
            f.write("=" * 72 + "\n\n")
            for v, c in folds:
                f.write(f"  {v!r:45s}  =>  {c!r}\n")

    con.close()
    print(f"wrote {args.out}  ({len(sample)} entries, {len(folds)} merge examples)")


if __name__ == "__main__":
    main()
