from __future__ import annotations
# Phase A corpus validation: embedding coverage, hyperedge well-formedness, degree distribution,
# find_vec grounding. Read-only. Run: PYTHONPATH=src python scripts/phaseA_health.py [--store PATH]
import argparse, sqlite3, sys, statistics as st
from pathlib import Path
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from graph.store import GraphStore

def run(path):
    con = sqlite3.connect(path)
    n_nodes = con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    n_edges = con.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    n_he = con.execute("SELECT COUNT(*) FROM edges WHERE source_id LIKE 'e_%' OR target_id LIKE 'e_%'").fetchone()[0]
    nv = con.execute("SELECT COUNT(*) FROM nodes WHERE info_vector IS NOT NULL").fetchone()[0]
    ev = con.execute("SELECT COUNT(*) FROM edges WHERE info_vector IS NOT NULL").fetchone()[0]
    print(f"STORE {path}")
    print(f"  nodes={n_nodes}  edges={n_edges}  hyperedges(e_ endpoint)={n_he}")
    print(f"  node info_vector: {nv}/{n_nodes} ({100*nv/max(n_nodes,1):.1f}%)   "
          f"edge info_vector: {ev}/{n_edges} ({100*ev/max(n_edges,1):.1f}%)")

    # hyperedge well-formedness: e_ endpoints must resolve to a real edges row
    bad = con.execute(
        "SELECT COUNT(*) FROM edges e WHERE (e.source_id LIKE 'e_%' AND NOT EXISTS "
        "(SELECT 1 FROM edges x WHERE x.id=e.source_id)) OR (e.target_id LIKE 'e_%' AND NOT EXISTS "
        "(SELECT 1 FROM edges y WHERE y.id=e.target_id))").fetchone()[0]
    # n_ endpoints must resolve to a real nodes row
    badn = con.execute(
        "SELECT COUNT(*) FROM edges e WHERE (e.source_id LIKE 'n_%' AND NOT EXISTS "
        "(SELECT 1 FROM nodes x WHERE x.id=e.source_id)) OR (e.target_id LIKE 'n_%' AND e.target_id!='' "
        "AND NOT EXISTS (SELECT 1 FROM nodes y WHERE y.id=e.target_id))").fetchone()[0]
    print(f"  dangling e_ endpoints={bad}   dangling n_ endpoints={badn}  (0/0 = well-formed)")

    # degree distribution over node endpoints (genericity health: not all-hub / all-leaf)
    degs = [r[0] for r in con.execute(
        "SELECT cnt FROM (SELECT source_id id, COUNT(*) cnt FROM edges GROUP BY source_id "
        "UNION ALL SELECT target_id, COUNT(*) FROM edges GROUP BY target_id) "
        "WHERE id LIKE 'n_%' GROUP BY id").fetchall()]
    if degs:
        degs.sort()
        p = lambda q: degs[min(len(degs)-1, int(q*len(degs)))]
        print(f"  node degree: min={degs[0]} p50={p(.5)} p90={p(.9)} p99={p(.99)} max={degs[-1]} "
              f"mean={st.mean(degs):.1f}")
        leaf = sum(1 for d in degs if d <= 1); hub = sum(1 for d in degs if d >= 50)
        print(f"  leaf(deg<=1)={100*leaf/len(degs):.1f}%  hub(deg>=50)={100*hub/len(degs):.1f}%")
    con.close()

    # find_vec grounding sanity (needs vectors)
    if nv > 0:
        store = GraphStore(path)
        print("  find_vec grounding:")
        for q in ["black hole", "neutron star", "dark matter", "gravitational wave", "redshift"]:
            hits = store.find_vec(q, 3)
            txt = ", ".join((store.text(h) or "?") for h in (hits or [])[:3])
            print(f"    {q:<20} -> {txt or 'NO MATCH'}")
        store.close()
    else:
        print("  find_vec: SKIPPED (no embeddings — store built with DI_GRAPH_EMBED=0)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=".di-ui/graph.astro.sqlite")
    run(ap.parse_args().store)
