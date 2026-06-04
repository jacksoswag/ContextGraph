from __future__ import annotations
# Phase 1.1 relevance-spectrum probe (Gate 1). Settle the field per query, sort ‖x*‖², and decide
# whether the spectrum is BIMODAL (hot cluster | dead tail ⇒ §3.2 energy-defined readout buildable)
# or SMOOTH (gradual decay ⇒ any threshold is arbitrary, keep top-K). Sweeps the leak (decay_gamma),
# the spec's named lever for forcing bimodality.
# Run: PYTHONPATH=src python scripts/phase1_spectrum.py [--store PATH]
import argparse, dataclasses, sys
from pathlib import Path
import numpy as np
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from graph.store import GraphStore
from field.config import DEFAULT_CFG
from field.gather import gather_from_store

QUERIES = ["black hole", "neutron star", "gravitational wave", "dark matter", "cosmic ray",
           "dark energy", "supernova", "redshift", "quasar", "galaxy cluster",
           "white dwarf", "cosmic microwave background"]

# per-query spectrum descriptors on the sorted (desc) relevance r = ‖x*‖²
def _np(r): return np.asarray(r.detach().cpu()) if hasattr(r, "detach") else np.asarray(r)
def descriptors(r):
    r = np.sort(_np(r))[::-1]
    rmax = float(r[0]) + 1e-12
    rn = r / rmax
    active = rn[rn > 1e-3]                                   # nodes with non-trivial relevance
    n_act = int(active.size)
    pr = float((r.sum() ** 2) / (np.square(r).sum() + 1e-12))  # participation ratio (effective support)
    # largest log-gap (cliff) within the active band — a bimodal spectrum has one big isolated drop
    if n_act >= 3:
        ratios = active[:-1] / (active[1:] + 1e-12)
        gi = int(np.argmax(ratios)); gap = float(ratios[gi]); gpos = (gi + 1) / n_act
    else:
        gap, gpos = 1.0, 1.0
    csum = np.cumsum(r) / (r.sum() + 1e-12)
    return dict(n_act=n_act, pr=pr, gap=gap, gpos=gpos,
                m10=float(csum[min(9, len(csum)-1)]), m34=float(csum[min(33, len(csum)-1)]))

def cfg(**kw):
    return dataclasses.replace(DEFAULT_CFG, N_max=200, H_max=3000, eps_x=1e-3, decay=1.5, **kw)

def run(store_path):
    store = GraphStore(store_path)
    print(f"STORE {store_path}\n")
    for gamma in (0.0, 1.0, 2.0, 4.0):
        c = cfg(decay_gamma=gamma)
        rows = []
        for q in QUERIES:
            hits = store.find_vec(q, 1) or store.find(q, 1)
            if not hits: continue
            res = gather_from_store(store, hits, c)
            rows.append(descriptors(res.relevance()))
        if not rows: print("no groundable queries"); break
        agg = lambda k: float(np.median([d[k] for d in rows]))
        print(f"decay_gamma={gamma:<4}  n_act={agg('n_act'):5.0f}  PR={agg('pr'):5.1f}  "
              f"gap(cliff)={agg('gap'):6.2f}  gap_pos={agg('gpos'):.2f}  "
              f"mass@10={agg('m10'):.2f}  mass@34={agg('m34'):.2f}")
    # eyeball: print head of the sorted spectrum for two queries at the default leak
    print("\nsorted ‖x*‖² head (decay_gamma=1.0):")
    c = cfg(decay_gamma=1.0)
    for q in ("black hole", "gravitational wave"):
        hits = store.find_vec(q, 1) or store.find(q, 1)
        if not hits: continue
        r = np.sort(_np(gather_from_store(store, hits, c).relevance()))[::-1]
        head = "  ".join(f"{v:.3f}" for v in (r[:14] / (r[0] + 1e-12)))
        print(f"  {q:<20} {head}")
    store.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=".di-ui/graph.astro.sqlite")
    run(ap.parse_args().store)
