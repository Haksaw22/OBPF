"""Z2 diagnostic under the SECONDARY (pre-registered) merge operator: mass-weighted
parameter averaging. Same optimizer-free landscape test as diag_z2.py: is truth a
local optimum of G_K(mass_avg) + lam_d*overlap? Decides whether the Z2 failure is
specific to update-sum or generic to reading (C) at this scale.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SEEDS = list(range(12))


def one_seed(args):
    seed, K, lam_d = args
    import torch
    from gapkit import (get_env, warm_init, gk_eval, onehot_w, overlap, ami_of,
                        random_hard_partition)
    env = get_env(seed)
    model0, X, Y = warm_init(env, seed)
    T, P = env.n_tasks, env.n_groups
    true = env.group_labels()

    def fit(labels):
        w = onehot_w(labels, P)
        g = gk_eval(w, env, model0, X, Y, K, merge="mass_avg")["G_K"]
        return -(g + lam_d * overlap(w)), g

    f_true, g_true = fit(true)
    # separation check under this operator too
    g_rand = [fit(random_hard_partition(1000 * seed + j, T, P))[1] for j in range(3)]
    neigh_fits = []
    n_better = 0
    best = None
    for t in range(T):
        for gnew in range(P):
            if gnew == int(true[t]):
                continue
            lab = true.clone()
            lab[t] = gnew
            f, g = fit(lab)
            rec = {"task": t, "to": gnew, "fitness": f, "G": g, "ami": ami_of(lab, env)}
            neigh_fits.append(rec)
            if f > f_true + 1e-9:
                n_better += 1
            if best is None or f > best["fitness"]:
                best = rec
    return {"seed": seed, "fit_true": f_true, "G_true": g_true,
            "G_random_mean": statistics.fmean(g_rand),
            "n_better_neighbours": n_better, "best_neighbour": best,
            "truth_is_local_opt": n_better == 0}


def main():
    params = json.loads((ROOT / "results" / "params.json").read_text())
    K, lam_d = params["K"], params["lam_d"]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(one_seed, [(s, K, lam_d) for s in SEEDS]))
    n_local = sum(r["truth_is_local_opt"] for r in results)
    out = {"config": {"K": K, "lam_d": lam_d, "merge": "mass_avg"},
           "n_truth_local_opt": n_local, "per_seed": results,
           "wall_s": time.time() - t0}
    (ROOT / "results" / "diag_z2_avg.json").write_text(json.dumps(out, indent=2))
    for r in results:
        bn = r["best_neighbour"]
        print(f"seed {r['seed']:2d}: G(true) {r['G_true']:+.4f}  G(rand) {r['G_random_mean']:+.4f}  "
              f"better-neighbours {r['n_better_neighbours']:2d}  "
              f"best-n fit {bn['fitness']:+.4f}  "
              f"-> {'LOCAL OPT' if r['truth_is_local_opt'] else 'prefers wrong'}")
    print(f"\nDIAG(mass_avg): truth is local opt on {n_local}/12 seeds  ({out['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
