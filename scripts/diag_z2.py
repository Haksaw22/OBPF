"""Z2 diagnostic (optimizer-free): is the true partition a LOCAL optimum of the
objective G_K + lam_d*overlap?

For each seed: fitness at the true hard partition vs all T*(P-1)=36 single-task
reassignment neighbours (hard partitions). If a neighbour is strictly better, the
objective genuinely prefers a wrong partition (formulation-level failure at this
operational instantiation); if truth beats all neighbours, Z2's walk-away was ES
drift (optimizer-level failure). Also classifies each Z2 seed by final-vs-true
fitness. Diagnostic only — no gate, no new claims.
"""
from __future__ import annotations

import json
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
    from gapkit import get_env, warm_init, gk_eval, onehot_w, overlap, ami_of
    env = get_env(seed)
    model0, X, Y = warm_init(env, seed)
    T, P = env.n_tasks, env.n_groups
    true = env.group_labels()

    def fit(labels):
        w = onehot_w(labels, P)
        g = gk_eval(w, env, model0, X, Y, K)["G_K"]
        return -(g + lam_d * overlap(w)), g

    f_true, g_true = fit(true)
    neigh = []
    for t in range(T):
        for gnew in range(P):
            if gnew == int(true[t]):
                continue
            lab = true.clone()
            lab[t] = gnew
            f, g = fit(lab)
            neigh.append({"task": t, "to": gnew, "fitness": f, "G": g,
                          "ami": ami_of(lab, env)})
    better = [n for n in neigh if n["fitness"] > f_true + 1e-9]
    best = max(neigh, key=lambda n: n["fitness"])
    return {"seed": seed, "fit_true": f_true, "G_true": g_true,
            "n_better_neighbours": len(better),
            "best_neighbour": best,
            "truth_is_local_opt": len(better) == 0}


def main():
    params = json.loads((ROOT / "results" / "params.json").read_text())
    K, lam_d = params["K"], params["lam_d"]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(one_seed, [(s, K, lam_d) for s in SEEDS]))

    z2 = json.loads((ROOT / "results" / "z2.json").read_text())
    for r, z in zip(results, z2["per_seed"]):
        r["z2_final_fitness"] = z["final_fitness"]
        r["z2_ami"] = z["ami"]
        r["classification"] = ("OBJECTIVE_PREFERS_WRONG"
                               if z["final_fitness"] > r["fit_true"] + 1e-9
                               else "ES_DRIFT")
        if not r["truth_is_local_opt"]:
            r["classification"] = "OBJECTIVE_PREFERS_WRONG (local)"
    n_obj = sum("OBJECTIVE" in r["classification"] for r in results)
    out = {"config": {"K": K, "lam_d": lam_d},
           "n_objective_prefers_wrong": n_obj,
           "n_es_drift": len(results) - n_obj,
           "per_seed": results, "wall_s": time.time() - t0}
    (ROOT / "results" / "diag_z2.json").write_text(json.dumps(out, indent=2))
    for r in results:
        bn = r["best_neighbour"]
        print(f"seed {r['seed']:2d}: fit(true) {r['fit_true']:+.4f}  "
              f"better-neighbours {r['n_better_neighbours']:2d}  "
              f"best-neighbour fit {bn['fitness']:+.4f} (ami {bn['ami']:.2f})  "
              f"z2 final {r['z2_final_fitness']:+.4f} (ami {r['z2_ami']:.2f})  "
              f"-> {r['classification']}")
    print(f"\nDIAG: objective-prefers-wrong on {n_obj}/12 seeds; "
          f"ES-drift on {12 - n_obj}/12  ({out['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
