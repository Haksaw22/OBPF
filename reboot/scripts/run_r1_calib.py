"""R1b — (K x lr) window sweep under the damped-sum operator (DESIGN §R1).

Grid K in {50,100,300} x lr in {0.005,0.02,0.05}, 6 seeds: G_damped for true,
random x5, uniform. Choose the best-separating cell meeting the Z1 criterion
(median random > mean true + 3*sd_seed(true)). If none separates: repair fails at
calibration -> report and stop.
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

KS = [50, 100, 300]
LRS = [0.005, 0.02, 0.05]
SEEDS = list(range(6))
N_RANDOM = 5


def one_cell_seed(args):
    K, lr, s = args
    import torch
    from gapkit import (get_env, warm_init, gk_eval, true_logits, onehot_w,
                        random_hard_partition)
    env = get_env(s)
    model0, X, Y = warm_init(env, s)
    w_true = torch.softmax(true_logits(env), dim=1)
    g_true = gk_eval(w_true, env, model0, X, Y, K, merge="damped_sum", lr=lr)
    w_unif = torch.full((env.n_tasks, env.n_groups), 1.0 / env.n_groups)
    g_unif = gk_eval(w_unif, env, model0, X, Y, K, merge="damped_sum", lr=lr)
    g_rand = []
    for j in range(N_RANDOM):
        lab = random_hard_partition(1000 * s + j, env.n_tasks, env.n_groups)
        g_rand.append(gk_eval(onehot_w(lab, env.n_groups), env, model0, X, Y, K,
                              merge="damped_sum", lr=lr)["G_K"])
    return {"K": K, "lr": lr, "seed": s, "G_true": g_true["G_K"],
            "alpha_true": g_true.get("alpha"), "G_uniform": g_unif["G_K"],
            "alpha_uniform": g_unif.get("alpha"), "G_random": g_rand,
            "L_joint": g_true["L_joint"]}


def main():
    cells = [(K, lr, s) for K in KS for lr in LRS for s in SEEDS]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(one_cell_seed, cells))
    out = {"config": {"Ks": KS, "lrs": LRS, "seeds": SEEDS, "n_random": N_RANDOM},
           "rows": rows, "cells": {}}
    best_cell, best_ratio = None, None
    for K in KS:
        for lr in LRS:
            sub = [r for r in rows if r["K"] == K and r["lr"] == lr]
            tr = [r["G_true"] for r in sub]
            rd = [g for r in sub for g in r["G_random"]]
            un = [r["G_uniform"] for r in sub]
            m_true, sd_true = statistics.fmean(tr), statistics.stdev(tr)
            med_rand = statistics.median(rd)
            sep = med_rand > m_true + 3 * sd_true
            ratio = (med_rand - m_true) / (sd_true + 1e-12)
            out["cells"][f"K{K}_lr{lr}"] = {
                "G_true_mean": m_true, "G_true_sd": sd_true,
                "G_random_median": med_rand, "G_uniform_mean": statistics.fmean(un),
                "separated": bool(sep), "sep_ratio": ratio,
            }
            print(f"K={K:3d} lr={lr:5.3f}: true {m_true:+.4f} (sd {sd_true:.4f})  "
                  f"rand-med {med_rand:+.4f}  unif {statistics.fmean(un):+.4f}  "
                  f"sep={sep} (ratio {ratio:.1f})", flush=True)
            if sep and (best_ratio is None or ratio > best_ratio):
                best_cell, best_ratio = (K, lr), ratio
    out["chosen_cell"] = best_cell
    out["wall_s"] = time.time() - t0
    (ROOT / "results" / "r1_calib.json").write_text(json.dumps(out, indent=2))
    print(f"\nR1b chosen cell: {best_cell}  ({out['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
