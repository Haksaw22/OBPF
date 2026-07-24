"""C3 — the coupling curve (DESIGN §4).

Env: CoupledQuadratic (12 base atoms in 4 groups with private variable blocks + 3
cross-term atoms c*(x_i - x_j)^2 linking consecutive groups; knob = coupling_strength).
Optimisation is directly over the decision variables x (16 replicates, mean loss).

Per (knob, seed): ES minimises G_K(w) + lam_d*overlap over soft w [15 x 4]; record the
G_K of the best solution. Curve = mean min-gap vs knob.
Gate: Spearman rho >= 0.8 over knob settings. Kill: rho < 0.5.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, os.environ.get("OPBF_SRC", "<path-to-OPBF-checkout>/opbf2/src"))

KNOBS = [0.0, 0.5, 1.0, 2.0, 5.0]
SEEDS = list(range(6))
K_INNER = 100
LR_X = 0.02
REPLICATES = 16
P = 4
PAIRS, GENS, SIGMA, LR_ES = 16, 30, 0.6, 0.3


def one_cell(args):
    knob, seed, lam_d = args
    import torch
    from opbf2.envs.synthetic_objectives import CoupledQuadratic

    env = CoupledQuadratic(seed=seed, coupling_strength=knob)
    N = env.n_atoms
    gen = torch.Generator().manual_seed(seed)
    x0 = torch.randn(REPLICATES, env.n_vars, generator=gen)

    def train_x(weights, K):
        x = x0.clone()
        for _ in range(K):
            xg = x.clone().requires_grad_(True)
            loss = (env.atoms(xg) * weights).sum(-1).mean()
            g, = torch.autograd.grad(loss, xg)
            x = x - LR_X * g
        return x

    def L(x):
        with torch.no_grad():
            return float(env.atoms(x).sum(-1).mean())

    def gk(w):
        joint = train_x(torch.ones(N), K_INNER)
        merged = x0.clone()
        for i in range(P):
            merged = merged + (train_x(w[:, i], K_INNER) - x0)
        return L(merged) - L(joint)

    def fitness(logits):
        w = torch.softmax(logits, dim=1)
        ov = float((1.0 - (w ** 2).sum(dim=1)).mean())
        g = gk(w)
        return -(g + lam_d * ov), g, ov

    gen_es = torch.Generator().manual_seed(400_000 + seed)
    gen_init = torch.Generator().manual_seed(500_000 + seed)
    mu = 0.5 * torch.randn(N, P, generator=gen_init)
    for _ in range(GENS):
        eps = [torch.randn(mu.shape, generator=gen_es) for _ in range(PAIRS)]
        cand = []
        for e in eps:
            cand += [mu + SIGMA * e, mu - SIGMA * e]
        fits = [fitness(c)[0] for c in cand]
        order = sorted(range(len(cand)), key=lambda i: fits[i])
        util = torch.zeros(len(cand))
        for rank, idx in enumerate(order):
            util[idx] = rank / (len(cand) - 1) - 0.5
        grad = torch.zeros_like(mu)
        for j, e in enumerate(eps):
            grad += (util[2 * j] - util[2 * j + 1]) * e
        mu = mu + LR_ES * grad / SIGMA
    f_mu, g_mu, ov_mu = fitness(mu)
    g_true_partition = None
    return {"knob": knob, "seed": seed, "min_gap": g_mu, "overlap": ov_mu,
            "fitness": f_mu}


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, idx in enumerate(order):
            r[idx] = float(rank)
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def main():
    params = json.loads((ROOT / "results" / "params.json").read_text())
    lam_d = params["lam_d"]
    cells = [(k, s, lam_d) for k in KNOBS for s in SEEDS]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(one_cell, cells))
    per_knob = {str(k): [r["min_gap"] for r in results if r["knob"] == k] for k in KNOBS}
    means = {k: statistics.fmean(v) for k, v in per_knob.items()}
    # primary: per-seed pairs (30 points); secondary: per-knob means (5 points)
    rho_pairs = spearman([r["knob"] for r in results], [r["min_gap"] for r in results])
    rho_means = spearman(KNOBS, [means[str(k)] for k in KNOBS])
    verdict = ("PASS" if rho_means >= 0.8 else
               "KILL" if rho_means < 0.5 else "BETWEEN")
    out = {"config": {"knobs": KNOBS, "seeds": SEEDS, "K": K_INNER, "lr_x": LR_X,
                      "lam_d": lam_d, "replicates": REPLICATES},
           "per_knob_min_gap": per_knob, "knob_means": means,
           "spearman_means": rho_means, "spearman_pairs": rho_pairs,
           "verdict": verdict, "cells": results, "wall_s": time.time() - t0}
    (ROOT / "results" / "c3.json").write_text(json.dumps(out, indent=2))
    for k in KNOBS:
        print(f"knob {k:4.1f}: min-gap mean {means[str(k)]:+.5f}  "
              f"(per-seed {[round(v, 4) for v in per_knob[str(k)]]})")
    print(f"\nC3: Spearman(means) {rho_means:+.3f}  (pairs {rho_pairs:+.3f})  -> {verdict}  "
          f"({out['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
