"""Z1 — K calibration (validity precondition; DESIGN §2).

Pick smallest K in {100, 300, 1000} with median G_K(random) > mean G_K(true) + 3*sigma_seed.
6 seeds, 5 random hard partitions per seed. Also logs G_K scale and overlap-penalty scale
so lam_d can be set (logged as a dated PARAMS note before Z2/C1 run).
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch  # noqa: E402

from gapkit import (get_env, warm_init, gk_eval, true_logits, onehot_w,  # noqa: E402
                    random_hard_partition, ami_of, ci95)

SEEDS = range(6)
KS = [100, 300, 1000]
N_RANDOM = 5

out = {"config": {"seeds": list(SEEDS), "Ks": KS, "n_random": N_RANDOM}, "per_K": {}}
t0 = time.time()
for K in KS:
    true_g, rand_g, rand_ami, unif_g = [], [], [], []
    for s in SEEDS:
        env = get_env(s)
        model0, X, Y = warm_init(env, s)
        w_true = torch.softmax(true_logits(env), dim=1)
        r = gk_eval(w_true, env, model0, X, Y, K)
        true_g.append(r["G_K"])
        # scaled-copy family (uniform rows) — measured for the P1 lambda_d check
        w_unif = torch.full((env.n_tasks, env.n_groups), 1.0 / env.n_groups)
        unif_g.append(gk_eval(w_unif, env, model0, X, Y, K)["G_K"])
        for j in range(N_RANDOM):
            lab = random_hard_partition(1000 * s + j, env.n_tasks, env.n_groups)
            rr = gk_eval(onehot_w(lab, env.n_groups), env, model0, X, Y, K)
            rand_g.append(rr["G_K"])
            rand_ami.append(ami_of(lab, env))
        print(f"K={K} seed {s}: G(true)={r['G_K']:+.5f}  G(unif)={unif_g[-1]:+.5f}  "
              f"L_joint={r['L_joint']:.4f}  L_theta0={r['L_theta0']:.4f}", flush=True)
    m_true, _ = ci95(true_g)
    sd_true = statistics.stdev(true_g)
    med_rand = statistics.median(rand_g)
    passed = med_rand > m_true + 3 * sd_true
    out["per_K"][str(K)] = {
        "G_true": true_g, "G_true_mean": m_true, "G_true_sd": sd_true,
        "G_random": rand_g, "G_random_median": med_rand,
        "G_uniform": unif_g, "G_uniform_mean": statistics.fmean(unif_g),
        "random_ami_mean": statistics.fmean(rand_ami),
        "separated": bool(passed),
    }
    print(f"== K={K}: true {m_true:+.5f} (sd {sd_true:.5f}) | random median {med_rand:+.5f} "
          f"| uniform {statistics.fmean(unif_g):+.5f} | separated={passed}", flush=True)

chosen = next((K for K in KS if out["per_K"][str(K)]["separated"]), None)
out["chosen_K"] = chosen
out["wall_s"] = time.time() - t0
res = Path(__file__).resolve().parent.parent / "results"
res.mkdir(exist_ok=True)
(res / "z1.json").write_text(json.dumps(out, indent=2))
print(f"\nZ1 chosen K = {chosen}  (wall {out['wall_s']:.0f}s)")
