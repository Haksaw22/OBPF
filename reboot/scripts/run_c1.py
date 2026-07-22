"""C1 — recovery, the main pre-registered claim (DESIGN §3).

Random-init ES on G_K + lam_d*overlap, 12 seeds. Gates (fixed at signing):
  PASS    mean AMI >= 0.95 AND >= kmeans_mean - 0.03
  PARTIAL mean AMI >= 0.88
  KILL    mean AMI <= 0.84
Baselines per seed on the same coupling data: kmeans-affinity (incumbent),
MERIT-style conflict split, c-BTM-style embedding kmeans, random partitions (20),
uniform shards, oracle. Scaled-copy is a mechanism check (rejected by lam_d), not AMI.
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
    from gapkit import (es_optimize, get_env, random_logits, kmeans_affinity,
                        conflict_split, embed_kmeans, uniform_shards, ami_of,
                        random_hard_partition, gk_eval, warm_init, true_logits,
                        onehot_w, overlap)
    env = get_env(seed)
    T, P = env.n_tasks, env.n_groups
    res = es_optimize(seed, K, lam_d, random_logits(seed, T, P), quiet=True)

    base = {}
    base["kmeans"] = ami_of(kmeans_affinity(env, seed), env)
    base["conflict_split"] = ami_of(conflict_split(env, seed), env)
    base["embed_kmeans"] = ami_of(embed_kmeans(env, seed), env)
    base["uniform_shards"] = ami_of(uniform_shards(T, P), env)
    base["random_mean"] = statistics.fmean(
        ami_of(random_hard_partition(5000 * seed + j, T, P), env) for j in range(20))

    # mechanism check: fitness of learned vs scaled-copy under the SAME objective
    model0, X, Y = warm_init(env, seed)
    w_unif = torch.full((T, P), 1.0 / P)
    g_unif = gk_eval(w_unif, env, model0, X, Y, K)["G_K"]
    fit_unif = -(g_unif + lam_d * overlap(w_unif))
    w_true = onehot_w(env.group_labels(), P)
    g_true = gk_eval(w_true, env, model0, X, Y, K)["G_K"]
    fit_true = -(g_true + lam_d * overlap(w_true))
    res["baselines"] = base
    res["mech"] = {"fit_uniform": fit_unif, "fit_true": fit_true,
                   "G_uniform": g_unif, "G_true": g_true,
                   "scaled_copy_rejected": bool(res["final_fitness"] > fit_unif or fit_true > fit_unif)}
    return res


def main():
    params = json.loads((ROOT / "results" / "params.json").read_text())
    K, lam_d = params["K"], params["lam_d"]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(one_seed, [(s, K, lam_d) for s in SEEDS]))
    amis = [r["ami"] for r in results]
    mean_ami = statistics.fmean(amis)
    km_mean = statistics.fmean(r["baselines"]["kmeans"] for r in results)
    if mean_ami >= 0.95 and mean_ami >= km_mean - 0.03:
        verdict = "PASS"
    elif mean_ami >= 0.88:
        verdict = "PARTIAL"
    elif mean_ami <= 0.84:
        verdict = "KILL"
    else:
        verdict = "BETWEEN (0.84, 0.88) — pre-registered as neither; report as weak-PARTIAL"
    out = {"config": {"K": K, "lam_d": lam_d, "seeds": SEEDS},
           "amis": amis, "mean_ami": mean_ami, "kmeans_mean": km_mean,
           "baseline_means": {k: statistics.fmean(r["baselines"][k] for r in results)
                              for k in results[0]["baselines"]},
           "verdict": verdict,
           "per_seed": [{k: r[k] for k in ("seed", "ami", "final_fitness", "detail",
                                           "pred", "baselines", "mech")} for r in results],
           "wall_s": time.time() - t0}
    (ROOT / "results" / "c1.json").write_text(json.dumps(out, indent=2))
    for r in results:
        b = r["baselines"]
        print(f"seed {r['seed']:2d}: ES-AMI {r['ami']:.3f} | km {b['kmeans']:.3f} "
              f"conf {b['conflict_split']:.3f} emb {b['embed_kmeans']:.3f} "
              f"rand {b['random_mean']:.3f} | G_K {r['detail']['G_K']:+.4f}")
    print(f"\nC1: mean AMI {mean_ami:.3f} vs kmeans {km_mean:.3f} -> {verdict}  "
          f"({out['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
