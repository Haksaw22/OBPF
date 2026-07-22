"""Z2 — warm-start non-degradation gate (DESIGN §2).

Init w at the TRUE partition; run the full ES outer optimisation on G_K + lam_d*overlap.
Gate: mean AMI >= 0.95 over 12 seeds AND no seed < 0.8.
KILL: mean AMI drop > 0.10 (mean < 0.90) -> the gap objective also walks away from truth.

K and lam_d come from results/params.json (P1 note, set from Z1 scales).
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
    from gapkit import es_optimize, get_env, true_logits
    env = get_env(seed)
    return es_optimize(seed, K, lam_d, true_logits(env), quiet=True)


def main():
    params = json.loads((ROOT / "results" / "params.json").read_text())
    K, lam_d = params["K"], params["lam_d"]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(one_seed, [(s, K, lam_d) for s in SEEDS]))
    amis = [r["ami"] for r in results]
    mean_ami = sum(amis) / len(amis)
    gate_pass = mean_ami >= 0.95 and min(amis) >= 0.8
    kill = mean_ami < 0.90
    out = {"config": {"K": K, "lam_d": lam_d, "seeds": SEEDS},
           "amis": amis, "mean_ami": mean_ami, "min_ami": min(amis),
           "gate_pass": bool(gate_pass), "KILL": bool(kill),
           "per_seed": [{k: r[k] for k in ("seed", "ami", "final_fitness", "detail", "pred")}
                        for r in results],
           "wall_s": time.time() - t0}
    (ROOT / "results" / "z2.json").write_text(json.dumps(out, indent=2))
    for r in results:
        print(f"seed {r['seed']:2d}: AMI {r['ami']:.3f}  G_K {r['detail']['G_K']:+.4f}  "
              f"overlap {r['detail']['overlap']:.3f}")
    print(f"\nZ2: mean AMI {mean_ami:.3f}  min {min(amis):.3f}  "
          f"GATE {'PASS' if gate_pass else 'FAIL'}  KILL={kill}  ({out['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
