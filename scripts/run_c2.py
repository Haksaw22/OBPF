"""C2 — parallel fidelity with the C1-learned partition (DESIGN §5).

Only runs if C1 >= PARTIAL. For each seed, take C1's learned partition and run the
predecessor's factor-parallel-vs-joint machinery (equal TOTAL capacity — the honest
split). Gate: mean loss gap <= 5% of joint.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, r"C:/Users/kulbi/Documents/Coding/OPBF/opbf2/src")

import torch  # noqa: E402
from opbf2.eval.e7_mtl import factor_parallel_vs_joint, _pred_to_groups  # noqa: E402
from gapkit import get_env  # noqa: E402


def main():
    c1 = json.loads((ROOT / "results" / "c1.json").read_text())
    if c1["verdict"] not in ("PASS", "PARTIAL"):
        print(f"C1 verdict is {c1['verdict']}; C2 does not run per DESIGN §5.")
        return
    t0 = time.time()
    gaps = []
    for rec in c1["per_seed"]:
        s = rec["seed"]
        env = get_env(s)
        groups = _pred_to_groups(torch.tensor(rec["pred"]), env.n_groups)
        r = factor_parallel_vs_joint(env, groups, seed=s, batch=512, steps=600)
        gaps.append(r["loss_gap_pct"])
        print(f"seed {s:2d}: loss gap {r['loss_gap_pct']:+.2f}%  "
              f"(joint {r['joint_loss']:.4f} parallel {r['parallel_loss']:.4f})")
    mean_gap = statistics.fmean(gaps)
    gate = mean_gap <= 5.0
    out = {"gaps_pct": gaps, "mean_gap_pct": mean_gap, "gate_pass": bool(gate),
           "wall_s": time.time() - t0}
    (ROOT / "results" / "c2.json").write_text(json.dumps(out, indent=2))
    print(f"\nC2: mean gap {mean_gap:+.2f}%  GATE {'PASS' if gate else 'FAIL'}")


if __name__ == "__main__":
    main()
