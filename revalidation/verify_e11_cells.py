"""Re-run the decisive (b)-closure cells from probe_e11_tpcgrad.py (read-only verification).

Cells: coupling=0.3 (transient) x noise in {0.0, 1.0}. Logic copied verbatim from
C:/Users/kulbi/Documents/Coding/OPBF/opbf2/scripts/probe_e11_tpcgrad.py
"""
from __future__ import annotations

import math
import statistics
import sys

sys.path.insert(0, r"C:/Users/kulbi/Documents/Coding/OPBF/opbf2/src")

import torch  # noqa: E402

from opbf2.envs.conflict_curriculum import CurriculumTwoObjective  # noqa: E402
from opbf2.optim.grad_combine import pcgrad, GradVac, TemporalPCGrad  # noqa: E402

LRS = (0.05, 0.1, 0.2)
BETAS = (0.5, 0.9)


def _make(method, beta):
    if method == "pcgrad":
        return ("stateless", pcgrad)
    if method == "gradvac":
        return ("stateful", GradVac(beta=beta))
    if method == "tpcgrad":
        return ("stateful", TemporalPCGrad(beta=beta))
    if method == "tsum":
        ema = {"v": None}
        def f(G):
            ema["v"] = G if ema["v"] is None else beta * ema["v"] + (1 - beta) * G
            return ema["v"].sum(0)
        return ("stateful_fn", f)
    if method == "pcgrad_mom":
        v = {"v": None}
        def f(G):
            d = pcgrad(G)
            v["v"] = d if v["v"] is None else beta * v["v"] + d
            return v["v"]
        return ("stateful_fn", f)
    raise ValueError(method)


def _episode(env, method, lr, beta, steps):
    kind, comb = _make(method, beta)
    step = comb.combine if kind == "stateful" else comb
    th = env.theta0.clone()
    tot = 0.0
    for t in range(steps):
        th = th - lr * step(env.grads(th))
        tot += float(env.losses(th).sum())
    return tot / steps


def _auc(method, lr, beta, coupling, noise, seeds, steps):
    return [_episode(CurriculumTwoObjective(d=4, coupling=coupling, seed=int(s), noise=noise),
                     method, lr, beta, steps) for s in seeds]


def _paired_p(a, b):
    if all(x == y for x, y in zip(a, b)):
        return 1.0
    from scipy.stats import wilcoxon
    return float(wilcoxon(a, b).pvalue)


def main():
    methods = ["pcgrad", "gradvac", "tpcgrad", "tsum", "pcgrad_mom"]
    sel, hold = range(12), range(100, 112)
    steps = 400
    for coupling, noise in ((0.3, 0.0), (0.3, 1.0)):
        print(f"\n#### coupling={coupling}  noise={noise}  (transient if c<=0.5) ####", flush=True)
        frozen = {}
        for m in methods:
            best = None
            for lr in LRS:
                for beta in (BETAS if m != "pcgrad" else (0.0,)):
                    a = _auc(m, lr, beta, coupling, noise, sel, steps)
                    ma = statistics.fmean(a)
                    if best is None or ma < best[0]:
                        best = (ma, lr, beta)
            frozen[m] = best[1:]
        ho = {m: _auc(m, frozen[m][0], frozen[m][1], coupling, noise, hold, steps) for m in methods}
        for m in methods:
            mm = statistics.fmean(ho[m])
            ci = 1.96 * statistics.stdev(ho[m]) / math.sqrt(len(ho[m]))
            print(f"  {m:11s} holdout AUC {mm:.4f} +/-{ci:.4f}  [lr={frozen[m][0]}, beta={frozen[m][1]}]", flush=True)
        dpc = statistics.fmean(ho["tpcgrad"]) - statistics.fmean(ho["pcgrad"])
        dgv = statistics.fmean(ho["tpcgrad"]) - statistics.fmean(ho["gradvac"])
        dts = statistics.fmean(ho["tsum"]) - statistics.fmean(ho["pcgrad"])
        ppc, pgv = _paired_p(ho["tpcgrad"], ho["pcgrad"]), _paired_p(ho["tpcgrad"], ho["gradvac"])
        win = (dpc < 0 and ppc < 0.05) and (dgv < 0 and pgv < 0.05)
        print(f"  tpcgrad vs pcgrad d={dpc:+.4f} p={ppc:.4f} | vs gradvac d={dgv:+.4f} p={pgv:.4f}"
              f"  -> WIN_BOTH_HOLDOUT={win}", flush=True)
        print(f"  [momentum control] tsum vs pcgrad d={dts:+.4f} p={_paired_p(ho['tsum'], ho['pcgrad']):.4f}", flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
