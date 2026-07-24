"""Missed-exit probes on the E7 recovery task (read-only wrt the OPBF repo).

Per seed:
  A. KMeans AMI on the coupling matrix (reference baseline).
  B. Soft OPBF (reproduce committed recover_task_groups, softmax).
  C. Objective values (the exact training loss) at: true one-hot partition,
     KMeans one-hot partition, learned-soft solution -> is the objective even
     minimised at the correct partition?
  D. WARM-START: distill assign_head to the KMeans partition (300 CE steps),
     then fine-tune 1500 steps on the real coupling+sparsity objective.
     AMI before/after fine-tune. (The classic 'start at the baseline' move.)
  E. ANNEALED soft->hard: softmax with temperature annealed 1.0 -> 0.05.
  F. ANNEALED Gumbel-ST: gumbel_hard with tau annealed 2.0 -> 0.1
     (vs the committed fixed tau=1.0 straight-through).
"""
from __future__ import annotations

import os
import statistics
import sys

sys.path.insert(0, os.environ.get("OPBF_SRC", "<path-to-OPBF-checkout>/opbf2/src"))

import torch
import torch.nn.functional as F

from opbf2.atoms.atom_types import AtomBatch
from opbf2.envs.mtl import StructuredMultiTaskRegression
from opbf2.losses.coupling import signed_coupling_matrices, signed_coupling_loss
from opbf2.losses.sparsity import sparsity_total
from opbf2.metrics.clustering import ami_score
from opbf2.models.factorisers import LinearAssignmentFactoriser
from opbf2.models.mixers import AdditiveMixer
from opbf2.utils.seeding import set_seed

torch.set_num_threads(2)

CW, LC, LU, LAE, LR, STEPS = 2.0, 1.0, 0.05, 0.01, 1e-2, 1500


def make_batch(T):
    return AtomBatch(atoms=torch.zeros(1, T), atom_mask=torch.ones(1, T),
                     atom_metadata=torch.eye(T).unsqueeze(0), context=torch.zeros(1, 1),
                     target_loss=torch.zeros(1))


def objective(A, R, C):
    with torch.no_grad():
        A = A.detach()
        return float(CW * signed_coupling_loss(A, R, C, LC)
                     + sparsity_total(A, torch.ones(1, A.shape[-1]), LAE, LU, 0.0, 0.0))


def onehot(labels, P):
    return F.one_hot(labels.long(), P).to(torch.float32).unsqueeze(0)


def train(f, bt, R, C, steps=STEPS, anneal=None):
    opt = torch.optim.Adam(f.parameters(), lr=LR)
    for i in range(steps):
        if anneal is not None:
            t0, t1 = anneal
            f.temperature = t0 + (t1 - t0) * i / (steps - 1)
        out = f(bt)
        loss = CW * signed_coupling_loss(out.assignments, R, C, LC) \
            + sparsity_total(out.assignments, out.gates, LAE, LU, 0.0, 0.0)
        opt.zero_grad(); loss.backward(); opt.step()
    f.eval()
    return f(bt).assignments[0].argmax(-1)


def main(n=6):
    cols = {k: [] for k in ("km", "soft", "hardST", "anneal_soft", "anneal_gumbel",
                            "warm_init", "warm_ft")}
    obj = {k: [] for k in ("true", "km", "soft_final", "warm_final")}
    for s in range(n):
        env = StructuredMultiTaskRegression(seed=s)
        set_seed(s)
        gen = torch.Generator().manual_seed(s)
        X, Y = env.sample(512, gen)
        Cmat = env.trained_coupling(X, Y, seed=s)
        R, C = signed_coupling_matrices(Cmat)
        T, G = env.n_tasks, env.n_groups
        true = env.group_labels()
        bt = make_batch(T)

        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=G, n_init=10, random_state=s).fit(Cmat.detach().numpy())
        km_lab = torch.tensor(km.labels_, dtype=torch.long)
        cols["km"].append(ami_score(km_lab, true))
        obj["true"].append(objective(onehot(true, G), R, C))
        obj["km"].append(objective(onehot(km_lab, G), R, C))

        # B. committed soft
        set_seed(s)
        f = LinearAssignmentFactoriser(G, T, 1, AdditiveMixer(), use_context=False,
                                       use_gates=False, assign_mode="softmax")
        pred = train(f, bt, R, C)
        cols["soft"].append(ami_score(pred, true))
        obj["soft_final"].append(objective(f(bt).assignments, R, C))

        # committed hard-ST (fixed tau=1.0)
        set_seed(s)
        f = LinearAssignmentFactoriser(G, T, 1, AdditiveMixer(), use_context=False,
                                       use_gates=False, assign_mode="gumbel_hard")
        cols["hardST"].append(ami_score(train(f, bt, R, C), true))

        # E. annealed softmax 1.0 -> 0.05
        set_seed(s)
        f = LinearAssignmentFactoriser(G, T, 1, AdditiveMixer(), use_context=False,
                                       use_gates=False, assign_mode="softmax")
        cols["anneal_soft"].append(ami_score(train(f, bt, R, C, anneal=(1.0, 0.05)), true))

        # F. annealed gumbel 2.0 -> 0.1
        set_seed(s)
        f = LinearAssignmentFactoriser(G, T, 1, AdditiveMixer(), use_context=False,
                                       use_gates=False, assign_mode="gumbel_hard")
        cols["anneal_gumbel"].append(ami_score(train(f, bt, R, C, anneal=(2.0, 0.1)), true))

        # D. warm-start from KMeans, then fine-tune on the real objective
        set_seed(s)
        f = LinearAssignmentFactoriser(G, T, 1, AdditiveMixer(), use_context=False,
                                       use_gates=False, assign_mode="softmax")
        opt = torch.optim.Adam(f.parameters(), lr=LR)
        for _ in range(300):                                   # distill to KMeans partition
            logits = f._assignment_logits(bt)[0]
            loss = F.cross_entropy(logits, km_lab)
            opt.zero_grad(); loss.backward(); opt.step()
        f.eval()
        cols["warm_init"].append(ami_score(f(bt).assignments[0].argmax(-1), true))
        f.train()
        pred = train(f, bt, R, C)                              # fine-tune 1500 steps
        cols["warm_ft"].append(ami_score(pred, true))
        obj["warm_final"].append(objective(f(bt).assignments, R, C))

        print(f"seed {s}: " + "  ".join(f"{k}={cols[k][-1]:.3f}" for k in cols), flush=True)
        print(f"        obj: " + "  ".join(f"{k}={obj[k][-1]:+.4f}" for k in obj), flush=True)

    print("\nMEANS over", n, "seeds")
    for k, v in cols.items():
        print(f"  AMI {k:13s} {statistics.fmean(v):.3f}")
    for k, v in obj.items():
        print(f"  OBJ {k:13s} {statistics.fmean(v):+.4f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
