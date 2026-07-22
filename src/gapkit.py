"""OBPF gap-closure harness (pre-registered in DESIGN-DRAFT.md, signed 2026-07-22).

Reading (C), matched-parallel budget: from a shared warm init, train one copy per
component on D_i = sum_t w_it * l_t for K steps, merge (update-sum primary), score
against K steps of joint training on the same fixed batch.

The OPBF source tree is imported READ-ONLY (env + AMI metric only). All new code and
all outputs live in OBPF/.
"""
from __future__ import annotations

import copy
import statistics
import sys

sys.path.insert(0, r"C:/Users/kulbi/Documents/Coding/OPBF/opbf2/src")

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from opbf2.envs.mtl import StructuredMultiTaskRegression
from opbf2.metrics.clustering import ami_score

torch.set_num_threads(2)

LR = 1e-2          # warmup Adam lr (predecessor protocol; warmup is shared, pre-branch)
INNER_LR = 0.02    # inner SGD lr — see DESIGN DEVIATIONS D1: Adam's per-parameter
                   # normalisation moves zero-gradient parameters at full step size,
                   # which destroys any additive merge regardless of partition quality
                   # (found at smoke test, before any gate ran). SGD keeps steps
                   # gradient-proportional, which the merge operators assume.
WARMUP = 10        # joint warmup steps to theta0 (predecessor protocol)
BATCH = 512


# ------------------------------------------------------------------ environment
def get_env(seed: int) -> StructuredMultiTaskRegression:
    return StructuredMultiTaskRegression(seed=seed)


def warm_init(env, seed: int):
    """Fixed batch + shared warm init theta0 for this seed (deterministic)."""
    gen = torch.Generator().manual_seed(seed)
    X, Y = env.sample(BATCH, gen)
    model = env.make_model(seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(WARMUP):
        opt.zero_grad()
        env.task_losses(model, X, Y).sum().backward()
        opt.step()
    return model, X, Y


def _train_copy(model0, env, X, Y, weights: torch.Tensor, K: int):
    """Fresh copy of model0 trained K full-batch SGD steps on sum_t weights_t * l_t."""
    m = copy.deepcopy(model0)
    opt = torch.optim.SGD(m.parameters(), lr=INNER_LR)
    for _ in range(K):
        opt.zero_grad()
        (env.task_losses(m, X, Y) * weights).sum().backward()
        opt.step()
    return m


def total_loss(model, env, X, Y) -> float:
    with torch.no_grad():
        return float(env.task_losses(model, X, Y).sum())


# ------------------------------------------------------------------ the objective
def gk_eval(w: torch.Tensor, env, model0, X, Y, K: int, merge: str = "update_sum") -> dict:
    """w: [T, P] rows on the simplex. Returns G_K and components."""
    T, P = w.shape
    joint = _train_copy(model0, env, X, Y, torch.ones(T), K)
    v0 = parameters_to_vector(model0.parameters()).detach()
    if merge == "update_sum":
        v = v0.clone()
        for i in range(P):
            bi = _train_copy(model0, env, X, Y, w[:, i], K)
            v += parameters_to_vector(bi.parameters()).detach() - v0
    elif merge == "mass_avg":
        v = torch.zeros_like(v0)
        mass = w.sum(dim=0)
        for i in range(P):
            bi = _train_copy(model0, env, X, Y, w[:, i], K)
            v += mass[i] * parameters_to_vector(bi.parameters()).detach()
        v /= mass.sum()
    else:
        raise ValueError(merge)
    merged = copy.deepcopy(model0)
    vector_to_parameters(v, merged.parameters())
    L_m, L_j = total_loss(merged, env, X, Y), total_loss(joint, env, X, Y)
    return {"G_K": L_m - L_j, "L_merged": L_m, "L_joint": L_j,
            "L_theta0": total_loss(model0, env, X, Y)}


def overlap(w: torch.Tensor) -> float:
    """Cross-component co-assignment mass, normalised to [0, 1-1/P]; 0 = hard partition."""
    return float((1.0 - (w ** 2).sum(dim=1)).mean())


def fitness(logits: torch.Tensor, env, model0, X, Y, K: int, lam_d: float) -> tuple[float, dict]:
    w = torch.softmax(logits, dim=1)
    r = gk_eval(w, env, model0, X, Y, K)
    ov = overlap(w)
    return -(r["G_K"] + lam_d * ov), {**r, "overlap": ov}


# ------------------------------------------------------------------ ES (primary outer optimiser)
def es_optimize(seed: int, K: int, lam_d: float, init_logits: torch.Tensor,
                pairs: int = 16, gens: int = 30, sigma: float = 0.6, lr_es: float = 0.3,
                log_every: int = 5, quiet: bool = False):
    """Antithetic NES with rank-shaped utilities. Deterministic given (seed, init)."""
    env = get_env(seed)
    model0, X, Y = warm_init(env, seed)
    mu = init_logits.clone()
    gen_noise = torch.Generator().manual_seed(100_000 + seed)
    history = []
    for g in range(gens):
        eps = [torch.randn(mu.shape, generator=gen_noise) for _ in range(pairs)]
        cand, fits = [], []
        for e in eps:
            cand += [mu + sigma * e, mu - sigma * e]
        for c in cand:
            f, _ = fitness(c, env, model0, X, Y, K, lam_d)
            fits.append(f)
        order = sorted(range(len(cand)), key=lambda i: fits[i])
        util = torch.zeros(len(cand))
        for rank, idx in enumerate(order):                       # rank-shaped in [-0.5, 0.5]
            util[idx] = rank / (len(cand) - 1) - 0.5
        grad = torch.zeros_like(mu)
        for j, e in enumerate(eps):
            grad += (util[2 * j] - util[2 * j + 1]) * e
        grad /= (2 * pairs * sigma)
        mu = mu + lr_es * grad * (2 * pairs)                     # scale-free NES step
        best = max(fits)
        history.append(best)
        if not quiet and g % log_every == 0:
            print(f"  seed {seed} gen {g:3d} best_fitness {best:+.5f}", flush=True)
    f_mu, detail = fitness(mu, env, model0, X, Y, K, lam_d)
    w = torch.softmax(mu, dim=1)
    pred = w.argmax(dim=1)
    return {"seed": seed, "ami": float(ami_score(pred, env.group_labels())),
            "final_fitness": f_mu, "detail": detail, "history": history,
            "pred": pred.tolist(), "w": w.tolist()}


def true_logits(env, scale: float = 4.0) -> torch.Tensor:
    lab = env.group_labels()
    T, P = env.n_tasks, env.n_groups
    z = torch.full((T, P), -scale)
    z[torch.arange(T), lab] = scale
    return z


def random_logits(seed: int, T: int, P: int, scale: float = 0.5) -> torch.Tensor:
    g = torch.Generator().manual_seed(200_000 + seed)
    return scale * torch.randn(T, P, generator=g)


def random_hard_partition(seed: int, T: int, P: int) -> torch.Tensor:
    """Random hard partition guaranteed to use all P groups."""
    g = torch.Generator().manual_seed(300_000 + seed)
    while True:
        lab = torch.randint(0, P, (T,), generator=g)
        if len(lab.unique()) == P:
            return lab


def onehot_w(labels: torch.Tensor, P: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels.long(), P).float()


# ------------------------------------------------------------------ baselines (C1)
def kmeans_affinity(env, seed: int) -> torch.Tensor:
    """The incumbent: k-means on the mid-training signed-cosine coupling matrix."""
    from sklearn.cluster import KMeans
    gen = torch.Generator().manual_seed(seed)
    X, Y = env.sample(BATCH, gen)
    C = env.trained_coupling(X, Y, seed=seed).detach().numpy()
    km = KMeans(n_clusters=env.n_groups, n_init=10, random_state=seed).fit(C)
    return torch.tensor(km.labels_)


def conflict_split(env, seed: int, restarts: int = 12, passes: int = 60) -> torch.Tensor:
    """MERIT-style: hard partition minimising within-group conflict mass, greedy local search."""
    gen = torch.Generator().manual_seed(seed)
    X, Y = env.sample(BATCH, gen)
    cos = env.trained_coupling(X, Y, seed=seed).detach()
    C = (-cos).clamp_min(0.0)
    T, P = env.n_tasks, env.n_groups

    def cost(lab):
        same = lab.unsqueeze(0) == lab.unsqueeze(1)
        return float((C * same).sum())

    best_lab, best_c = None, None
    for r in range(restarts):
        lab = random_hard_partition(seed * 37 + r, T, P)
        improved = True
        p = 0
        while improved and p < passes:
            improved = False
            p += 1
            for t in range(T):
                cur = cost(lab)
                for gnew in range(P):
                    if gnew == int(lab[t]):
                        continue
                    trial = lab.clone()
                    trial[t] = gnew
                    if len(trial.unique()) < P:
                        continue
                    if cost(trial) < cur - 1e-12:
                        lab, cur, improved = trial, cost(trial), True
        if best_c is None or cost(lab) < best_c:
            best_lab, best_c = lab, cost(lab)
    return best_lab


def embed_kmeans(env, seed: int, probe: int = 256) -> torch.Tensor:
    """c-BTM-style: k-means on generic per-task embeddings (per-example loss profile at theta0)."""
    from sklearn.cluster import KMeans
    model0, X, Y = warm_init(env, seed)
    with torch.no_grad():
        pred = model0(X[:probe])
        prof = ((pred - Y[:probe]) ** 2).T.numpy()               # [T, probe]
    km = KMeans(n_clusters=env.n_groups, n_init=10, random_state=seed).fit(prof)
    return torch.tensor(km.labels_)


def uniform_shards(T: int, P: int) -> torch.Tensor:
    return torch.tensor([min(t * P // T, P - 1) for t in range(T)])


def ami_of(labels: torch.Tensor, env) -> float:
    return float(ami_score(labels, env.group_labels()))


def ci95(xs) -> tuple[float, float]:
    m = statistics.fmean(xs)
    h = 1.96 * statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0
    return m, h
