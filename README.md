# The Price of Coupling — two honest negatives on factorising loss functions

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haksaw22/OBPF/blob/main/playground.ipynb)

Can a learned decomposition split a training objective $L = \sum_t Q_t$ into a few
near-independent components — skills you could train in parallel on curated data,
transfer as a library, and read as an anatomy of the objective? The gap
$\min_\theta \sum_t Q_t - \sum_i \min_\theta D_i \ge 0$ (the *price of coupling*,
zero iff the components share a minimiser) makes the question exact.

**Two projects, two answers, both negative, each killing a different escape:**

| | Objective descended | How it died |
|---|---|---|
| **Part I — OPBF** | a proxy: gradient-affinity + reconstruction + entropy | lost to k-means / a generic encoder / a *random* partition; warm-started at the truth, its objective walks away (animated in the article) |
| **Part II — the reboot** | the desideratum itself: finite-K independent-training-then-merge fidelity | pre-registered gate zero killed it: the truth is not a local optimum on 8/12 seeds (merge-operator physics dominates); the repair round showed that removing the artifact removes the signal too, and the artifact-free objective *prefers redundant generalist copies over any decomposition* |

Untouched by both: the idealised equality at exact minimisation — the form nobody can
descend. The sharpened open question: *what descendable objective has the true
decomposition as its minimum?*

## Map

- **[BLOG.md](BLOG.md) — the article.** The bet, the three motivations, both deaths
  with mechanisms, and what survives. Start here.
- **[playground.ipynb](playground.ipynb)** — rebuilds the headline figures of both
  parts from raw data; minutes on CPU; pre-executed so it renders without running.
- **[figures/](figures/)** — including
  [objective_walks_away.gif](figures/objective_walks_away.gif) (Part I: start at the
  right answer, watch the proxy leave) and
  [r1_nowindow.png](figures/r1_nowindow.png) (Part II: the no-signal-window result).
- **Part I materials:** [data/](data/) (raw result JSONs, copied verbatim from the
  closed research repo), [revalidation/](revalidation/) (independent re-runs and
  probe scripts behind every cited number), [REVALIDATION.md](REVALIDATION.md).
- **Part II materials:** [reboot/](reboot/) — the complete research repo, vendored
  with its git history intact: [THEORY-DRAFT.md](reboot/THEORY-DRAFT.md) (the
  formulation, three readings, degeneracies), [NOVELTY.md](reboot/NOVELTY.md)
  (nine-literature prior-art sweep), [DESIGN-DRAFT.md](reboot/DESIGN-DRAFT.md) (the
  signed pre-registration — gate numbers dated in git before any run, deviations
  logged), [VERDICT-STAGE1.md](reboot/VERDICT-STAGE1.md) (the final verdict),
  [src/](reboot/src/) + [scripts/](reboot/scripts/) (the harness),
  [results/](reboot/results/) (every gate's raw JSON).

## Reproduce

Figures: `jupyter execute playground.ipynb` (numpy + matplotlib). Part II gates
end-to-end: `python reboot/scripts/run_z1.py`, `run_z2.py`, `diag_z2.py`,
`run_r1_calib.py` (CPU; the whole empirical arc cost ~3 CPU-hours). Part I's
underlying experiments live in the closed predecessor repo; the decisive comparisons
were re-run independently ~8 weeks after closure and reproduced to 2–3 decimals
(logs in [revalidation/](revalidation/)).

## Process notes, briefly

Every gate number in Part II was fixed and committed before its experiment ran
(first commit in [reboot/](reboot/) history). Two deviations, both logged with cause
before use. The pre-committed one-repair-round limit was honoured. The main claims
(C1/C3/C2) never ran because their precondition — gate zero — never held. Part I's
numbers were independently revalidated before publication; one error in its closing
report was caught and corrected ([REVALIDATION.md](REVALIDATION.md)).
