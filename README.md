# OPBF — an honest negative result

**Optimisation-Preserving Behavioural Factorisation**: can a learned, differentiable
factoriser decompose a training objective (`L = Σᵢ Zᵢ`) into a few near-independent
latent factors — for parallel training, transfer, and interpretability? In spirit,
*ICA for objectives*.

**Answer: no.** Across seven applications and pivots, every learned OPBF mechanism lost
to a fair — usually trivial — baseline, under pre-registered numeric gates and
adversarial verification:

| Claim | Fair baseline that wins | Score |
|---|---|---|
| Group recovery (soft assignment) | k-means on the same signal | 0.84 vs **0.98** AMI |
| Group recovery (learned *hard* assignment) | k-means | 0.73 vs **0.98** AMI, p=5e-4 |
| Frozen-factor transfer | generic frozen encoder (equal size/budget) | 0.888 vs **0.997**, p=5e-5 |
| Interpretability (do-intervention localisation) | a random partition | spillover 0.47/0.71 vs **0.36/0.44** |
| Conflict-aware optimisation under drift | GradVac | tie; smoothing *hurts* under fast drift |
| Conflict signal as a training diagnostic | PCGrad + momentum | 0.057 vs **0.042** |
| Active probing for identification (RL-pivot precondition) | uniform coverage | tie at AMI 1.0 |

**Root cause:** learned soft grouping is a *dominated middle* — when structure is
recoverable, hard clustering on the same signal recovers it better; when it isn't, soft
assignment is lossy and every downstream payoff inherits the loss. Independent
revalidation sharpened this: even warm-started *at* the k-means solution, the
factoriser's own training objective walks away from the true partition — the objective,
not the optimiser, is misaligned.

## Honest status

- The project is **closed** (self-closed 2026-05-30 after ~70 hours across three
  generations; the closing commit reads "project banked — does not have wings").
- Every load-bearing number here was **independently revalidated** (2026-07-22) against
  the raw code and data before publication: baseline fairness audited at code level,
  decisive comparisons re-run and reproduced, two untried rescue routes tried (both
  fail), one erroneous figure in the closure report caught and corrected. See
  [REVALIDATION.md](REVALIDATION.md).
- Not done, stated plainly: the co-adaptation regime (the one place differentiable
  grouping could in principle win) was closed by argument, not experiment; drift of the
  *conflict structure* was never exercised; everything is CPU-scale synthetic/bandit/toy-MTL.

## Map

- **[BLOG.md](BLOG.md) — "The Price of Coupling."** The write-up: the idea, the three
  reasons it should have worked, the scoreboard, and the mechanism of the failure —
  including an animation of the factoriser being handed the correct answer and
  training itself away from it. Start here.
- **[REVALIDATION.md](REVALIDATION.md)** — the independent verification pass behind
  every number cited.
- **[playground.ipynb](playground.ipynb)** — rebuilds the headline figures from the raw
  data in minutes on CPU; pre-executed so it renders without running anything.
- **[figures/](figures/)** — the blog's figures, including
  [objective_walks_away.gif](figures/objective_walks_away.gif) (a real training run:
  initialise at the true partition, minimise OPBF's own objective, watch agreement
  with the truth fall while the loss goes down).
- **[data/](data/)** — raw result JSONs copied verbatim from the closed research repo
  (E6 transfer, E7 recovery, E9 interpretability, E10 optimiser, E1 smoke).
- **[revalidation/](revalidation/)** — re-run logs and probe scripts from the
  revalidation pass (hard-vs-k-means re-run, tpcgrad arc reproduction, missed-exit
  probes).

## Reproduce

The scoreboard figures: `jupyter execute playground.ipynb` (matplotlib + numpy only).

The underlying experiments live in the closed source project (three generations: a spec
suite, a first implementation, and the clean rewrite `opbf2` whose committed harness
produced everything cited here). The revalidation re-ran the decisive comparisons from
that committed code ~8 weeks after closure and reproduced them to 2–3 decimals; the
re-run logs are archived here.

## What's reusable

If you're tempted by learned objective factorisation, the transferable assets are:
the fair-baseline suite (k-means / generic encoder / random partition / PCGrad+momentum
/ GradVac), the best-learning-rate-per-method protocol, out-of-sample holdout as a
non-negotiable, and the scoreboard above as prior evidence.

---

*Status: draft pending author review; not yet published.*
