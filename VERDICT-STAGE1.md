# OBPF Stage 1 verdict — gate zero KILLED the operational objective (2026-07-22)

*Pre-registered gates: DESIGN-DRAFT.md (signed 2026-07-22, commit 000a696). Total
compute spent: ~2.5 CPU-hours, within caps. C1/C3/C2 were NOT run — the
pre-registration stops everything on a Z2 kill, and Z2 killed.*

## What happened, in order

1. **Z1 (K calibration): PASS.** K=100 — random partitions separate from the true one
   (median G 2.64 vs 1.01 ± 0.40); at K=300/1000 seed variance outgrows separation.
   Bonus: the operational objective alone annihilates the scaled-copy degeneracy
   (G(uniform) ≈ 191 — update-sum punishes duplicated mass by compounding overshoot).
2. **Z2 (warm-start non-degradation): GATE FAIL, KILL fired.** Initialised at the
   true partition, the full outer optimisation ends at mean AMI **0.594** (min 0.232)
   over 12 seeds — far below the 0.90 kill line. ([results/z2.json](results/z2.json))
3. **Diagnosis (optimizer-free, both pre-registered merge operators):**
   - **update-sum**: the true partition is not a local optimum of
     $G_K + \lambda_d \cdot \text{overlap}$ on **8/12 seeds** — single-task-moved
     neighbours (AMI 0.80) strictly beat it. ES drift explains only 4/12.
     ([results/diag_z2.json](results/diag_z2.json))
   - **mass-average**: globally inverted — $G(\text{true}) > G(\text{random})$ on
     **12/12 seeds**; truth is a local optimum on 0/12.
     ([results/diag_z2_avg.json](results/diag_z2_avg.json))

## The finding

**Finite-K merge fidelity is not a faithful surrogate for the separability
equality at this scale.** The operational gap decomposes as

> G_K = (coupling signal) + (merge-operator artifact),

and the artifact term dominates the landscape near the truth. Each operator fails in
its own characteristic way: update-sum's overshoot penalty prefers partitions that
minimise *travel geometry* (e.g. unbalanced branches move less), which is why moving
one task out of its true group improves the objective on most seeds; parameter
averaging dilutes specialists by 1/P, so it structurally rewards branches that each
train a bit of everything — mixing, the exact opposite of decomposition.

This is a different failure from the predecessor's. OPBF's affinity proxy had an
argmin unrelated to the truth for *representational* reasons. Here the desideratum
itself was the objective — and its cheap operational form injects operator physics
that swamps the signal it was supposed to measure. Gate zero was designed for exactly
this, caught it before the main claims ran, and cost ~2.5 CPU-hours.

The idealised reading (A) — argmin-set intersection at exact minimisation — remains
**untested, not refuted**: it is precisely the intractable-to-descend form that the
finite-K surrogate was meant to stand in for. What died is the surrogate, at this
scale, under both pre-registered merge operators.

## Honest scope notes

- Toy scale (12-task synthetic MTL, ~2k-param model), K=100, one λ_d. The artifact/
  signal ratio could differ at other scales; nothing here says it improves.
- The Z2 kill is gate-approved as *written* and *as intended* (the diagnosis confirms
  objective preference, not optimizer noise, as the dominant cause).
- A side observation with independent value for the model-merging/BTM literature:
  end-to-end optimising a partition-train-merge pipeline at this scale recovers
  merge-friendliness, not semantic structure.

## Options (Kulbir decides; no further compute without a new signed design)

1. **Stop and bank.** Two-layer honest negative: proxy objectives mis-point (OPBF);
   the desideratum's cheap surrogate is operator-dominated (OBPF Stage 1). Strong,
   coherent, article-ready arc — "we then trained on the thing itself, and the
   measuring instrument bent."
2. **One repair iteration (new design round, ~1 day):** damped update-sum with a
   per-evaluation line search $\theta_0 + \alpha \sum_i \Delta_i$, $\alpha \in (0,1]$
   — removes the overshoot term that produces the artifact; plus a (K × lr) window
   sweep asking whether any regime exists where coupling signal > artifact > noise.
   Gate-zero rerun under the repaired operator decides in half a day. Risk: operator
   whack-a-mole; mitigations: pre-commit to ONE repair round.
3. **Reframe around the artifact:** the merge-artifact-dominates finding, pointed at
   the BTM/merging literature, as the deliverable. Smaller claim, closer to their
   incumbent interest.
