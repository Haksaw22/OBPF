# OBPF verdict — FINAL: the operational objective cannot see the structure (2026-07-22)

*Pre-registered gates: DESIGN.md (signed 2026-07-22, commit 000a696; R1 repair
round authorized same day, pre-committed to one round). Total compute: ~3 CPU-hours,
within caps. C1/C3/C2 never ran — gate zero killed, the repair round failed at its
own calibration, and the pre-commitment stops there.*

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

## The repair round (option 2, authorized 2026-07-22; R1 in DESIGN)

The diagnosed artifact had a specific cure: damped update-sum with a per-evaluation
line search over the merge step (every partition gets its best damping), plus a full
(K × lr) window sweep asking whether *any* regime has coupling signal > artifact >
noise. Result ([results/r1_calib.json](results/r1_calib.json)):

**No cell in {50, 100, 300} × {0.005, 0.02, 0.05} separates true from random.**
Separation ratios −1.2 to 0.5 against a required 3. Worse — with the artifact
removed, the **uniform (scaled-copy) family becomes the best partition in most
cells** (e.g. K=300, lr=0.02: G(uniform) 0.13 vs G(true) 1.70): four damped
generalists reconstruct joint training almost exactly, while four specialists leave
each other's features untrained — a deficit no scalar damping can repair.

Two conclusions follow. First, **Z1's original "separation" was entirely the
overshoot artifact** — what made random partitions look worse than the truth at
α = 1 was overshoot magnitude, not coupling. Second, once the artifact is removed,
**the finite-K merge-fidelity gap contains no usable trace of the true structure at
any probed scale — and actively prefers non-decomposition** (every branch a
generalist). Merge fidelity, as an objective, opposes specialisation here.

## Final disposition — BANKED, three-layer negative

Per the pre-commitment (one repair round), the project closes. The layered result:

1. **OPBF:** a *proxy* objective for decomposition (gradient affinity) has its
   argmin away from the true structure — warm-started at the truth, it walks away.
2. **OBPF gate zero:** the *desideratum itself*, operationalised as finite-K merge
   fidelity, has its landscape dominated by merge-operator physics — the truth is
   not a local optimum under either pre-registered operator.
3. **OBPF R1:** removing the operator artifact removes the signal too, and the
   artifact-free objective prefers generalist copies over any decomposition. The
   window in which this objective family could identify structure does not exist at
   this scale.

Untouched by all three: the idealised equality (reading (A), exact minimisation) —
intractable to descend, which is the entire practical problem. The residual open
question is unchanged from the theory draft, now sharpened: *what descendable
objective has the true decomposition as its minimum?* Three natural candidates are
now dead: affinity proxies, finite-K merge fidelity, and its damped repair.

Side finding with standalone value: for the branch-train-merge literature, this says
end-to-end optimising a partition-train-merge pipeline at small scale does not
recover semantic structure — it recovers merge-friendliness, and merge-friendliness
prefers redundant generalists. (Consistent with why c-BTM's clusters help at LLM
scale for *capacity* reasons, not structure-recovery reasons.)

The article can now be written on a complete, coherent arc.
