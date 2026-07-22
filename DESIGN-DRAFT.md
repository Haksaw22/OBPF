# OBPF reboot — experiment design (pre-registration draft)

*Status: **SIGNED by Kulbir 2026-07-22.** This document is the pre-registration: every
gate number was fixed before any result was seen; deviations get logged in the dated
DEVIATIONS section at the bottom; demotions get published. Theory basis:
THEORY-DRAFT.md (approved 2026-07-22, reading (C) operational). Tier: **M** — CPU
only, hard cap 2 days wall-clock including analysis.*

## 1. Objects

**Environments.** Reused from the predecessor's committed harness (imported read-only
from `OPBF/opbf2/src`, as the revalidation probes did; all new code lives here):

- **E-MTL**: `StructuredMultiTaskRegression` — T=12 task losses in G=4 known groups,
  the exact env where affinity scored 0.84 and k-means 0.98. Used for gate zero, C1, C2.
- **E-COUPLE**: the synthetic-objectives family with its ground-truth coupling-strength
  knob (the predecessor's E2 sweep machinery). Used for C3.

**Granularity staging (deliberate).** Stage 1 uses task-level units (t indexes the 12
task losses) for exact comparability with the predecessor's numbers. The raw
per-timestep stream (t indexes examples) is Stage 2, only reached if Stage 1 gates
pass — walking before running, and the walk is on the ground where the proxy approach
demonstrably stumbled.

**The decomposition.** Soft conserved partition $w \in \mathbb{R}^{T \times P}$,
rows on the simplex ($w_{it} \ge 0$, $\sum_i w_{it} = 1$), $P = G = 4$ (oracle count,
same concession every baseline gets). $D_i(\theta) = \sum_t w_{it}\,\ell_t(\theta)$.

**The objective (reading C, operational).** From a shared warm init $\theta_0$
(fixed per seed, 10 warmup steps of joint training — the predecessor's protocol):

- Independent branch: $P$ copies, each trained $K$ steps of Adam on its $D_i$.
- Merge: primary operator **update-sum** $\theta_0 + \sum_i (\theta_i - \theta_0)$;
  secondary **mass-weighted parameter average**. (Primary chosen because update-sum
  is the natural additive-decomposition merge; both reported.)
- Score: $G_K(w) = L(\text{merged}) - L(\theta^K_{\text{joint}})$, **matched-parallel**
  budget as primary ($K$ steps per worker vs $K$ joint steps — parallelism is the
  motivation), matched-total ($P \cdot K$ joint steps) reported as secondary.
- Regulariser: disjointness penalty $\lambda_d \sum_{i \ne j} \sum_t w_{it} w_{jt}$
  (excludes the scaled-copy family; the $\lambda_d$ sweep IS the coupling curve).

**The outer optimiser.** Primary: simple ES / NES over $w$-logits (population 32,
~30 generations) — no unrolling bias, few knobs. Secondary (if ES succeeds and budget
remains): unrolled gradient through small-$K$ inner training, to check the
differentiable version agrees. Pre-committed: conclusions ride on the primary.

## 2. Gate zero — the two sanity gates (run FIRST; everything stops on failure)

**Z1 — K calibration (validity precondition).** Choose the smallest $K$ (from
{100, 300, 1000}) such that a random hard partition is separated from the true
partition: median $G_K(\text{random}) > G_K(\text{true}) + 3\sigma_{\text{seed}}$,
6 seeds. If no $K$ in budget separates them, the env cannot test the claim →
**redesign env before any C-experiment** (this is a validity failure, not a result).

**Z2 — warm-start non-degradation (the gate the predecessor failed).** Initialise $w$
at the true partition; run the full outer optimisation ($G_K + \lambda_d$ at the
mid-sweep $\lambda_d$). Gate: mean AMI after optimisation ≥ 0.95 over 12 seeds and no
seed below 0.8. **Kill criterion: mean AMI drop > 0.10 → the gap objective also
prefers to leave the truth → the formulation is wrong; project stops and the write-up
says so.** (This is the exact test that exposed the predecessor; it is run here
*first*, not during revalidation-after-the-fact.)

## 3. C1 — recovery (main pre-registered claim)

**Prediction.** Outcome-partitioning beats proxy-partitioning where the proxy
demonstrably underperformed: optimising $G_K$ + disjointness over $w$ from random
inits recovers the true grouping.

**Protocol.** 12 seeds; per seed the same batches/coupling data every method sees
(seed-matched); AMI of hard-argmax($w$) vs truth.

**Gates (fixed now).**
- **PASS**: mean AMI ≥ 0.95 AND ≥ (k-means mean − 0.03) — ties or beats the 0.98
  incumbent within noise.
- **PARTIAL**: mean AMI ≥ 0.88 (above affinity's best probe) but below k-means —
  honest reading: "the desideratum objective beats the proxy objective; clustering
  still suffices on a clean signal." Publishable as such; Stage 2 (raw stream, where
  no clean affinity matrix exists) becomes the decisive ground.
- **KILL**: mean AMI ≤ 0.84 (the affinity predecessor) — the desideratum objective
  cannot even match the proxy it was meant to replace; dominated-middle redux;
  write-up says so.

**Baselines (fixed by the novelty sweep; every one gets the same coupling data,
oracle $P$, and seed-matched evaluation).** k-means on the affinity matrix (0.98
incumbent); MERIT-style gradient-conflict hard split; c-BTM-style embedding k-means;
random partitions (20 draws/seed); uniform shards; scaled-copy family (checked to be
*rejected* by the disjointness term — a mechanism test, not a competitor); oracle
truth (ceiling).

## 4. C3 — the coupling curve (the distinctive instrument)

**Prediction.** On E-COUPLE across ≥ 5 coupling-knob settings × 6 seeds: the minimum
achievable $G_K$ at fixed $\lambda_d$ increases with true coupling strength.

**Gate**: Spearman ρ ≥ 0.8 (knob vs min-gap). **Kill**: ρ < 0.5 → the residual gap is
not a usable coupling measurement at this scale; the "achievable-Γ frontier"
instrument claim is dropped (C1's verdict is unaffected).

## 5. C2 — parallel fidelity (cheap addendum, only if C1 ≥ PARTIAL)

With the C1-learned partition: factor-parallel training (per-group trunks, the
predecessor's Part B machinery, equal-total-capacity variant) within **5%** of joint
final per-task loss, 10 seeds. Same gate the predecessor pre-registered; now with a
learned-by-outcome partition feeding it.

## 6. Budget, cost model, caps

One $G_K$ evaluation = $(P{+}1) \times K$ inner steps of a ~2k-param MLP ≈ seconds on
CPU. ES at pop 32 × 30 generations × 12 seeds ≈ 11.5k evaluations → est. 4–8 h with
8-way multiprocessing. **Hard caps**: gate zero ≤ 0.5 day; C1 ≤ 1 day; C3 ≤ 0.5 day;
C2 ≤ 2 h. Total ≤ 2 days wall. If an estimate is exceeded 2×, stop and report rather
than silently downscope. No GPU. Overage plan: drop to 6 seeds for a first pass and
confirm survivors at 12 — logged as a deviation if used.

## 7. What this design does NOT claim

No scale claims (CPU toys throughout); no language-model claims; identifiability
caveat inherited from THEORY-DRAFT §4 (gap-minimisation finds *a* compatible
decomposition — uniqueness is not claimed); Stage-2 raw-stream results are out of
scope for these gates and get their own design if Stage 1 passes.

## 8. Order of execution

Z1 → Z2 → C1 → C3 → C2 → verdict doc (per-gate outcomes, PASS/PARTIAL/KILL stated
plainly) → only then the article question reopens.

## DEVIATIONS

*(empty at pre-registration; dated entries only)*
