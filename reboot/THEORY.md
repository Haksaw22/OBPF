# OBPF reboot — theory draft v0

*Status: **Approved 2026-07-22** — three-readings fork resolved as proposed
((C) merge-fidelity operational, (A) compatible-demands theory, (B) parameter-partition
baseline); degeneracy handling approved. Experiment design lives in DESIGN.md
(separate sign-off gates any compute). Predecessor (OPBF: affinity-proxy factorisation
of given loss terms) is closed and its negative stands; this is a different mechanism.*

## 0. The formulation (2026-07-22)

Raw per-timestep contributions $Q_t$ — data per timestep, not hand-enumerated loss
terms. A learnt decomposition $D_i = f_i(\{Q_t\})$, possibly $D_i = \sum_j d_{ij}$,
with the defining property

$$\min_\theta \sum_t Q_t(\theta) \;=\; \sum_i \min_\theta D_i(\theta).$$

The equality (in general: the gap between the two sides) is the **training objective
of the decomposition itself**, not an after-the-fact audit. The predecessor project
never optimised this — that is the delta that reopens the question.

## 1. Setup and the gap

Data stream indexed by $t$; per-datum loss $\ell_t(\theta)$ (the $Q_t$ above); total
$L(\theta) = \sum_t \ell_t(\theta)$. The simplest learnable decomposition family is a
**soft partition of the stream**: weights $w_{it} \ge 0$, $\sum_i w_{it} = 1$,

$$D_i(\theta) = \sum_t w_{it}\, \ell_t(\theta), \qquad \sum_i D_i = L \text{ automatically.}$$

(The general $f_i$ / learned-$d_{ij}$ version generalises this — re-featurisation, not
just reweighting — but the partition family is where analysis and first experiments
should start.) Define the **gap**

$$G(w) \;=\; \min_\theta L(\theta) \;-\; \sum_i \min_\theta D_i(\theta) \;\ge\; 0,$$

and learn $w$ (or $f$) by minimising $G$, subject to a non-degeneracy pressure (§3).
$G \ge 0$ always; the formulation asks for decompositions that make it small.

## 2. Three inequivalent readings — pick one consciously

The one-line version hides a fork. These are different projects:

**(A) Common-minimiser / compatible-demands reading.** $G(w) = 0$ iff the components'
argmin *sets* have a common point. The right mental picture is set-valued: a
component that has "one skill's data" should pin down the parameter directions that
skill needs and be *flat* (indifferent) everywhere else. Flat components have large
argmin sets; the equality asks that all these sets intersect — every skill's demands
are simultaneously satisfiable, and the joint optimum sits in the intersection. This
is the "skills" story told properly: each part self-sufficient on its own directions,
silent on the others. Note what it is *not*: if a component's argmin is a single
point equal to $\theta^*$, that component alone identifies the entire solution — a
coreset, not a skill.

**(B) Parameter-partition reading (modularity).** If $\theta$ splits into disjoint
blocks $\theta_i$ and $D_i$ touches only $\theta_i$, the equality holds *by
construction* — $\min_\theta \sum_i D_i(\theta_i) = \sum_i \min_{\theta_i} D_i$ is an
identity for separable variables. All content moves into whether the task tolerates
the block structure (this is just separate trunks; the predecessor's E7 Part B tested
it: quality survives when per-block capacity is adequate — known, not novel). Reading
(B) is the trivially-achievable corner of the space; interesting only when the *data*
grouping and the *parameter* blocks are learned jointly.

**(C) Merge reading (the operational one).** Nobody evaluates $\min_\theta D_i$
exactly. The runnable version: from shared init, train a copy on each $D_i$
independently for $K$ steps, merge (average / block-concat / update-sum), and compare
against $K$ steps of joint training at matched total budget:

$$G_K(w) \;=\; L\big(\mathrm{merge}(\theta_1^K, \dots, \theta_P^K)\big) \;-\; L\big(\theta_{\text{joint}}^K\big),$$

minimised over $w$ by unrolled differentiation (MAML-style) or evolution strategies.
This is the version an experiment actually tests, and the version the parallel-training
motivation actually needs. (A) is its $K \to \infty$, exact-merge idealisation.

**Working choice:** target (C) as the operational objective, keep (A)
as the theory story, treat (B) as a baseline (given parameter blocks) rather than a
result.

## 3. Degeneracies — name them before they bite

1. **The scaled-copy family.** $D_i = c_i L$ (all weight-rows identical up to scale)
   satisfies the equality exactly and decomposes nothing. Harmless for parallelism
   (each worker trains the whole loss — no speedup, no damage), useless for
   interpretability and transfer. Exclusion: a disjointness pressure, e.g. penalise
   row overlap $\sum_t w_{it} w_{jt}$, or hard capacity per component. *Predecessor's
   lesson, reincarnated: reconstruction-only was gauge-free; equality-only is
   degeneracy-free only modulo this family. The second criterion is again where the
   identifying content lives — but see §4 for why the situation is structurally
   better this time.*
2. **The small-$K$ linearisation regime.** For tiny $K$, updates are near-linear and
   *everything* merges fine — $G_K \approx 0$ regardless of $w$. The probe must run
   at $K$ large enough that interference is visible for a known-bad partition
   (calibrate on random $w$ before trusting any learned $w$).
3. **Easy-data dumping.** Components can lower their $\min$ by hoarding easy data.
   Conservation ($\sum_i w_{it} = 1$) limits this globally but not locally; watch the
   per-component data-difficulty profile as a diagnostic.
4. **Exact equality is generally unachievable — and that is a feature.** If the task
   genuinely couples skills, no nontrivial decomposition reaches $G = 0$. The object
   of interest is the trade-off curve: minimum achievable gap as a function of the
   disjointness constraint — a **coupling curve** for the task. Its shape (how much
   independence can be bought, at what price) is a measurable property of the
   problem, not of the learner. This is the cleanest new instrument the formulation
   offers.

## 4. Why this evades the predecessor's specific failure — and what it does not evade

The predecessor died of proxy misalignment: its descended objective (gradient-affinity
+ reconstruction + entropy) had minima that were not the true structure, demonstrated
by warm-starting at the truth and watching the objective walk away. Here the descended
objective **is the desideratum**: $G_K$ is measured against real joint training — an
external, non-learnable anchor. A decomposition cannot score well by gaming a
surrogate; it scores well only if independent training actually works. The warm-start
test still applies and becomes a *design gate*: initialise $w$ at the true partition
(where truth is known) and verify $G_K$ does **not** prefer to leave — if it does,
the formulation is wrong and we stop.

What it does not evade: (i) identifiability — if the data stream simply doesn't
determine a unique partition, gap-minimisation finds *a* compatible decomposition,
not *the* one (fine for parallelism, weaker for interpretability; say so up front);
(ii) cost — the objective contains inner training runs; every outer step is $P$
inner runs of $K$ steps (toy-scale CPU feasible; scaling is an open question, not a
promise); (iii) the degeneracy family of §3.1, which the disjointness term must
carry alone.

## 5. Claims worth testing (mapped to the original motivations)

Design-gate drafts only — numbers/kill-criteria to be fixed in DESIGN before any run.

- **C1 (decoupling — recovery).** On synthetic MTL with known groups (predecessor's
  env, truth known, affinity's failure documented at 0.84 vs k-means 0.98):
  gap-minimisation over soft data-partitions recovers the true grouping. Gate sketch:
  AMI meaningfully above the affinity predecessor; kill if ≤ affinity. Baselines:
  k-means-on-affinity, random partitions, given-label partition, scaled-copy family,
  uniform shards.
- **C2 (decoupling — parallel fidelity).** The learned partition trains
  factor-parallel at matched budget without quality loss (machinery exists from the
  predecessor). Gate sketch: within pre-registered tolerance of joint.
- **C3 (measurement — the coupling curve).** Across environments with a ground-truth
  coupling knob (the predecessor's synthetic family has one), the coupling curve
  (min gap vs disjointness) orders environments by true coupling strength. Gate
  sketch: rank correlation with the knob. This claim is the most distinctive.
- **C4 (nested optimisation — stretch, later).** An outer loop over the learned
  decomposition (scheduling/reweighting components) beats monolithic descent at
  matched compute. Riskiest; not part of the first design.
- **Gate zero (before all of the above): the warm-start sanity gate of §4.**

## 6. Novelty status (sweep complete — see NOVELTY.md)

Nine literatures swept: 0 owned, 5 adjacent, 4 open-delta. The composite is
unclaimed; the ingredients are not. Headline finding: the gap $G$ already has a name
— the FedAvg **heterogeneity constant** $\Gamma = F^* - \sum_k p_k F_k^*$ — used
throughout local-SGD convergence theory as an *assumed property of a given partition*,
never as a trainable objective. Must-cite-and-differentiate: Grimm & Singh 2019
(independence-optimised conserved *reward* decomposition — the closest single work),
QTRAN/IGM (the argmax-isomorph of reading (A), given partition, joint TD training),
MERIT 2026 + c-BTM (the partition-train-merge pipeline with proxy partitions — our
incumbent baselines). The coupling curve of §3.4 is, in the field's language, the
achievable-$\Gamma$ frontier. Details, false friends (e.g. "sum-of-minimum
optimization" is the opposite nesting), and the narrowly-stated novelty claim are in
NOVELTY.md.

## 7. Naming

Working name **OBPF** (matching this project's GitHub repo). Backronym candidate:
*Objective decomposition By Parallel-training Fidelity* — accurate, since fidelity of
independent-then-merged training to joint training is literally the loss.
