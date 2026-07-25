# OBPF reboot — novelty verdict (2026-07-22)

*Nine-literature sweep (multi-agent value factorisation, reward decomposition,
distributed optimisation, model merging/local SGD, bilevel/unrolled, MTL task
grouping, modular DL/MoE, continual/data partitioning, direct phrase & theory).
Per-area verdicts: 0 × OWNED, 5 × ADJACENT, 4 × OPEN_DELTA. Full per-area verdicts
omitted here; this is the synthesis.*

## Verdict

**The composite is unclaimed; every ingredient is not.** No found work learns a
conserved decomposition of the raw per-timestep loss stream by minimising the
min-of-sum minus sum-of-mins gap over shared parameters, with the residual read as a
coupling measurement. But three works perform pieces of the core move closely enough
that positioning lives or dies on stating the deltas, and one literature already
*names our objective* — as a constant nobody optimises.

**The best single sentence the sweep produced:** in FedAvg convergence theory the
quantity $\Gamma = F^* - \sum_k p_k F_k^*$ — literally our gap, zero iff the parts
share a common minimiser — is a standard *assumed heterogeneity constant* of a given
client partition, used in bounds. Nobody has ever treated $\Gamma$ as a loss and
learned the partition by descending it. "The gap has had a name since 2020; it has
never had a gradient" is both the novelty claim and the article hook, if the project
survives its gates.

## The three works that must be cited or reviewers do it for us

1. **Grimm & Singh 2019, "Learning Independently-Obtainable Reward Functions"** —
   the closest work found anywhere. Learns a conservation-respecting decomposition
   (of *reward*) by optimising an independence criterion, with explicit
   anti-triviality machinery (they too fought the scaled-copy degeneracy). Deltas:
   substrate is environment reward over states, not a training-loss stream; their
   independence is pairwise disjointness across *separate policies'* value functions
   — the joint optimum never appears and there is no shared $\theta$, so the min-sum
   gap is not their objective; no train-then-merge; no coupling readout. The reboot
   is, honestly stated, "the Grimm–Singh move transplanted to supervised loss streams
   with the gap itself as the objective."
2. **QTRAN 2019 / the IGM lineage (VDN, QMIX, QPLEX)** — the equality "independent
   optima compose to the joint optimum" as an explicit training constraint on a
   learned factorisation. Deltas: over joint *actions* at each state, partition given
   by the agent roster, one joint TD loss trains everything, components never trained
   independently. IGM is the argmax-isomorph of reading (A) in THEORY-DRAFT; say so
   before a MARL reviewer does.
3. **MERIT 2026 (conflict-aware data splitting) and c-BTM 2023 (clustered
   branch-train-merge)** — the operational pipeline (partition data, train
   independently, merge) exists at scale. Deltas: MERIT splits by a first-order
   gradient-conflict heuristic (structurally the same affinity move the predecessor
   project already watched lose) with hard assignment; c-BTM partitions by embedding
   k-means, optimised against nothing. Neither descends the joint-vs-merged gap.
   These are not just citations — **they are the incumbent baselines the experiments
   must beat**, and they make the comparison story clean: proxy-partition (affinity /
   k-means) vs outcome-partition (gap-trained).

## Corroborating gaps (the areas' own words)

- The **Modular Deep Learning survey** (Pfeiffer et al. 2023) states that automated
  module *discovery* and formal modularity metrics are underdeveloped — the canonical
  reference declaring the slot open.
- **MergeProbe 2026** (predicting mergeability of fine-tuning updates) names
  data-mixture feedback for merge compatibility as *future work* — the delta is being
  circled right now. Timeliness cuts both ways: room to move, and months not years.
- **Task grouping** (Standley 2020; TAG; DMTG 2024) already selects groups by
  separate-training outcomes — but over hand-enumerated tasks, as discrete selection
  or joint-training loss, never as a differentiable objective of a learned
  decomposition of the stream.

## False friends (disclaim in any write-up)

- **"Sum-of-minimum optimization"** (Cui, Ding et al. 2024): the min sits *inside*
  the sum (per-datum best-of-k assignment — clustering). Opposite nesting. The name
  collision is unfortunate; flag it.
- **Rockafellar–Wets decomposable spaces / interchange of min and integration**:
  existence theory for when the equality holds; verified, never searched-for.
- **Lagrangian/dual decomposition, ADMM, Shapley–Folkman duality-gap bounds**: the
  complement of this project — coordinate a *given* split; we learn the split so
  coordination is unnecessary. Good theory ancestry for the gap-as-coupling reading.
- **Inf-convolution of risk measures**: allocation among given agents; terminology
  ancestry only.

## Implications adopted into the design

1. **Novelty claim, narrowly stated:** "the FedAvg heterogeneity constant $\Gamma$ as
   a trainable objective over a learned, conserved, soft decomposition of the
   training stream; residual $\Gamma$ as a coupling measurement." Not "learning
   decompositions" (owned in pieces everywhere), not "train-and-merge" (BTM's), not
   "independence-optimised decomposition" (Grimm & Singh's).
2. **Baselines fixed by the sweep:** c-BTM-style embedding k-means; MERIT-style
   gradient-conflict split; random partition; given-label partition; scaled-copy
   family; uniform shards. The first two are the field's actual incumbents, and both
   are proxy-partitioners — the experiment is precisely proxy vs outcome.
3. **Vocabulary to adopt:** heterogeneity $\Gamma$ (local SGD), IGM (MARL),
   independently-obtainable (Grimm & Singh). The coupling curve of THEORY-DRAFT §3.4
   is "the achievable-$\Gamma$ frontier" in the field's own language.
4. **Fun-investigation fallback is safe:** even if a referee collapses the delta onto
   Grimm & Singh + BTM, "we gave the oldest constant in federated learning a gradient
   and watched what it discovers" stands on its own as an investigation.
