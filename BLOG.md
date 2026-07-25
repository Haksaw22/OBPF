# The Price of Coupling

*Every training loss is a sum, and the sum hides whatever anatomy the task actually has. This piece writes down exactly what it would mean to learn that anatomy — choose the decomposition that closes the gap between the minimum of the sum and the sum of the minima — and reports what happened when I optimised that directly, under kill criteria fixed before any experiment ran. The gate built to catch a lying objective caught this one, and the autopsy is worth more than the result would have been. Total bill: about three CPU-hours, two dead objectives, one question sharper than the one I started with.*

**Code, data, and the honest revalidation trail live in the accompanying repository, under [reboot/](reboot/). I am now reworking this project from the ground up; what follows is the record of the first attempt, kept honest.**

---

## A loss is a sum, and sums hide anatomy

Almost everything we train is graded by a sum: $L = \sum_t Q_t$, thousands of
per-timestep loss contributions handed to an optimiser as one number — the index $t$
is just bookkeeping. If the objective is secretly built from a few underlying skills, the
sum does not say which contributions belong to which skill — and gradient descent,
which only ever sees the sum, cannot care.

I wanted the anatomy, for three reasons — none of which really depend on how this ends.
Partly for parallelism: if the skills separate even approximately, you can curate
data for one skill and train on it without trampling the others — every skill at
once, on separate workers, ideally with autonomy granted gradually as each part
proves it can train alone without regressing the whole. Partly because a factored
loss is a richer game than a monolithic one. A single number admits one move,
descend it; a decomposition admits scheduling, reweighting, and restructuring *of
the decomposition* — editing source instead of the compiled binary. And partly on
the conviction that the most relevant coordinates for a problem are a property of
relationships between states or their latents, not something fixed in advance — so
the decomposition should be learned, and learnable *as it moves*, not bolted on
afterwards by a clustering pass.

The test of whether a split is real has a pleasingly blunt form: train the parts
apart, glue the results together, and see whether the glued model matches the one
trained jointly. If the syllabus really was four courses, four students studying
separately should reconstruct the class.

## The object: close the gap by choice of decomposition

For any decomposition of the stream into components $D_i = f_i(\{Q_t\})$ with
$\sum_i D_i = \sum_t Q_t$,

$$\min_\theta \sum_t Q_t(\theta) \;\geq\; \sum_i \min_\theta D_i(\theta),$$

with equality exactly when the components share a common minimiser. The gap is the
**price of coupling**: what you lose by optimising the parts separately instead of
together. My proposal was to make the gap itself the training objective of the
decomposition — why not ask directly for the split whose parts, trained apart, still
add up to the whole?

The picture that makes this a *skills* story: a component holding one skill's data
should pin down the parameter directions that skill needs and be flat —
indifferent — everywhere else. Flat components have large sets of minimisers; the
equality asks that all those sets intersect, with the joint optimum in the
intersection. (A component whose minimiser is a single point would be something
else: a dataset that determines the entire solution by itself.) And a bonus if it
works: the *residual* gap, at the best achievable decomposition, is a measurement of
the task — how much coupling is intrinsic, in loss units.

Exact inner minimisation is intractable — evaluating the right side means running
$P$ optimisations to convergence for components that are themselves being learned —
so the operational form is **merge fidelity**: from a shared warm start, train one
copy per component for $K$ steps, merge the updates (merging here means the
bluntest thing available — add the weight updates together), and score the merged
model against $K$ steps of ordinary joint training:

![the machine](figures/gap_concept.png)
*The soft partition $w$ is the learned object. $G_K(w) \ge 0$ in expectation;
minimising it over $w$ asks for components that can be trained apart without cost.*

One degeneracy is visible from the armchair and needs naming before any experiment:
components that are *scaled copies* of the whole loss ($D_i = c_i L$) share a
minimiser trivially and decompose nothing. I guarded the exclusion with a
disjointness penalty on the partition weights — though, as it turned out, the
objective had stronger opinions about this family than the penalty did.

I searched nine adjacent literatures for this object before building it ([the
sweep, per-area verdicts included](reboot/NOVELTY.md)). The nearest relatives: reward decompositions optimised for independence (Grimm & Singh
2019), the IGM condition in multi-agent RL — equality of argmaxes by construction,
over actions, with the partition given — and partition-train-merge pipelines whose
partitions come from proxies like embedding clustering (c-BTM) or gradient conflict
(MERIT). In federated learning the gap itself has a standard name, the heterogeneity
constant $\Gamma = F^* - \sum_k p_k F_k^*$, where it is assumed for a given client
partition and used in convergence bounds — and, as far as I could find, never used
as a trainable objective over a learned decomposition. Make of that what you will —
I took it as reason enough to run the experiment.

## Rules before results

Everything below was pre-registered — gate numbers fixed and committed (dated, in
[the repo history](reboot/DESIGN.md)) before any experiment ran, deviations
logged with cause, and one repair round pre-committed as the maximum.

The load-bearing gate deserves its one sentence of history. An earlier attempt at
this problem through a *proxy* objective (gradient affinity) died of a specific
disease — warm-started at the true decomposition, its objective preferred to leave —
and that autopsy ([archived here](legacy/OPBF-autopsy.md), deprecated) contributed
exactly one thing to the present project: the test became **gate zero**. Before any
recovery claims run: calibrate $K$ so that a random partition is measurably worse
than the true one (Z1 — if no affordable $K$ separates them, the environment cannot
test the claim at all); then initialise the decomposition *at the true partition*
and let the gap objective keep training (Z2). If it walks away from the answer, the
objective is wrong and the project stops — before, not after, the headline
experiments.

Testbed: synthetic multi-task regression with known ground truth — twelve
per-timestep loss channels in four latent skill groups, a small shared trunk, soft
partition weights $w \in \mathbb{R}^{12 \times 4}$ optimised by evolution strategies
(no unrolling bias), inner training by SGD. That last choice was itself a logged
finding: Adam's per-parameter normalisation moves zero-gradient parameters at full
step size, which quietly destroys *any* additive merge regardless of partition
quality. Adam breaks merging — file that one away.

## Gate zero

Z1 passed at $K = 100$, with one pleasant surprise: the scaled-copy family I had
planned to exclude by penalty is annihilated by the objective itself — duplicated
mass makes the summed updates overshoot catastrophically ($G \approx 191$ against
the truth's $\approx 1.0$). Then Z2 ran.

![gate zero](figures/z2_walkaway.png)
*Twelve seeds, each initialised at the true partition, each fully optimised on the
gap objective. Mean final agreement with the truth: 0.59, far below the
pre-registered kill line of 0.90.* ([raw results](reboot/results/z2.json))

The kill criterion fired. And the important question — is this the objective's
preference, or just optimiser noise? — has an optimiser-free answer: evaluate the
objective at the true partition and at all 36 single-task-moved neighbours. On
**8 of 12 seeds the truth is not even a local optimum** — moving one task out of its
true group strictly improves the objective ([diagnostics](reboot/results/diag_z2.json)).
Under the pre-registered secondary merge operator, parameter averaging, it is worse:
the global ordering *inverts* on 12 of 12 seeds — averaging four specialists dilutes
every skill to quarter strength, while averaging four generalists merely averages
four decent models. Averaging rewards mixing; mixing is the opposite of anatomy.

So the finding at gate zero: the operational gap decomposes as **coupling signal
plus merge-operator artifact, and near the truth the artifact is most of what the
objective sees**. Update-sum prefers partitions with favourable travel geometry;
averaging prefers redundancy. The objective *was* the thing I actually wanted this
time — and the measuring instrument bent anyway.

## The repair round

The artifact had a mechanistically understood cause — overshoot — and therefore a
specific cure: damp the merged update, with a per-evaluation line search so that
*every* candidate partition gets its own best step size, and sweep the full
$(K \times lr)$ grid asking whether any regime puts coupling signal above artifact
above noise. One round, pre-committed, gates fixed before running.

![no window](figures/r1_nowindow.png)
*Nine calibration cells. Blue: the true partition. Orange: random partitions. With
the overshoot artifact removed they are statistically indistinguishable in every
cell — and the uniform-copies family (green) is frequently the best "decomposition"
in the grid.* ([raw results](reboot/results/r1_calib.json))

No cell separates — two conclusions, each worth the day it cost. First, the
separation that Z1 had certified was **entirely the artifact**: what made random
partitions look worse than the truth was overshoot magnitude, never coupling.
Second, with the artifact gone the objective does not merely fail to find the
structure — **it actively prefers non-decomposition**. Four redundant generalists
reconstruct joint training almost exactly; four specialists leave each other's
features untrained, and no scalar damping repairs that deficit. Merge fidelity, as
an objective, rewards exactly the thing decomposition exists to eliminate.

Per the pre-commitment, that was the round. Stop.

## What died, and what is standing

Dead, by direct experiment: finite-$K$ merge fidelity as an identifier of structure
(what it scores is operator physics), and its artifact-corrected repair (nothing
left underneath, and the remainder anti-correlated with anatomy). Dead earlier, for
a different reason, and archived: the affinity proxy. Standing, untested rather than
refuted: the idealised equality itself, at exact minimisation — which is precisely
the form nobody can descend. That is the residue, and it is sharper than where I
started: **what descendable objective has the true decomposition as its minimum?**
Having run the honest version of the experiment, I really do think this is a
question about objectives, not an engineering gap, and the corpses above are its
boundary markers. It is the question the ground-up rework is aimed at.

A side observation that stands alone: for the branch-train-merge line of work, this
says end-to-end optimising the partition-train-merge pipeline at small scale would
not recover semantic structure — it recovers *merge-friendliness*, and
merge-friendliness prefers redundant generalists. Proxy partitions like embedding
clustering are probably doing those systems a favour precisely by not being optimal
for the merge.

Three things survive the wreck. The negative itself, pre-registered end to end:
every gate number written and dated before its experiment, two deviations logged
with cause, the repair-round limit honoured, and the headline claims (recovery vs
k-means, the coupling curve, parallel fidelity) never run — because their
precondition never held. The full gate trail, harness, and raw results are in
[reboot/](reboot/).

The warm-start-at-the-answer test, cheap and brutal and now two-for-two against
structure-finding objectives. If you are designing a loss that is supposed to
*discover* something, initialise at the ground truth and check the loss wants to
stay. It costs minutes of compute, and it would probably have saved each of these
projects its illusion.

And two portable facts. Adam silently breaks additive model merging (per-parameter
normalisation walks zero-gradient parameters). And a surrogate can fail in two
distinct ways: pointing at the wrong thing, or bending the ruler that measures the
right thing — the second is quieter and survives better intentions.

## Coda

The three reasons to want factored objectives stand: parallel per-skill training, an
expressive outer loop, coordinates that track what matters. What this project
establishes is the current price of the wanting. The equality that defines a good
decomposition is exactly the thing you cannot descend, and the descendable stand-ins
tried here — including the most faithful one I could construct — either measure
their own operators or prefer redundancy. The gap between wanting structure and
writing down a loss whose argmin *is* that structure was the whole project. It is
still open — but now it has a fence around it with the dead ends labelled, which is
what negative results are for.

---

*Reproduce everything: the pre-registered research repo — theory, novelty sweep,
signed design with dated gates, harness, raw results, diagnostics — is under
[reboot/](reboot/) with its commit history intact.
[playground.ipynb](playground.ipynb) rebuilds the figures on CPU in minutes,
pre-executed so it renders without running. The deprecated proxy-attempt autopsy,
with its own data and independent revalidation, is in
[legacy/OPBF-autopsy.md](legacy/OPBF-autopsy.md).*
