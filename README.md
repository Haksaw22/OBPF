# The Price of Coupling — learning a decomposition by closing the gap

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haksaw22/OBPF/blob/main/playground.ipynb)

If a training loss $L = \sum_t Q_t$ is secretly a sum of skills, the exact statement
of "a good decomposition" is the separability gap

$$\min_\theta \sum_t Q_t \;-\; \sum_i \min_\theta D_i \;\ge\; 0, \qquad \sum_i D_i = \sum_t Q_t,$$

zero exactly when the components share a minimiser. This project **made the gap the
training objective of the decomposition** — operationally: train one copy per
component, merge, score against joint training — and tested it under pre-registered
kill criteria.

**Result: a clean negative with a diagnosable mechanism.** The gate designed to
catch a lying objective (initialise at the *true* decomposition; check the objective
wants to stay) killed it: the truth is not a local optimum on 8/12 seeds, and under
the secondary merge operator the ordering fully inverts. The pre-authorized repair round
showed the deeper fact — the apparent signal had been merge-operator artifact all
along, and the artifact-free objective *prefers redundant generalist copies over any
decomposition*. Full verdict: [reboot/VERDICT-STAGE1.md](reboot/VERDICT-STAGE1.md).
The sharpened open question: *what descendable objective has the true decomposition
as its minimum?*

## Map

- **[BLOG.md](BLOG.md) — the article** (abstract up top; formulation, gates, both
  deaths, mechanisms, what survives). Start here.
- **[reboot/](reboot/) — the research repo, vendored with its git history intact:**
  [THEORY.md](reboot/THEORY.md) (formulation, readings, degeneracies),
  [NOVELTY.md](reboot/NOVELTY.md) (nine-literature prior-art sweep — nearest
  relatives: Grimm & Singh 2019, IGM/QTRAN, c-BTM/MERIT; the gap is FedAvg's
  heterogeneity constant $\Gamma$, assumed-never-descended),
  [DESIGN.md](reboot/DESIGN.md) (the signed pre-registration — gate
  numbers dated in git before any run; deviations logged),
  [VERDICT-STAGE1.md](reboot/VERDICT-STAGE1.md) (final verdict),
  [src/](reboot/src/) + [scripts/](reboot/scripts/) (harness),
  [results/](reboot/results/) (raw JSON for every gate).
- **[playground.ipynb](playground.ipynb)** — rebuilds the figures from raw data;
  minutes on CPU; pre-executed so it renders without running.
- **[figures/](figures/)** — including [gap_concept.png](figures/gap_concept.png)
  (the machine), [z2_walkaway.png](figures/z2_walkaway.png) (gate zero) and
  [r1_nowindow.png](figures/r1_nowindow.png) (the repair round).

## Reproduce

Figures: `jupyter execute playground.ipynb` (numpy + matplotlib). The gates
end-to-end: `python reboot/scripts/run_z1.py`, `run_z2.py`, `diag_z2.py`,
`run_r1_calib.py` — the entire empirical arc cost ~3 CPU-hours.

## Process notes

Every gate number was fixed and committed before its experiment ran (first commit in
the [reboot/](reboot/) history). Two deviations, both logged with cause before use
(inner optimiser Adam→SGD — Adam's per-parameter normalisation breaks additive
merges; ES best-seen elitism). The pre-committed one-repair-round limit was
honoured. The headline claims (recovery vs k-means, coupling curve, parallel
fidelity) never ran because their precondition — gate zero — never held.

## Legacy

An earlier attempt approached the same wish through a proxy objective (gradient
affinity) and died differently — its objective walks away from the truth for
representational reasons. Its autopsy is **deprecated** but kept for the record,
with raw data and an independent revalidation of every number:
[legacy/OPBF-autopsy.md](legacy/OPBF-autopsy.md), [data/](data/),
[revalidation/](revalidation/), [REVALIDATION.md](REVALIDATION.md). Its one lasting
contribution is this project's gate zero.

Code and text: [MIT License](LICENSE).
