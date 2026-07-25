# Independent revalidation of the OPBF closure (2026-07-22)

Before anything from the OPBF project's own closure report was repeated publicly, its
load-bearing claims were re-verified with fresh eyes against the actual code and raw
data, about eight weeks after the project closed. The reason for the ceremony: the
project leaned heavily on AI agents to execute, and a self-report from that kind of
run can inherit the run's own blind spots — unfair baselines, premature closure,
miscalibrated hyperparameters. So nothing public is cited on the project's word
alone; every load-bearing number below was either reproduced from the committed code
or corrected.

**Verdict: REVISED — the close stands and is strengthened; specific claims corrected.**

## What was checked, and what was found

### 1. E7 recovery — was k-means given the same signal? (FAIR, confirmed by re-run)

Code-level check: `kmeans_coupling_ami` clusters the bit-identical `trained_coupling`
matrix (same seed, same 512-sample batch, same 10-warmup/8-probe estimation window)
that the learned factoriser trains on. Neither side gets a longer or cleaner estimate.
The k-means baseline was added *against* the project's first-pass positive headline by
its own adversarial verification — the direction that strengthens the negative.

Re-run of the project's own probe (12 seeds, [log](revalidation/probe_e7_hard_rerun.txt)):

| | reported | reproduced |
|---|---|---|
| hard-OPBF AMI | 0.732 ± 0.056 | **0.736 ± 0.053** |
| k-means AMI | 0.983 ± 0.033 | **0.983 ± 0.033** |
| paired p | 5e-4 | **5e-4** |

Noted asymmetries, all mild and all *against* OPBF's excuse budget: the factoriser ran
at one fixed hyperparameter setting (a documented 5000-step probe reached only 0.88),
single init vs k-means' best-of-10 restarts, and the Gumbel variant used fixed
temperature with no annealing. None plausibly closes a 0.73–0.88 vs 0.98 gap; annealing
was tested anyway (see §3).

### 2. E6 transfer — was the winning baseline fair? (FAIR, confirmed by re-run)

The "Successor Features" baseline is, precisely, a generic frozen MLP encoder — same
pretraining data, steps, learning rate; **identical parameter count (1,092 vs 1,092,
measured)**; identical feature dimension; identical closed-form ridge learner downstream
for every frozen-feature condition. Seeds 0–5 of the committed harness were re-run and
reproduced the saved JSON bit-for-bit; the doc's key robustness claim — that a purely
*linear* encoder of the same budget also beats OPBF — was independently reproduced.
The headline p-value (4.8e-5) was recomputed directly from the raw per-seed arrays.

One naming note for honesty: no successor-measure machinery is involved (in a one-step
bandit, SF correctly degenerates to reward-feature regression); the blog calls it a
"generic frozen encoder". One pre-registration deviation found: the committed gate uses
the *mean* where the plan said *median* for the vs-scratch criterion (with the median it
would have passed that sub-criterion); this affects only the vs-scratch nuance — which
the project's own doc already reported honestly in prose — not the decisive vs-encoder
result.

### 3. The missed-exit hunt — did the project stop too early? (No; two untried exits tested, both fail)

Escape routes a stronger researcher might have tried, checked against the repo and, where
untried and cheap, actually run ([probe script](revalidation/probe_missed_exits.py)):

- **Learned hard assignment** — tried by the project (the "(c) falsification nail");
  code matches the report; reproduces (§1).
- **Annealed soft→hard temperature schedule** — *never actually tried* (only fixed
  temperatures were swept, on a different task; the "anneal hypothesis refuted" wording
  in the project overstated). Run now: annealed softmax 1.0→0.05 gives AMI 0.816 (vs
  0.800 fixed); annealed Gumbel 2.0→0.1 gives 0.732 (vs 0.698 fixed). K-means: 0.966.
  No rescue.
- **Warm-start from the k-means solution** — never tried; the classic
  "start-at-the-baseline" move. Run now: distilling the assignment to the k-means
  partition reaches parity (AMI 0.966); **fine-tuning on the project's own
  coupling+sparsity objective then degrades it to 0.838**, walking off perfect
  partitions on 2/6 seeds — and the loss values show why: on those seeds the objective
  assigns a *better* score to the wrong soft solution than to the true partition.
  **The training objective's optimum is not the true partition.** This corrects the
  closure report's mechanism story ("straight-through gradients get stuck", tested only
  cold-start) to something stronger: objective misalignment. "Fix the factoriser" would
  require fixing the objective, not the optimiser.
- **A different clusterer (spectral/agglomerative)** — claimed tested at ~0.98 with no
  committed code; immaterial (all reasonable clusterers saturate the signal).
- **K-means-as-factoriser + learned mixer through the downstream payoffs** — never run
  as a condition, but bracketed by committed oracles: in transfer the perfect-partition
  oracle (1.000) statistically ties the generic encoder (0.997), so there is no headroom
  for any factoriser to *beat* the baseline; in interpretability a hard partition passes
  the gate but contains no learned component — it concedes the thesis rather than
  rescuing it.
- **The co-adaptation regime** (grouping must co-adapt with a downstream policy — the
  project's own twice-named "only place differentiable grouping could win") — **never
  built or tested; closed by argument, not experiment.** This is the one genuinely open
  exit. Days of bespoke-environment work, real rigging risk, tempered promise given the
  oracle ceilings above. Named in the blog as not-done.

### 4. The (b) conflict-auditing arc — does the three-round story reproduce? (Yes, to 3–4 decimals)

The full arc re-ran ([script](revalidation/verify_e11_cells.py),
[log](revalidation/e11_tpcgrad_repro.txt)): the round-2 out-of-sample win (tpcgrad beats
PCGrad *and* GradVac, frozen HP, disjoint holdout seeds, p=5e-4 reproduced), the round-3
dissolution (PCGrad+momentum 0.0417 vs tpcgrad 0.0565, matching the reported
0.042/0.057; the momentum control beating PCGrad, p=0.0034), and the noise cell
(win loses significance, p=0.0640, matching the reported 0.064 to three decimals).

### 5. Raw-data spot-checks (5 match; 1 mismatch; 1 provenance gap)

- **Match:** E7 (0.841/0.980 in the saved JSON), E6 (0.888/0.997; p recomputed from raw
  arrays = 4.77e-5), E9 (0.465/0.711 vs random 0.359/0.444), E10 (fast-drift β=0 −10.90
  beats β=0.9 −10.14; GradVac tie p=0.23). Seed counts all match claimed.
- **Mismatch (corrected in the blog):** the closure report and theory doc cite the E1
  smoke reconstruction error as "NMAE ≈ 0.003"; the only raw file says **0.162**, and a
  live re-run of the exact config reproduces ~0.159. The AMI ≈ 0 half of the claim
  holds. The "0.003" figure appears in no raw data and is not cited here.
- **Provenance gap (now closed):** the two closure headlines — the (c) hard-assignment
  nail and the (b) tpcgrad arc — were never persisted to disk by their scripts
  (stdout only). Both reproduce from the committed code (§1, §4) and the re-run logs
  are archived in [revalidation/](revalidation/). A minor aside in the closure report
  ("worse than even the soft version, 0.859") matches neither the saved 10-seed soft
  result (0.841) nor the 12-seed re-run (0.883); the aside is dropped rather than cited.

## Net

Every load-bearing negative reproduces from committed code, several to three decimals,
roughly eight weeks and one library-stack drift later. The two cheap rescues the project
missed were tried and failed — one of them (warm-start) turning up a *deeper* root cause
than the project claimed. One decorative number was wrong and is corrected; two headline
numbers lacked archived data and now have it. The closure verdict — every learned OPBF
mechanism loses to a fair, usually trivial, baseline — **stands, slightly
over-determined**.
