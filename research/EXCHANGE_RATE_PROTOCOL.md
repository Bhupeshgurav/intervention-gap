# Logs-versus-experiments protocol (v2) — 16 October 2026

Written before running `research/process/run_exchange_rate.py`. It responds to an independent review (`review/INDEPENDENT_REVIEW.md`) that found (a) no off-policy-evaluation baseline, (b) a ten-seed acquisition ranking driven by single catastrophic seeds, and (c) weak adaptive comparators. All v1 and follow-up results stay unchanged. Everything here is a simulator study and is exploratory in the sense that the task was chosen after earlier results were seen; seeds, estimators, budgets and summaries below are fixed before this run.

## Question

How many routine operator-logged episodes, analysed with standard off-policy estimators that are even given the operator's **true** action probabilities, does it take to choose actions as well as a small number of deliberate, randomized interventions? And among intervention-collection rules, which avoid catastrophic seeds?

## Shared setting

Same PC-Gym CSTR adapter, step 0.000125, horizon 1 source time unit, five concentration strata (0.72–0.96), commands 290/296/302 K, hidden initial temperature uniform on [320,340] K, no warmup. Primary cost: integrated squared temperature error (K²). Secondary: absolute error and pointwise capped squared error. Reference: per-concentration expected cost averaged over 976 midpoint temperatures, checked against 488 (fail if squared means move > 1 K²). Metric: matched-information regret, averaged over the five strata.

## Part A — logs versus experiments

- Seeds 1001–1100 (100 seeds). Operator preference q ∈ {0.9 (primary), 0.6}; preferred command by temperature thirds, as in v1.
- Nested log sizes N ∈ {512, 2048, 8192}: the first N of one 8192-episode log per seed.
- Estimators per concentration × command, then choose the lowest estimated cost per concentration:
  - naive cell mean;
  - IPW (Horvitz–Thompson) with true logged propensities, normalized by the stratum count;
  - self-normalized IPW (Hájek) with true propensities;
  - doubly robust with the naive cell mean as the outcome model.
- A cell with no observations gets +∞ estimated cost (never chosen); count such events.
- True propensities are unavailable in a factory; supplying them makes these estimators optimistic, not pessimistic.
- Comparator: randomized intervention budgets K ∈ {15, 30, 60, 120, 240} under the "stratified" rule (equal counts per cell, K/15 each) and the "random" rule (uniform cells; empty cells get +∞), using the same seeds.

## Part B — acquisition, 100 seeds

Repeat the v1 acquisition protocol (`research/ACQUISITION_PROTOCOL.md`) unchanged except: seeds 2001–2100; only the primary interval [320,340]; add **top-two Thompson sampling** (Russo 2016, β = 0.5) as a fair adaptive comparator. For each batch slot, contexts are assigned round-robin (three slots per concentration per 15-outcome batch); within a context draw one Gaussian posterior sample per command from N(mean, SE²) using the protocol's existing SE formula; the leader is the argmin; with probability 0.5 take it, otherwise redraw up to 100 times for a different argmin (fallback: uniform among the other commands). The posterior is frozen within each batch. Selectors still receive only counts, sums and sums of squares.

## Reporting (both parts)

Per method and budget: mean, median, 90th percentile, fraction of seeds with regret > 5 K², fraction exactly zero. Pairwise comparisons against stratified use the per-seed paired difference with a 10,000-resample percentile bootstrap 95% CI (seed 0). Win/tie/loss counts are reported. No result is called an improvement if its CI includes zero. All per-seed values are saved.

## Information-audit addendum

Add simple matched-information reference rows to the existing information audit (wider interval) from the saved evaluator: the single best fixed command, and the best command per initial-concentration stratum (reconstructed from the frozen evaluator seed). These are reference rules, not learned models.
