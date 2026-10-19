# Acquisition feasibility protocol — 16 October 2026

Recorded before running this comparison. No claim of a new algorithm. Prior work includes [Russo 2016](https://proceedings.mlr.press/v49/russo16.html) on best-arm identification and [Gupta et al. 2023](https://proceedings.mlr.press/v213/gupta23a.html) on active sampling under causal confusion. Our heuristic is not a reproduction of either method or entitled to their theoretical guarantees.

Use the original five concentration strata and three actions; no warmup. Development-domain temperature interval [320,340]; one separately reported, previously unused transfer interval [322,338]. These are simulator distribution changes, not cross-plant transfer. Ten independent seeds 101–110. Costs are squared temperature error (primary), absolute error, and pointwise capped squared error (secondary). Primary allocation scores depend only on squared cost.

All five arms share 30 initial interventional outcomes: two per concentration/action cell, each temperature independently uniform. These 30 count toward total budgets 30,60,120,240. No unpaid base-data initialization. Additional batches contain 15 outcomes; choose the entire batch before seeing its outcomes. Every selected cell produces a fresh hidden-temperature draw. All methods may select the concentration context: this is a resettable simulator assumption, not factory feasibility.

Arms:
- passive: random concentration, operator's temperature-dependent command (q=.9, thirds of interval); fit cell averages to all available records, including the common initial intervention set. This is explicitly a naive mixture of intervention and observational data.
- random: independent uniform concentration/action pairs.
- stratified: exactly one sample per cell per additional batch, a strong coverage baseline.
- uncertainty: sample cells proportional to estimated reduction of mean-cost variance, with 20% uniform exploration.
- decision: 128 independent Gaussian draws of cell means using empirical standard errors; estimate each action's probability of minimizing cost within its concentration. Weight variance-reduction scores by p(1-p)+.01; add the same 20% uniform exploration. This is an uncalibrated heuristic, not a causal-confidence guarantee.

Standard errors use cell-centered sums of squares and two pseudo-observations at pooled within-cell variance: variance = (cell_SSE + 2*pooled_variance)/(n+1), SE² = variance/n. Clamp pooled variance to at least 1e-8 only for numerical positivity. Predicted costs are raw cell sample means; do not tune shrinkage or mix observational data into intervention-only arms. Every cell is initialized, so no empty-cell default is needed.

The selector receives only count, sum, sum-of-squares of acquired costs and its private sampling RNG. It does not receive temperatures, simulator equations, evaluator means, or alternative outcomes. The simulator/evaluator retains separate privileged data. Save selected cells, probabilities, observed outcomes, and hidden state in an audit ledger; hidden state is never passed to selection.

Evaluate against expected action costs conditional on concentration via 488 and 976 midpoint temperatures, with max differences <1 K² squared, <.1 K absolute, and <.5 K² capped; fail evaluation if not met. All branch outcomes belong only to this evaluator. Score matched-information regret at every budget and retain per-seed values. A comparison may not be described as an improvement if it loses to stratified or random acquisition on the held-out interval. Do not tune after seeing results. Secondary objectives are evaluated using their own empirical cost-head recommendations, although allocation was optimized for squared cost only.
