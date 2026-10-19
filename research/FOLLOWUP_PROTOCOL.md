# Follow-up protocol — recorded before follow-up model evaluation

16 October 2026. This is a local prospective specification, not an external registration. The v1 results informed these questions. Keep v1 unchanged. Report all follow-up conditions, including failures.

## A. Information and thermal-tail audit

Generate independent reset states with concentration uniformly selected from the existing five strata. Compare initial temperature ranges [320,340] and [320,330] K. The latter is a narrower initial condition, not a certified safe region. Warm up each reset for 0.02 source-time units at coolant 296 K, recording concentration at 0, .01, .02. Decision time is the end of warmup.

The learner receives either current concentration; the three past/current concentration readings; or current concentration and temperature. No future information enters those features. Behavior chooses using actual decision-time temperature with preference probabilities 1/3, .6, .9; temperature bands use the reset interval and are clipped to the outer band. Predict three direct costs over the next unit of time: integrated squared error, integrated absolute error, and integrated squared error capped pointwise at 400 K². The capped objective is a sensitivity test, not a safety guarantee.

Train 512 episodes per seed (11,23,47). Use fixed quadratic ridge and 64x64 MLP, 160 epochs, train-only scaling, no test tuning. Evaluate on 1000 independent held-out resets per temperature range (fixed evaluator seed 20261016). All three action branches belong only to evaluation. Report excess cost over the **full-state per-episode oracle** for every observation mode. This is a privileged lower bound for snapshot/history, NOT the matched-information regret in v1. Also report prediction error and observed maximum temperatures. Compare modes on this common physical evaluation, without equating their available information.

Integrator max step .000125; output step .0025. Use the previously validated equations. Save configuration and source hashes before evaluation. Do not claim realistic sensor robustness from noiseless, short synthetic histories.

## B. Targeted collection feasibility, separate from A

Build a transparent, conventional decision-focused allocation heuristic, not a claimed new causal algorithm. At fixed equal outcome budgets, compare passive collection, random interventions, predictive-uncertainty allocation, and allocation based on uncertainty about the best action. The selector must receive only observed concentration/action/cost records and its own acquisition provenance. Never pass latent temperature, evaluator outcomes, or simulator parameters into selection.

Use independently drawn outcomes for each chosen concentration/action. Conditioning acquisition on concentration assumes selectable/resettable contexts; explicitly disclose this limitation for factory deployment. Charge each outcome to the ledger. Do not use policy-selected base outcomes as unbiased interventional estimates; initialization and any randomized calibration outcomes count toward the budget. Keep this experiment distinct from the v1 pooled-data question.

Before executing B, write the exact scoring rules, initialization, budgets, seeds, and held-out conditions. A baseline win is a result; do not tune the heuristic until it wins.

## Company-facing decision

Only propose a factory pilot, not proven factory benefit. Link research to fewer intervention-linked samples needed for reliable advisory decisions. No revenue, valuation, customer ROI, or fundraising outcome can be inferred from simulator performance. See COMPANY_DIRECTION_AND_EVIDENCE.md for source-grounded strategy and unknowns.
