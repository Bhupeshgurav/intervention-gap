# Independent review (technical sections)

An independent reviewer that had not seen the work produced reviewed this repository on 16 October 2026. Sections on job-application strategy are omitted here; the technical findings, spot-checks and ranked fixes are unchanged. The follow-up responses are `research/EXCHANGE_RATE_PROTOCOL.md`, `research/process/information_audit_references.py` and `research/process/run_difference_control.py`.

## Verdict: REVISE (do not send as-is)

The engineering hygiene is good: frozen configs, hashed sources, ledgers, no leakage that I could find, and negative results kept. The scientific content is thin and mostly known. The headline acquisition claim overreads 10 noisy seeds. The writing undersells the work by leading with retractions. It needs about one to two days of fixes plus one positive experiment before a founder sees it.

---

## 1. Spot-check of headline numbers (all reproduced)

| Claim | Source | Result |
|---|---|---|
| RESULTS.md v1 regret table (cell 181.69/181.69/34.17/1.45; ridge 226.09/…/60.63; MLP 29.16/37.51/12.50/1.45) and MLP per-seed 81.24/6.23/0.00 | `results/process_v1/metrics.json` (`regret_K2`, grouped by model/arm/budget/fit) | **Match** |
| FOLLOWUP acquisition table at 240 (182.03/62.70/11.74/39.88/23.37; 102.82/34.31/0.09/7.41/19.51) | `results/acquisition_v1/metrics.json` (`regret`, cost=`squared`, budget=240) | **Match** |
| FOLLOWUP info audit MLP [320,340] (45.62/36.88/7.64; 43.02/45.25/7.23; 73.52/55.64/7.44); abs 2.52/2.45/0.17; capped 36.25/36.25/1.57; narrow interval all zero, 302 K optimal for 1000/1000 | `results/information_audit_v1/metrics.json` (`excess_full_state_oracle`), `evaluator_330.npz` | **Match** |
| STATE_ESTIMATION observer table (all 10 rows) | `results/state_estimation_v1/metrics.json` | **Match** (noisy RMSE is a mean of per-seed RMSEs, which is a minor convention issue) |
| Robot 100 ms table (all 24 MLP cells); uncertainty Spearman medians and Sines-T3 ratio 0.848 | `results/real_robot_v1/metrics_by_rollout.json`, `results/robot_uncertainty_v1/metrics.json` | **Match** |

I found no numeric mismatches. There is one stale statement: `RESULTS.md:62` says "Four automated tests pass", but 12 exist.

## 2. Bugs and leakage

**I found no demonstrated code bugs.** I checked specifically:
- **Selector isolation.** `run_acquisition.py:11` `probabilities()` receives only counts, sums and sum-of-squares plus its own RNG. Only the passive arm uses `temp` (line 80), and that use is legitimate because the operator sees temperature.
- **Budget accounting.** The 30 shared initial outcomes are charged, and each 15-outcome batch is charged. The loop ends at exactly 240 outcomes. The test that verifies the ledger passes.
- **Regret.** `run_study.py:99` and `run_acquisition.py:68` compute matched-information regret against quadrature-converged conditional means. The full-state oracle path is separate (`run_study.py` `full=True`; info audit labels it explicitly).
- **Features.** Snapshot and history features exclude temperature (`run_study.py:35`, `run_information_audit.py:22`). The observer receives only the noisy concentration history (`run_state_estimation.py:53`).
- **Robot loader.** Features are taken only at the anchor (`run_baselines.py:31`). Test splits are loaded only after fitting (line 88). The official splits are used and validation-based epoch selection is train-side. I see no split leakage. Overlapping stride-50 windows are correlated, and the docs acknowledge this.

**Demonstrated interpretation errors (from the saved data):**

1. **The acquisition ranking is not supported per seed.** `FOLLOWUP_RESULTS.md` §1 and the memo say evenly distributed sampling "beats our uncertainty and decision-focused heuristics in both evaluated intervals." `results/acquisition_v1/metrics.json` at budget 240 shows otherwise:
   - On the held-out [322,338] interval, stratified regret is 0.12 in 7 of 10 seeds. Uncertainty is exactly 0 in 9 of 10 seeds and 73.97 in one. Decision is 0 or 0.12 in 9 of 10 seeds and 194.75 in one.
   - The heuristics therefore beat stratified on *more seeds*: uncertainty wins 6 seeds and loses 1; decision wins 4 and loses 2. The mean ranking comes entirely from one catastrophic seed per heuristic.
   - On [320,340], stratified vs decision is 3 wins / 4 ties / 3 losses (paired bootstrap 95% CI on the mean difference: [-31, +6] K²).
   - The "random beats stratified at 60/120" statement is also within noise (CIs span 0).

   The defensible statement is narrower: stratified has no catastrophic seeds, while the variance-driven heuristics occasionally fail badly. That is a tail-risk result, and it should be reported with P(regret > τ) and worst-seed numbers, not with means.
2. **The information-audit table is missing its matched-information reference.** I computed it from `evaluator_340.npz`. The best constant action per initial-concentration stratum has 37.7 K² excess over the full-state oracle, and "always 296 K" has 51.2. The q=0.9 snapshot MLP (73.5) is therefore worse than a constant rule. The q=1/3 history MLP (36.9) already beats the concentration-only optimum. Without these rows the table cannot separate confounding cost from information loss.
3. **The matched-parameter observer result is close to tautological.** With exact equations, noiseless data and a 2-state identifiable ODE, the model-predictive choice using the true simulator *must* match the oracle. `STATE_ESTIMATION_RESULTS.md` treats this as a finding ("the history contains the missing information"). It is a sanity check. The informative rows are the noise and k0-mismatch rows.

**Speculative or design concerns (not errors):**
- **No off-policy-evaluation baseline.** I ran IPW and self-normalized IPW with the operator's logged propensities on the v1 ledgers. They do **not** fix v1: SNIPW selects the same actions as naive regression in all 3 seeds, and IPW is worse in seed 47. The reason is that the thermal-runaway tail (302 K at high T, mean cost up to 1577 K²) sits exactly in the 5%-propensity region, which leaves about 1 sample per cell. This is a sharper and more interesting finding than "confounding exists", because the binding problem is **positivity and tail variance, not just bias**. The applicant should add it.
- **Physically inverted observation model.** In a real CSTR, temperature is the cheap, reliable online measurement, and concentration needs a lagged analyzer. Hiding T and observing Ca is the opposite of plant reality. A process-savvy founder will notice.
- **The heuristics are strawmen of best-arm identification.** The `p(1-p)+.01` weighting still lets the 10⁶-variance runaway cell dominate. Knowledge gradient or top-two Thompson sampling are the fair "targeted" baselines.
- **Too few seeds.** v1 uses 3 seeds, and regret takes a few discrete values (0, 4.3, 35.6, 293, 557). Means of 3 such values are fragile.
- **Robot section is disconnected.** It is forecasting-only, and the docs admit it cannot test decisions. Ridge+desired beats every MLP on Sines T1/T3, which is not mentioned.

## 3. Prior work (verified-to-my-knowledge; uncertain ones flagged)

- de Haan, Jayaraman, Levine, *Causal Confusion in Imitation Learning*, NeurIPS 2019 (confident).
- Gupta, Rudner, McAllister, Gaidon, Gal, *Can Active Sampling Reduce Causal Confusion in Offline RL?*, CLeaR 2023, PMLR v213 (confident; cited by the applicant).
- Bareinboim, Forney, Pearl, *Bandits with Unobserved Confounders*, NeurIPS 2015; Lattimore, Lattimore, Reid, *Causal Bandits*, NeurIPS 2016 (confident).
- OPE: Precup, Sutton, Singh 2000; Dudík, Langford, Li, *Doubly Robust Policy Evaluation*, ICML 2011 (confident). Confounded OPE: Kallus & Zhou, *Confounding-Robust Policy Improvement*, NeurIPS 2018 (confident); Namkoong et al. 2020 and Tennenholtz, Shalit, Mannor (AAAI 2020) on OPE under unobserved confounding / partial observability (fairly confident).
- Best-arm identification: Audibert & Bubeck, COLT 2010 (uniform allocation is a strong baseline); Russo, COLT 2016 (top-two Thompson); Frazier, Powell, Dayanik 2008, knowledge gradient (confident).
- Prediction-vs-decision: Lambert et al., *Objective Mismatch in Model-based RL*, L4DC 2020; Elmachtoub & Grigas, *Smart "Predict, then Optimize"*, Mgmt Sci 2022 (confident).
- State estimation: moving-horizon estimation (Rao, Rawlings, Mayne, IEEE TAC 2003; fairly confident) and EKF observers for CSTRs (textbook).
- Robot data: Agudelo-España et al., ICRA 2020 transferability dataset (confident).
- Lu, Schölkopf, Hernández-Lobato 2018, *Deconfounding RL* (**uncertain** on details).

**Originality: none established.** Each finding is a textbook instance: omitted-variable bias, uniform allocation competitive at small budgets, and observability of a known ODE. The contribution is a clean, auditable *engineering harness*. It is not publishable as-is. The IPW/positivity-tail observation above, made systematic, would be the closest thing to a workshop-grade note.

## 7. Ranked fixes

**Must-fix**
1. Rewrite the acquisition conclusion using per-seed, tail and paired-CI reporting. Raise to ≥ 50 seeds (cheap). Drop "beats in both intervals".
2. Add the matched-information reference rows (concentration-stratum oracle, constant-296 rule) to the info-audit table.
3. Add IPW/SNIPW/DR baselines with logged propensities to v1, and report the positivity/tail failure.
4. Remove internal and planning files from the shared bundle. Fix the stale "Four tests" line. Pin an install that works.
5. Rewrite the memo to lead with the positive headline in §6. Keep disclaimers to one boundary section.

**High-value**
6. Run the §6 transfer experiment.
7. Replace the heuristics with top-two Thompson sampling or knowledge gradient, plus a fixed-confidence stopping variant.
8. Add a history-differences feature control, so that "generic model failure" is a tested claim.
9. Flip to realistic observability (T measured, Ca hidden or lagged).

**Optional**
10. Move the robot study to an appendix, and note that ridge+desired wins on Sines T1/T3.
11. Expand the one-liner scripts into readable code.
12. Add a sequential/closed-loop variant.
