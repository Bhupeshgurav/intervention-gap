# Intervention Gap: CSTR experiment protocol

Design recorded 16 October 2026. This is a prospective specification for the first empirical experiment, written after the analytic counterexample and before the final training/evaluation runs. It is not an externally registered study. Numerical development checks may change the proposed defaults below; the final run must save its complete configuration, a SHA-256 hash, and the development changes **before** final evaluation. A change made after inspecting final results creates a new exploratory run, not a revision of the original test.

## Question and scope

Can an action-conditioned predictor improve prediction on operator-selected records while recommending worse commands than a predictor trained with a small amount of randomized command data? At the same additional interaction budget, compare collecting more records under the source operator with randomizing admissible commands.

The physical equations come from PC-Gym's continuous stirred-tank reactor (CSTR). The observation mask, reset distribution, operator policy, objectives, and evaluation protocol are our experimental constructions. This is a controlled simulator study of a possible failure mechanism. It does not test Celesnity's models, validate a factory process, establish a new causality theorem, or show that a sufficiently informative history cannot resolve the hidden state.

The first study uses a **snapshot learner**, not a recurrent learner with an arbitrarily shortened history. Withholding reactor temperature deliberately creates omitted information. A future experiment should test whether ordinary telemetry histories remove the gap.

## Physical substrate and development checks

Use the CSTR equations and default constants from PC-Gym commit `fbfb36bf84b494cd157cff88292e1fd1b864cee5`. State is reactor concentration `Ca` and reactor temperature `T`; input is jacket temperature `Tc`; feed concentration `Caf` and feed temperature `Ti` remain fixed in the primary experiment. Retain the downloaded upstream source and MIT license. A NumPy/RK4 adapter is acceptable if its derivatives and numerical trajectories are checked against the source equations and step refinement.

Primary sources:

- [PC-Gym CSTR documentation](https://maximilianb2.github.io/pc-gym/env/cstr/) describes the two states, jacket-temperature input, and a 290–302 K action interval.
- [Pinned model source](https://github.com/MaximilianB2/pc-gym/blob/fbfb36bf84b494cd157cff88292e1fd1b864cee5/src/pcgym/model_classes.py) supplies the actual equations and constants.
- [PC-Gym paper](https://arxiv.org/abs/2410.22093) describes the benchmark platform.

Temperature units are kelvin. The inherited source/reference material contains inconsistent labels for concentration and time; until independently reconciled, use **source concentration units** and **source time units**. Do not turn a numerical horizon of 1 into a claim of one physical second or minute. Temperature-tracking costs below use a time average and have units K².

Proposed defaults, subject only to the documented development pilot:

| Item | Proposed value |
| --- | --- |
| Observed concentration strata | `0.72, 0.78, 0.84, 0.90, 0.96` |
| Hidden initial temperature | Independent uniform draw on `[320, 340]` K |
| Command alternatives | Constant jacket temperatures `290, 296, 302` K |
| Evaluation horizon | `H = 1` source time unit |
| Integration | RK4, internal step `0.01`; compare with `0.005` |
| Recorded target samples | Ten equally spaced future states, excluding the initial state |
| Temperature target | `330` K, fixed and known |
| Source behavior preference probability | `q = 0.90` |
| Base passive sample count | `512` independent command episodes |
| Additional outcome budgets | `0, 16, 32, 64, 128` episodes |
| Independent acquisition/training seeds | At least 3; proposed `11, 23, 47` |
| Test latent-state quadrature | 61 uniformly spaced midpoint temperatures per observed concentration stratum |

The pilot may check finite integration, derivative agreement, trajectory ranges, command effects exceeding solver error, and runtime. It may not search for a temperature range, target, or horizon maximizing the preferred method's advantage. Report any changed defaults and their numerical/physical reason. If different commands are essentially indistinguishable at the selected horizon, record this before testing the hypothesis and choose a physically meaningful development horizon transparently.

Do not silently clip internal reactor states to make a run stable. Save numerical failures and the initial conditions that caused them. No fitted model is needed for the physical pilot. The primary evaluation must use a region where all candidates produce valid numerical trajectories; rejection of regions is disclosed, not described as an industrial safety certificate.

## Information available at the decision

At `t0`, the partial-observation learner receives only `Ca0`, the known target/horizon if supplied, and the candidate requested command for action-conditioned models. The full-state control additionally receives `T0`. The operator that collects source records sees both `Ca0` and `T0`.

The initial temperature is sampled independently of the observed concentration. Thus a command selected using `T0` can reveal information absent from the partial learner's snapshot. This is a deliberately induced omitted-variable regime. It is not proof of nonidentifiability for every conceivable model with known physics or richer observations.

Neither model receives initial-temperature bands, policy sampling probabilities, policy identity, episode IDs, seeds, evaluator quadrature indices, future acknowledgement, future applied command, future sensor values, or alternative-branch outcomes as inference features. Normalization is fitted on allowed training records only. Acquisition provenance can determine which training subset a method uses; it cannot become a proxy input revealing the hidden temperature.

Training targets may include the future trajectory and realized episode cost. Outcome supervision is identical for observation-only and action-conditioned models. Observation-only means **no candidate-action input**, not absence of outcome supervision.

## Behavior policies and isolated shifts

Divide the hidden-temperature interval into three equal-width bands. The source operator prefers `302 K` for the lowest band, `296 K` for the middle band, and `290 K` for the highest band. It selects its preferred command with probability `q`; each other command has probability `(1-q)/2`. Every command therefore has positive probability throughout the hidden-temperature range.

The policy-reversal test swaps the low/high band preferences while retaining the middle preference. The underlying initial-state distribution, plant parameters, execution, command support, and target remain fixed. With the symmetric latent distribution, source and reversed policies have the same command marginals. Verify and report this property from the weighted evaluation records; finite sampled training counts need not be exactly equal.

The randomized policy selects each candidate with probability `1/3`, independently of the hidden temperature. Record the actual sampling probability and policy version in the acquisition ledger, separately from model features.

**Immediate, exact execution is mandatory in the primary experiment.** Requested and applied jacket temperature coincide. This isolates operator selection. Delay, clipping, or rejection is an optional later experiment with a separate result table; it must not be mixed into the main comparison. A physical-parameter shift is likewise separate and is not required for the first deliverable.

Source-policy and reversed-policy forecasting evaluate different observational conditionals. A change in forecasting error after reversal is not alone evidence of bad physics: the association between action and hidden state has changed. The command intervention test below determines decision usefulness.

## Acquisition contract and splits

Each command episode starts from a fresh reset, holds exactly one command through the horizon, and returns one observed outcome/trajectory. It counts as **one acquired outcome**, regardless of how many future samples it contains. This is a finite-horizon command-selection experiment, not a closed-loop control experiment.

For each seed, every method starts with the identical base passive dataset. The two acquisition arms use the same added initial-state sequence and differ only in command assignment:

1. **More passive:** use the source behavior policy.
2. **Randomized:** select an admissible command uniformly.

Use nested prefixes for the five budgets. The base pool and each additional pool are independent. Do not give the randomized arm both/all outcomes from a state while charging one interaction. Paired counterfactual simulation is reserved for evaluation and never enters training for free.

Store a separate development set for physical checks and a training-only validation set if tuning is used. Final test states, quadrature, labels, and alternative branches are evaluator-only. Fit no parameters, preprocessing, stopping criteria, feature choices, or data weights on final outcomes.

The concentration strata intentionally recur across partitions: this isolates causal selection from extrapolation in the observed input. Sample continuous hidden temperatures independently in training, and use reserved quadrature nodes for final evaluation, avoiding copied full trajectories. All branches of any given reset stay within one partition. Describe this as held-out reset episodes under known concentration strata, **not unseen machines, sites, or observed-state extrapolation**.

With deterministic equations the evaluator's branches need no stochastic disturbance replicates. If noise is introduced later, use a separately keyed disturbance tape shared across candidate branches; merely resetting a random number generator is insufficient when actions change draw order. Never perturb the primary plant or add noise to rescue a preferred result.

## Targets and action selection

For each candidate command `a` and initial state, compute the finite-horizon trajectory and the time-averaged squared tracking cost:

`J(a) = (1/H) integral_0^H (T_a(t) - 330 K)^2 dt`.

Use a documented quadrature over the integration path, not only the final temperature. `J` has units K². Preserve final and trajectory temperatures in K, and concentration in the explicitly labelled source units. Report RMS tracking error `sqrt(mean J)` in K as an accessible companion statistic.

The model predicts expected cost **directly**, in addition to any trajectory heads. Choose the candidate with the lowest predicted expected cost. Do not square the predicted mean temperature and call it expected quadratic cost: hidden-state uncertainty makes `E[(T-target)^2]` different from `(E[T]-target)^2`. An alternative predicting the required second moments must be declared before evaluation.

The observation-only model's predictions are identical for all commands. Its decision rule must therefore be fixed explicitly, for example the middle command `296 K`; it cannot use hidden-state tie breaking. Also report a uniform-random command baseline and the best fixed command selected using training/development data only.

## Primary metric: regret at matched information

For each observed concentration stratum `c`, compute `mu(c,a) = E[J(a) | Ca0=c]` by averaging the simulator's candidate branches over the fixed hidden-temperature distribution. Define `a_star(c) = argmin_a mu(c,a)`. This reference sees exactly the learner's observation and knows the distribution/plant; it cannot choose a different command after inspecting the hidden temperature.

For a deterministic learned rule `a_hat(c)`, raw action regret is:

`R = mean_c [mu(c,a_hat(c)) - min_a mu(c,a)]`, in K².

For a stochastic rule, take its expectation over commands. The primary plotted quantity is `R / R_uniform`, where `R_uniform` is the regret of a uniform-random command rule under the same evaluator. The matched-information reference has normalized regret 0 and uniform random has 1. Values above 1 are possible. The denominator is fixed across methods and budgets. If it is near zero relative to numerical uncertainty, report only raw regret and declare normalization uninformative; do not select a denominator after seeing model results.

Also report a **privileged per-state oracle** as a diagnostic only. Its use of `T0` can yield a lower cost than any snapshot policy. The difference between the matched-information and privileged reference quantifies the price of missing information; do not count that irreducible difference as model failure.

Check test quadrature adequacy by a predeclared doubled-resolution numerical check, without using model rankings to choose resolution. If oracle actions or the normalization denominator change materially, refine the evaluator uniformly and record the change.

## Models and matched training rules

Required model families are a compact PyTorch MLP and at least one strong simple baseline. A practical neural default is two 32-unit hidden layers with a fixed activation, shared trajectory/cost outputs, and fewer than 10,000 parameters. A ridge model with quadratic/interacting input features and a stratum/action empirical-mean model are particularly useful here. ExtraTrees is optional. No architecture search or large-model training is necessary.

Use the same episode records, inference information, targets, scaling rules, model-selection budget, and optimizer/epoch rules within each model-family comparison. Save actual hyperparameters and parameter counts in the final manifest. Scale training targets using training data so a numerically large cost head cannot accidentally drown out the trajectory targets. Action selection always uses the unscaled expected-cost prediction.

Freeze these **two distinct fitting procedures** before final runs:

1. **Pooled refit:** refit on the common base plus the additional records, with equal per-episode weights. This tests whether naive addition of a small randomized set is diluted by the source distribution.
2. **New-data-only recalibration:** fit the final outcome/cost predictor on only the additional records, with the same family and rule in both acquisition arms. If a representation is pretrained on the common base, freeze it identically for both arms and state this explicitly. At budget zero, use the common base fit. Report the effective fitting count and do not label recalibration as using all records with equal weights.

The intervention arm does not get a better optimizer, more epochs, test-informed weights, or a special causal estimator unavailable to its passive comparator. New-data-only fitting can have high variance at small budgets; that is part of the result. Sparse cell-mean fallback/shrinkage, if used, must be specified identically in both arms before evaluation.

If optional decision-disagreement acquisition is attempted, it follows completion of the passive/randomized comparison, pays for all outcomes, and is scored against uniform random at identical budgets. Ensemble agreement is not a causal certificate. It is acceptable and informative to omit this extension today.

## Required controls and diagnostics

1. **Observation-only versus action-conditioned:** compare source-policy forecast error using matched model families and identical supervised targets. Quantify the information gained from the command.
2. **Full-state control:** reveal `T0` to both simple and neural models. A failure persisting equally strongly here may indicate estimation, optimization, or objective errors rather than omitted-state confounding.
3. **Randomized-source control:** replace source behavior with `q=1/3`. In this control, the two acquisition arms have the same population distribution. Any stable systematic acquisition advantage warrants an implementation audit.
4. **Policy-reversal forecast:** evaluate weighted errors with the source and reversed command assignments on the same test physical branches. Preserve state and command marginals; report the association that changed.
5. **Action-effect diagnostic:** compare predicted command contrasts to simulator contrasts **averaged at the same observed concentration**. Do not compare a partial learner's expected effect to an individual hidden-state effect as if both were the same estimand.
6. **Solver refinement:** confirm outcome contrasts and reference decisions are not integration artifacts.
7. **Feature audit:** assert that forbidden fields cannot enter the feature tensor. Save feature names and one traceable example record.
8. **Simple baselines:** report uniform random, fixed-middle command, best training-selected fixed command, and the conventional learned model. A neural failure is not by itself a causal discovery.

Action permutation is only a supplementary negative control. Shuffling actions destroys both useful action-response information and action-as-state information, so degradation after shuffling does not distinguish causal understanding from the shortcut.

## Reporting, uncertainty, and claim gates

Save every model prediction, selected command, raw cost, reference cost, acquisition assignment, training seed, fit rule, and configuration hash. Use identical test records across all comparisons. Report source/reversed factual errors, direct expected-cost error, raw/normalized regret, and budget curves separately rather than merging them into one aggregate score.

Show all independent seeds and paired acquisition-arm differences. With three seeds, report the dispersion and individual outcomes; do not present a narrow bootstrap interval over correlated trajectory samples as certainty. Deterministic quadrature reduces evaluator Monte Carlo error but does not remove finite-training-data uncertainty. If more seeds fit the CPU budget, add the same predetermined seeds to every arm, without retaining only favorable runs.

Evidence for the main hypothesis requires both a factual-prediction/decision discrepancy and a consistent advantage for randomized over passive acquisition at the same budget. Report whether the advantage holds for simple as well as neural learners and whether it depends on the fitting procedure. Claim only the tested simulator, observation regime, reset distribution, action set, and horizon.

Useful nulls include: no factual/decision discrepancy; passive collection matching randomized collection; simple cell means beating the MLP; randomized benefits disappearing once `T0` is observed; recalibration variance overwhelming bias reduction; and no gain from targeted acquisition. Report these without changing the target or hiding regimes.

The experiment is invalidated or requires relabelling as exploratory if any final test feedback changes physics, target, hidden-state distribution, feature mask, learner weighting, tuning, or method selection; if alternative evaluator branches leak into training; if models receive future application/outcome information; if interaction counts differ; if numerical error determines the winning command; or if the headline reference exploits hidden state denied to the learner.

## What this first study cannot establish

The primary experiment concerns **issuing a command** with immediate execution. A later state-dependent execution wrapper changes the estimand. Randomizing requests does not automatically identify the effect of forcing an applied actuator value when hidden conditions also affect application. Keep those results separate.

The study does not establish safe closed-loop control, causal identification from arbitrary real logs, performance on packaging cells, simulation-to-real transfer, general architectural superiority, or novelty of causal confusion itself. Its intended contribution is an inspectable qualification test and an evidence-allocation measurement that can inform a future real-machine experiment.
