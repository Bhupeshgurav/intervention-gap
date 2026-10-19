# State-estimation control — 16 October 2026

Prospective follow-up to the already observed history/sensor comparison. This is a mechanistic control, not a new learned method or an independent holdout study.

Question: can a known-physics observer extract temperature from the same three concentration readings that were supplied to the history MLP?

Two fixed estimators:
1. Estimate the current concentration derivative with the three-point backward difference `(3*c2-4*c1+c0)/(2*dt)`, dt=.01. Invert the concentration equation: `k=(q/V*(Caf-c2)-dc/dt)/c2`, `T=(EA/R)/log(k0/k)`. Mark nonphysical inversions invalid; do not silently clip temperature or substitute the true state.
2. Fix initial concentration to its measured value and fit initial temperature in [300,360] K by 36 iterations of golden-section search on squared concentration residuals at .01 and .02. Simulate the known 296 K warmup; use the fitted final state. This is a bounded numerical estimate; report near-boundary solutions. It assumes exact warmup timing and command, deterministic dynamics, and known initial concentration up to measurement noise.

Evaluate the previously saved 1000 episodes in each [320,340] and [320,330] initial-temperature interval. Gaussian concentration noise standard deviations 0, .0001, .001 source units; noise seeds 101,102,103 for nonzero noise, one deterministic run for zero noise. These levels are sensitivity settings, not sensor specifications. For model mismatch, repeat only zero noise with assumed k0 factors .9 and 1.1; truth remains unchanged. Use the assumed parameters both for observer and forecast. No parameter selection after results.

Each estimate produces candidate costs using the known equations, the existing integration settings, and the three existing objectives. Compare chosen actions against the saved full-state oracle. Report valid-estimate fraction, temperature RMSE, boundary fraction, conditional decision excess, and decision excess if invalid estimates fall back to fixed command 296 K. An invalid estimate is an abstention; fallback exists only to make a full-population numeric comparison, not to recommend a physical fallback action. Record its frequency.

The observer has privileged knowledge of the true governing equations in the matched case. It receives no actual temperature, future outcome, or evaluator branch cost as an input. It does not prove such parameters are available in a factory. Lower error than a learned model is an explanatory control, not an architecture victory. The original histories are noiseless; this follow-up challenges an information-loss interpretation and tests its fragility to noise/mismatch.
