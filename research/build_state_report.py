"""Generate the known-physics control report from saved measurements."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'results/state_estimation_v1/metrics.json').read_text())
rows=[]
for sigma,factor in [(0,1),(.0001,1),(.001,1),(0,.9),(0,1.1)]:
 for method in ['finite_difference','trajectory_fit']:
  z=[x for x in r if x['range_high']==340 and x['noise_sd']==sigma and x['k0_factor']==factor and x['method']==method and x['cost']=='squared']
  rows.append(dict(noise_sd=sigma,k0_factor=factor,method=method,**{k:float(np.mean([x[k] for x in z])) for k in ['temperature_rmse_K','valid_fraction','boundary_fraction','fallback_population_excess']}))
(ROOT/'results/state_estimation_v1/summary.json').write_text(json.dumps(rows,indent=2)+'\n')
fig,axs=plt.subplots(1,2,figsize=(10.5,4.6))
for method,label,color in [('finite_difference','Derivative inversion','#b46a35'),('trajectory_fit','Trajectory fit','#167c80')]:
 z=[x for x in rows if x['method']==method and x['k0_factor']==1]
 for ax,key in zip(axs,['temperature_rmse_K','fallback_population_excess']):
  ax.plot(range(3),[x[key] for x in z],'-o',label=label,color=color)
  ax.set_xticks(range(3),['0','0.0001','0.001']);ax.set_xlabel('Concentration noise SD (source units)');ax.grid(alpha=.15);ax.set_ylim(bottom=0)
axs[0].set_ylabel('Temperature RMSE on valid cases (K)')
axs[1].set_ylabel('Full-population decision excess (K²)');axs[1].legend(frameon=False)
fig.suptitle('Known physics resolves noiseless history; measurement noise changes the result\nWider interval; noise-run means; invalid inversions use the declared 296 K fallback',fontsize=11)
fig.tight_layout();fig.savefig(ROOT/'results/figures/state_estimation.png')
text='''# State estimation: the history contains the missing information

16 October 2026. Mechanistic follow-up on previously examined evaluation episodes, with assumptions fixed in `research/STATE_ESTIMATION_PROTOCOL.md` before execution.

## Conclusion

**The history-model failure is not evidence that these noiseless histories lack temperature information.** Both known-physics estimators recover temperature and choose optimal actions on all 1000 episodes in each evaluated interval, for all three cost objectives, in the noiseless matched-parameter setting.

The three-point derivative estimate followed by algebraic inversion has temperature RMSE 0.000694 K on the wider interval. Fitting an initial temperature to the short concentration trajectory reduces that RMSE to approximately 3e-7 K. Neither receives the true temperature or future outcomes as an input. Both know the governing equations; the trajectory fit also knows the historical command and timing. This is additional physical knowledge unavailable to the generic learned models, not an architecture-matched victory.

## Wider-interval sensitivity results

Nonzero-noise rows average three independent noise realizations on the same 1000 physical episodes. Zero-noise rows are deterministic, not three independent replications. Temperature RMSE is conditional on usable estimates. Decision excess includes all episodes and the prospectively declared fixed-command fallback for invalid estimates.

| Noise SD | Assumed k0 / true k0 | Estimator | Temperature RMSE, K | Valid fraction | Boundary fraction | Decision excess, K² |
| --- | --- | --- | ---: | ---: | ---: | ---: |
'''
for x in rows:
 text+=f"| {x['noise_sd']:g} | {x['k0_factor']:g} | {x['method'].replace('_',' ')} | {x['temperature_rmse_K']:.6g} | {x['valid_fraction']:.4f} | {x['boundary_fraction']:.4f} | {x['fallback_population_excess']:.4f} |\n"
text+='''
![Observer sensitivity](results/figures/state_estimation.png)

The larger noise setting invalidates approximately 22.5% of derivative inversions in the wider interval. The trajectory fit always returns a finite bounded estimate, but 3.4% of solutions approach the search limits. **Finite is not the same as reliable.** A bounded solver can hide poor identifiability behind a plausible number; boundary hits and residuals deserve attention.

A ±10% error in the assumed kinetic pre-factor produces roughly 1.1–1.3 K temperature RMSE and nonzero decision error even with noiseless observations. These are sensitivity examples, not realistic factory parameter distributions. The same assumed parameters are used for estimation and subsequent forecasting.

## How the observer works

The concentration equation provides `k = ((q/V)(Caf-C) - dC/dt)/C`, then `T = (EA/R)/log(k0/k)`. The derivative uses only the three concentration samples ending at the decision. Invalid logarithms are reported as abstentions. Differentiation magnifies measurement noise.

The trajectory fit fixes initial concentration to its measured value and searches initial temperature over [300,360] K for the best fit to the next two historical concentration readings. It uses 36 golden-section iterations and the known 296 K warmup. This is a simple conditional least-squares observer, not a full statistical state estimator: initial-concentration uncertainty, correlated sensor error, and parameter uncertainty are not jointly inferred. Its boundary fraction is not a calibrated confidence score.

## Claims this removes

- Do not describe the history experiment as proving an irreducible hidden-state problem.
- Do not infer that adding a temperature sensor is always necessary from the MLP comparison.
- Do not claim learned world models are necessary when known physical equations already solve the tested noiseless problem.
- Do not describe these results as evidence of industrial sensing quality or factory ROI.

The original snapshot confounding construction remains valid in its stated information setting. The history experiment has different information, and its conclusions must be narrower.

## Practical implication

Before spending a collection budget or training a larger model, ask what state information is recoverable from existing measurements under available physical knowledge. Then measure sensitivity to noise, model mismatch, and incomplete execution records. A useful industrial contribution would combine these checks with an actual operational decision and quantify when the simplest adequate model stops being adequate.

This study still needs external methodological review and a real, nontrivial decision domain. The review request is in `review/STATE_ESTIMATION_REVIEW_PROMPT.md`; preparing that prompt is not an independent review.

## Artifacts

The complete 108 metric rows, per-episode estimates, validity flags, boundary flags, action choices, frozen configuration, and source hashes are in `results/state_estimation_v1`. Reproduce in a fresh output version with `python -m research.process.run_state_estimation`. All numerical code uses the previously validated small integration step.
'''
(ROOT/'STATE_ESTIMATION_RESULTS.md').write_text(text)
print('Wrote state-estimation report and figure')
