# The Intervention Gap

**Better prediction from operator logs does not guarantee better action choices.** This repository measures the gap, and measures how many deliberate trials it takes to close it, on an independently authored reactor model and a public real-robot dataset. It is a CPU-scale research prototype: every number regenerates from the code, configurations are frozen with hashes, and negative results are kept.

## Findings

1. **A model can pass the usual "world model" test and still pick worse actions.**
   - In an exact example, action-aware prediction halves error (MSE 18 → 9) yet recommends the harmful choice (`research/action_confounding_counterexample.py`).
   - In the reactor study, every concentration/action cell had data, yet a network trained on operator logs chose worse actions than a fixed per-concentration rule (73.5 vs 37.7 K²).
2. **Randomized trials are worth several logged episodes each** (100 seeds, `EXCHANGE_RATE_RESULTS.md`).
   - At equal count, 240 randomized trials give 7.6× lower regret than 240 operator-logged episodes (27 vs 206 K²).
   - Naive regression on unlimited logs still converges to a wrong command (4.3 K²).
   - Reweighting by the operator's true action probabilities fixes this only with thousands of logs, and a plant log does not contain those probabilities.
3. **Among randomized designs, the choice matters less than claimed earlier.**
   - With 100 seeds, stratified, random, top-two Thompson sampling and a decision-focused heuristic cannot be separated.
   - An uncertainty heuristic is reliably worse. This corrects a 10-seed claim in `FOLLOWUP_RESULTS.md`.
4. **The history wasn't missing information; the model was missing a feature.**
   - Known-physics observers recover the hidden temperature from short noiseless histories (`STATE_ESTIMATION_RESULTS.md`).
   - Handing the network two concentration differences cuts its excess cost from 55.6 to 11.9 K², near a real temperature sensor (7.4).
   - Noise and ±10% parameter error degrade the observers measurably.
5. **On real robot data, command inputs help most forecasts but not every controller shift** (`RESULTS.md`, section 2).

![Logs versus experiments](results/figures/exchange_rate.png)

## What this is not

- **Not a new algorithm.** Each ingredient has textbook roots: causal confusion (de Haan et al., 2019), off-policy evaluation (Precup et al., 2000; Dudík et al., 2011), best-arm identification (Russo, 2016). The contribution is a reproducible qualification harness and measured trade-offs.
- **Not real-plant evidence.** Results come from one simulated reactor with a constructed operator, plus a public robot dataset.
- **Not proof of plant value.** It is not a safe controller, and it is not evidence of factory ROI.

An independent technical review and the responses to it are in `review/INDEPENDENT_REVIEW.md`.

## Reproduce

Python 3.13, `pip install -r requirements.txt`. Everything runs on CPU.

```sh
python -m pytest -q                                   # 12 tests
python -m research.action_confounding_counterexample  # exact example
python -m research.process.validate                   # solver checks vs SciPy Radau
python -m research.process.run_exchange_rate          # v2, 100 seeds (~1 h on 2 cores)
python research/build_exchange_report.py              # tables + figure
```

Earlier studies have their own runners: `run_study`, `run_information_audit`, `run_acquisition` and `run_state_estimation` in `research/process/`. The real-robot baselines are in `research/real_data/` (download the data first with `python -m research.real_data.download --help`). Runners refuse to overwrite frozen outputs, so choose a new output directory to rerun.

## Layout

| Path | What |
| --- | --- |
| `research/*_PROTOCOL.md` | Protocols written before each run |
| `research/process/` | Reactor adapter (pinned PC-Gym equations), studies, observers |
| `research/real_data/` | Robot dataset loader and baselines |
| `research/records/` | Action/outcome record checker (synthetic example only) |
| `results/` | Frozen configs, per-seed metrics, ledgers, figures |
| `RESULTS.md`, `FOLLOWUP_RESULTS.md`, `STATE_ESTIMATION_RESULTS.md`, `EXCHANGE_RATE_RESULTS.md` | Write-ups in the order the work was done |

## Provenance and licence

- **Code:** MIT (`LICENSE`).
- **Reactor equations:** from [PC-Gym](https://github.com/MaximilianB2/pc-gym) at a pinned commit, MIT; see `vendor/pcgym_reference`.
- **Robot data:** the CC0 [Transferable Dynamics dataset](https://doi.org/10.17617/3.ZT6K7P) (Agudelo-España et al., ICRA 2020). It is downloaded separately; URLs and SHA-256 hashes are in `data/transferable_dynamics/manifest.json`.
- **Frozen configs:** one stores an absolute local path. It is left unchanged so the config hash still matches.

Developed with AI assistance.
