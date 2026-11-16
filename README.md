# Physics-Guarded Experiment Agent

An LLM agent that runs modelling experiments on mechanical-engineering systems.
Every model it proposes has to pass physics checks (conservation laws, bounds,
scaling laws, monotonicity) before its accuracy is allowed to count.

**Status: scaffold (work in progress).** The benchmarks, harness, guardrails and
loop are built and tested offline. The LLM-driven runs have not been done yet,
so there are no agent results to report.

## The question

Autonomous experiment loops optimise one number. On scientific problems that is
risky: a model can lower its error while breaking the physics, and still look
like progress. This project asks:

1. Can an LLM agent find models that are both accurate and physically
   consistent, when inconsistent ones are rejected outright?
2. Does physical consistency on the training range carry over to conditions the
   model has never seen?

## Benchmarks

Three simulated systems with known ground truth. Each has 300 training points
with 3% multiplicative measurement noise, and a held-out extrapolation set
outside the training range that the agent never sees.

| Task | Area | Ground truth | Physics checks | Extrapolation |
|---|---|---|---|---|
| `heat_exchanger` | Thermal | Counterflow ε-NTU, water both sides | Second law (Q ≤ Q_max); invariance to shifting both inlet temperatures; Q rises with UA | UA beyond training range |
| `pipe_flow` | Fluids | Darcy-Weisbach, laminar / transitional / Colebrook turbulent | Δp ≥ 0; Δp ∝ L; Δp rises with velocity | Longer pipes |
| `vibration` | Dynamics | Forced damped SDOF steady-state amplitude | 0 ≤ X ≤ F0/(cω); X ∝ F0 (linearity) | Forcing at 3-5× natural frequency |

The simulators were checked against textbook values: ε = 0.775 at NTU 2, Cr 0.5;
Colebrook f = 0.0185 at Re 10⁵, ε/D 10⁻⁴; amplitude ratio 5 at resonance with ζ = 0.1.

## How it works

```
propose  ->  static check  ->  run (subprocess, time limit)  ->  physics check  ->  keep / revert  ->  log
```

| File | Role | Who edits it |
|---|---|---|
| `runs/<task>/best.py` | The current model; starts as `baselines/generic.py` | The agent |
| `program.md` | Brief template, filled in per task | Human |
| `tasks/` | Simulators, inputs, physics checks | Human |
| `harness.py` | Fixed 5-fold CV scoring, physics measurements, held-out extrapolation | Nobody |
| `guardrails.py` | Static code check and pass/fail on the physics measurements | Human |
| `proposers.py` | LLM proposer, plus a scripted one for testing the loop | Human |
| `agent.py` | The loop, logging and report | Human |

A model is kept only if it passes every physics check and improves the CV error
by at least 2%. If the current model fails the checks, the first passing model
replaces it. The static check stops a model reading data files or importing the
ground-truth simulators; it is a filter, not a security sandbox.

## Results so far

Black-box baseline (`baselines/generic.py`: log features, averaged small MLPs),
scored by `harness.py` (scikit-learn 1.9.1, Linux):

| Task | CV rel. RMSE | Physics checks | Extrap. rel. RMSE |
|---|---|---|---|
| heat_exchanger | 5.9% | **Fails all three** (second law 5.7% of predictions, shift 4.7%, UA monotonic 6.7%) | 10.7% |
| pipe_flow | 9.0% | Passes | 8.6% |
| vibration | 21.6% | **Fails both** (damping bound 2.0%, linearity 5.8%) | 7.9% |

A model with good accuracy can still break basic physics: the heat-exchanger
baseline predicts more heat transfer than the second law allows on 5.7% of
validation points.

Loop test with hand-written edits (`--proposer scripted`): importing the
simulator was rejected by the static check, negative predictions by the bounds
check, a crash was logged and skipped, and a valid edit was scored and
discarded for no gain. These test the machinery; they are not agent results.

## Run it

```bash
pip install -r requirements.txt
python harness.py --task heat_exchanger                          # score the baseline
python agent.py --task pipe_flow --iters 4 --proposer scripted   # offline loop test

export LLM_API_KEY=...        # any OpenAI-compatible endpoint
export LLM_BASE_URL=...       # optional
export LLM_MODEL=...          # optional
python agent.py --task heat_exchanger --iters 30
```

The LLM proposer has not yet been run against a live API.

## Next

- [ ] First LLM-driven run on each task (30-50 experiments) and a written analysis
- [ ] Ablation: the same agent with the physics check off, compared on extrapolation
- [ ] Track whether kept models extrapolate better or worse than the baseline
- [ ] More systems: fin heat transfer, beam deflection, fatigue life (S-N curve)
- [ ] Agent-written summary of what it learned about each system

## Credits

Loop design adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
(one editable file, fixed evaluation, keep-or-revert), moved from GPU
language-model training to CPU engineering benchmarks, with the physics
guardrail layer added.
