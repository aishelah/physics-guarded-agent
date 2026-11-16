# Brief for the experiment agent

You are improving a model of a physical system from noisy measurements.

## The system: {title}

{brief}

Inputs (DataFrame columns, SI units, all positive):
{inputs}

Target: `{target}`

## What you control

You rewrite one Python file. It must keep this contract:

- `fit(X, y)` returns the fitted state. `X` is a DataFrame of the inputs above,
  `y` a NumPy array of measured `{target}`.
- `predict(state, X)` returns one prediction per row of `X`.

You cannot change the data, the cross-validation splits or the scoring. You may
hard-code physical constants and known relations from engineering theory.

## How you are judged

1. **Static check.** Imports are limited to numpy, pandas, scipy, sklearn, math
   and warnings. No file access, no `eval`/`exec`.
2. **Physics check.** A model is rejected, whatever its score, if it breaks any
   of these:
{constraints}
3. **Score.** Mean 5-fold cross-validated relative RMSE, lower is better. The
   measurement noise is about {noise_pct}%, so that is roughly the floor. A
   change is kept only if it passes the physics check and beats the current
   best by at least {min_gain_pct}% (relative). If the current best does not
   pass the physics check, any passing model is kept.

Each evaluation must finish within {timeout} seconds.

## How to work

- Change one idea per experiment, so the result tells you something.
- State a hypothesis first: what you are changing and which physical or
  statistical reason says it will help.
- Read the history. Do not repeat an experiment that already failed; build on
  the ones that were kept.
- Prefer models whose structure follows the physics (dimensionless groups,
  known limits, conservation laws) over extra capacity.

## Reply format

HYPOTHESIS: <one sentence>

```python
<the complete new file>
```
