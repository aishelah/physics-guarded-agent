"""Fixed evaluation harness. The agent may NOT edit this file.

For one task and one candidate model it measures:
  - cv_rel_rmse   5-fold cross-validated relative RMSE against noisy measurements
                  (the score the agent is judged on)
  - constraints   each physics check from the task, on the validation inputs
  - extrap_rel_rmse  error against the exact ground truth on inputs outside the
                  training range. Held out from the agent: reported, never used
                  to keep or reject a proposal.

Usage:
    python harness.py --task pipe_flow                       # scores baselines/generic.py
    python harness.py --task pipe_flow runs/pipe_flow/best.py
Prints one JSON object on the last line of stdout.
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

import tasks
from tasks.common import make_dataset

ROOT = Path(__file__).resolve().parent
TRAIN_SEED, EXTRAP_SEED, CV_SEED = 1, 2, 42
N_SPLITS, N_EXTRAP = 5, 200
EPS = 1e-12


def load_candidate(path):
    spec = importlib.util.spec_from_file_location("candidate_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _predict(cand, state, X):
    pred = np.asarray(cand.predict(state, X.copy()), dtype=float).ravel()
    if pred.shape != (len(X),):
        raise ValueError(f"predict() returned shape {pred.shape}, expected ({len(X)},)")
    if not np.all(np.isfinite(pred)):
        raise ValueError("predict() returned NaN or infinite values")
    return pred


def rel_rmse(y, pred):
    return float(np.sqrt(np.mean(((pred - y) / y) ** 2)))


def _value(spec, X):
    return spec(X) if callable(spec) else np.full(len(X), float(spec))


def measure(c, cand, state, X, pred):
    """Return one number per constraint; compared with c['limit'] (pass if <=)."""
    kind = c["type"]
    if kind == "bounds":
        bad = np.zeros(len(pred), dtype=bool)
        if c.get("lower") is not None:
            lo = _value(c["lower"], X)
            bad |= pred < lo - c["tol"] * np.abs(lo)
        if c.get("upper") is not None:
            bad |= pred > _value(c["upper"], X) * (1.0 + c["tol"])
        return float(bad.mean())

    Xp = X.copy()
    if kind == "scaling":
        Xp[c["var"]] = Xp[c["var"]] * c["factor"]
        expected = pred * c["factor"] ** c["exponent"]
        got = _predict(cand, state, Xp)
        return float(np.mean(np.abs(got - expected) / np.maximum(np.abs(expected), EPS)))
    if kind == "shift":
        for v in c["vars"]:
            Xp[v] = Xp[v] + c["delta"]
        got = _predict(cand, state, Xp)
        return float(np.mean(np.abs(got - pred) / np.maximum(np.abs(pred), EPS)))
    if kind == "monotonic":
        Xp[c["var"]] = Xp[c["var"]] * c["factor"]
        got = _predict(cand, state, Xp)
        wrong = (got - pred) * c["direction"] < -c["tol"] * np.abs(pred)
        return float(wrong.mean())
    raise ValueError(f"unknown constraint type {kind}")


def evaluate(task, path):
    cand = load_candidate(path)
    X, y, _ = make_dataset(task, task.N_TRAIN, TRAIN_SEED)
    t0 = time.time()
    scores = []
    per_constraint = {c["name"]: [] for c in task.CONSTRAINTS}

    for tr, val in KFold(N_SPLITS, shuffle=True, random_state=CV_SEED).split(X):
        state = cand.fit(X.iloc[tr].reset_index(drop=True).copy(), y[tr].copy())
        X_val = X.iloc[val].reset_index(drop=True)
        pred = _predict(cand, state, X_val)
        scores.append(rel_rmse(y[val], pred))
        for c in task.CONSTRAINTS:
            per_constraint[c["name"]].append(measure(c, cand, state, X_val, pred))

    # Held-out extrapolation check: fit on all training data, compare with exact truth
    state = cand.fit(X.copy(), y.copy())
    X_ex, _, y_ex_true = make_dataset(task, N_EXTRAP, EXTRAP_SEED, extrapolate=True)
    extrap = rel_rmse(y_ex_true, _predict(cand, state, X_ex))

    constraints = {}
    for c in task.CONSTRAINTS:
        v = float(np.mean(per_constraint[c["name"]]))
        constraints[c["name"]] = {"value": round(v, 4), "limit": c["limit"], "pass": v <= c["limit"]}

    return {
        "task": task.NAME,
        "cv_rel_rmse": round(float(np.mean(scores)), 4),
        "cv_rel_rmse_std": round(float(np.std(scores)), 4),
        "noise_floor": task.NOISE,
        "constraints": constraints,
        "extrap_rel_rmse": round(extrap, 4),
        "seconds": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(tasks.TASKS))
    ap.add_argument("candidate", nargs="?", default=str(ROOT / "baselines" / "generic.py"))
    args = ap.parse_args()
    print(json.dumps(evaluate(tasks.get(args.task), args.candidate)))
