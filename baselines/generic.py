"""Starting model for every task: a black box with no physics in it.

Contract (checked by harness.py and guardrails.py):
    fit(X, y)          -> any object holding the fitted state
    predict(state, X)  -> 1-D array, one prediction per row of X

X is a DataFrame of the task's input columns (all positive, SI units).
y is the measured response (positive).

Log-transforms inputs and target (most engineering relations are close to
power laws), then averages a few small neural networks.
"""
import warnings

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

N_SEEDS = 5
HIDDEN = (32, 32)
ALPHA = 1e-3


def _design(X):
    return np.log(np.clip(X.to_numpy(dtype=float), 1e-12, None))


def fit(X, y):
    scaler = StandardScaler().fit(_design(X))
    Z = scaler.transform(_design(X))
    t = np.log(np.clip(y, 1e-12, None))
    models = [MLPRegressor(hidden_layer_sizes=HIDDEN, alpha=ALPHA, solver="lbfgs",
                           max_iter=3000, random_state=s).fit(Z, t)
              for s in range(N_SEEDS)]
    return {"scaler": scaler, "models": models}


def predict(state, X):
    Z = state["scaler"].transform(_design(X))
    preds = np.array([m.predict(Z) for m in state["models"]])
    return np.exp(np.mean(preds, axis=0))
