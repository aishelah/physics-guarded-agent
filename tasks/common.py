"""Shared helpers for the simulated benchmark tasks."""
import numpy as np
import pandas as pd


def loguniform(rng, lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


def make_dataset(task, n, seed, extrapolate=False, noise=None):
    """Sample inputs, compute the true response, add measurement noise.

    Noise is multiplicative and log-normal (a constant percentage error, as from
    a calibrated sensor). Returns (X, y_noisy, y_true).
    """
    rng = np.random.default_rng(seed)
    X = task.sample_inputs(n, rng, extrapolate=extrapolate)
    y_true = np.asarray(task.truth(X), dtype=float)
    sigma = task.NOISE if noise is None else noise
    y = y_true * np.exp(sigma * rng.normal(size=n))
    return X.reset_index(drop=True), y, y_true


def as_frame(columns):
    return pd.DataFrame(columns)
