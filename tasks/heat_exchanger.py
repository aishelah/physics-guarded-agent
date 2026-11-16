"""Counterflow heat exchanger, water on both sides (thermal engineering).

Ground truth: the effectiveness-NTU method.
    C = m * cp,  C_min, C_max,  Cr = C_min / C_max,  NTU = UA / C_min
    eps = (1 - exp(-NTU (1 - Cr))) / (1 - Cr exp(-NTU (1 - Cr)))     (Cr < 1)
    eps = NTU / (1 + NTU)                                             (Cr = 1)
    Q   = eps * C_min * (T_hot_in - T_cold_in)
"""
import numpy as np

from .common import as_frame, loguniform

NAME = "heat_exchanger"
TITLE = "Counterflow heat exchanger"
TARGET = "Q_kW"
NOISE = 0.03
N_TRAIN = 300
CP = 4180.0  # J/(kg K), water

INPUTS = {
    "m_hot_kg_s": "hot-side mass flow rate (kg/s)",
    "m_cold_kg_s": "cold-side mass flow rate (kg/s)",
    "T_hot_in_K": "hot-side inlet temperature (K)",
    "T_cold_in_K": "cold-side inlet temperature (K)",
    "UA_W_K": "overall heat-transfer conductance UA (W/K)",
}

BRIEF = (
    "A counterflow heat exchanger with water on both sides (cp = 4180 J/kg K). "
    "Predict the heat duty Q in kW from the two flow rates, the two inlet "
    "temperatures and the conductance UA. Measurements carry about 3% noise. "
    "The held-out assessment includes exchangers with larger UA than any seen "
    "in training."
)


def sample_inputs(n, rng, extrapolate=False):
    ua_range = (8000, 30000) if extrapolate else (200, 8000)
    return as_frame({
        "m_hot_kg_s": loguniform(rng, 0.05, 2.0, n),
        "m_cold_kg_s": loguniform(rng, 0.05, 2.0, n),
        "T_hot_in_K": rng.uniform(330, 420, n),
        "T_cold_in_K": rng.uniform(280, 320, n),
        "UA_W_K": loguniform(rng, *ua_range, n),
    })


def _c_min_max(X):
    ch = X["m_hot_kg_s"].to_numpy() * CP
    cc = X["m_cold_kg_s"].to_numpy() * CP
    return np.minimum(ch, cc), np.maximum(ch, cc)


def q_max_kw(X):
    """Largest duty the second law allows: C_min times the inlet temperature difference."""
    c_min, _ = _c_min_max(X)
    return c_min * (X["T_hot_in_K"].to_numpy() - X["T_cold_in_K"].to_numpy()) / 1000.0


def truth(X):
    c_min, c_max = _c_min_max(X)
    cr = c_min / c_max
    ntu = X["UA_W_K"].to_numpy() / c_min
    near_one = np.abs(1.0 - cr) < 1e-6
    e = np.exp(-ntu * (1.0 - cr))
    eps = np.where(near_one, ntu / (1.0 + ntu),
                   (1.0 - e) / np.where(near_one, 1.0, 1.0 - cr * e))
    return eps * q_max_kw(X)


CONSTRAINTS = [
    {"name": "second_law", "type": "bounds", "lower": 0.0, "upper": q_max_kw,
     "tol": 0.03, "limit": 0.01,
     "desc": "0 <= Q <= C_min (T_hot_in - T_cold_in). No exchanger can transfer more "
             "than the maximum duty (3% tolerance; at most 1% of predictions may break it)."},
    {"name": "inlet_shift_invariance", "type": "shift",
     "vars": ["T_hot_in_K", "T_cold_in_K"], "delta": 15.0, "limit": 0.03,
     "desc": "Raising both inlet temperatures by 15 K must leave Q unchanged (constant "
             "cp: only the temperature difference matters). Mean relative change <= 3%."},
    {"name": "UA_monotonic", "type": "monotonic", "var": "UA_W_K", "factor": 1.25,
     "direction": +1, "tol": 0.005, "limit": 0.05,
     "desc": "Increasing UA by 25% must not reduce Q. At most 5% of predictions may "
             "move the wrong way."},
]
