"""Steady-state response of a forced, damped single-degree-of-freedom system
(dynamics of machines).

Ground truth:
    m x'' + c x' + k x = F0 sin(omega t)
    X = F0 / sqrt((k - m omega^2)^2 + (c omega)^2)
"""
import numpy as np

from .common import as_frame, loguniform

NAME = "vibration"
TITLE = "Forced damped vibration"
TARGET = "X_mm"
NOISE = 0.03
N_TRAIN = 300

INPUTS = {
    "m_kg": "mass (kg)",
    "k_N_m": "spring stiffness (N/m)",
    "c_N_s_m": "viscous damping coefficient (N s/m)",
    "F0_N": "amplitude of the harmonic force (N)",
    "omega_rad_s": "forcing frequency (rad/s)",
}

BRIEF = (
    "A mass-spring-damper machine mount driven by a harmonic force. Predict the "
    "steady-state displacement amplitude in mm from mass, stiffness, damping, "
    "force amplitude and forcing frequency. Training covers forcing frequencies "
    "up to 3 times the natural frequency, including resonance. Measurements "
    "carry about 3% noise. The held-out assessment uses forcing frequencies "
    "between 3 and 5 times the natural frequency."
)


def sample_inputs(n, rng, extrapolate=False):
    m = loguniform(rng, 1.0, 50.0, n)
    k = loguniform(rng, 1e3, 1e5, n)
    zeta = loguniform(rng, 0.02, 0.5, n)
    r = rng.uniform(3.0, 5.0, n) if extrapolate else rng.uniform(0.1, 3.0, n)
    return as_frame({
        "m_kg": m,
        "k_N_m": k,
        "c_N_s_m": 2.0 * zeta * np.sqrt(k * m),
        "F0_N": loguniform(rng, 10.0, 1000.0, n),
        "omega_rad_s": r * np.sqrt(k / m),
    })


def damping_limit_mm(X):
    """X can never exceed F0 / (c omega): the damping term alone bounds the response."""
    return 1000.0 * X["F0_N"].to_numpy() / (X["c_N_s_m"].to_numpy() * X["omega_rad_s"].to_numpy())


def truth(X):
    m, k, c = X["m_kg"].to_numpy(), X["k_N_m"].to_numpy(), X["c_N_s_m"].to_numpy()
    w = X["omega_rad_s"].to_numpy()
    return 1000.0 * X["F0_N"].to_numpy() / np.sqrt((k - m * w ** 2) ** 2 + (c * w) ** 2)


CONSTRAINTS = [
    {"name": "damping_bound", "type": "bounds", "lower": 0.0, "upper": damping_limit_mm,
     "tol": 0.03, "limit": 0.01,
     "desc": "0 <= X <= F0 / (c omega). The response can never exceed the damping-limited "
             "amplitude (3% tolerance; at most 1% of predictions may break it)."},
    {"name": "force_linearity", "type": "scaling", "var": "F0_N", "factor": 2.0,
     "exponent": 1.0, "limit": 0.05,
     "desc": "The system is linear: doubling F0 must double X. Mean relative deviation "
             "from exact doubling <= 5%."},
]
