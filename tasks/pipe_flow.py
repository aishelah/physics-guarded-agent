"""Pressure drop in a straight circular pipe (fluid mechanics).

Ground truth: Darcy-Weisbach with a friction factor that changes regime.
    Re = V D / nu
    f  = 64 / Re                          Re < 2300   (laminar)
    f  = Colebrook-White, solved          Re > 4000   (turbulent)
    f  = linear blend in Re               in between  (transitional)
    dp = f (L / D) rho V^2 / 2
"""
import numpy as np

from .common import as_frame, loguniform

NAME = "pipe_flow"
TITLE = "Pipe flow pressure drop"
TARGET = "dp_kPa"
NOISE = 0.03
N_TRAIN = 300
RHO = 1000.0  # kg/m^3

INPUTS = {
    "V_m_s": "mean flow velocity (m/s)",
    "D_m": "pipe inner diameter (m)",
    "L_m": "pipe length (m)",
    "roughness_m": "absolute wall roughness (m)",
    "nu_m2_s": "kinematic viscosity of the fluid (m^2/s)",
}

BRIEF = (
    "Fully developed flow of a liquid (density 1000 kg/m^3) through a straight "
    "circular pipe. Predict the pressure drop in kPa from velocity, diameter, "
    "length, wall roughness and viscosity. The data spans laminar, transitional "
    "and turbulent flow. Measurements carry about 3% noise. The held-out "
    "assessment includes pipes longer than any seen in training."
)


def sample_inputs(n, rng, extrapolate=False):
    l_range = (100, 500) if extrapolate else (1, 100)
    d = loguniform(rng, 0.01, 0.2, n)
    return as_frame({
        "V_m_s": loguniform(rng, 0.05, 3.0, n),
        "D_m": d,
        "L_m": loguniform(rng, *l_range, n),
        "roughness_m": d * loguniform(rng, 1e-6, 1e-2, n),
        "nu_m2_s": loguniform(rng, 1e-6, 1e-4, n),
    })


def _colebrook(re, rel_rough, iters=30):
    # Swamee-Jain as the starting guess, then fixed-point iteration on 1/sqrt(f)
    f = 0.25 / np.log10(rel_rough / 3.7 + 5.74 / re ** 0.9) ** 2
    x = 1.0 / np.sqrt(f)
    for _ in range(iters):
        x = -2.0 * np.log10(rel_rough / 3.7 + 2.51 * x / re)
    return 1.0 / x ** 2


def friction_factor(re, rel_rough):
    lam = 64.0 / re
    turb = _colebrook(np.maximum(re, 4000.0), rel_rough)
    f_lo = 64.0 / 2300.0
    w = np.clip((re - 2300.0) / (4000.0 - 2300.0), 0.0, 1.0)
    trans = (1 - w) * f_lo + w * turb
    return np.where(re < 2300, lam, np.where(re > 4000, turb, trans))


def truth(X):
    v, d = X["V_m_s"].to_numpy(), X["D_m"].to_numpy()
    re = v * d / X["nu_m2_s"].to_numpy()
    f = friction_factor(re, X["roughness_m"].to_numpy() / d)
    return f * (X["L_m"].to_numpy() / d) * RHO * v ** 2 / 2.0 / 1000.0


CONSTRAINTS = [
    {"name": "non_negative", "type": "bounds", "lower": 0.0, "upper": None,
     "tol": 0.0, "limit": 0.0,
     "desc": "dp >= 0. Friction never adds pressure."},
    {"name": "length_scaling", "type": "scaling", "var": "L_m", "factor": 2.0,
     "exponent": 1.0, "limit": 0.05,
     "desc": "For fully developed flow, doubling the length must double dp. Mean "
             "relative deviation from exact doubling <= 5%."},
    {"name": "velocity_monotonic", "type": "monotonic", "var": "V_m_s", "factor": 1.2,
     "direction": +1, "tol": 0.005, "limit": 0.05,
     "desc": "Increasing velocity by 20% must not reduce dp. At most 5% of predictions "
             "may move the wrong way."},
]
