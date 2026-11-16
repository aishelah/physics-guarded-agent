"""Simulated benchmark problems with known ground truth.

Each task module defines:
    NAME, TITLE, TARGET, NOISE, N_TRAIN, INPUTS (column -> description), BRIEF
    sample_inputs(n, rng, extrapolate=False) -> DataFrame
    truth(X) -> array of the exact response
    CONSTRAINTS -> list of physics checks, measured by harness.py
"""
from . import heat_exchanger, pipe_flow, vibration

TASKS = {m.NAME: m for m in (heat_exchanger, pipe_flow, vibration)}


def get(name):
    if name not in TASKS:
        raise SystemExit(f"unknown task '{name}'. Choose from: {', '.join(TASKS)}")
    return TASKS[name]
