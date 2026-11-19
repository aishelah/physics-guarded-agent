# Run report: Counterflow heat exchanger

- Baseline: CV relative RMSE 0.0592, physics check failed
- Best passing model: 0.0305
- Measurement noise floor: about 0.03
- Experiments: 5 (2 discarded, 1 kept, 2 proposal_failed)

Extrapolation error is measured on held-out inputs outside the training range and is never shown to the agent.

| id | status | CV rel. RMSE | extrap. rel. RMSE | hypothesis | reason |
|---|---|---|---|---|---|
| 000 | baseline_fail | 0.0592 | 0.1051 | starting model | second_law: 0.06 > limit 0.01; inlet_shift_invariance: 0.0481 > limit 0.03; UA_monotonic: 0.06 > limit 0.05 |
| 001 | kept | 0.0305 | 0.0 | Replacing the black‑box neural net with the exact NTU effectiveness formula for a counter‑flow heat exchanger (using the known water heat capacity) will enforce the thermodynamic constraints and improve predictive accuracy. |  |
| 002 | discarded | 0.0305 | 0.0 | ** Adding a small monotonic, UA‑only correction factor to the NTU‑based prediction will capture systematic bias (e.g., imperfect UA calibration) and lower the cross‑validated RMSE while still obeying all physics checks. | no gain over 0.0305 |
| 003 | discarded | 0.0307 | 0.0027 | Adding a non‑negative linear correction term proportional to UA will capture systematic bias in the NTU prediction (e.g., imperfect UA calibration) and reduce RMSE while preserving all physics constraints. | no gain over 0.0305 |
| 004 | proposal_failed |  |  | (no proposal) | HTTP Error 429: Too Many Requests |
| 005 | proposal_failed |  |  | (no proposal) | HTTP Error 429: Too Many Requests |
