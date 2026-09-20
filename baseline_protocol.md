# Baseline revision protocol

Frozen design: 2026-09-08, before new-rule validation and all revised baseline test runs. This addendum is part of PLAN.md. Inputs, preprocessing, chronological 240/60/65 day split, causal forecasts, full lighting, terminal battery energy, common QP and daily reset remain the existing experiment contract. The historical holdout has been inspected; no claim of untouched external validation is made.

Research question: how sensitive are the implemented-controller comparisons to the original rule's net-load behavior and the MPC planner's hard future-ramp constraints? The null is that these targeted changes do not materially alter the observed performance trade-offs; a changed ranking is a valid result and must be retained. This is a bounded computational sensitivity analysis, not a new safety algorithm.

## R2: current-box-anchored, future-ramp-relaxed MPC

Retain `MPC-H24+QP` with its original 45 kW hard ramp constraint over the whole remaining-day battery feasibility tail. Add `MPC-H24-FRR+QP`, called “MPC relaxed” in prose. For this sensitivity, the first battery command lies in the exact current interval returned by the repaired common `env._bounds()`; first-step EV bounds are also explicit. Future planned battery powers (j >= 1) have no hard ramp constraint. Battery energy bounds, 100 kW power bounds, efficiencies, the 125 kWh terminal target and prohibition of simultaneous charge/discharge remain. Objective horizon H = min(24, remaining steps), objective coefficients, piecewise-linear peak approximation, causal exogenous information and terminal feasibility tail remain unchanged. No exogenous values beyond H enter the tail.

This isolates the requirement to plan a future hard-ramp path while anchoring the current proposal to the actual execution box, whose energy/terminal priority can relax ramp. At every actual step the same QP applies the same conditional ramp rule. The future plan does not reproduce those recursive conditional execution boxes, grid corrections or forecast errors: it is a future-ramp-relaxed sensitivity, not a claim of fully equivalent recursive feasible sets. Unchanged one-second MILP limit, 0.001 relative gap, incumbent feasibility check and LegacyRule fallback are retained; every solver call is logged.

## R3: validation-selected net-load support rule

Retain original `Rule+QP`, called “LegacyRule” in prose; do not label its historical formula a proved sign error. Add `NetLoadRule+QP`. At the current state obtain battery bounds [lo, hi] and EV bounds [emin, emax]. Keep the legacy EV heuristic and price/carbon quantiles: e = clip(max(emin, 1 if PV > load else 0.65), emin, emax); inactive EV has e = 0. Full lighting power is L = 30 times the known light profile. Define current proposed service net load N = load + L + 60e - PV.

For a fixed candidate cap c, in kW, the unbounded battery proposal is:

1. If N < 0: p = -min(100, -N), prioritizing absorption of the actual proposed-service PV surplus.
2. Otherwise, if current price >= causal price quantile 0.70 or current carbon >= causal carbon quantile 0.65: p = min(c, N).
3. Otherwise, if current price <= causal price quantile 0.35: p = -c.
4. Otherwise: p = 0.

The returned battery proposal is clip(p, lo, hi), EV is e and lighting fraction is 1.0. Thus service/terminal feasibility takes precedence over the heuristic power cap. c = 35, 60 and 100 kW are the only candidates. The candidate cap limits high-price/high-carbon discharge and low-price charging; surplus charging retains the physical 100 kW maximum for all candidates. Quantile levels and EV heuristic are not tuned.

Evaluate each candidate on all 60 validation days (day IDs 240–299). Rank lexicographically by mean service-failure indicator, mean grid excess energy, mean cost, mean action correction, then smaller c. Service failure is exactly the PPO validation definition: battery violation, absolute terminal error > 1e-5 kWh, incomplete EV, or lighting error > 1e-5 kWh. Retain all candidate/day records. Freeze selected c, validation hashes, source/config/protocol hashes and pre-test status before test evaluation. No test metric enters selection.

## Execution, verification and outputs

Minimal code map: one optional planning mode in `src/matched_mpc.py`; independent `src/review_baselines.py` provides the rule, evaluator, selection, runner and audit. No change to the original rule, reward, data or causal evaluator is part of this subtask. Before formal execution, a bounded training/validation-state smoke check verifies proposal feasibility, the net-load direction and evaluator output; it provides no test evidence. Formal execution waits for the shared bounds fix and root PLAN freeze.

Evaluate each of four deterministic controllers once on each of the 65 test days (300–364), seed label 0 only. Persist `baseline_<controller>.csv`, `trajectory_<controller>.csv`, and for each MPC `mpc_diagnostics_<controller>.jsonl`. Required daily/step columns follow `causal_run.evaluate`; extra diagnostics are supplementary. Baselines are not repeated or counted as five independent seeds. Statistical contrasts use paired day clusters downstream under the unchanged analysis contract.

Before acceptance: verify all required daily metrics finite, 65 distinct daily rows per method, 96 steps per day, energy/EV/lighting service, all 96 QP paths accounted, trajectories agree with daily sums and maxima, all MPC solves retained, source/protocol/selection hashes unchanged since freeze. Preserve inferior/null findings, planning fallbacks, time-limit incumbents and ramp relaxations. Record commands and runtime environment in the baseline run contract and completion summary. The acceptance gate is computational evidence ready for synthesis; it does not imply author approval, public release or formal submission.
