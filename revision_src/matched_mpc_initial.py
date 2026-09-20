from __future__ import annotations

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack

from run_experiment import raw_from_physical, rule_action


def matched_mpc_action(env, horizon=24, future_ramp_relaxed=False, export_priority=False):
    started = time.perf_counter()
    H = min(int(horizon), env.n - env.t, env.h)
    L = env.n - env.t
    d = env.days[env.day]
    forecast = {k: np.asarray(env._forecast(k)[:H], float) for k in ("load", "pv", "price", "carbon")}
    active = np.asarray(d["ev_active"][env.t:env.t + H], float)
    lighting = env.cfg["loads"]["lighting_base_kw"] * np.asarray(d["light_profile"][env.t:env.t + H], float)
    base = forecast["load"] + lighting - forecast["pv"]
    ix = {}
    n = 0
    for key, size in (("ch", L), ("dis", L), ("charge_on", L), ("soc", L + 1), ("ev", H), ("gi", H), ("ge", H), ("over_import", H), ("over_export", H), ("peak", H)):
        ix[key] = np.arange(n, n + size)
        n += size
    lower = np.zeros(n)
    upper = np.full(n, np.inf)
    objective = np.zeros(n)
    integer = np.zeros(n, int)
    upper[ix["ch"]] = env.P
    upper[ix["dis"]] = env.P
    upper[ix["charge_on"]] = 1.0
    integer[ix["charge_on"]] = 1
    lower[ix["soc"]] = env.soc_min
    upper[ix["soc"]] = env.soc_max
    lower[ix["soc"][0]] = upper[ix["soc"][0]] = env.soc
    target = env.cfg["bess"]["soc_initial"] * env.E
    lower[ix["soc"][-1]] = upper[ix["soc"][-1]] = target
    upper[ix["ev"]] = np.minimum(env.ev_max * active, env.ev_remaining / env.dt)
    current_bounds = env._bounds() if future_ramp_relaxed else None
    if current_bounds is not None:
        lower[ix["ev"][0]] = current_bounds[2] * env.ev_max
        upper[ix["ev"][0]] = current_bounds[3] * env.ev_max
    upper[ix["gi"]] = np.maximum(0.0, base + env.ev_max * active + env.P)
    upper[ix["ge"]] = np.maximum(0.0, -base + env.P)
    objective[ix["gi"]] = (forecast["price"] / 10.0 + 1.8 * forecast["carbon"]) * env.dt
    objective[ix["ch"][:H]] = 0.0015 * env.dt
    objective[ix["dis"][:H]] = 0.0015 * env.dt
    objective[ix["over_import"]] = 0.25
    objective[ix["over_export"]] = 0.25
    objective[ix["peak"]] = 0.00035 * env.dt
    rows, columns, values, lo, hi = [], [], [], [], []

    def add(coefficients, low=-np.inf, high=np.inf):
        row = len(lo)
        for column, value in coefficients.items():
            rows.append(row)
            columns.append(int(column))
            values.append(float(value))
        lo.append(float(low))
        hi.append(float(high))

    for j in range(L):
        add({ix["ch"][j]: 1, ix["charge_on"][j]: -env.P}, high=0)
        add({ix["dis"][j]: 1, ix["charge_on"][j]: env.P}, high=env.P)
        add({ix["soc"][j + 1]: 1, ix["soc"][j]: -1, ix["ch"][j]: -env.eta_c * env.dt, ix["dis"][j]: env.dt / env.eta_d}, 0, 0)
        ramp = {ix["dis"][j]: 1, ix["ch"][j]: -1}
        if future_ramp_relaxed:
            if j == 0:
                add(ramp, current_bounds[0], current_bounds[1])
        elif j:
            ramp.update({ix["dis"][j - 1]: -1, ix["ch"][j - 1]: 1})
            add(ramp, -env.ramp, env.ramp)
        else:
            add(ramp, env.prev_p - env.ramp, env.prev_p + env.ramp)
    for j in range(H):
        add({ix["gi"][j]: 1, ix["ge"][j]: -1, ix["ev"][j]: -1, ix["ch"][j]: -1, ix["dis"][j]: 1}, base[j], base[j])
        add({ix["gi"][j]: 1, ix["over_import"][j]: -1}, high=env.grid_max + 0.01)
        add({ix["ge"][j]: 1, ix["over_export"][j]: -1}, high=-env.grid_min + 0.01)
        max_excess = max(0.0, upper[ix["gi"][j]] - 220.0)
        for left in np.arange(0.0, max_excess, 10.0):
            right = min(left + 10.0, max_excess)
            add({ix["peak"][j]: 1, ix["gi"][j]: -(left + right)}, low=-(left + right) * 220.0 - left * right)
    future_ev_capacity = float(np.sum(np.asarray(d["ev_active"][env.t + H:], float)) * env.ev_max * env.dt)
    ev_minimum = max(0.0, env.ev_remaining - future_ev_capacity)
    add({i: env.dt for i in ix["ev"]}, ev_minimum, env.ev_remaining)
    matrix = coo_matrix((values, (rows, columns)), shape=(len(lo), n)).tocsc()
    def solve(cost, active_matrix, active_lo, active_hi):
        active_constraint = LinearConstraint(active_matrix, active_lo, active_hi)
        answer = milp(cost, integrality=integer, bounds=Bounds(lower, upper), constraints=active_constraint,
                      options={"time_limit": 1.0, "mip_rel_gap": 0.001})
        accepted = answer.x is not None and np.isfinite(answer.x).all()
        residual = np.inf
        if accepted:
            candidate = answer.x
            ax = active_matrix @ candidate
            residual = float(max(np.max(np.maximum(active_lo - ax, 0.0)), np.max(np.maximum(ax - active_hi, 0.0)), np.max(np.maximum(lower - candidate, 0.0)), np.max(np.maximum(candidate - upper, 0.0)), np.max(np.abs(candidate[ix["charge_on"]] - np.round(candidate[ix["charge_on"]])))))
            accepted = residual <= 1e-5
        return answer, accepted, residual

    export_objective = np.zeros(n)
    export_objective[ix["over_export"]] = env.dt
    phase1 = None
    phase1_feasible = None
    phase1_violation = None
    phase2_feasible = None
    used_phase1_solution = False
    if export_priority:
        phase1, phase1_feasible, phase1_violation = solve(export_objective, matrix, np.asarray(lo), np.asarray(hi))
        if phase1_feasible:
            export_limit = float(export_objective @ phase1.x) + 1e-6
            priority_row = coo_matrix((np.full(H, env.dt), (np.zeros(H, int), ix["over_export"])), shape=(1, n)).tocsc()
            priority_matrix = vstack([matrix, priority_row], format='csc')
            result, feasible, violation = solve(objective, priority_matrix, np.r_[np.asarray(lo), -np.inf], np.r_[np.asarray(hi), export_limit])
            phase2_feasible = feasible
            if not feasible:
                result, feasible, violation = phase1, True, phase1_violation
                used_phase1_solution = True
        else:
            result, feasible, violation = phase1, False, phase1_violation
    else:
        result, feasible, violation = solve(objective, matrix, np.asarray(lo), np.asarray(hi))
    if feasible:
        x = result.x
    info = {
        "solver": "scipy.optimize.milp/HiGHS", "status": int(result.status), "message": str(result.message),
        "fallback": not feasible, "seconds": time.perf_counter() - started, "objective_horizon": H,
        "battery_feasibility_horizon": L, "mip_gap": float(result.mip_gap) if getattr(result, "mip_gap", None) is not None else None,
        "node_count": int(result.mip_node_count) if getattr(result, "mip_node_count", None) is not None else None,
        "constraint_violation_max": violation if np.isfinite(violation) else None,
        "peak_approximation": "convex piecewise-linear chords, maximum interval 10 kW",
        "peak_objective_error_bound": H * 0.00035 * env.dt * 10.0 ** 2 / 4.0,
        "export_revenue": 0.0, "pv_curtailment_control": False,
        "tail_scope": "battery feasibility only; no exogenous values outside the objective horizon",
        "planning_ramp_mode": "future relaxed; current command uses common execution box" if future_ramp_relaxed else "hard 45 kW throughout remaining day",
        "current_battery_bounds_kw": list(current_bounds[:2]) if current_bounds is not None else None,
        "objective_mode": "lexicographic export exceedance then operating objective" if export_priority else "operating objective",
        "phase1_status": int(phase1.status) if phase1 is not None else None,
        "phase1_feasible": bool(phase1_feasible) if phase1_feasible is not None else None,
        "phase1_export_kwh": float(export_objective @ phase1.x) if phase1_feasible else None,
        "phase1_constraint_violation_max": phase1_violation if phase1_feasible else None,
        "phase2_feasible": bool(phase2_feasible) if phase2_feasible is not None else None,
        "used_phase1_solution": used_phase1_solution,
    }
    if feasible:
        p = float(x[ix["dis"][0]] - x[ix["ch"][0]])
        ev = float(x[ix["ev"][0]] / env.ev_max) if active[0] else 0.0
        info.update({"objective": float(objective @ x), "terminal_soc_kwh": float(x[ix["soc"][-1]]),
                     "charge_discharge_overlap_kw": float(np.max(np.minimum(x[ix["ch"]], x[ix["dis"]]))),
                     "import_export_overlap_kw": float(np.max(np.minimum(x[ix["gi"]], x[ix["ge"]]))),
                     "max_ramp_kw": float(np.max(np.abs(np.diff(np.r_[env.prev_p, x[ix["dis"]] - x[ix["ch"]]])))),
                     "ev_planned_kwh": float(x[ix["ev"]].sum() * env.dt), "ev_required_within_horizon_kwh": ev_minimum,
                     "planned_first_grid_net_kw": float(x[ix["gi"][0]] - x[ix["ge"][0]])})
        raw = raw_from_physical(p, ev, 1.0, env.P)
    else:
        raw = rule_action(env)
        raw[2] = 1.0
        info["fallback_controller"] = "LegacyRule+QP; caller applies the common QP execution layer"
    env.mpc_last_info = info
    return raw
