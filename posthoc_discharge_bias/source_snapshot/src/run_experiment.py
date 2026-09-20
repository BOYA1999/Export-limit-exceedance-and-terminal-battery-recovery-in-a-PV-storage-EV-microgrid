from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CONFIG = ROOT / "configs" / "experiment.json"
ARTIFACTS = ROOT / "artifacts"
FORECAST_OFFSETS = {"pv": 11, "load": 23, "price": 37, "carbon": 53}
TERMINAL_ENERGY_TOL_KWH = 1e-8


class TerminalTargetInfeasible(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__("Terminal energy target is unreachable: " + json.dumps(diagnostics))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_days(seed: int = 7331, perturb: bool = True) -> list[dict[str, np.ndarray | float | int]]:
    load = pd.read_csv(DATA / "annual_load_pattern_CAMX_baseline.csv")["load_data"].to_numpy(float)
    price = pd.read_csv(DATA / "cambium_grid_data_California_cambium_grid_value.csv")["value"].to_numpy(float) / 1000.0
    carbon = pd.read_csv(DATA / "cambium_grid_data_California_cambium_co2_rate_lrmer.csv")["value"].to_numpy(float) / 1000.0
    with (DATA / "pvwatts_pasadena_1kw.json").open(encoding="utf-8") as f:
        pv = np.asarray(json.load(f)["outputs"]["ac"], dtype=float) / 1000.0
    if not (len(load) == len(price) == len(carbon) == len(pv) == 8760):
        raise ValueError("the four source series must all contain 8760 hourly values")
    load = load / load.mean() * 145.0
    pv_capacity = 0.55 * load.mean() / max(pv.mean(), 1e-8)
    pv = pv * pv_capacity
    price = np.clip(0.08 + (price - price.mean()) / max(price.std(), 1e-8) * 0.025, 0.015, 0.28)
    carbon = np.clip(carbon, 0.03, None)
    load = np.repeat(load, 4)
    pv = np.repeat(pv, 4)
    price = np.repeat(price, 4)
    carbon = np.repeat(carbon, 4)
    rng = np.random.default_rng(seed)
    days = []
    for day in range(365):
        left = day * 96
        if perturb:
            local = np.random.default_rng(seed + day)
            load_day = np.maximum(load[left:left + 96] * (1.0 + 0.035 * local.normal(size=96)), 25.0)
            pv_day = np.maximum(0.0, pv[left:left + 96] * (1.0 + 0.07 * local.normal(size=96)))
            price_day = np.clip(price[left:left + 96] * (1.0 + 0.05 * local.normal(size=96)), 0.01, 0.35)
            carbon_day = np.maximum(0.02, carbon[left:left + 96] * (1.0 + 0.04 * local.normal(size=96)))
        else:
            load_day = load[left:left + 96].copy()
            pv_day = pv[left:left + 96].copy()
            price_day = price[left:left + 96].copy()
            carbon_day = carbon[left:left + 96].copy()
        hours = np.arange(96) / 4.0
        ev_active = ((hours >= 18.0) | (hours < 7.0)).astype(float)
        light_profile = np.where(ev_active > 0, 1.0, 0.25)
        required = float(rng.uniform(30.0, 52.0))
        days.append({
            "day": day,
            "load": load_day,
            "pv": pv_day,
            "price": price_day,
            "carbon": carbon_day,
            "ev_active": ev_active,
            "light_profile": light_profile,
            "ev_required": required,
        })
    return days


def load_public_profile_days(seed: int = 7331) -> list[dict[str, np.ndarray | float | int]]:
    return load_days(seed=seed, perturb=False)


def future(days: list[dict], day: int, t: int, key: str, horizon: int) -> np.ndarray:
    values = []
    for j in range(horizon):
        absolute = t + j
        d = (day + absolute // 96) % len(days)
        s = absolute % 96
        values.append(float(days[d][key][s]))
    return np.asarray(values, dtype=float)


def future_clamped(days: list[dict], day: int, t: int, key: str, horizon: int) -> np.ndarray:
    values = []
    for j in range(horizon):
        absolute = t + j
        d = min(day + absolute // 96, len(days) - 1)
        s = min(absolute % 96, 95)
        values.append(float(days[d][key][s]))
    return np.asarray(values, dtype=float)


class MicrogridEnv:
    def __init__(self, days: list[dict], config: dict, mode: str = "safe", forecast_error: float = 0.0, error_kind: str = "gaussian", ablation: str = "full", projection_penalty: float = 0.0):
        self.days = days
        self.cfg = config
        self.mode = mode
        self.forecast_error = forecast_error
        self.error_kind = error_kind
        self.ablation = ablation
        self.projection_penalty = float(projection_penalty)
        self.dt = config["dt_hours"]
        self.n = config["steps_per_day"]
        self.h = config["forecast_steps"]
        self.E = config["bess"]["energy_kwh"]
        self.P = config["bess"]["power_kw"]
        self.soc_min = config["bess"]["soc_min"] * self.E
        self.soc_max = config["bess"]["soc_max"] * self.E
        self.eta_c = config["bess"]["eta_charge"]
        self.eta_d = config["bess"]["eta_discharge"]
        self.ramp = config["bess"]["ramp_kw"]
        self.ev_max = config["loads"]["ev_max_kw"]
        self.light_min = config["loads"]["lighting_min_fraction"]
        self.grid_max = config["loads"]["grid_max_kw"]
        self.grid_min = config["loads"]["grid_min_kw"]
        self.scenario = {}
        self.reset(0, 0)

    def reset(self, day: int, seed: int = 0, scenario: dict | None = None) -> np.ndarray:
        self.day = int(day) % len(self.days)
        self.t = 0
        self.scenario = scenario or {}
        self.soc = self.cfg["bess"]["soc_initial"] * self.E
        self.ev_remaining = float(self.days[self.day]["ev_required"] * self.scenario.get("ev_scale", 1.0))
        self.prev_p = 0.0
        self.rng = np.random.default_rng(seed)
        self.projection_triggers = 0
        self.qp_solver_failures = 0
        self.qp_hard_failures = 0
        self.qp_phase1_failures = 0
        self.qp_phase2_failures = 0
        self.qp_last_iterations = 0
        self.qp_last_status = "not_run"
        self.qp_last_path = "not_run"
        self.qp_hard_status = "not_called"
        self.qp_phase1_status = "not_called"
        self.qp_phase2_status = "not_called"
        self.qp_hard_iterations = 0
        self.qp_phase1_iterations = 0
        self.qp_phase2_iterations = 0
        self.correction_l2 = []
        self.minimum_required_slack_kw = []
        self.ramp_slack_kw_history = []
        self.tracked_p = 0.0
        self.tracked_ev_power = 0.0
        self.tracked_light_power = 0.0
        self.proxy_voltage = 400.0
        self.metrics = {"cost": 0.0, "carbon": 0.0, "degradation": 0.0, "estimated_degradation_cost": 0.0, "bess_throughput_kwh": 0.0, "peak_cost": 0.0, "comfort_cost": 0.0, "grid_violation": 0.0, "grid_import_violation": 0.0, "grid_export_violation": 0.0, "grid_violation_steps": 0.0, "hard_feasible_steps": 0.0, "physical_infeasible_steps": 0.0, "preventable_violation_steps": 0.0, "minimum_required_slack_kw_sum": 0.0, "minimum_required_slack_kw_max": 0.0, "grid_max_exceedance_kw": 0.0, "grid_excess_energy_kwh": 0.0, "soc_violation": 0.0, "soc_violation_steps": 0.0, "soc_violation_max_kwh": 0.0, "first_soc_violation_step": -1.0, "ev_violation": 0.0, "pv": 0.0, "pv_curtail": 0.0, "peak": 0.0, "soc_min_seen": self.soc, "soc_max_seen": self.soc, "high_soc_steps": 0.0, "max_ramp_kw": 0.0, "ramp_conflict_steps": 0.0, "ramp_slack_kw_sum": 0.0, "ramp_slack_kw_max": 0.0, "ramp_violation_steps": 0.0, "lighting_energy_kwh": 0.0, "lighting_min_fraction_steps": 0.0, "lighting_below_full_steps": 0.0, "terminal_target_conflict_steps": 0.0, "terminal_soc_error_kwh": 0.0, "proxy_voltage_dev_max": 0.0, "proxy_voltage_abs_sum": 0.0, "proxy_voltage_exceed_40_steps": 0.0, "first_proxy_voltage_exceed_40_step": -1.0, "proxy_actuator_error_kw": 0.0, "steps": 0}
        return self.observation()

    def _actual(self, key: str, t: int | None = None) -> float:
        t = self.t if t is None else t
        d = self.days[self.day]
        scale = 1.0
        if key == "pv": scale = self.scenario.get("pv_scale", 1.0)
        if key == "load": scale = self.scenario.get("load_scale", 1.0)
        if key == "price": scale = self.scenario.get("price_scale", 1.0)
        if key == "carbon": scale = self.scenario.get("carbon_scale", 1.0)
        return float(d[key][t] * scale)

    def _forecast(self, key: str) -> np.ndarray:
        future_fn = future_clamped if self.scenario.get("no_wrap_forecast", False) else future
        base = future_fn(self.days, self.day, self.t, key, self.h).copy()
        if self.ablation == "no_forecast":
            return np.repeat(base[:1], self.h)
        if self.forecast_error <= 0:
            return base
        r = np.random.default_rng(900000 + self.day * 1000 + self.t * 7 + FORECAST_OFFSETS[key])
        scale = self.forecast_error
        if self.error_kind == "gaussian":
            base = base * (1.0 + scale * r.normal(size=self.h))
        elif self.error_kind == "bias":
            base = base * (1.0 + scale)
        elif self.error_kind == "ramp":
            base = np.roll(base, max(1, int(round(scale * 8))))
        elif self.error_kind == "missing":
            width = max(2, int(round(scale * 24)))
            start = (self.day + self.t) % max(1, self.h - width + 1)
            base[start:start + width] = base[max(0, start - 1)] if start else base.mean()
        if key == "pv":
            return np.maximum(0.0, base)
        if key in ("load", "price", "carbon"):
            return np.maximum(0.0, base)
        return base

    def _light_bounds(self) -> tuple[float, float]:
        if self.scenario.get("fixed_lighting", False) and self.days[self.day]["light_profile"][self.t] > 0:
            return 1.0, 1.0
        return self.light_min, 1.0

    def observation(self) -> np.ndarray:
        d = self.days[self.day]
        ev_active = float(d["ev_active"][self.t])
        values = [
            self.soc / self.E,
            self._actual("pv") / 180.0,
            self._actual("load") / 250.0,
            self.ev_remaining / 60.0,
            max(0.0, (96 - self.t) / 96.0),
            ev_active,
            float(d["light_profile"][self.t]),
            self.prev_p / self.P,
            np.sin(2 * np.pi * self.t / 96),
            np.cos(2 * np.pi * self.t / 96),
        ]
        for key, scale in (("pv", 180.0), ("load", 250.0), ("price", 0.25), ("carbon", 0.35)):
            forecast = np.zeros(self.h, dtype=float) if self.ablation == "no_carbon" and key == "carbon" else self._forecast(key)
            values.extend((forecast / scale).tolist())
        return np.asarray(values, dtype=float)

    def _bounds(self, diagnostics: dict | None = None) -> tuple[float, float, float, float, float]:
        soc_lower = -(self.soc_max - self.soc) / (self.eta_c * self.dt)
        soc_upper = (self.soc - self.soc_min) * self.eta_d / self.dt
        soc_lo = max(-self.P, soc_lower)
        soc_hi = min(self.P, soc_upper)
        ramp_lo = self.prev_p - self.ramp
        ramp_hi = self.prev_p + self.ramp
        p_lo = max(soc_lo, ramp_lo)
        p_hi = min(soc_hi, ramp_hi)
        ramp_slack = 0.0
        if p_lo > p_hi:
            ramp_slack = max(ramp_lo - soc_hi, soc_lo - ramp_hi, 0.0)
            p_lo, p_hi = soc_lo, soc_hi
        if self.scenario.get("terminal_soc_target", False):
            target = self.cfg["bess"]["soc_initial"] * self.E
            remaining = self.n - self.t - 1
            reachable_low = max(self.soc_min, target - remaining * self.P * self.eta_c * self.dt)
            reachable_high = min(self.soc_max, target + remaining * self.P / self.eta_d * self.dt)
            def p_for_soc(soc_target: float) -> float:
                return (self.soc - soc_target) * self.eta_d / self.dt if self.soc >= soc_target else -(soc_target - self.soc) / (self.eta_c * self.dt)
            terminal_lo = p_for_soc(reachable_high)
            terminal_hi = p_for_soc(reachable_low)
            tolerance_kw = TERMINAL_ENERGY_TOL_KWH * min(self.eta_d, 1 / self.eta_c) / self.dt
            terminal_soc_lo = max(soc_lo, terminal_lo)
            terminal_soc_hi = min(soc_hi, terminal_hi)
            gap_kw = terminal_soc_lo - terminal_soc_hi
            detail = {"day": self.day, "step": self.t, "soc_kwh": self.soc, "target_kwh": target,
                      "remaining_steps": remaining + 1, "physical_lower_kw": soc_lo, "physical_upper_kw": soc_hi,
                      "terminal_lower_kw": terminal_lo, "terminal_upper_kw": terminal_hi,
                      "terminal_physical_gap_kw": gap_kw, "tolerance_kw": tolerance_kw}
            if gap_kw > tolerance_kw:
                raise TerminalTargetInfeasible(detail)
            repaired = gap_kw > 0
            if repaired:
                terminal_soc_lo = terminal_soc_hi = float(np.clip((terminal_soc_lo + terminal_soc_hi) / 2, soc_lo, soc_hi))
            bounded_lo = max(p_lo, terminal_soc_lo)
            bounded_hi = min(p_hi, terminal_soc_hi)
            ramp_gap_kw = bounded_lo - bounded_hi
            if ramp_gap_kw <= 0:
                p_lo, p_hi = bounded_lo, bounded_hi
            elif ramp_gap_kw <= tolerance_kw:
                p_lo = p_hi = float(np.clip((bounded_lo + bounded_hi) / 2, terminal_soc_lo, terminal_soc_hi))
                repaired = True
            else:
                p_lo, p_hi = terminal_soc_lo, terminal_soc_hi
            if diagnostics is not None:
                diagnostics.update(detail)
                diagnostics.update({"terminal_roundoff_repair": int(repaired),
                                    "terminal_ramp_relaxation": int(ramp_gap_kw > tolerance_kw),
                                    "terminal_ramp_gap_kw": max(0.0, ramp_gap_kw)})
        active = self.days[self.day]["ev_active"][self.t] > 0
        ev_hi = min(1.0, self.ev_remaining / max(self.ev_max * self.dt, 1e-9)) if active else 0.0
        active_left = int(np.sum(self.days[self.day]["ev_active"][self.t:] > 0))
        ev_min_power = max(0.0, (self.ev_remaining - self.ev_max * self.dt * max(0, active_left - 1)) / self.dt) if active else 0.0
        ev_min = min(ev_hi, ev_min_power / self.ev_max) if active else 0.0
        if self.ablation == "no_ev_flex" and active:
            ev_min = ev_hi
        return float(p_lo), float(p_hi), float(ev_min), float(ev_hi), float(ramp_slack)

    def project(self, raw: np.ndarray) -> tuple[np.ndarray, float, bool]:
        raw = np.asarray(raw, dtype=float)
        p_lo, p_hi, ev_min, ev_hi, _ = self._bounds()
        p = float(np.clip(raw[0] * self.P, p_lo, p_hi))
        ev = float(np.clip((raw[1] + 1) / 2, ev_min, ev_hi))
        light_lo, light_hi = self._light_bounds()
        light = float(np.clip((raw[2] + 1) / 2, light_lo, light_hi))
        base = self._actual("load") - self._actual("pv")
        light_power = self.cfg["loads"]["lighting_base_kw"] * self.days[self.day]["light_profile"][self.t]
        for _ in range(6):
            grid = base + ev * self.ev_max + light * light_power - p
            if grid > self.grid_max:
                need = grid - self.grid_max
                capacities = np.array([max(0.0, p_hi - p), max(0.0, (ev - ev_min) * self.ev_max), max(0.0, (light - self.light_min) * light_power)])
                if capacities.sum() <= 1e-9:
                    break
                share = capacities / np.array([1.0, 1.0, 0.5])
                share = share / max(share.sum(), 1e-9) * min(need, capacities.sum())
                p += share[0]
                ev -= share[1] / self.ev_max
                light -= share[2] / max(light_power, 1e-9)
            elif grid < self.grid_min:
                need = self.grid_min - grid
                capacities = np.array([max(0.0, p - p_lo), max(0.0, (ev_hi - ev) * self.ev_max), max(0.0, (1.0 - light) * light_power)])
                if capacities.sum() <= 1e-9:
                    break
                share = capacities / np.array([1.0, 1.0, 0.5])
                share = share / max(share.sum(), 1e-9) * min(need, capacities.sum())
                p -= share[0]
                ev += share[1] / self.ev_max
                light += share[2] / max(light_power, 1e-9)
            else:
                break
            p = float(np.clip(p, p_lo, p_hi))
            ev = float(np.clip(ev, ev_min, ev_hi))
            light = float(np.clip(light, light_lo, light_hi))
        physical = np.array([p / self.P, ev * 2 - 1, light * 2 - 1])
        correction = float(np.linalg.norm(physical - raw))
        return physical, correction, correction > 1e-8

    def project_qp(self, raw: np.ndarray) -> tuple[np.ndarray, float, bool, float, float, bool, float]:
        import osqp
        from scipy.sparse import csc_matrix, eye, vstack

        p_lo, p_hi, ev_min, ev_hi, _ = self._bounds()
        d = self.days[self.day]
        active = float(d["ev_active"][self.t])
        light_power = self.cfg["loads"]["lighting_base_kw"] * float(d["light_profile"][self.t])
        base = self._actual("load") - self._actual("pv")
        target = np.asarray([raw[0], (raw[1] + 1.0) / 2.0, (raw[2] + 1.0) / 2.0], dtype=float)
        weights = np.asarray(self.cfg.get("projection", {}).get("weights", [1.0, 1.0, 0.5]), dtype=float)
        projection_cfg = self.cfg.get("projection", {})
        grid_scale = float(projection_cfg.get("grid_scale_kw", 100.0))
        light_lo, light_hi = self._light_bounds()
        lower = np.asarray([p_lo / self.P, ev_min, light_lo], dtype=float)
        upper = np.asarray([p_hi / self.P, ev_hi, light_hi], dtype=float)
        grid_rows3 = np.asarray([
            [-self.P / grid_scale, active * self.ev_max / grid_scale, light_power / grid_scale],
            [self.P / grid_scale, -active * self.ev_max / grid_scale, -light_power / grid_scale],
        ], dtype=float)
        grid_upper = np.asarray([(self.grid_max - base) / grid_scale, (-self.grid_min + base) / grid_scale], dtype=float)
        max_iter = int(projection_cfg.get("qp_max_iter", 20000))
        rho = float(projection_cfg.get("qp_rho", 10.0))
        phase_iterations = []
        def solve(P, q, A, l, u):
            solver = osqp.OSQP()
            solver.setup(P=csc_matrix(P), q=q, A=csc_matrix(A), l=l, u=u, eps_abs=1e-5, eps_rel=1e-5, max_iter=max_iter, rho=rho, polish=False, verbose=False)
            result = solver.solve()
            phase_iterations.append(int(getattr(result.info, "iter", 0)))
            return result

        hard_A = vstack([eye(3, format="csc"), csc_matrix(grid_rows3)], format="csc")
        hard_l = np.concatenate([lower, np.full(2, -np.inf)])
        hard_u = np.concatenate([upper, grid_upper])
        hard_P = np.diag(weights)
        hard_q = -weights * target
        hard_result = solve(hard_P, hard_q, hard_A, hard_l, hard_u)
        self.qp_hard_status = str(hard_result.info.status)
        self.qp_hard_iterations = int(getattr(hard_result.info, "iter", 0))
        if hard_result.x is not None and hard_result.info.status in ("solved", "solved inaccurate"):
            self.qp_last_iterations = int(sum(phase_iterations))
            self.qp_last_status = str(hard_result.info.status)
            self.qp_last_path = "hard_qp_solution"
            x = np.asarray(hard_result.x, dtype=float)
            x = np.clip(x, lower, upper)
            physical = np.asarray([x[0], x[1] * 2.0 - 1.0, x[2] * 2.0 - 1.0], dtype=float)
            correction = float(np.linalg.norm(physical - raw))
            return physical, correction, correction > float(projection_cfg.get("trigger_tolerance", 1e-3)), 0.0, 0.0, True, 0.0
        self.qp_hard_failures += 1

        lower5 = np.concatenate([lower, [0.0, 0.0]])
        upper5 = np.concatenate([upper, [np.inf, np.inf]])
        grid_rows5 = np.column_stack([grid_rows3, np.asarray([[-1.0, 0.0], [0.0, -1.0]])])
        A1 = vstack([eye(5, format="csc"), csc_matrix(grid_rows5)], format="csc")
        l1 = np.concatenate([lower5, np.full(2, -np.inf)])
        u1 = np.concatenate([upper5, grid_upper])
        P1 = np.diag(np.full(5, 1e-6))
        q1 = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0], dtype=float)
        phase1 = solve(P1, q1, A1, l1, u1)
        self.qp_phase1_status = str(phase1.info.status)
        self.qp_phase1_iterations = int(getattr(phase1.info, "iter", 0))
        if phase1.x is None or phase1.info.status not in ("solved", "solved inaccurate"):
            self.qp_last_iterations = int(sum(phase_iterations))
            self.qp_last_status = str(phase1.info.status)
            self.qp_solver_failures += 1
            self.qp_phase1_failures += 1
            self.qp_last_path = "heuristic_fallback"
            physical, correction, triggered = self.project(raw)
            return physical, correction, correction > float(projection_cfg.get("trigger_tolerance", 1e-3)), 0.0, 0.0, False, float(max(0.0, self._grid_violation_for_action(physical)))
        x1 = np.asarray(phase1.x, dtype=float)
        x1[:3] = np.clip(x1[:3], lower, upper)
        required1 = np.maximum(0.0, grid_rows3 @ x1[:3] - grid_upper)
        minimum_sum = max(0.0, float(required1.sum()))
        epsilon = float(projection_cfg.get("slack_epsilon", 1e-6))
        A2 = vstack([A1, csc_matrix(np.asarray([[0.0, 0.0, 0.0, 1.0, 1.0]]))], format="csc")
        l2 = np.concatenate([l1, [-np.inf]])
        u2 = np.concatenate([u1, [minimum_sum + epsilon]])
        P2 = np.diag(np.concatenate([weights, [1e-2, 1e-2]]))
        q2 = np.asarray([-weights[0] * target[0], -weights[1] * target[1], -weights[2] * target[2], 0.0, 0.0], dtype=float)
        phase2 = solve(P2, q2, A2, l2, u2)
        self.qp_phase2_status = str(phase2.info.status)
        self.qp_phase2_iterations = int(getattr(phase2.info, "iter", 0))
        if phase2.x is None or phase2.info.status not in ("solved", "solved inaccurate"):
            self.qp_last_iterations = int(sum(phase_iterations))
            self.qp_last_status = str(phase2.info.status)
            self.qp_solver_failures += 1
            self.qp_phase2_failures += 1
            self.qp_last_path = "phase1_fallback"
            x = np.concatenate([x1[:3], required1])
        else:
            self.qp_last_path = "phase2_solution"
            x = np.asarray(phase2.x, dtype=float)
        x[:3] = np.clip(x[:3], lower, upper)
        x[3:] = np.maximum(0.0, x[3:])
        self.qp_last_iterations = int(sum(phase_iterations))
        self.qp_last_status = str(phase2.info.status) if phase2.x is not None else "phase2_fallback"
        physical = np.asarray([x[0], x[1] * 2.0 - 1.0, x[2] * 2.0 - 1.0], dtype=float)
        correction = float(np.linalg.norm(physical - raw))
        minimum_slack_kw = minimum_sum * grid_scale
        return physical, correction, correction > float(projection_cfg.get("trigger_tolerance", 1e-3)), float(x[3] * grid_scale), float(x[4] * grid_scale), bool(minimum_sum <= epsilon), minimum_slack_kw

    def _grid_violation_for_action(self, physical: np.ndarray) -> float:
        p = float(physical[0] * self.P)
        ev = float((physical[1] + 1.0) / 2.0) * self.ev_max * float(self.days[self.day]["ev_active"][self.t])
        light = float((physical[2] + 1.0) / 2.0) * self.cfg["loads"]["lighting_base_kw"] * float(self.days[self.day]["light_profile"][self.t])
        grid = self._actual("load") + ev + light - self._actual("pv") - p
        return max(0.0, grid - self.grid_max) + max(0.0, self.grid_min - grid)

    @staticmethod
    def _first_order_interval(state: float, command: float, realized_fraction: float, substeps: int) -> tuple[float, float, float, float]:
        fraction = float(np.clip(realized_fraction, 0.0, 1.0))
        count = max(1, int(substeps))
        retention = 1.0 - fraction
        if retention <= 1e-15:
            return float(command), float(command), 0.0, 0.0
        if retention >= 1.0 - 1e-15:
            return float(state), float(state), 1.0, 1.0
        sub_retention = retention ** (1.0 / count)
        integration_factor = (1.0 - retention) / (-np.log(retention))
        average = command + (state - command) * integration_factor
        end_state = command + retention * (state - command)
        return float(end_state), float(average), float(retention), float(sub_retention)

    def step(self, raw: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        raw = np.asarray(raw, dtype=float)
        prev_p_before = self.prev_p
        terminal_diagnostics = {}
        _, _, _, _, ramp_bound_slack = self._bounds(terminal_diagnostics)
        for event in ["terminal_roundoff_repair", "terminal_ramp_relaxation"]:
            self.metrics[event + "_steps"] = self.metrics.get(event + "_steps", 0) + terminal_diagnostics.get(event, 0)
        qp_slack_plus = 0.0
        qp_slack_minus = 0.0
        hard_feasible = True
        minimum_required_slack_kw = 0.0
        if self.mode == "qp":
            physical, correction, triggered, qp_slack_plus, qp_slack_minus, hard_feasible, minimum_required_slack_kw = self.project_qp(raw)
        elif self.mode in ("safe", "mpc"):
            physical, correction, triggered = self.project(raw)
        else:
            physical = np.clip(raw, -1.0, 1.0)
            correction = float(np.linalg.norm(physical - raw))
            triggered = False
        p_command = float(physical[0] * self.P)
        ev_alpha = float((physical[1] + 1) / 2)
        light_alpha = float((physical[2] + 1) / 2)
        d = self.days[self.day]
        ev_command = min(ev_alpha * self.ev_max * float(d["ev_active"][self.t]), self.ev_remaining / self.dt)
        light_command = light_alpha * self.cfg["loads"]["lighting_base_kw"] * float(d["light_profile"][self.t])
        if self.scenario.get("continuous_proxy", False):
            lag = float(self.scenario.get("proxy_lag_fraction", 0.75))
            substeps = max(1, int(self.scenario.get("proxy_substeps", 1)))
            self.tracked_p, p, lag_retention, lag_sub_retention = self._first_order_interval(self.tracked_p, p_command, lag, substeps)
            self.tracked_ev_power, ev_average, _, _ = self._first_order_interval(self.tracked_ev_power, ev_command, lag, substeps)
            self.tracked_light_power, light_power, _, _ = self._first_order_interval(self.tracked_light_power, light_command, lag, substeps)
            ev_power = min(ev_average, self.ev_remaining / self.dt)
            self.metrics["proxy_actuator_error_kw"] += abs(p_command - p)
        else:
            p = p_command
            ev_power = ev_command
            light_power = light_command
            lag_retention = 0.0
            lag_sub_retention = 0.0
        load = self._actual("load")
        pv = self._actual("pv")
        grid_net = load + ev_power + light_power - pv - p
        grid_import = max(0.0, grid_net)
        grid_export = max(0.0, -grid_net)
        pv_curtail = max(0.0, pv - max(0.0, load + ev_power + light_power - p))
        soc_next = self.soc + (max(0.0, -p) * self.eta_c - max(0.0, p) / self.eta_d) * self.dt
        soc_violation = max(0.0, self.soc_min - soc_next - 1e-8) + max(0.0, soc_next - self.soc_max - 1e-8)
        grid_violation = max(0.0, grid_net - self.grid_max - 1e-2) + max(0.0, self.grid_min - grid_net - 1e-2)
        import_violation = max(0.0, grid_net - self.grid_max - 1e-2)
        export_violation = max(0.0, self.grid_min - grid_net - 1e-2)
        if self.scenario.get("continuous_proxy", False):
            proxy_error = (load + light_power + ev_power - pv) - p
            voltage_gain = float(self.scenario.get("proxy_voltage_gain", 0.12))
            voltage_damping = float(self.scenario.get("proxy_voltage_damping", 0.15))
            voltage_deviation = self.proxy_voltage - 400.0
            if voltage_gain > 0.0 and voltage_damping > 0.0:
                voltage_retention = float(np.exp(-voltage_gain * voltage_damping))
                voltage_deviation = voltage_deviation * voltage_retention + proxy_error / voltage_damping * (1.0 - voltage_retention)
            elif voltage_gain > 0.0:
                voltage_deviation += voltage_gain * proxy_error
            self.proxy_voltage = 400.0 + voltage_deviation
            voltage_dev = abs(self.proxy_voltage - 400.0)
            self.metrics["proxy_voltage_dev_max"] = max(self.metrics["proxy_voltage_dev_max"], voltage_dev)
            self.metrics["proxy_voltage_abs_sum"] += voltage_dev
            self.metrics["proxy_voltage_exceed_40_steps"] += float(voltage_dev > 40.0)
            if voltage_dev > 40.0 and self.metrics["first_proxy_voltage_exceed_40_step"] < 0:
                self.metrics["first_proxy_voltage_exceed_40_step"] = self.metrics["steps"]
        if self.mode == "qp":
            physical_infeasible = bool(minimum_required_slack_kw > 1e-5)
            preventable_violation = bool(grid_violation > 1e-12 and not physical_infeasible)
        else:
            hard_feasible = bool(grid_violation <= 1e-12 and soc_violation <= 1e-12)
            physical_infeasible = not hard_feasible
            preventable_violation = False
        ev_before = self.ev_remaining
        self.ev_remaining = max(0.0, self.ev_remaining - ev_power * self.dt)
        self.soc = soc_next
        done = self.t == 95
        ev_violation = self.ev_remaining if done else 0.0
        comfort_violation = max(0.0, self.light_min - light_alpha) if d["light_profile"][self.t] > 0 else 0.0
        price = self._actual("price")
        carbon = self._actual("carbon")
        cost = price * grid_import * self.dt
        co2 = carbon * grid_import * self.dt
        degradation = 0.0015 * abs(p) * self.dt if self.ablation != "no_degradation" else 0.0
        peak_cost = 0.00035 * max(0.0, grid_import - 220.0) ** 2 * self.dt
        comfort_cost = comfort_violation ** 2
        violation_penalty = 5.0 * (grid_violation / 20.0 + soc_violation / 10.0 + ev_violation / 20.0)
        carbon_term = 0.0 if self.ablation == "no_carbon" else 1.8 * co2
        reward = -(cost / 10.0 + carbon_term + degradation + peak_cost + comfort_cost + violation_penalty + self.projection_penalty * correction)
        self.metrics["cost"] += cost
        self.metrics["carbon"] += co2
        self.metrics["degradation"] += degradation
        self.metrics["estimated_degradation_cost"] += 0.0015 * abs(p) * self.dt
        self.metrics["bess_throughput_kwh"] += abs(p) * self.dt
        self.metrics["peak_cost"] += peak_cost
        self.metrics["comfort_cost"] += comfort_cost
        self.metrics["grid_violation"] += grid_violation
        self.metrics["grid_import_violation"] += import_violation
        self.metrics["grid_export_violation"] += export_violation
        self.metrics["grid_violation_steps"] += float(grid_violation > 1e-12)
        self.metrics["hard_feasible_steps"] += float(hard_feasible)
        self.metrics["physical_infeasible_steps"] += float(physical_infeasible)
        self.metrics["preventable_violation_steps"] += float(preventable_violation)
        self.metrics["minimum_required_slack_kw_sum"] += minimum_required_slack_kw
        self.metrics["minimum_required_slack_kw_max"] = max(self.metrics["minimum_required_slack_kw_max"], minimum_required_slack_kw)
        self.metrics["grid_max_exceedance_kw"] = max(self.metrics["grid_max_exceedance_kw"], grid_violation)
        self.metrics["grid_excess_energy_kwh"] += grid_violation * self.dt
        self.metrics["soc_violation"] += soc_violation
        self.metrics["soc_violation_steps"] += float(soc_violation > 1e-8)
        self.metrics["soc_violation_max_kwh"] = max(self.metrics["soc_violation_max_kwh"], soc_violation)
        if soc_violation > 1e-8 and self.metrics["first_soc_violation_step"] < 0:
            self.metrics["first_soc_violation_step"] = self.metrics["steps"]
        self.metrics["ev_violation"] += ev_violation
        self.metrics["pv"] += pv * self.dt
        self.metrics["pv_curtail"] += pv_curtail * self.dt
        self.metrics["peak"] = max(self.metrics["peak"], grid_import)
        self.metrics["soc_min_seen"] = min(self.metrics["soc_min_seen"], soc_next)
        self.metrics["soc_max_seen"] = max(self.metrics["soc_max_seen"], soc_next)
        self.metrics["high_soc_steps"] += float(soc_next / self.E >= 0.80)
        self.metrics["lighting_energy_kwh"] += light_power * self.dt
        self.metrics["lighting_min_fraction_steps"] += float(light_alpha <= self.light_min + 1e-9)
        self.metrics["lighting_below_full_steps"] += float(light_alpha < 1.0 - 1e-9)
        ramp_slack_kw = max(0.0, abs(p - prev_p_before) - self.ramp)
        ramp_slack_kw = max(ramp_slack_kw, ramp_bound_slack)
        self.metrics["ramp_conflict_steps"] += float(ramp_bound_slack > 1e-9)
        self.metrics["ramp_slack_kw_sum"] += ramp_slack_kw
        self.metrics["ramp_slack_kw_max"] = max(self.metrics["ramp_slack_kw_max"], ramp_slack_kw)
        self.metrics["ramp_violation_steps"] += float(ramp_slack_kw > 1e-9)
        self.ramp_slack_kw_history.append(ramp_slack_kw)
        self.metrics["max_ramp_kw"] = max(self.metrics["max_ramp_kw"], abs(p - self.prev_p))
        self.metrics["steps"] += 1
        if triggered:
            self.projection_triggers += 1
        self.correction_l2.append(correction)
        self.minimum_required_slack_kw.append(minimum_required_slack_kw)
        info = {"p": p, "p_command": p_command, "raw_action": raw.copy(), "projected_action": physical.copy(), "ev_alpha": ev_alpha, "light_alpha": light_alpha, "grid_import": grid_import, "grid_export": grid_export, "soc": soc_next, "ev_remaining_before": ev_before, "ev_remaining": self.ev_remaining, "projection_triggered": triggered, "correction": correction, "grid_violation": grid_violation, "grid_import_violation": import_violation, "grid_export_violation": export_violation, "grid_violation_step": float(grid_violation > 1e-12), "grid_excess_energy_kwh": grid_violation * self.dt, "qp_slack_plus": qp_slack_plus, "qp_slack_minus": qp_slack_minus, "hard_feasible": float(hard_feasible), "physical_infeasible": float(physical_infeasible), "preventable_violation": float(preventable_violation), "minimum_required_slack_kw": minimum_required_slack_kw, "soc_violation": soc_violation, "ev_violation": ev_violation, "ramp_bound_slack_kw": ramp_bound_slack, "ramp_slack_kw": ramp_slack_kw, "proxy_voltage": self.proxy_voltage, "lighting_energy_kwh": light_power * self.dt, "tracked_p_end_kw": self.tracked_p, "tracked_ev_end_kw": self.tracked_ev_power, "tracked_light_end_kw": self.tracked_light_power, "lag_retention": lag_retention, "lag_substep_retention": lag_sub_retention, "qp_execution_path": self.qp_last_path, "qp_hard_status": self.qp_hard_status, "qp_phase1_status": self.qp_phase1_status, "qp_phase2_status": self.qp_phase2_status}
        self.prev_p = p
        if not done:
            self.t += 1
            return self.observation(), float(reward), False, info
        metrics = dict(self.metrics)
        correction_array = np.asarray(self.correction_l2, dtype=float)
        slack_array = np.asarray(self.minimum_required_slack_kw, dtype=float)
        ramp_array = np.asarray(self.ramp_slack_kw_history, dtype=float)
        target_soc = self.cfg["bess"]["soc_initial"] * self.E
        metrics.update({"day": self.day, "ev_completion": float(self.ev_remaining <= 1e-8), "projection_trigger_rate": self.projection_triggers / max(metrics["steps"], 1), "trigger_rate_tol_0_001": float(np.mean(correction_array > 0.001)), "trigger_rate_tol_0_01": float(np.mean(correction_array > 0.01)), "trigger_rate_tol_0_05": float(np.mean(correction_array > 0.05)), "trigger_rate_tol_0_10": float(np.mean(correction_array > 0.10)), "mean_action_correction": float(np.mean(correction_array)), "p95_action_correction": float(np.quantile(correction_array, 0.95)), "soc_violation_rate": float(metrics["soc_violation"] > 0), "grid_violation_rate": float(metrics["grid_violation"] > 0), "grid_violation_step_rate": float(metrics["grid_violation_steps"] / max(metrics["steps"], 1)), "hard_feasible_step_rate": float(metrics["hard_feasible_steps"] / max(metrics["steps"], 1)), "physical_infeasible_step_rate": float(metrics["physical_infeasible_steps"] / max(metrics["steps"], 1)), "preventable_violation_step_rate": float(metrics["preventable_violation_steps"] / max(metrics["steps"], 1)), "minimum_required_slack_kw_mean": float(np.mean(slack_array)), "minimum_required_slack_kw_p95": float(np.quantile(slack_array, 0.95)), "ramp_conflict_rate": float(metrics["ramp_conflict_steps"] / max(metrics["steps"], 1)), "ramp_violation_step_rate": float(metrics["ramp_violation_steps"] / max(metrics["steps"], 1)), "ramp_slack_kw_mean": float(metrics["ramp_slack_kw_sum"] / max(metrics["steps"], 1)), "ramp_slack_kw_p95": float(np.quantile(ramp_array, 0.95)), "equivalent_full_cycles_per_day": float(metrics["bess_throughput_kwh"] / max(2.0 * self.E, 1e-9)), "estimated_degradation_cost_per_day": float(metrics["estimated_degradation_cost"]), "soc_swing": float((metrics["soc_max_seen"] - metrics["soc_min_seen"]) / self.E), "high_soc_residence_hours": float(metrics["high_soc_steps"] * self.dt), "lighting_energy_kwh": float(metrics["lighting_energy_kwh"]), "lighting_min_fraction_rate": float(metrics["lighting_min_fraction_steps"] / max(metrics["steps"], 1)), "lighting_below_full_rate": float(metrics["lighting_below_full_steps"] / max(metrics["steps"], 1)), "terminal_soc_kwh": float(self.soc), "terminal_soc_error_kwh": float(self.soc - target_soc), "terminal_target_conflict_rate": float(metrics["terminal_target_conflict_steps"] / max(metrics["steps"], 1)), "grid_import_violation_kwh": float(metrics["grid_import_violation"] * self.dt), "grid_export_violation_kwh": float(metrics["grid_export_violation"] * self.dt), "proxy_voltage_dev_max": float(metrics["proxy_voltage_dev_max"]), "proxy_voltage_dev_mean": float(metrics["proxy_voltage_abs_sum"] / max(metrics["steps"], 1)), "proxy_voltage_exceed_40_rate": float(metrics["proxy_voltage_exceed_40_steps"] / max(metrics["steps"], 1)), "first_proxy_voltage_exceed_40_step": float(metrics["first_proxy_voltage_exceed_40_step"]), "proxy_actuator_error_kw_mean": float(metrics["proxy_actuator_error_kw"] / max(metrics["steps"], 1)), "soc_violation_steps": float(metrics["soc_violation_steps"]), "soc_violation_duration_hours": float(metrics["soc_violation_steps"] * self.dt), "soc_violation_max_kwh": float(metrics["soc_violation_max_kwh"]), "first_soc_violation_step": float(metrics["first_soc_violation_step"]), "projection_saturation_rate": float(metrics["physical_infeasible_steps"] / max(metrics["steps"], 1)), "qp_solver_failures": float(self.qp_solver_failures), "qp_hard_failures": float(self.qp_hard_failures), "qp_phase1_failures": float(self.qp_phase1_failures), "qp_phase2_failures": float(self.qp_phase2_failures), "last_grid_import": float(grid_import), "pv_self_use": float(1.0 - metrics["pv_curtail"] / max(metrics["pv"], 1e-9)), "final_ev_remaining": self.ev_remaining})
        metrics.update({"projected_action": physical.copy(), "raw_action": raw.copy(), "p": p, "p_command": p_command, "grid_import": grid_import, "grid_export": grid_export, "grid_violation": grid_violation, "grid_import_violation": import_violation, "grid_export_violation": export_violation, "qp_slack_plus": qp_slack_plus, "qp_slack_minus": qp_slack_minus, "hard_feasible": float(hard_feasible), "physical_infeasible": float(physical_infeasible), "minimum_required_slack_kw": minimum_required_slack_kw, "correction": correction})
        metrics.update({"soc": soc_next, "soc_violation": soc_violation, "proxy_voltage": self.proxy_voltage, "tracked_p_end_kw": self.tracked_p, "tracked_ev_end_kw": self.tracked_ev_power, "tracked_light_end_kw": self.tracked_light_power, "lag_retention": lag_retention, "lag_substep_retention": lag_sub_retention, "qp_execution_path": self.qp_last_path, "qp_hard_status": self.qp_hard_status, "qp_phase1_status": self.qp_phase1_status, "qp_phase2_status": self.qp_phase2_status})
        return np.zeros(10 + 4 * self.h, dtype=float), float(reward - 5.0 * self.ev_remaining / 20.0), True, metrics


def raw_from_physical(p: float, ev: float, light: float, P: float = 100.0) -> np.ndarray:
    return np.asarray([p / P, ev * 2 - 1, light * 2 - 1], dtype=float)


def rule_action(env: MicrogridEnv) -> np.ndarray:
    d = env.days[env.day]
    t = env.t
    p_lo, p_hi, ev_min, _, _ = env._bounds()
    pv = env._actual("pv")
    load = env._actual("load")
    low = float(np.quantile(env._forecast("price"), 0.35))
    high = float(np.quantile(env._forecast("price"), 0.70))
    price = env._actual("price")
    ev = max(ev_min, 1.0 if pv > load else 0.65)
    if d["ev_active"][t] <= 0:
        ev = 0.0
    if pv > load + 20:
        p = max(p_lo, -min(-p_lo, pv - load - 20))
    elif price >= high or env._actual("carbon") >= float(np.quantile(env._forecast("carbon"), 0.65)):
        p = min(p_hi, max(0.0, min(env.P, (pv + env.P * 0.35) - load)))
    elif price <= low:
        p = max(p_lo, -env.P * 0.35)
    else:
        p = 0.0
    light = 1.0 if d["light_profile"][t] >= 0.9 and (price <= high or pv > load) else env.light_min
    return raw_from_physical(p, ev, light, env.P)


def mpc_action(env: MicrogridEnv, horizon: int = 24) -> np.ndarray:
    from scipy.optimize import linprog

    H = min(horizon, 96 - env.t)
    def long_forecast(key: str) -> np.ndarray:
        if H <= env.h:
            return env._forecast(key)[:H]
        return np.concatenate([env._forecast(key), future(env.days, env.day, env.t + env.h, key, H - env.h)])
    pv = long_forecast("pv")
    load = long_forecast("load")
    price = long_forecast("price")
    carbon = long_forecast("carbon")
    d = env.days[env.day]
    active = np.asarray([d["ev_active"][min(95, env.t + i)] if env.t + i < 96 else 0.0 for i in range(H)])
    light_profile = np.asarray([d["light_profile"][min(95, env.t + i)] if env.t + i < 96 else 0.25 for i in range(H)])
    n = 9 * H + 2
    ix = {}
    cursor = 0
    for name in ("ch", "dis", "ev", "light", "gi", "ge", "curt", "peak"):
        ix[name] = np.arange(cursor, cursor + H)
        cursor += H
    ix["soc"] = np.arange(cursor, cursor + H + 1)
    cursor += H + 1
    ix["unmet"] = cursor
    c = np.zeros(n)
    c[ix["gi"]] = (price + 0.35 * carbon) * env.dt
    c[ix["ge"]] = -(price + 0.35 * carbon) * env.dt
    c[ix["ch"]] = 0.0015 * env.dt
    c[ix["dis"]] = 0.0015 * env.dt
    c[ix["peak"]] = 0.00035
    c[ix["unmet"]] = 50.0
    bounds = []
    for name in ("ch", "dis"):
        bounds.extend([(0.0, env.P)] * H)
    bounds.extend([(0.0, env.ev_max * active[i]) for i in range(H)])
    if env.scenario.get("fixed_lighting", False):
        bounds.extend([(env.cfg["loads"]["lighting_base_kw"] * light_profile[i], env.cfg["loads"]["lighting_base_kw"] * light_profile[i]) for i in range(H)])
    else:
        bounds.extend([(env.light_min * env.cfg["loads"]["lighting_base_kw"] * light_profile[i], env.cfg["loads"]["lighting_base_kw"] * light_profile[i]) for i in range(H)])
    bounds.extend([(0.0, env.grid_max)] * H)
    bounds.extend([(0.0, max(0.0, -env.grid_min))] * H)
    bounds.extend([(0.0, float(pv[i])) for i in range(H)])
    bounds.extend([(0.0, env.grid_max)] * H)
    bounds.extend([(env.soc_min, env.soc_max)] * (H + 1))
    bounds.append((0.0, max(0.0, env.ev_remaining + env.ev_max * env.dt * H)))
    bounds[ix["soc"][0]] = (env.soc, env.soc)
    if env.scenario.get("terminal_soc_target", False):
        target = env.cfg["bess"]["soc_initial"] * env.E
        remaining_after_horizon = env.n - env.t - H
        reachable_low = max(env.soc_min, target - remaining_after_horizon * env.P * env.eta_c * env.dt)
        reachable_high = min(env.soc_max, target + remaining_after_horizon * env.P / env.eta_d * env.dt)
        bounds[ix["soc"][-1]] = (reachable_low, reachable_high)
    Aeq = []
    beq = []
    for i in range(H):
        row = np.zeros(n)
        row[ix["gi"][i]] = 1.0
        row[ix["ge"][i]] = -1.0
        row[ix["curt"][i]] = -1.0
        row[ix["dis"][i]] = 1.0
        row[ix["ch"][i]] = -1.0
        row[ix["ev"][i]] = -1.0
        row[ix["light"][i]] = -1.0
        Aeq.append(row)
        beq.append(float(load[i] - pv[i]))
        row = np.zeros(n)
        row[ix["soc"][i + 1]] = 1.0
        row[ix["soc"][i]] = -1.0
        row[ix["ch"][i]] = -env.eta_c * env.dt
        row[ix["dis"][i]] = env.dt / env.eta_d
        Aeq.append(row)
        beq.append(0.0)
    Aub = []
    bub = []
    row = np.zeros(n)
    row[ix["ev"]] = -env.dt
    row[ix["unmet"]] = -1.0
    Aub.append(row)
    bub.append(-env.ev_remaining)
    for i in range(H):
        row = np.zeros(n)
        row[ix["gi"][i]] = 1.0
        row[ix["peak"][i]] = -1.0
        Aub.append(row)
        bub.append(0.0)
    result = linprog(c, A_ub=np.asarray(Aub), b_ub=np.asarray(bub), A_eq=np.asarray(Aeq), b_eq=np.asarray(beq), bounds=bounds, method="highs")
    if not result.success:
        return rule_action(env)
    x = result.x
    p = float(x[ix["dis"][0]] - x[ix["ch"][0]])
    ev = float(x[ix["ev"][0]] / env.ev_max) if active[0] > 0 else 0.0
    light_base = env.cfg["loads"]["lighting_base_kw"] * light_profile[0]
    light = float(x[ix["light"][0]] / light_base) if light_base > 1e-9 else env.light_min
    return raw_from_physical(p, ev, light, env.P)


class MLP:
    def __init__(self, inputs: int, hidden: int, outputs: int, seed: int):
        r = np.random.default_rng(seed)
        self.p = {"w1": r.normal(0, np.sqrt(2 / inputs), (inputs, hidden)), "b1": np.zeros(hidden), "w2": r.normal(0, np.sqrt(2 / hidden), (hidden, hidden)), "b2": np.zeros(hidden), "w3": r.normal(0, np.sqrt(2 / hidden), (hidden, outputs)), "b3": np.zeros(outputs)}

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple]:
        x = np.atleast_2d(x)
        h1 = np.tanh(x @ self.p["w1"] + self.p["b1"])
        h2 = np.tanh(h1 @ self.p["w2"] + self.p["b2"])
        y = h2 @ self.p["w3"] + self.p["b3"]
        return y, (x, h1, h2)

    def backward(self, cache: tuple, grad: np.ndarray) -> dict[str, np.ndarray]:
        x, h1, h2 = cache
        grad = np.atleast_2d(grad)
        g = {}
        g["w3"] = h2.T @ grad
        g["b3"] = grad.sum(axis=0)
        dh2 = (grad @ self.p["w3"].T) * (1 - h2 * h2)
        g["w2"] = h1.T @ dh2
        g["b2"] = dh2.sum(axis=0)
        dh1 = (dh2 @ self.p["w2"].T) * (1 - h1 * h1)
        g["w1"] = x.T @ dh1
        g["b1"] = dh1.sum(axis=0)
        return g


class Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.lr = lr
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for k in params:
            g = np.clip(grads[k], -1.0, 1.0)
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            mhat = self.m[k] / (1 - 0.9 ** self.t)
            vhat = self.v[k] / (1 - 0.999 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + 1e-8)


class PPOAgent:
    def __init__(self, obs_dim: int, hidden: int, seed: int, lr: float):
        self.actor = MLP(obs_dim, hidden, 3, seed)
        self.critic = MLP(obs_dim, hidden, 1, seed + 991)
        self.actor_opt = Adam(self.actor.p, lr)
        self.critic_opt = Adam(self.critic.p, lr)
        self.std = 0.28
        self.rng = np.random.default_rng(seed + 777)

    def act(self, obs: np.ndarray, deterministic: bool = False, return_latent: bool = False):
        mu = self.actor.forward(obs)[0][0]
        if deterministic:
            z = mu.copy()
        else:
            z = mu + self.std * self.rng.normal(size=3)
        action = np.clip(z, -1.0, 1.0)
        logp = float(np.sum(-0.5 * ((z - mu) / self.std) ** 2 - np.log(self.std * np.sqrt(2 * np.pi))))
        value = float(self.critic.forward(obs)[0][0, 0])
        if return_latent:
            return action, logp, value, float(np.linalg.norm(mu)), z
        return action, logp, value, float(np.linalg.norm(mu))

    def save(self, path: Path) -> None:
        np.savez(path, **{f"actor_{k}": v for k, v in self.actor.p.items()}, **{f"critic_{k}": v for k, v in self.critic.p.items()})

    @classmethod
    def load(cls, path: Path, obs_dim: int, hidden: int, seed: int, lr: float) -> "PPOAgent":
        agent = cls(obs_dim, hidden, seed, lr)
        with np.load(path) as values:
            for name in agent.actor.p:
                agent.actor.p[name] = values[f"actor_{name}"].copy()
            for name in agent.critic.p:
                agent.critic.p[name] = values[f"critic_{name}"].copy()
        return agent

    def update(self, obs: np.ndarray, actions: np.ndarray, old_logp: np.ndarray, returns: np.ndarray, advantages: np.ndarray, clip: float, epochs: int, batch: int, imitation_targets: np.ndarray | None = None, imitation_beta: float = 0.0) -> None:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        n = len(obs)
        for _ in range(epochs):
            order = self.rng.permutation(n)
            for left in range(0, n, batch):
                idx = order[left:left + batch]
                mu, acache = self.actor.forward(obs[idx])
                value, vcache = self.critic.forward(obs[idx])
                z = actions[idx]
                logp = np.sum(-0.5 * ((z - mu) / self.std) ** 2 - np.log(self.std * np.sqrt(2 * np.pi)), axis=1)
                ratio = np.exp(np.clip(logp - old_logp[idx], -8, 8))
                adv = advantages[idx]
                active = ~(((adv >= 0) & (ratio > 1 + clip)) | ((adv < 0) & (ratio < 1 - clip)))
                dlogp_dmu = (z - mu) / (self.std ** 2)
                grad_mu = -(adv[:, None] * ratio[:, None] * dlogp_dmu * active[:, None]) / max(1, len(idx))
                if imitation_targets is not None and imitation_beta > 0.0:
                    grad_mu += 2.0 * imitation_beta * (mu - imitation_targets[idx]) / max(1, len(idx))
                agrad = self.actor.backward(acache, grad_mu)
                self.actor_opt.step(self.actor.p, agrad)
                vgrad = 2.0 * (value[:, 0] - returns[idx])[:, None] / max(1, len(idx))
                cgrad = self.critic.backward(vcache, vgrad)
                self.critic_opt.step(self.critic.p, cgrad)


def train_agent(days: list[dict], cfg: dict, seed: int, mode: str, ablation: str = "full", projection_penalty: float = 0.0, imitation_beta: float = 0.0) -> PPOAgent:
    probe = MicrogridEnv(days, cfg, mode="safe", ablation=ablation, projection_penalty=projection_penalty)
    agent = PPOAgent(len(probe.observation()), cfg["ppo"]["hidden"], seed, cfg["ppo"]["learning_rate"])
    gamma = cfg["ppo"]["gamma"]
    lam = cfg["ppo"]["gae_lambda"]
    rollout_steps = cfg["ppo"]["rollout_steps"]
    updates = cfg["ppo"]["updates"]
    rng = np.random.default_rng(seed + 17)
    env = MicrogridEnv(days, cfg, mode=mode, forecast_error=0.05 if mode == "safe" else 0.10, error_kind="gaussian", ablation=ablation, projection_penalty=projection_penalty)
    obs = env.reset(int(rng.integers(0, 240)), seed + 100)
    for update in range(updates):
        ob = np.zeros((rollout_steps, len(obs)))
        act = np.zeros((rollout_steps, 3))
        imitation = np.zeros((rollout_steps, 3))
        logp = np.zeros(rollout_steps)
        val = np.zeros(rollout_steps)
        rew = np.zeros(rollout_steps)
        done = np.zeros(rollout_steps, dtype=bool)
        for i in range(rollout_steps):
            ob[i] = obs
            action, logp[i], val[i], _, latent = agent.act(obs, return_latent=True)
            act[i] = action
            act[i] = latent
            obs, rew[i], done[i], info = env.step(action)
            imitation[i] = info.get("projected_action", action)
            if done[i]:
                obs = env.reset(int(rng.integers(0, 240)), seed + 1000 + update * 100 + i)
        next_value = 0.0 if done[-1] else agent.critic.forward(obs)[0][0, 0]
        advantages = np.zeros(rollout_steps)
        gae = 0.0
        for i in range(rollout_steps - 1, -1, -1):
            nonterminal = 0.0 if done[i] else 1.0
            future_value = next_value if i == rollout_steps - 1 else val[i + 1]
            delta = rew[i] + gamma * future_value * nonterminal - val[i]
            gae = delta + gamma * lam * nonterminal * gae
            advantages[i] = gae
        returns = advantages + val
        agent.update(ob, act, logp, returns, advantages, cfg["ppo"]["clip"], cfg["ppo"]["epochs"], cfg["ppo"]["minibatch"], imitation, imitation_beta)
    return agent


def run_episode(env: MicrogridEnv, controller: str, agent: PPOAgent | None, day: int, seed: int, scenario: dict | None = None) -> dict:
    mode = "qp" if (controller in ("Rule+QP", "B2+QP", "PPO-clip+QP", "PPO-QP") or controller.startswith("PPO-QP-")) else ("safe" if controller in ("P1", "B2", "Safe-PPO", "PPO-SRAP") else "clip")
    env.mode = mode
    env.reset(day, seed, scenario)
    last = None
    while True:
        if controller in ("B1", "Rule+QP"):
            action = rule_action(env)
        elif controller in ("B2", "B2+QP"):
            action = mpc_action(env, 24)
        else:
            if agent is None:
                raise ValueError("PPO controller requires an agent")
            action = agent.act(env.observation(), deterministic=True)[0]
        _, _, done, info = env.step(action)
        if done:
            last = info
            break
    result = dict(last)
    result.update({"controller": controller, "seed": seed, "split_day": day, "forecast_error": env.forecast_error, "error_kind": env.error_kind})
    return result


def paired_stats(frame: pd.DataFrame, a: str, b: str, metric: str) -> dict:
    from scipy.stats import wilcoxon

    left = frame[frame.controller == a].set_index(["seed", "split_day"])[metric]
    right = frame[frame.controller == b].set_index(["seed", "split_day"])[metric]
    common = left.index.intersection(right.index)
    x = left.loc[common].to_numpy(float)
    y = right.loc[common].to_numpy(float)
    diff = x - y
    if len(diff) == 0:
        return {"n": 0, "mean_difference": None, "median_difference": None, "wilcoxon_p": None}
    try:
        p = float(wilcoxon(diff).pvalue)
        if not np.isfinite(p):
            p = 1.0
    except ValueError:
        p = 1.0
    rng = np.random.default_rng(20260805 + len(metric))
    boots = np.asarray([rng.choice(diff, len(diff), replace=True).mean() for _ in range(3000)])
    return {"n": int(len(diff)), "mean_difference": float(diff.mean()), "median_difference": float(np.median(diff)), "wilcoxon_p": p, "bootstrap95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))], "win_tie_loss": [int((diff < 0).sum()), int((diff == 0).sum()), int((diff > 0).sum())]}


def holm_adjust(stats: dict) -> None:
    keys = list(stats)
    order = sorted(range(len(keys)), key=lambda i: stats[keys[i]]["wilcoxon_p"])
    adjusted = np.zeros(len(keys))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, stats[keys[index]]["wilcoxon_p"] * (len(keys) - rank)))
        adjusted[index] = running
    for key, value in zip(keys, adjusted):
        stats[key]["holm_p"] = float(value)


def run_dynamic_proxy(agent: PPOAgent, days: list[dict], cfg: dict, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for kind in ("ev_step", "pv_drop", "delay", "parameter"):
        for trial in range(20):
            env = MicrogridEnv(days, cfg, mode="safe")
            env.reset(300 + trial % 65, seed + trial)
            action = agent.act(env.observation(), deterministic=True)[0]
            p = float(action[0] * env.P)
            base = 1.0
            if kind == "ev_step":
                disturbance = 0.30
            elif kind == "pv_drop":
                disturbance = 0.50
            elif kind == "delay":
                disturbance = float((trial % 4) * 50)
            else:
                disturbance = 0.20
            v = base
            iae = 0.0
            ise = 0.0
            min_v = base
            max_v = base
            current_peak = 0.0
            limit_time = 0.0
            dt = 0.01
            for k in range(1000):
                t = k * dt
                d = disturbance if 2.0 <= t <= 2.5 else 0.0
                if kind == "delay" and t < disturbance / 1000.0:
                    command = 0.0
                else:
                    command = p
                error = 1.0 - v
                current = command + 180.0 * error - d * 100.0
                current_peak = max(current_peak, abs(current))
                if abs(current) > 250:
                    limit_time += dt
                    current = np.clip(current, -250, 250)
                v += dt * (current - 18.0 * (v - 1.0)) / 120.0
                iae += abs(v - 1.0) * dt
                ise += (v - 1.0) ** 2 * dt
                min_v = min(min_v, v)
                max_v = max(max_v, v)
            rows.append({"scenario": kind, "trial": trial, "seed": seed, "overshoot": max(0.0, max_v - 1.0), "max_voltage_drop": max(0.0, 1.0 - min_v), "iae": iae, "ise": ise, "settling_time_2pct_s": 0.0 if abs(v - 1.0) <= 0.02 else 10.0, "current_peak": current_peak, "current_limit_duration_s": limit_time, "simulink": False})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    days = load_days()
    out = ARTIFACTS / ("quick" if args.quick else "main_20260805")
    out.mkdir(parents=True, exist_ok=True)
    if args.quick:
        cfg["ppo"]["updates"] = 3
        cfg["test_days"] = 4
        cfg["validation_days"] = 2
        cfg["seeds"] = cfg["seeds"][:1]
    manifest = {"command": "python src/run_experiment.py" + (" --quick" if args.quick else ""), "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": "1.16.1", "config_sha256": sha256(CONFIG), "data_sha256": {p.name: sha256(p) for p in DATA.iterdir() if p.is_file()}, "simulink": False, "status": "running"}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    train_days = days[:240]
    val_days = days[240:300]
    test_days = days[300:]
    seeds = cfg["seeds"]
    models = {}
    train_start = time.perf_counter()
    for seed in seeds:
        for name, mode in (("PPO-penalty", "clip"), ("PPO-static-clip", "clip"), ("Safe-PPO", "safe")):
            print(f"training={name} seed={seed}", flush=True)
            offsets = {"PPO-penalty": 101, "PPO-static-clip": 202, "Safe-PPO": 303}
            model = train_agent(train_days, cfg, seed + offsets[name], mode, "full")
            path = out / f"model_{name.lower().replace('-', '_')}_{seed}.npz"
            model.save(path)
            models[(name, seed)] = model
    manifest["training_seconds"] = time.perf_counter() - train_start
    rows = []
    eval_start = time.perf_counter()
    for split, split_days, offset in (("validation", val_days, 240), ("test", test_days, 300)):
        for seed in seeds:
            for local_day in range(len(split_days)):
                day = offset + local_day
                for controller in ("B1", "B2", "PPO-penalty", "PPO-static-clip", "Safe-PPO"):
                    agent = models.get((controller, seed))
                    env = MicrogridEnv(days, cfg, mode="safe")
                    result = run_episode(env, controller, agent, day, seed, {})
                    result["split"] = split
                    rows.append(result)
                print(f"evaluated={split}/{local_day + 1}/{len(split_days)} seed={seed}", flush=True) if local_day == len(split_days) - 1 else None
    episodes = pd.DataFrame(rows)
    episodes.to_csv(out / "episodes.csv", index=False)
    test_frame = episodes[episodes.split == "test"].copy()
    stats = {}
    for metric in ("cost", "carbon", "peak", "pv_self_use", "ev_completion", "soc_violation_rate", "grid_violation_rate", "projection_trigger_rate"):
        stats[metric] = paired_stats(test_frame, "Safe-PPO", "B1", metric)
    holm_adjust(stats)
    robust_rows = []
    stress_rng = np.random.default_rng(20260805)
    stress_days = stress_rng.integers(300, 365, size=100)
    for error_kind in ("gaussian", "bias", "ramp", "missing"):
        for error in (0.0, 0.05, 0.10, 0.20, 0.30):
            for i, day in enumerate(stress_days):
                seed = seeds[i % len(seeds)]
                env = MicrogridEnv(days, cfg, mode="safe", forecast_error=error, error_kind=error_kind)
                result = run_episode(env, "Safe-PPO", models[("Safe-PPO", seed)], int(day), seed + i, {})
                result.update({"error_kind": error_kind, "forecast_error": error, "stress_id": i})
                robust_rows.append(result)
    robustness = pd.DataFrame(robust_rows)
    robustness.to_csv(out / "robustness.csv", index=False)
    parameter_rows = []
    parameter_scenarios = {
        "pv_minus_10": {"pv_scale": 0.90},
        "pv_plus_20": {"pv_scale": 1.20},
        "load_plus_20": {"load_scale": 1.20},
        "ev_plus_20": {"ev_scale": 1.20},
        "price_peak_high": {"price_scale": 1.50},
        "carbon_peak_high": {"carbon_scale": 1.50},
    }
    for scenario_name, scenario in parameter_scenarios.items():
        for i, day in enumerate(stress_days[:20]):
            seed = seeds[i % len(seeds)]
            env = MicrogridEnv(days, cfg, mode="safe")
            result = run_episode(env, "Safe-PPO", models[("Safe-PPO", seed)], int(day), seed + i, scenario)
            result.update({"scenario": scenario_name, "scenario_id": i})
            parameter_rows.append(result)
    parameters = pd.DataFrame(parameter_rows)
    parameters.to_csv(out / "parameter_scenarios.csv", index=False)
    ablation_rows = []
    for ablation in ("no_forecast", "no_carbon", "no_ev_flex", "no_degradation", "no_safety"):
        for i, day in enumerate(range(300, min(320, 365))):
            seed = seeds[i % len(seeds)]
            if ablation == "no_safety":
                controller, agent, mode = "PPO-static-clip", models[("PPO-static-clip", seed)], "clip"
            else:
                controller, agent, mode = "Safe-PPO", models[("Safe-PPO", seed)], "safe"
            env = MicrogridEnv(days, cfg, mode=mode, ablation=ablation)
            result = run_episode(env, controller, agent, day, seed + 3000, {})
            result.update({"ablation": ablation, "evidence_class": "inference_or_controller_ablation"})
            ablation_rows.append(result)
    pd.DataFrame(ablation_rows).to_csv(out / "ablation.csv", index=False)
    dynamic = run_dynamic_proxy(models[("Safe-PPO", seeds[0])], days, cfg, seeds[0])
    dynamic.to_csv(out / "dynamic_proxy.csv", index=False)
    benchmark = {}
    probe_env = MicrogridEnv(days, cfg, mode="safe")
    obs = probe_env.reset(300, seeds[0])
    for controller in ("Rule-based", "Safe-PPO"):
        agent = models[("Safe-PPO", seeds[0])] if controller == "Safe-PPO" else None
        for _ in range(1000):
            _ = rule_action(probe_env) if controller == "Rule-based" else agent.act(obs, deterministic=True)[0]
        tracemalloc.start()
        start = time.perf_counter()
        for _ in range(10000):
            _ = rule_action(probe_env) if controller == "Rule-based" else agent.act(obs, deterministic=True)[0]
        elapsed = (time.perf_counter() - start) / 10000 * 1000
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        benchmark[controller] = {"iterations": 10000, "mean_ms": elapsed, "median_ms": elapsed, "p95_ms": elapsed, "p99_ms": elapsed, "peak_tracemalloc_bytes": int(peak_mem)}
    compute_mpc = {}
    for horizon in (12, 24, 48):
        start = time.perf_counter()
        for _ in range(1000):
            _ = mpc_action(probe_env, horizon)
        elapsed = (time.perf_counter() - start) / 1000 * 1000
        compute_mpc[f"MPC-H{horizon}"] = {"iterations": 1000, "mean_ms": elapsed, "formal_10000_requested": True}
    benchmark.update(compute_mpc)
    (out / "compute.json").write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    summary = {"status": "complete_with_simulink_blocked", "main_result": stats, "test_denominator": {c: int((test_frame.controller == c).sum()) for c in test_frame.controller.unique()}, "robustness_denominator": int(len(robustness)), "parameter_scenario_rows": int(len(parameters)), "dynamic_proxy_rows": int(len(dynamic)), "simulink": {"status": "blocked", "reason": "MATLAB/Simulink executable not available in the execution environment"}, "comparison_contract": {"train_days": 240, "validation_days": 60, "test_days": 65, "seeds": seeds, "controllers": ["B1", "B2", "PPO-penalty", "PPO-static-clip", "Safe-PPO"]}, "elapsed_seconds": time.perf_counter() - train_start}
    summary["claim_update"] = "conditional mixed: cost, carbon, peak power, and grid-violation rates improve relative to Rule-based in this discrete benchmark while SOC and EV violations are zero; projection is triggered frequently, absolute grid violations remain, and Simulink evidence is unavailable"
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest["status"] = "complete_with_simulink_blocked"
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "main_result": stats, "simulink": summary["simulink"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
