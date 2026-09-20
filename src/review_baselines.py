from pathlib import Path
import argparse
import json
import os
import platform
import sys
import time
import numpy as np
import pandas as pd
import scipy
import causal_run as cr
from matched_mpc import matched_mpc_action

ROOT, OUT = cr.ROOT, cr.OUT
CONTROLLERS = ['Rule+QP', 'MPC-H24+QP', 'NetLoadRule+QP', 'MPC-H24-FRR+QP']
CAPS = [35, 60, 100]


def net_load_rule_action(env, cap):
    lo, hi, emin, emax, _ = env._bounds()
    pv, load = env._actual('pv'), env._actual('load')
    ev = float(np.clip(max(emin, 1.0 if pv > load else 0.65), emin, emax))
    light = env.cfg['loads']['lighting_base_kw'] * env.days[env.day]['light_profile'][env.t]
    net = load + light + ev * env.ev_max - pv
    price = env._actual('price')
    if net < 0:
        p = -min(env.P, -net)
    elif price >= np.quantile(env._forecast('price'), 0.70) or env._actual('carbon') >= np.quantile(env._forecast('carbon'), 0.65):
        p = min(cap, net)
    elif price <= np.quantile(env._forecast('price'), 0.35):
        p = -cap
    else:
        p = 0.0
    return cr.core.raw_from_physical(float(np.clip(p, lo, hi)), ev, 1.0, env.P)


def evaluate(days, controller, day, cap=None):
    env = cr.CausalEnv(days, cr.CFG, mode='qp', projection_penalty=0.0)
    env.reset(day, 0)
    steps, diagnostics, timings = [], [], []
    while True:
        started = time.perf_counter()
        mpc = controller.startswith('MPC-')
        if mpc:
            action = matched_mpc_action(env, future_ramp_relaxed=controller == 'MPC-H24-FRR+QP')
            diagnostics.append({'controller': controller, 'day': day, 'step': env.t, **env.mpc_last_info})
        elif controller == 'Rule+QP':
            action = cr.core.rule_action(env)
        else:
            action = net_load_rule_action(env, cap)
        _, _, done, info = env.step(action)
        timings.append((time.perf_counter() - started) * 1000)
        row = {'controller': controller, 'seed': 0, 'day': day, 'step': len(timings) - 1,
               **{k: info[k] for k in ['p', 'soc', 'grid_import', 'grid_export', 'grid_violation', 'qp_execution_path']},
               **env.step_audit, 'mpc_status': env.mpc_last_info['status'] if mpc else -1,
               'mpc_fallback': int(env.mpc_last_info['fallback']) if mpc else 0, 'runtime_ms': timings[-1],
               'raw_battery_kw': float(action[0] * env.P), 'raw_ev_fraction': float((action[1] + 1) / 2),
               'executed_ev_fraction': float((info['projected_action'][1] + 1) / 2), 'action_correction': float(info['correction'])}
        if mpc:
            row.update({'mpc_' + k: env.mpc_last_info[k] for k in ['mip_gap', 'seconds', 'constraint_violation_max', 'message']})
        steps.append(row)
        if done:
            break
    result = {k: float(v) for k, v in info.items() if isinstance(v, (int, float, np.number))}
    result.update({'controller': controller, 'seed': 0, 'split_day': day,
                   'split': 'validation' if day < 300 else 'test',
                   'mpc_fallbacks': sum(int(x['fallback']) for x in diagnostics),
                   'mpc_time_limit_incumbents': sum(int(x['status'] == 1 and not x['fallback']) for x in diagnostics),
                   'runtime_mean_ms': float(np.mean(timings)), 'runtime_p95_ms': float(np.quantile(timings, 0.95))})
    assert len(steps) == sum(result[k] for k in ['hard_qp_solution', 'phase2_solution', 'phase1_fallback', 'heuristic_fallback']) == 96
    return result, steps, diagnostics


def service_failure(frame):
    return ((frame.soc_violation_rate > 0) | (frame.terminal_soc_error_kwh.abs() > 1e-5) |
            (frame.ev_completion < 1) | ((frame.lighting_energy_kwh - 472.5).abs() > 1e-5)).astype(int)


def hashes():
    files = [ROOT / 'PLAN.md', ROOT / 'baseline_protocol.md', ROOT / 'configs/experiment.json']
    files += [ROOT / 'src' / name for name in ['run_experiment.py', 'causal_run.py', 'matched_mpc.py', 'review_baselines.py']]
    return {str(p.relative_to(ROOT)): cr.core.sha256(p) for p in files}


def smoke():
    days, _ = cr.make_days()
    rows = []
    for controller in CONTROLLERS:
        row, steps, diagnostics = evaluate(days, controller, 240, 60)
        assert service_failure(pd.DataFrame([row])).sum() == 0
        assert np.isfinite([row[k] for k in cr.METRICS]).all()
        if controller == 'MPC-H24-FRR+QP':
            for step, diag in zip(steps, diagnostics):
                if not diag['fallback']:
                    lo, hi = diag['current_battery_bounds_kw']
                    assert lo - 1e-5 <= step['raw_battery_kw'] <= hi + 1e-5
        rows.append(row)
    reference, trace = cr.evaluate(days, 'Rule+QP', None, 240, keep_steps=True)
    for key in set(cr.METRICS) - {'runtime_mean_ms', 'runtime_p95_ms'}:
        assert abs(reference[key] - rows[0][key]) < 1e-8
    env = cr.CausalEnv(days, cr.CFG, mode='qp')
    env.reset(240)
    env._bounds = lambda: (-100.0, 100.0, 0.0, 0.0, 0.0)
    env._forecast = lambda key: np.linspace(0.0, 1.0, 24)
    env._actual = lambda key: {'pv': 0.0, 'load': 80.0, 'price': 1.0, 'carbon': 0.0}[key]
    assert all(abs(net_load_rule_action(env, cap)[0] * 100 - cap) < 1e-9 for cap in CAPS)
    env._actual = lambda key: {'pv': 130.0, 'load': 80.0, 'price': 1.0, 'carbon': 0.0}[key]
    assert abs(net_load_rule_action(env, 35)[0] * 100 + 20.0) < 1e-9
    cr.save_json(OUT / 'baseline_smoke.json', {'status': 'passed', 'validation_day': 240,
                 'controller_results': rows, 'legacy_evaluator_equivalence': True,
                 'net_load_direction_and_candidate_caps': True, 'test_evidence': False})
    print('baseline smoke passed on validation day 240', flush=True)


def validate():
    assert not any(OUT.glob('baseline_*.csv'))
    frozen = hashes()
    cr.save_json(OUT / 'baseline_run_contract.json', {'hashes': frozen, 'controllers': CONTROLLERS,
                 'rule_caps_kw': CAPS, 'selection_order': ['service_failure', 'grid_excess_energy_kwh', 'cost', 'mean_action_correction', 'cap_kw'],
                 'command': [sys.executable, *sys.argv], 'python': sys.version, 'platform': platform.platform(),
                 'numpy': np.__version__, 'scipy': scipy.__version__,
                 'thread_environment': {k: os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']},
                 'started_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'test_started': False})
    days, _ = cr.make_days()
    all_rows = []
    for cap in CAPS:
        rows = []
        for day in range(240, 300):
            row, _, _ = evaluate(days, 'NetLoadRule+QP', day, cap)
            rows.append({**row, 'cap_kw': cap})
        pd.DataFrame(rows).to_csv(OUT / f'validation_NetLoadRule_cap{cap}.csv', index=False)
        all_rows.extend(rows)
        print(f'rule validation cap={cap} days=60 complete', flush=True)
    frame = pd.DataFrame(all_rows)
    frame['service_failure'] = service_failure(frame)
    summary = frame.groupby('cap_kw')[cr.METRICS + ['service_failure']].mean().reset_index()
    order = ['service_failure', 'grid_excess_energy_kwh', 'cost', 'mean_action_correction', 'cap_kw']
    winner = summary.sort_values(order).iloc[0]
    summary.to_csv(OUT / 'validation_NetLoadRule_means.csv', index=False)
    assert frozen == hashes()
    cr.save_json(OUT / 'baseline_selection.json', {'selected_cap_kw': int(winner.cap_kw), 'hashes': frozen,
                 'selection_inputs': {p.name: cr.core.sha256(p) for p in OUT.glob('validation_NetLoadRule_cap*.csv')},
                 'selected_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'test_evaluation_not_started': not any(OUT.glob('baseline_*.csv'))})
    print(f'NetLoadRule cap frozen: {int(winner.cap_kw)} kW', flush=True)


def test(controller):
    selection = json.loads((OUT / 'baseline_selection.json').read_text(encoding='utf-8'))
    assert selection['hashes'] == hashes()
    assert all(cr.core.sha256(OUT / name) == value for name, value in selection['selection_inputs'].items())
    assert selection['test_evaluation_not_started']
    record = {'controller': controller, 'command': [sys.executable, *sys.argv],
              'started_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'selection_sha256': cr.core.sha256(OUT / 'baseline_selection.json')}
    cr.save_json(OUT / f'baseline_execution_{controller}.json', record)
    days, _ = cr.make_days()
    rows, traces, diagnostics = [], [], []
    started = time.perf_counter()
    for day in range(300, 365):
        row, steps, diags = evaluate(days, controller, day, selection['selected_cap_kw'])
        rows.append(row)
        traces.extend(steps)
        diagnostics.extend(diags)
        if (day - 299) % 5 == 0:
            pd.DataFrame(rows).to_csv(OUT / f'baseline_{controller}.csv', index=False)
            print(f'baseline {controller} days={day-299}/65 seconds={time.perf_counter()-started:.1f}', flush=True)
    pd.DataFrame(rows).to_csv(OUT / f'baseline_{controller}.csv', index=False)
    pd.DataFrame(traces).to_csv(OUT / f'trajectory_{controller}.csv', index=False)
    if diagnostics:
        (OUT / f'mpc_diagnostics_{controller}.jsonl').write_text('\n'.join(json.dumps(x, allow_nan=False) for x in diagnostics) + '\n', encoding='utf-8')
    assert selection['hashes'] == hashes()
    cr.save_json(OUT / f'baseline_execution_{controller}.json', {**record, 'completed_utc': pd.Timestamp.now(tz='UTC').isoformat(),
                 'elapsed_seconds': time.perf_counter() - started, 'episodes': len(rows), 'steps': len(traces)})


def verify():
    selection = json.loads((OUT / 'baseline_selection.json').read_text(encoding='utf-8'))
    assert selection['hashes'] == hashes()
    rows = []
    for controller in CONTROLLERS:
        daily = pd.read_csv(OUT / f'baseline_{controller}.csv')
        trace = pd.read_csv(OUT / f'trajectory_{controller}.csv')
        assert len(daily) == daily.split_day.nunique() == 65
        assert set(daily.split_day) == set(range(300, 365))
        assert len(trace) == 6240 and (trace.groupby('day').size() == 96).all()
        assert np.isfinite(daily[cr.METRICS].to_numpy()).all()
        assert service_failure(daily).sum() == 0
        for _, day in daily.iterrows():
            part = trace[trace.day == day.split_day]
            assert abs(part.grid_violation.sum() * 0.25 - day.grid_excess_energy_kwh) < 1e-7
            assert abs(part.grid_import.max() - day.peak) < 1e-7
            assert abs(np.abs(np.diff(np.r_[0.0, part.p.to_numpy()])).max() - day.max_ramp_kw) < 1e-7
            for name in ['hard_qp_solution', 'phase2_solution', 'phase1_fallback', 'heuristic_fallback']:
                assert int((part.qp_execution_path == name).sum()) == int(day[name])
        if controller.startswith('MPC-'):
            diags = [json.loads(x) for x in (OUT / f'mpc_diagnostics_{controller}.jsonl').read_text(encoding='utf-8').splitlines()]
            assert len(diags) == 6240
            assert sum(int(x['fallback']) for x in diags) == daily.mpc_fallbacks.sum()
            assert sum(int(x['status'] == 1 and not x['fallback']) for x in diags) == daily.mpc_time_limit_incumbents.sum()
            for step, diag in zip(trace.to_dict('records'), diags):
                assert (step['day'], step['step']) == (diag['day'], diag['step'])
                if not diag['fallback']:
                    assert abs(diag['terminal_soc_kwh'] - 125.0) < 1e-5
                    assert diag['charge_discharge_overlap_kw'] < 1e-5
                    if controller == 'MPC-H24-FRR+QP':
                        lo, hi = diag['current_battery_bounds_kw']
                        assert lo - 1e-5 <= step['raw_battery_kw'] <= hi + 1e-5
        rows.append({'controller': controller, **{key: float(daily[key].mean()) for key in cr.METRICS}})
    pd.DataFrame(rows).to_csv(OUT / 'baseline_revision_means.csv', index=False)
    cr.save_json(OUT / 'baseline_verification.json', {'status': 'passed', 'controllers': CONTROLLERS,
                 'episodes': 260, 'steps': 24960, 'selected_rule_cap_kw': selection['selected_cap_kw'],
                 'service_failures': 0, 'selection_hashes_unchanged': True,
                 'outputs': {p.name: cr.core.sha256(p) for p in OUT.iterdir() if p.name.startswith(('baseline_', 'trajectory_Rule', 'trajectory_NetLoadRule', 'trajectory_MPC', 'mpc_diagnostics_')) and p.name != 'baseline_verification.json'}})
    print(pd.DataFrame(rows)[['controller', 'cost', 'grid_excess_energy_kwh', 'peak', 'task_score']].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['smoke', 'validate', 'test', 'verify'])
    parser.add_argument('--controller', choices=CONTROLLERS)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.stage == 'test':
        for controller in [args.controller] if args.controller else CONTROLLERS:
            test(controller)
    else:
        {'smoke': smoke, 'validate': validate, 'verify': verify}[args.stage]()
