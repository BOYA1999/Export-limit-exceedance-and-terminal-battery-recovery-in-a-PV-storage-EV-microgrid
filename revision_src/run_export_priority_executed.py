from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
import causal_run as cr
from matched_mpc import matched_mpc_action

NAME = 'MPC-H24-XP+QP'
OUT = cr.OUT


def evaluate(days, day):
    env = cr.CausalEnv(days, cr.CFG, mode='qp', projection_penalty=0.0)
    env.reset(day, 0)
    steps, diagnostics, timings = [], [], []
    while True:
        started = time.perf_counter()
        action = matched_mpc_action(env, export_priority=True)
        diagnostics.append({'controller': NAME, 'day': day, 'step': env.t, **env.mpc_last_info})
        _, _, done, info = env.step(action)
        timings.append((time.perf_counter() - started) * 1000)
        steps.append({'controller': NAME, 'seed': 0, 'day': day, 'step': len(timings) - 1,
                      **{k: info[k] for k in ['p', 'soc', 'grid_import', 'grid_export', 'grid_violation', 'qp_execution_path']},
                      **env.step_audit, 'mpc_status': env.mpc_last_info['status'],
                      'mpc_fallback': int(env.mpc_last_info['fallback']), 'runtime_ms': timings[-1],
                      'raw_battery_kw': float(action[0] * env.P), 'raw_ev_fraction': float((action[1] + 1) / 2),
                      'executed_ev_fraction': float((info['projected_action'][1] + 1) / 2),
                      'action_correction': float(info['correction']),
                      **{'mpc_' + k: env.mpc_last_info[k] for k in ['mip_gap', 'seconds', 'constraint_violation_max', 'message']}})
        if done:
            break
    result = {k: float(v) for k, v in info.items() if isinstance(v, (int, float, np.number))}
    result.update({'controller': NAME, 'seed': 0, 'split_day': day, 'split': 'test',
                   'mpc_fallbacks': sum(int(x['fallback']) for x in diagnostics),
                   'mpc_time_limit_incumbents': sum(int(x['status'] == 1 and not x['fallback']) for x in diagnostics),
                   'runtime_mean_ms': float(np.mean(timings)), 'runtime_p95_ms': float(np.quantile(timings, 0.95))})
    return result, steps, diagnostics


def main():
    protocol = Path(__file__).resolve().parents[2] / 'PROTOCOL.md'
    record = {'controller': NAME, 'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'mpc_source_sha256': hashlib.sha256((Path(__file__).parent / 'matched_mpc.py').read_bytes()).hexdigest(),
              'command': [sys.executable, *sys.argv], 'started_utc': pd.Timestamp.now(tz='UTC').isoformat()}
    (OUT / f'baseline_execution_{NAME}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    days, _ = cr.make_days()
    rows, traces, diagnostics = [], [], []
    started = time.perf_counter()
    for day in range(300, 365):
        row, steps, diags = evaluate(days, day)
        rows.append(row)
        traces.extend(steps)
        diagnostics.extend(diags)
        if (day - 299) % 5 == 0:
            pd.DataFrame(rows).to_csv(OUT / f'baseline_{NAME}.csv', index=False)
            print(f'{NAME} days={day-299}/65 seconds={time.perf_counter()-started:.1f}', flush=True)
    daily = pd.DataFrame(rows)
    trace = pd.DataFrame(traces)
    daily.to_csv(OUT / f'baseline_{NAME}.csv', index=False)
    trace.to_csv(OUT / f'trajectory_{NAME}.csv', index=False)
    (OUT / f'mpc_diagnostics_{NAME}.jsonl').write_text('\n'.join(json.dumps(x, allow_nan=False) for x in diagnostics) + '\n', encoding='utf-8')
    service_failure = ((daily.soc_violation_rate > 0) | (daily.terminal_soc_error_kwh.abs() > 1e-5) |
                       (daily.ev_completion < 1) | ((daily.lighting_energy_kwh - 472.5).abs() > 1e-5))
    assert len(daily) == 65 and len(trace) == 6240 and not service_failure.any()
    assert (trace.groupby('day').size() == 96).all()
    for _, row in daily.iterrows():
        part = trace[trace.day == row.split_day]
        assert abs(part.grid_violation.sum() * 0.25 - row.grid_excess_energy_kwh) < 1e-7
        assert abs(part.grid_import.max() - row.peak) < 1e-7
    result = {**record, 'completed_utc': pd.Timestamp.now(tz='UTC').isoformat(),
              'elapsed_seconds': time.perf_counter() - started, 'episodes': 65, 'steps': 6240,
              'service_failures': 0, 'means': {k: float(daily[k].mean()) for k in cr.METRICS},
              'phase1_failures': sum(not x['phase1_feasible'] for x in diagnostics),
              'phase2_fallback_to_phase1': sum(x['used_phase1_solution'] for x in diagnostics)}
    (OUT / f'baseline_execution_{NAME}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['means'], indent=2), flush=True)


if __name__ == '__main__':
    main()
