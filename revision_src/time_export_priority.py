import hashlib
import json
import os
import time
from collections import Counter
import numpy as np
import pandas as pd
import causal_run as cr
from matched_mpc import matched_mpc_action

for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'

OUT = cr.OUT
NAME = 'MPC-H24-XP+QP'


def main():
    protocol = json.loads((OUT / 'timing_state_protocol.json').read_text(encoding='utf-8'))
    days, _ = cr.make_days()
    rows, warmups = [], []
    for state in protocol['states']:
        for repetition in range(6):
            env = cr.CausalEnv(days, cr.CFG, mode='qp', projection_penalty=0.0)
            env.reset(state['day'], 20260805)
            env.soc = state['soc_kwh']
            env.prev_p = state['previous_battery_power_kw']
            env.ev_remaining = state['ev_remaining_kwh']
            env.t = state['step']
            assert hashlib.sha256(env.observation().tobytes()).hexdigest() == state['observation_sha256']
            started = time.perf_counter_ns()
            action = matched_mpc_action(env, export_priority=True)
            _, _, _, info = env.step(action)
            row = {'state_id': state['state_id'], 'day': state['day'], 'step': state['step'], 'controller': NAME,
                   'repetition': repetition, 'elapsed_ms': (time.perf_counter_ns() - started) / 1e6,
                   'mpc_status': env.mpc_last_info['status'], 'mpc_fallback': int(env.mpc_last_info['fallback']),
                   'phase1_status': env.mpc_last_info['phase1_status'], 'phase2_feasible': env.mpc_last_info['phase2_feasible'],
                   'qp_execution_path': info['qp_execution_path'], 'executed_battery_power_kw': info['p'],
                   'soc_after_kwh': info['soc'], 'grid_violation_kw': info['grid_violation']}
            (warmups if repetition == 0 else rows).append(row)
    frame, warm = pd.DataFrame(rows), pd.DataFrame(warmups)
    assert len(frame) == 125 and len(warm) == 25 and frame.groupby('state_id').size().eq(5).all()
    frame.to_csv(OUT / 'timing_xp_rows.csv', index=False)
    warm.to_csv(OUT / 'timing_xp_warmup_rows.csv', index=False)
    elapsed = frame.elapsed_ms.to_numpy()
    summary = {'controller': NAME, 'calls': len(frame), 'unique_states': int(frame.state_id.nunique()),
               'mean_ms': float(elapsed.mean()), 'median_ms': float(np.median(elapsed)),
               'p95_ms': float(np.quantile(elapsed, .95)), 'maximum_ms': float(elapsed.max()),
               'mpc_status_counts': {str(k): int(v) for k, v in Counter(frame.mpc_status).items()},
               'mpc_fallback_calls': int(frame.mpc_fallback.sum()),
               'qp_execution_path_counts': {k: int(v) for k, v in Counter(frame.qp_execution_path).items()}}
    result = {'status': 'passed', 'scope': protocol['scope'], 'warmups': 25, 'summary': summary}
    (OUT / 'timing_xp_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
