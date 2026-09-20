import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys
import time
import numpy as np
import pandas as pd
import causal_run as driver
from matched_mpc import matched_mpc_action
from review_baselines import CONTROLLERS as BASELINES, net_load_rule_action, hashes as baseline_hashes

OUT = driver.OUT
CONTROLLERS = [*BASELINES, *driver.VARIANTS]
SEED = 20260805
VALIDATION_DAYS = [240, 252, 264, 276, 288]
STEPS = [0, 24, 48, 72, 95]


def restore(days, state, controller):
    env = driver.CausalEnv(days, driver.CFG, mode='qp', projection_penalty=driver.VARIANTS.get(controller, 0))
    env.reset(state['day'], SEED)
    env.soc = state['soc_kwh']
    env.prev_p = state['previous_battery_power_kw']
    env.ev_remaining = state['ev_remaining_kwh']
    env.t = state['step']
    return env


def main():
    started = datetime.now(timezone.utc).isoformat()
    days, _ = driver.make_days()
    selection = json.loads((OUT / 'selection.json').read_text(encoding='utf-8'))['selected_budgets']
    rule_selection = json.loads((OUT / 'baseline_selection.json').read_text(encoding='utf-8'))
    assert rule_selection['hashes'] == baseline_hashes()
    driver.verify_contract()
    models = {name: driver.load_model(name, SEED, selection[name]) for name in driver.VARIANTS}
    states = []
    for day in VALIDATION_DAYS:
        env = driver.CausalEnv(days, driver.CFG, mode='qp')
        env.reset(day, SEED)
        for step in range(96):
            if step in STEPS:
                states.append({'state_id': len(states), 'day': day, 'step': step, 'soc_kwh': env.soc, 'previous_battery_power_kw': env.prev_p, 'ev_remaining_kwh': env.ev_remaining, 'observation_sha256': hashlib.sha256(env.observation().tobytes()).hexdigest()})
            env.step(driver.core.rule_action(env))
    rows, warmups = [], []
    for state in states:
        offset = state['state_id'] % len(CONTROLLERS)
        order = CONTROLLERS[offset:] + CONTROLLERS[:offset]
        for controller in order:
            for repetition in range(6):
                env = restore(days, state, controller)
                assert hashlib.sha256(env.observation().tobytes()).hexdigest() == state['observation_sha256']
                before = time.perf_counter_ns()
                observation = env.observation()
                if controller == 'Rule+QP':
                    action = driver.core.rule_action(env)
                elif controller.startswith('MPC-'):
                    action = matched_mpc_action(env, future_ramp_relaxed=controller == 'MPC-H24-FRR+QP')
                elif controller == 'NetLoadRule+QP':
                    action = net_load_rule_action(env, rule_selection['selected_cap_kw'])
                else:
                    action = models[controller].act(observation, deterministic=True)[0]
                _, _, _, info = env.step(action)
                elapsed = (time.perf_counter_ns() - before) / 1e6
                row = {'state_id': state['state_id'], 'day': state['day'], 'step': state['step'], 'controller': controller, 'policy_seed': SEED if controller in models else 0, 'selected_budget': selection.get(controller, 0), 'repetition': repetition, 'elapsed_ms': elapsed, 'mpc_status': env.mpc_last_info['status'] if controller.startswith('MPC-') else -1, 'mpc_fallback': int(env.mpc_last_info['fallback']) if controller.startswith('MPC-') else 0, 'qp_execution_path': info['qp_execution_path'], 'executed_battery_power_kw': info['p'], 'soc_after_kwh': info['soc'], 'grid_violation_kw': info['grid_violation']}
                (warmups if repetition == 0 else rows).append(row)
    frame = pd.DataFrame(rows)
    warm = pd.DataFrame(warmups)
    frame.to_csv(OUT / 'timing_state_rows.csv', index=False)
    warm.to_csv(OUT / 'timing_state_warmup_rows.csv', index=False)
    summaries = []
    for controller in CONTROLLERS:
        group = frame[frame.controller == controller]
        elapsed = group.elapsed_ms.to_numpy()
        summaries.append({'controller': controller, 'calls': len(group), 'unique_states': int(group.state_id.nunique()), 'mean_ms': float(elapsed.mean()), 'median_ms': float(np.median(elapsed)), 'p90_ms': float(np.quantile(elapsed, .90)), 'p95_ms': float(np.quantile(elapsed, .95)), 'p99_ms': float(np.quantile(elapsed, .99)), 'maximum_ms': float(elapsed.max()), 'minimum_ms': float(elapsed.min()), 'mpc_status_counts': {str(k): int(v) for k, v in Counter(group.mpc_status).items()} if controller.startswith('MPC-') else {}, 'mpc_fallback_calls': int(group.mpc_fallback.sum()), 'qp_execution_path_counts': {k: int(v) for k, v in Counter(group.qp_execution_path).items()}, 'maximum_within_state_executed_power_spread_kw': float(group.groupby('state_id').executed_battery_power_kw.agg(lambda x: x.max() - x.min()).max())})
    checks = {'exact_25_requested_states': len(states) == 25 and {(s['day'], s['step']) for s in states} == {(d, t) for d in VALIDATION_DAYS for t in STEPS}, 'all_reconstructed_observations_identical': True, 'all_750_timed_calls': len(frame) == 750, 'all_150_warmup_calls': len(warm) == 150, 'five_repeats_per_controller_state': frame.groupby(['controller', 'state_id']).size().eq(5).all(), 'all_timed_numeric_finite': np.isfinite(frame.select_dtypes(include='number')).all().all(), 'all_warmup_numeric_finite': np.isfinite(warm.select_dtypes(include='number')).all().all(), 'positive_timing': frame.elapsed_ms.gt(0).all(), 'one_thread_environment_variables': all(os.environ[n] == '1' for n in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])}
    protocol = {'started_at_utc': started, 'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'Serial repeated execution costs on 25 common Rule-derived validation states; not a full state-distribution estimate or deployment worst-case bound.', 'validation_days': VALIDATION_DAYS, 'steps': STEPS, 'states': states, 'controllers': CONTROLLERS, 'policy_seed': SEED, 'selected_budgets': selection, 'selected_rule_cap_kw': rule_selection['selected_cap_kw'], 'baseline_selection_sha256': driver.core.sha256(OUT / 'baseline_selection.json'), 'warmups_per_controller_state': 1, 'timed_repetitions_per_controller_state': 5, 'controller_order': 'Rotate the fixed six-controller order by state_id modulo six; all calls execute serially in one process.', 'state_restoration': 'Construct a fresh CausalEnv, reset the same day and seed, then restore soc, previous battery power, remaining EV demand and step. No OSQP engine is copied. Diagnostic histories/counters are reset for each call; this is a one-step timing probe.', 'timed_scope': 'Current observation including causal forecast slots, controller proposal, common QP and full env.step including next-state update and returned observation/final-step diagnostics.', 'excluded_scope': 'Source/model loading, state-trajectory construction, environment construction/reset/restoration, observation-identity assertion and result serialization.', 'proxy': 'Not enabled in this reanalysis; the four restored physical fields determine current observation and execution constraints.', 'python': sys.version, 'platform': platform.platform(), 'processor': platform.processor(), 'logical_cpu_count': os.cpu_count(), 'thread_environment': {n: os.environ[n] for n in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']}, 'native_solver_threads': 'Existing MPC/OSQP solver options are unchanged; environment variables are recorded, not a claim of independently measured native-thread counts.', 'numpy': np.__version__, 'model_sha256': {name: driver.core.sha256(driver.model_path(name, SEED, budget)) for name, budget in selection.items()}, 'source_code_sha256': {name: driver.core.sha256(driver.ROOT / 'src' / name) for name in ['timing_common_states.py', 'causal_run.py', 'run_experiment.py', 'matched_mpc.py', 'review_baselines.py']}}
    driver.save_json(OUT / 'timing_state_protocol.json', protocol)
    result = {'status': 'passed' if all(checks.values()) else 'failed', 'scope': protocol['scope'], 'checks': {k: bool(v) for k, v in checks.items()}, 'summaries': summaries, 'warmup_mpc_status_counts': {str(k): int(v) for k, v in Counter(warm[warm.controller.str.startswith('MPC-')].mpc_status).items()}, 'warmup_mpc_fallback_calls': int(warm.mpc_fallback.sum()), 'warmup_qp_path_counts': {k: int(v) for k, v in Counter(warm.qp_execution_path).items()}, 'artifact_sha256': {name: driver.core.sha256(OUT / name) for name in ['timing_state_rows.csv', 'timing_state_warmup_rows.csv', 'timing_state_protocol.json']}}
    driver.save_json(OUT / 'timing_state_summary.json', result)
    print(json.dumps({'status': result['status'], 'summaries': summaries}))


if __name__ == '__main__':
    main()
