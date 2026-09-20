from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd
import causal_run as cr

OUT = cr.OUT
summaries, event_rows = [], []
for controller in ['MPC-H24+QP', 'MPC-H24-FRR+QP']:
    path = OUT / f'mpc_diagnostics_{controller}.jsonl'
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    trace = pd.read_csv(OUT / f'trajectory_{controller}.csv')
    daily = pd.read_csv(OUT / f'baseline_{controller}.csv')
    accepted = [row for row in records if not row['fallback']]
    gaps = np.array([row['mip_gap'] for row in accepted if row['mip_gap'] is not None], float)
    assert len(records) == len(trace) == 6240 and np.isfinite(gaps).all()
    nonoptimal = []
    for row in records:
        if row['status'] != 0 or row['fallback']:
            event = {key: row[key] for key in ['controller', 'day', 'step', 'status', 'fallback', 'seconds', 'mip_gap', 'constraint_violation_max', 'message']}
            event.update({'hour': row['step'] / 4, 'evaluation_day_block': '300-319' if row['day'] < 320 else '320-339' if row['day'] < 340 else '340-364',
                          'within_day_block': ['00-06', '06-12', '12-18', '18-24'][row['step'] // 24]})
            nonoptimal.append(event)
    event_rows.extend(nonoptimal)
    item = {'controller': controller, 'calls': len(records), 'status_counts': dict(Counter(str(row['status']) for row in records)),
            'optimal_accepted': sum(row['status'] == 0 and not row['fallback'] for row in records),
            'time_limit_incumbents': sum(row['status'] == 1 and not row['fallback'] for row in records),
            'fallbacks': sum(row['fallback'] for row in records),
            'fallback_status_counts': dict(Counter(str(row['status']) for row in records if row['fallback'])),
            'accepted_gap': {'count': len(gaps), 'minimum': float(gaps.min()), 'median': float(np.median(gaps)),
                             'p95': float(np.quantile(gaps, 0.95)), 'maximum': float(gaps.max()), 'above_0_001': int((gaps > .001).sum())},
            'accepted_constraint_violation_max': max(row['constraint_violation_max'] for row in accepted),
            'nonoptimal_by_evaluation_day_block': dict(Counter(row['evaluation_day_block'] for row in nonoptimal)),
            'nonoptimal_by_within_day_block': dict(Counter(row['within_day_block'] for row in nonoptimal)),
            'time_limit_by_evaluation_day_block': dict(Counter(row['evaluation_day_block'] for row in nonoptimal if row['status'] == 1)),
            'time_limit_by_within_day_block': dict(Counter(row['within_day_block'] for row in nonoptimal if row['status'] == 1)),
            'accepted_plans_with_max_ramp_above_45kw': sum(row['max_ramp_kw'] > 45 + 1e-5 for row in accepted),
            'executed_steps_with_ramp_above_45kw': int(sum((np.abs(np.diff(np.r_[0.0, part.p.to_numpy()])) > 45 + 1e-5).sum() for _, part in trace.groupby('day'))),
            'execution_record': json.loads((OUT / f'baseline_execution_{controller}.json').read_text(encoding='utf-8')),
            'diagnostics_sha256': cr.core.sha256(path)}
    if controller == 'MPC-H24+QP':
        old = pd.read_csv(cr.ROOT.parents[1] / 'submission_reanalysis_20260908/artifacts/baseline_MPC-H24+QP.csv')
        keys = ['cost', 'carbon', 'task_score', 'peak', 'grid_excess_energy_kwh', 'terminal_soc_error_kwh', 'mean_action_correction']
        item['historical_max_absolute_daily_metric_difference'] = {key: float((daily[key] - old[key]).abs().max()) for key in keys}
        item['historical_time_limit_incumbents'] = int(old.mpc_time_limit_incumbents.sum())
        item['historical_fallbacks'] = int(old.mpc_fallbacks.sum())
    summaries.append(item)
pd.DataFrame(event_rows).to_csv(OUT / 'baseline_mpc_nonoptimal_events.csv', index=False)
cr.save_json(OUT / 'baseline_solver_audit.json', {'status': 'passed', 'summaries': summaries,
             'scope': 'Actual deterministic baseline runs, conducted concurrently with retraining. Wall-clock limits may depend on machine load; no best-run selection or selective rerun was performed.',
             'source_sha256': cr.core.sha256(Path(__file__)), 'events_sha256': cr.core.sha256(OUT / 'baseline_mpc_nonoptimal_events.csv')})
print(json.dumps(summaries, indent=2))
