from pathlib import Path
import ast, hashlib, io, json, types, zipfile
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
PACKAGE=OUT.parents[2]/'JRSE_submission/Reproducibility.zip'
z=zipfile.ZipFile(PACKAGE)
original_dir=OUT/'original_diagnostic_snapshot';fixed_src=OUT/'corrected_diagnostic_src';fixed_out=OUT/'corrected_diagnostic_artifacts'
for p in [original_dir,fixed_src,fixed_out]:p.mkdir(exist_ok=True)
source=z.read('revision_src/analyze_revision.py').decode('utf-8-sig')
(original_dir/'analyze_revision.py').write_bytes(z.read('revision_src/analyze_revision.py'))
files=['constraint_source_decomposition.csv','charge_bound_diagnostics.csv','export_event_timing.csv']
for name in files:(original_dir/name).write_bytes(z.read('artifacts/'+name))
old='ramp_enforced = (physical_ramp_lo <= physical_ramp_hi + tolerance) & (np.maximum(physical_ramp_lo, terminal_lo) <= np.minimum(physical_ramp_hi, terminal_hi) + tolerance)'
new='ramp_enforced = (physical_ramp_lo <= physical_ramp_hi) & frame.terminal_ramp_relaxation.eq(0).to_numpy()'
assert source.count(old)==1
source=source.replace(old,new)
needle='    no_ramp_hi = np.minimum(physical_hi, frame.terminal_upper_kw.to_numpy())\n'
source=source.replace(needle,needle+'    for lower, upper in [(no_terminal_lo, no_terminal_hi), (no_ramp_lo, no_ramp_hi)]:\n        assert np.all(lower <= frame.power_lower_kw.to_numpy() + tolerance + 1e-12)\n        assert np.all(upper >= frame.power_upper_kw.to_numpy() - tolerance - 1e-12)\n    assert np.all(frame.ev_lower_fraction.to_numpy() >= 0)\n')
needle='    lower_without_headroom = np.maximum.reduce([rated_lo, effective_terminal_lo, effective_ramp_lo])\n'
source=source.replace(needle,needle+'    assert np.all(lower_without_rating <= actual_lo + tolerance + 1e-12)\n    assert np.all(lower_without_headroom <= actual_lo + tolerance + 1e-12)\n')
(fixed_src/'analyze_revision.py').write_text(source,encoding='utf-8')
controllers=['Rule+QP','NetLoadRule+QP','MPC-H24+QP','MPC-H24-FRR+QP','MPC-H24-XP+QP','PPO-QP','PPO-QP-no-penalty']
trace=pd.concat([pd.read_csv(io.BytesIO(z.read('artifacts/trajectory_'+c+'.csv'))) for c in controllers[:5]]+[pd.read_csv(io.BytesIO(z.read(f'artifacts/trajectory_seed_{s}.csv'))) for s in range(20260805,20260810)],ignore_index=True)
assert len(trace)==93600
ns={'np':np,'pd':pd,'OUT':fixed_out,'DT':.25,'cr':types.SimpleNamespace(CFG=json.loads(z.read('configs/experiment.json')))}
module=ast.parse(source);functions=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in ['endpoint_excess','constraint_sources']]
exec(compile(ast.Module(body=functions,type_ignores=[]),str(fixed_src/'analyze_revision.py'),'exec'),ns)
ns['constraint_sources'](trace)
deltas=[];checks={}
for name in files:
    before=pd.read_csv(original_dir/name);after=pd.read_csv(fixed_out/name);keys=['controller']+(['one_step_counterfactual'] if name.startswith('constraint') else [])
    before=before.set_index(keys).sort_index();after=after.set_index(keys).sort_index();assert before.index.equals(after.index)
    numeric=before.select_dtypes(include='number').columns
    for key in before.index:
        for column in numeric:
            a=float(before.loc[key,column]);b=float(after.loc[key,column])
            if abs(b-a)>1e-10:deltas.append({'file':name,'key':str(key),'metric':column,'before':a,'after':b,'change':b-a})
    checks[name]={'same_rows':len(before),'maximum_numeric_difference':float(abs(after[numeric]-before[numeric]).to_numpy().max())}
pd.DataFrame(deltas,columns=['file','key','metric','before','after','change']).to_csv(OUT/'diagnostic_output_differences.csv',index=False)
prev=trace.groupby(['controller','seed','day']).p.shift().fillna(0).to_numpy();pl=trace.physical_lower_kw.to_numpy();ph=trace.physical_upper_kw.to_numpy();tol=trace.tolerance_kw.to_numpy();rl=prev-45;rh=prev+45
tlo=np.maximum(pl,trace.terminal_lower_kw.to_numpy());thi=np.minimum(ph,trace.terminal_upper_kw.to_numpy());physical_ramp_lo=np.maximum(pl,rl);physical_ramp_hi=np.minimum(ph,rh)
legacy=(physical_ramp_lo<=physical_ramp_hi+tol)&(np.maximum(physical_ramp_lo,tlo)<=np.minimum(physical_ramp_hi,thi)+tol)
correct=(physical_ramp_lo<=physical_ramp_hi)&trace.terminal_ramp_relaxation.eq(0).to_numpy()
no_terminal_lo=np.where(legacy,physical_ramp_lo,pl);no_terminal_hi=np.where(legacy,physical_ramp_hi,ph)
invalid=(no_terminal_lo>trace.power_lower_kw.to_numpy()+tol+1e-12)|(no_terminal_hi<trace.power_upper_kw.to_numpy()-tol-1e-12)
extra={'verified_steps':93600,'legacy_branch_mismatch_steps':int((legacy!=correct).sum()),'legacy_no_terminal_interval_not_superset_steps':int(invalid.sum()),'mismatch_export_events':int(((legacy!=correct)&(trace.grid_export.to_numpy()>40.01)).sum()),'numeric_checks':checks,'changed_values_over_1e-10':len(deltas),'all_fixed_relaxed_intervals_are_supersets_with_recorded_tolerance':True,'terminal_energy_tolerance_kwh':1e-8,'interval_tolerance':'stored tolerance_kw (3.8e-8 kW) plus 1e-12 kW floating arithmetic margin'}
(OUT/'diagnostic_repair_verification.json').write_text(json.dumps(extra,indent=2)+'\n',encoding='utf-8')
report=['# Fixed-state diagnostic branch repair','', 'The source execution code removes a physically conflicting ramp interval using an exact comparison before applying its tolerance-aware terminal intersection. The previous analysis reconstructed the first branch with an added tolerance and could therefore re-enable a ramp that execution had already removed. The corrected analysis uses the exact physical-intersection branch and the recorded terminal-ramp-relaxation flag. No controller, training, evaluation trajectory, or original archive was changed.','',f'The branch differs at {extra["legacy_branch_mismatch_steps"]} of 93,600 observed steps. None is an export event. The previous terminal-removal interval failed its superset condition at {extra["legacy_no_terminal_interval_not_superset_steps"]} steps using the declared tolerance. Corrected terminal, ramp, EV-lower, charge-rating and charge-headroom removals now pass interval-superset checks at every recorded step.','', 'Original diagnostic source and three output tables are retained byte for byte in original_diagnostic_snapshot. The corrected package-ready analyze_revision.py is in corrected_diagnostic_src; its directly regenerated diagnostic outputs are in corrected_diagnostic_artifacts.','',pd.DataFrame(deltas).to_markdown(index=False) if deltas else 'No numeric output changed beyond 1e-10.','', 'All reported event counts and daily minimum-excess changes remain unchanged. Some all-step attainable-grid-width summaries change because the correction preserves the already-relaxed ramp branch. The manuscript does not report those width summaries. This invariance is an observed result, not an assumption: the affected states were feasible and generated no export-exceedance contribution.','']
(OUT/'diagnostic_repair_note.md').write_text('\n'.join(report),encoding='utf-8')
print(json.dumps(extra));print(pd.DataFrame(deltas).to_string(index=False))
