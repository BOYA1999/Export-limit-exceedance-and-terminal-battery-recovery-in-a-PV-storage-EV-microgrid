from pathlib import Path
import hashlib, io, json, zipfile
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PACKAGE = ROOT/'JRSE_submission/Reproducibility.zip'
SHA = lambda value: hashlib.sha256(value).hexdigest()
source_hash = SHA(PACKAGE.read_bytes())
source_members = {}
z = zipfile.ZipFile(PACKAGE)
def read(name):
    content = z.read(name)
    source_members[name] = SHA(content)
    return pd.read_csv(io.BytesIO(content))
controllers = ['Rule+QP','NetLoadRule+QP','MPC-H24+QP','MPC-H24-FRR+QP','MPC-H24-XP+QP','PPO-QP','PPO-QP-no-penalty','DischargeBias+QP']
labels = dict(zip(controllers,['LegacyRule','NetLoadRule','MPC H24','MPC relaxed','MPC export-first','PPO 0.1','PPO 0','DischargeBias']))
config = json.loads(z.read('configs/experiment.json'))
source_members['configs/experiment.json'] = SHA(z.read('configs/experiment.json'))
dt, power, ramp = config['dt_hours'], config['bess']['power_kw'], config['bess']['ramp_kw']
trace = pd.concat([read('artifacts/trajectory_'+c+'.csv') for c in controllers[:5]]+[read(f'artifacts/trajectory_seed_{s}.csv') for s in config['seeds']],ignore_index=True).sort_values(['controller','seed','day','step']).reset_index(drop=True)
baseline_path = OUT.parent/'baseline_experiment/trajectory.csv'
baseline_bytes = baseline_path.read_bytes()
source_members['additional_baseline/trajectory.csv'] = SHA(baseline_bytes)
baseline_trace = pd.read_csv(io.BytesIO(baseline_bytes)).assign(seed=0)
assert len(baseline_trace)==6240 and set(baseline_trace.controller)=={'DischargeBias+QP'}
trace = pd.concat([trace,baseline_trace],ignore_index=True).sort_values(['controller','seed','day','step']).reset_index(drop=True)
assert len(trace)==99840 and not trace.duplicated(['controller','seed','day','step']).any()
assert trace.groupby(['controller','seed','day']).size().eq(96).all()
previous = trace.groupby(['controller','seed','day']).p.shift().fillna(0).to_numpy()
physical_lo, physical_hi = trace.physical_lower_kw.to_numpy(),trace.physical_upper_kw.to_numpy()
terminal_lo, terminal_hi = trace.terminal_lower_kw.to_numpy(),trace.terminal_upper_kw.to_numpy()
actual_lo, actual_hi, tolerance = trace.power_lower_kw.to_numpy(),trace.power_upper_kw.to_numpy(),trace.tolerance_kw.to_numpy()
ramp_lo, ramp_hi = previous-ramp, previous+ramp
legacy_ramp_on = (np.maximum(physical_lo,ramp_lo)<=np.minimum(physical_hi,ramp_hi)+tolerance)&(np.maximum.reduce([physical_lo,ramp_lo,terminal_lo])<=np.minimum.reduce([physical_hi,ramp_hi,terminal_hi])+tolerance)
ramp_on = (np.maximum(physical_lo,ramp_lo)<=np.minimum(physical_hi,ramp_hi))&trace.terminal_ramp_relaxation.eq(0).to_numpy()
headroom_lo = -(config['bess']['soc_max']*config['bess']['energy_kwh']-trace.soc_kwh.to_numpy())/(config['bess']['eta_charge']*dt)
rated_lo = np.full(len(trace),-power)
terminal_on = terminal_lo>physical_lo+tolerance
effective_terminal = np.where(terminal_on,terminal_lo,-np.inf)
effective_ramp = np.where(ramp_on,ramp_lo,-np.inf)
lo_no_rating = np.maximum.reduce([headroom_lo,effective_terminal,effective_ramp])
lo_no_headroom = np.maximum.reduce([rated_lo,effective_terminal,effective_ramp])
base = trace.base_load_minus_pv_kw.to_numpy()+trace.lighting_kw.to_numpy()
def export_minimum(lower):
    grid_max = base+config['loads']['ev_max_kw']*trace.ev_upper_fraction.to_numpy()-lower
    return np.maximum(0,config['loads']['grid_min_kw']-grid_max-.01)
actual, no_rating, no_headroom = [export_minimum(v) for v in [actual_lo,lo_no_rating,lo_no_headroom]]
assert np.all(lo_no_rating<=actual_lo+tolerance+1e-12) and np.all(lo_no_headroom<=actual_lo+tolerance+1e-12)
trace.loc[legacy_ramp_on & ~ramp_on].to_csv(OUT/'diagnostic_ramp_branch_mismatches.csv',index=False)
assert np.all(no_rating<=actual+1e-7) and np.all(no_headroom<=actual+1e-7)
active = {'rated_charge_power':abs(actual_lo-rated_lo)<=1e-5,'energy_headroom':abs(actual_lo-headroom_lo)<=1e-5,'terminal_lower_envelope':terminal_on&(abs(actual_lo-terminal_lo)<=1e-5),'ramp_lower_bound':ramp_on&(abs(actual_lo-ramp_lo)<=1e-5)}
rows=[]
for c in controllers:
    idx=np.flatnonzero(trace.controller.to_numpy()==c);events=idx[actual[idx]>1e-9];episodes=trace.iloc[idx].groupby(['seed','day']).ngroups
    row={'controller':c,'controller_label':labels[c],'episodes':episodes,'steps':len(idx),'actual_export_infeasible_steps':len(events),'export_events_per_episode':len(events)/episodes,'export_infeasible_pct_all_steps':100*len(events)/len(idx)}
    for name,mask in active.items():
        count=int(mask[events].sum());row[f'active_{name}_steps']=count;row[f'active_{name}_pct_export_events']=100*count/len(events)
    for name,value in [('charge_rating',no_rating),('energy_headroom',no_headroom)]:
        count=int((value[events]<=1e-9).sum());row[f'resolved_export_steps_remove_{name}']=count;row[f'resolved_export_pct_remove_{name}']=100*count/len(events)
        row[f'mean_daily_export_minimum_change_remove_{name}_kwh']=float((value[idx]-actual[idx]).sum()*dt/episodes)
    rows.append(row)
charge=pd.DataFrame(rows);original=read('artifacts/charge_bound_diagnostics.csv').set_index('controller')
charge_error=max(abs(charge.set_index('controller').reindex(original.index)[column]-original[column]).max() for column in original.columns if column!='active_counts_nonexclusive')
assert charge_error<1e-7
charge.to_csv(OUT/'charge_bound_normalized.csv',index=False)

h24=read('artifacts/baseline_MPC-H24+QP.csv').set_index('split_day').sort_index()
xp=read('artifacts/baseline_MPC-H24-XP+QP.csv').set_index('split_day').sort_index()
assert list(h24.index)==list(xp.index)==list(range(300,365))
metrics=['cost','carbon','task_score','peak','grid_excess_energy_kwh','grid_export_violation_kwh','ramp_slack_kw_mean','max_ramp_kw']
diff=(xp[metrics]-h24[metrics]).to_numpy();rng=np.random.default_rng(20260918);rows=[]
for j,m in enumerate(metrics):
    lo,hi=np.quantile(diff[:,j][rng.integers(0,65,size=(20000,65))].mean(axis=1),[.025,.975])
    rows.append({'metric':m,'mean_difference':diff[:,j].mean(),'method':'day_cluster','ci_lower':lo,'ci_upper':hi,'rng_seed':20260918,'day_clusters':65,'bootstrap_replicates':20000})
block_indices={};rng=np.random.default_rng(20260920)
for length in [7,14]:
    starts=rng.integers(65,size=(20000,int(np.ceil(65/length))))
    index=((starts[:,:,None]+np.arange(length))%65).reshape(20000,-1)[:,:65];block_indices[length]=index
    ci=np.quantile(diff[index].mean(axis=1),[.025,.975],axis=0)
    for j,m in enumerate(metrics):rows.append({'metric':m,'mean_difference':diff[:,j].mean(),'method':f'circular_block_{length}','ci_lower':ci[0,j],'ci_upper':ci[1,j],'rng_seed':20260920,'day_clusters':65,'bootstrap_replicates':20000})
xp_intervals=pd.DataFrame(rows);old=read('artifacts/export_priority_comparison.csv').set_index('metric')
xp_error=max(abs(xp_intervals[xp_intervals.method=='day_cluster'].set_index('metric')[new]-old[oldname]).max() for new,oldname in [('mean_difference','mean_difference_export_priority_minus_original'),('ci_lower','day_cluster_ci_lower'),('ci_upper','day_cluster_ci_upper')])
assert xp_error<1e-12
xp_intervals.to_csv(OUT/'export_first_vs_h24_intervals.csv',index=False)

original_intervals=read('artifacts/paired_intervals.csv');peaks=original_intervals[original_intervals.metric=='peak'][['comparator','method','mean_difference','ci_lower','ci_upper','day_clusters','fixed_trained_seeds','bootstrap_replicates']].copy();peaks['rng_seed']=20260908
policy=pd.concat([read(f'artifacts/test_seed_{s}.csv') for s in config['seeds']],ignore_index=True)
policy_peak=policy[policy.controller=='PPO-QP'].groupby('split_day').peak.mean().reindex(h24.index)
old_xp=read('artifacts/figure2_export_priority_intervals.csv').set_index('metric').loc['peak']
peakdiff=policy_peak.to_numpy()-xp.peak.to_numpy();extra=[{'comparator':'MPC-H24-XP+QP','method':'day_cluster','mean_difference':peakdiff.mean(),'ci_lower':old_xp.day_cluster_ci_lower,'ci_upper':old_xp.day_cluster_ci_upper,'day_clusters':65,'fixed_trained_seeds':5,'bootstrap_replicates':20000,'rng_seed':20260918}]
for length,index in block_indices.items():
    lo,hi=np.quantile(peakdiff[index].mean(axis=1),[.025,.975]);extra.append({**extra[0],'method':f'circular_block_{length}','ci_lower':lo,'ci_upper':hi,'rng_seed':20260920})
peaks=pd.concat([peaks,pd.DataFrame(extra)],ignore_index=True);peaks['contains_zero']=(peaks.ci_lower<=0)&(peaks.ci_upper>=0)
peaks.to_csv(OUT/'figure2_peak_intervals.csv',index=False)

active_table=[];relax_table=[]
for r in charge.itertuples(index=False):
    active_table.append({'Controller':r.controller_label,'Episodes':r.episodes,'Export events':r.actual_export_infeasible_steps,'Events/day':f'{r.export_events_per_episode:.3f}','Rating active %':f'{r.active_rated_charge_power_pct_export_events:.2f}','Headroom active %':f'{r.active_energy_headroom_pct_export_events:.2f}','Terminal active %':f'{r.active_terminal_lower_envelope_pct_export_events:.2f}','Ramp active %':f'{r.active_ramp_lower_bound_pct_export_events:.2f}'})
    relax_table.append({'Controller':r.controller_label,'Resolved no rating n (%)':f'{r.resolved_export_steps_remove_charge_rating} ({r.resolved_export_pct_remove_charge_rating:.2f})','Resolved no headroom n (%)':f'{r.resolved_export_steps_remove_energy_headroom} ({r.resolved_export_pct_remove_energy_headroom:.2f})','Daily change no rating kWh':f'{r.mean_daily_export_minimum_change_remove_charge_rating_kwh:.3f}','Daily change no headroom kWh':f'{r.mean_daily_export_minimum_change_remove_energy_headroom_kwh:.3f}'})
active_table=pd.DataFrame(active_table);relax_table=pd.DataFrame(relax_table)
active_table.to_csv(OUT/'supplement_charge_bound_active_table.csv',index=False);relax_table.to_csv(OUT/'supplement_charge_bound_relaxation_table.csv',index=False)
def wide(table,key):
    result=[]
    for value,g in table.groupby(key,sort=False):
        row={key:value,'Mean difference':f'{g.mean_difference.iloc[0]:.3f}'}
        for r in g.itertuples():row[r.method]=f'[{r.ci_lower:.3f}, {r.ci_upper:.3f}]'
        result.append(row)
    return pd.DataFrame(result)
notes='Percentages use each controller\'s own export-infeasible event count as denominator; active and resolved counts can overlap. Events/day divides by 65 deterministic-controller episodes or 325 PPO seed/day episodes, without treating the latter as independent scenario days. Daily changes sum relaxed-minus-original endpoint-minimum export power over every recorded step, multiply by 0.25 h, and divide by the controller episode count. Negative changes indicate a smaller fixed-state lower bound on export exceedance; they are not predicted closed-loop gains. Displayed -0.000 is numerical zero.'
md=['# Supplementary statistical updates','', '## Export-side active battery bounds', '',active_table.to_markdown(index=False),'','## Fixed-state charge-bound relaxation','',relax_table.to_markdown(index=False),'',notes,'','## Export-first MPC minus hard-ramp MPC','',wide(xp_intervals,'metric').to_markdown(index=False),'','All eight means and day-cluster intervals reproduce Table S17. The added block intervals use the same 65 paired dates and hold the controllers fixed.','', '## Figure 2 peak-import contrasts: PPO-QP minus comparator','',wide(peaks,'comparator').to_markdown(index=False),'','The peak contrast versus relaxed MPC excludes zero under day resampling but includes zero under both block lengths. Near-zero PPO-variant contrasts reflect numerical agreement.','']
(OUT/'supplementary_tables.md').write_text('\n'.join(md),encoding='utf-8')
protocol={'source_archive_sha256':source_hash,'source_members_sha256':source_members,'numpy_version':np.__version__,'day_cluster':{'rng':'NumPy default_rng / PCG64','seed':20260918,'replicates':20000,'algorithm':'For S17, iterate metrics in stored order; independently draw a 20000 by 65 integer index matrix for each metric from one seeded RNG. Percentile endpoints use numpy.quantile at 0.025 and 0.975.','metric_order':metrics},'new_circular_blocks':{'rng':'NumPy default_rng / PCG64','seed':20260920,'replicates':20000,'length_order':[7,14],'algorithm':'At each length L, draw 20000 by ceil(65/L) uniform starts in [0,64]. Append L consecutive indices modulo 65 for each start; concatenate blocks and truncate to 65. Reuse these date-index matrices for every S17 metric and the PPO-versus-export-first peak contrast. Each replicate is the mean of 65 paired differences.','conditioning':'Fixed controllers, five fixed PPO policies where applicable, inspected 65-day chronological scenario; no seed, site, policy-selection or solver-rerun resampling.'},'existing_figure2_peak_intervals':'Original five comparators retain verified archive intervals, using seed 20260908 and day-cluster followed by length-7 and length-14 circular sampling. XP day interval retains seed 20260918 (cost then peak then export); new XP block intervals use the new shared block matrices.','checks':{'trace_rows':len(trace),'controller_count':len(charge),'charge_bound_recompute_max_absolute_error':float(charge_error),'s17_day_interval_recompute_max_absolute_error':float(xp_error),'all_removed_intervals_are_supersets_with_recorded_tolerance':True,'diagnostic_ramp_branch_mismatch_steps':int((legacy_ramp_on & ~ramp_on).sum()),'diagnostic_ramp_branch_mismatch_export_events':int(((legacy_ramp_on & ~ramp_on)&(actual>1e-9)).sum()),'minimum_export_never_increases':True,'no_original_archive_change':SHA(PACKAGE.read_bytes())==source_hash}}
assert all([protocol['checks']['no_original_archive_change'],len(xp_intervals)==24,len(peaks)==18])
(OUT/'verification.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'verification.md').write_text('# Verification\n\nAll statistics were computed from the current Reproducibility.zip without changing that archive or any manuscript.\n\n'+notes+'\n\nThe 99,840 analyzed steps (93,600 archived steps plus 6,240 DischargeBias baseline steps) cover exactly 96 steps per controller/seed/day. Every archived charge-bound count and daily change was independently reproduced; maximum absolute discrepancy: '+f'{charge_error:.3g}'+'. All eight existing S17 day intervals were recomputed; maximum discrepancy: '+f'{xp_error:.3g}'+'. The added block algorithm uses PCG64 seed 20260920, 20,000 replicates, block lengths 7 then 14, circular indices modulo 65, and truncation to 65 days. Detailed source hashes and sampling order are in verification.json.\n\nExisting Figure 2 peak intervals were independently checked against the daily records in the preceding read-only review and retained without changes; the new export-first block intervals are generated here. No multiplicity adjustment, equivalence test, fresh-policy inference, or closed-loop intervention effect is claimed.\n',encoding='utf-8')
(OUT/'manifest.sha256').write_text(''.join(SHA(p.read_bytes())+'  '+p.relative_to(OUT).as_posix()+'\n' for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.sha256'),encoding='utf-8')
print(json.dumps(protocol['checks'],ensure_ascii=False));print(xp_intervals[xp_intervals.metric=='grid_export_violation_kwh'].to_string(index=False))


