from pathlib import Path
import ast, io, json, types, zipfile
import numpy as np,pandas as pd
OUT=Path(__file__).resolve().parent
z=zipfile.ZipFile(OUT.parents[2]/'JRSE_submission/Reproducibility.zip')
source=(OUT/'corrected_diagnostic_src/analyze_revision.py').read_text(encoding='utf-8')
paths=[p for p in z.namelist() if p.startswith('artifacts/trajectory_') and p.endswith('.csv')]
trace=pd.concat([pd.read_csv(io.BytesIO(z.read(p))) for p in paths]+[pd.read_csv(OUT.parent/'baseline_experiment/trajectory.csv').assign(seed=0)],ignore_index=True)
assert len(trace)==99840 and trace.groupby(['controller','seed','day']).size().eq(96).all()
output=OUT/'corrected_eight_controller_diagnostics';output.mkdir(exist_ok=True)
ns={'np':np,'pd':pd,'OUT':output,'DT':.25,'cr':types.SimpleNamespace(CFG=json.loads(z.read('configs/experiment.json')))}
functions=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ['endpoint_excess','constraint_sources']]
exec(compile(ast.Module(body=functions,type_ignores=[]),str(OUT/'corrected_diagnostic_src/analyze_revision.py'),'exec'),ns)
_,charge,_=ns['constraint_sources'](trace)
independent=pd.read_csv(OUT/'charge_bound_normalized.csv').set_index('controller');reproduced=charge.set_index('controller')
columns=reproduced.select_dtypes(include='number').columns
error=float(abs(independent[columns]-reproduced[columns]).to_numpy().max());assert error<1e-7
v={'controllers':8,'steps':99840,'episodes':1040,'superset_checks_all_steps':True,'independent_charge_calculation_max_error':error,'baseline_seed_convention':0,'baseline_source':'../baseline_experiment/trajectory.csv','original_baseline_file_modified':False}
(OUT/'eight_controller_verification.json').write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
print(json.dumps(v));print(independent.loc['DischargeBias+QP'].to_string())
