from pathlib import Path, PurePosixPath
import argparse
import ast
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import urllib.parse
import zipfile
import numpy as np
import pandas as pd

sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
load=lambda path:json.loads(path.read_text(encoding='utf-8-sig'))
ENV=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')

def safe_name(name):
    parts=PurePosixPath(name)
    return bool(name) and not parts.is_absolute() and '..' not in parts.parts and not any(c in name for c in [chr(92),':',chr(0)])

def extract(archive,destination):
    destination.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as zipped:
        names=zipped.namelist()
        assert len(names)==len(set(names)) and zipped.testzip() is None
        for member in zipped.infolist():
            assert safe_name(member.filename) and not stat.S_ISLNK(member.external_attr>>16)
            assert (destination/member.filename).resolve().is_relative_to(destination.resolve())
        zipped.extractall(destination)
    return len(names)

def content_audit(root):
    findings=[]; models=0; files=0
    patterns=[r'(?i)(?<![a-z0-9])[a-z]:[\\/]',r'(?i)/(?:'+'Users|home'+r')/[^\s/]+',r'(?i)sk-[a-z0-9]{16,}',r'(?i)(?:api[_-]?key|password|access[_-]?token)\s*[:=]\s*["\']?([a-z0-9/+_=.-]{12,})',r'(?i)[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}',r'\\\\[A-Za-z0-9_.-]+\\[A-Za-z0-9_.-]+']
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        files+=1
        if path.suffix=='.npz':
            with zipfile.ZipFile(path) as zipped:
                assert zipped.testzip() is None and all(safe_name(name) for name in zipped.namelist())
            with np.load(path,allow_pickle=False) as arrays:
                assert all(arrays[key].dtype.kind in 'fiu' and np.isfinite(arrays[key]).all() for key in arrays)
            models+=1
            continue
        raw=path.read_bytes()
        if path.suffix=='.pdf':
            with fitz.open(path) as document:
                text=json.dumps(document.metadata)+'\n'+document.get_xml_metadata()+'\n'+'\n'.join(page.get_text() for page in document)
                text+='\n'+'\n'.join(document.xref_object(index) for index in range(1,document.xref_length()))
        elif path.suffix=='.png':
            with Image.open(path) as picture:
                picture.verify()
            with Image.open(path) as picture:
                text=str(picture.info)+str(picture.getexif())
        else:
            text=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')
        decoded=urllib.parse.unquote(text)
        for index,pattern in enumerate(patterns):
            if re.search(pattern,decoded):
                findings.append({'file':path.relative_to(root).as_posix(),'pattern_index':index})
        if path.suffix=='.py':
            ast.parse(text)
        elif path.suffix=='.json':
            json.loads(text)
        elif path.suffix=='.jsonl':
            for line in text.splitlines():
                json.loads(line)
    assert not findings,findings
    assert models==41  # 40 frozen checkpoints plus one duplicated post-hoc source snapshot
    return {'inspected_files':files,'finite_numeric_checkpoints':models,'content_findings':findings,'text_json_jsonl_svg_and_nested_npz_content_checked':True}

def integrity(root):
    expected={}
    for line in (root/'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1)
        assert safe_name(name) and name not in expected
        assert sha(root/name)==digest,name
        expected[name]=digest
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert actual==set(expected)|{'MANIFEST.sha256'}
    artifacts=root/'artifacts'; contract=load(artifacts/'run_contract.json')
    assert contract['protocol_sha256']==sha(root/'PLAN.md') and contract['config_sha256']==sha(root/'configs/experiment.json')
    for name,digest in contract['code_hashes'].items():
        assert sha(root/'src'/name)==digest
    selection=load(artifacts/'selection.json'); baseline=load(artifacts/'baseline_selection.json')
    assert selection['run_contract_sha256']==sha(artifacts/'run_contract.json')
    for record in [baseline,load(artifacts/'baseline_run_contract.json')]:
        for name,digest in record['hashes'].items():
            normalized=name.replace(chr(92),'/')
            assert safe_name(normalized) and sha(root/normalized)==digest
    for record in [selection,baseline]:
        assert record['test_evaluation_not_started']
        for name,digest in record['selection_inputs'].items():
            assert sha(artifacts/name)==digest
    for name,digest in selection['selected_model_hashes'].items():
        assert sha(artifacts/name)==digest
    for seed in range(20260805,20260810):
        assert load(artifacts/f'test_receipt_{seed}.json')['selection_sha256']==sha(artifacts/'selection.json')
        for controller in selection['selected_budgets']:
            summary=load(artifacts/f'train_{controller}_{seed}.json')
            for budget,digest in summary['checkpoints'].items():
                assert sha(artifacts/f'{controller}_{seed}_u{budget}.npz')==digest
    changes=load(root/'provenance/packaging_changes.json')
    for name,digest in load(artifacts/'baseline_verification.json')['outputs'].items():
        key='artifacts/'+name
        assert (sha(root/key)==digest) or (changes['curated_files'][key]['original_sha256']==digest)
    source=load(root/'provenance/source_verification.json')
    for item in source['csv_comparisons']+[source['pvwatts_comparison']]:
        assert sha((root/'provenance'/item['local_file']).resolve())==item['local_sha256']
    for item in source['bundled_licenses']:
        assert sha(root/item['bundled_file'])==item['full_notice_text_sha256']
    for name,digest in load(artifacts/'preprocessing.json')['source_hashes'].items():
        assert sha(root/'data'/name)==digest
    for controller,budget in selection['selected_budgets'].items():
        frame=pd.concat([pd.read_csv(artifacts/f'validation_{controller}_{seed}.csv') for seed in range(20260805,20260810)])
        assert len(frame)==1200
        frame['service_failure']=((frame.soc_violation_rate>0)|(frame.terminal_soc_error_kwh.abs()>1e-5)|(frame.ev_completion<1)|(frame.lighting_energy_kwh.sub(472.5).abs()>1e-5)).astype(int)
        grouped=frame.groupby('budget')[['service_failure','grid_excess_energy_kwh','cost','mean_action_correction']].mean().reset_index()
        assert int(grouped.sort_values(['service_failure','grid_excess_energy_kwh','cost','mean_action_correction','budget']).iloc[0].budget)==budget
    rules=pd.concat([pd.read_csv(artifacts/f'validation_NetLoadRule_cap{cap}.csv') for cap in [35,60,100]])
    assert len(rules)==180
    rules['service_failure']=((rules.soc_violation_rate>0)|(rules.terminal_soc_error_kwh.abs()>1e-5)|(rules.ev_completion<1)|(rules.lighting_energy_kwh.sub(472.5).abs()>1e-5)).astype(int)
    grouped=rules.groupby('cap_kw')[['service_failure','grid_excess_energy_kwh','cost','mean_action_correction']].mean().reset_index()
    assert int(grouped.sort_values(['service_failure','grid_excess_energy_kwh','cost','mean_action_correction','cap_kw']).iloc[0].cap_kw)==baseline['selected_cap_kw']
    assert len(pd.read_csv(artifacts/'paired_intervals.csv'))==120
    counts={name:sum(1 for _ in (artifacts/f'mpc_diagnostics_{name}.jsonl').open(encoding='utf-8')) for name in ['MPC-H24+QP','MPC-H24-FRR+QP']}
    assert all(count==6240 for count in counts.values())
    mechanism=load(artifacts/'mechanism_analysis.json')
    assert mechanism['script_sha256']==sha(root/'src/analyze_mechanisms.py')
    for key in ['source_sha256','output_sha256']:
        for name,digest in mechanism[key].items():
            assert sha(artifacts/name)==digest
    timing=load(artifacts/'timing_state_protocol.json')
    for name,digest in timing['source_code_sha256'].items():
        assert sha(root/'src'/name)==digest
    for name,digest in load(artifacts/'timing_state_summary.json')['artifact_sha256'].items():
        assert sha(artifacts/name)==digest
    return {'manifest_files':len(actual),'frozen_source_config_protocol_hashes_match':True,'selection_inputs_and_winners_recomputed':True,'baseline_windows_keys_resolve_as_posix_paths':True,'source_arrays_and_complete_notices_match':True,'solver_records':counts,'intervals':120,**content_audit(root)}

def reproduce(root):
    names=['summary.csv','seed_means.csv','paired_intervals.csv','aggregation_verification.json']
    before={name:sha(root/'artifacts'/name) for name in names}
    commands=[['src/aggregate_causal.py'],['artifacts/training_diagnostics_qa.py'],['artifacts/policy_test_qa.py'],['portable_runner.py','synthetic-bounds'],['portable_runner.py','baseline','verify']]
    results=[]
    for command in commands:
        result=subprocess.run([sys.executable,*command],cwd=root,env=ENV,capture_output=True,text=True,check=True)
        results.append({'command':['python',*command],'stdout':result.stdout.strip()})
    same={name:sha(root/'artifacts'/name)==digest for name,digest in before.items()}
    assert all(same.values()),same
    code='''import sys,json,numpy as np,pandas as pd
sys.path.insert(0,'src')
import causal_run as run
run.verify_contract()
days,_=run.make_days(); selection=json.loads((run.OUT/'selection.json').read_text())['selected_budgets']; recorded=pd.read_csv(run.OUT/'test_seed_20260805.csv'); traced=pd.read_csv(run.OUT/'trajectory_seed_20260805.csv'); rows=[]
for controller,budget in selection.items():
 actual,steps=run.evaluate(days,controller,run.load_model(controller,20260805,budget),300,20260805,keep_steps=True); expected=recorded[(recorded.controller==controller)&(recorded.split_day==300)].iloc[0]; g=traced[(traced.controller==controller)&(traced.day==300)].sort_values('step'); replay=pd.DataFrame(steps); metrics=['cost','carbon','task_score','peak','grid_excess_energy_kwh','terminal_soc_error_kwh','lighting_energy_kwh','ev_completion']; fields=['p','soc','grid_import','grid_export','grid_violation']; delta=max(abs(float(actual[m])-float(expected[m])) for m in metrics); step_delta=float(np.max(np.abs(replay[fields].to_numpy()-g[fields].to_numpy()))); assert delta<1e-8 and step_delta<1e-8; assert list(replay.qp_execution_path)==list(g.qp_execution_path); rows.append({'controller':controller,'seed':20260805,'day':300,'budget':budget,'steps':96,'maximum_daily_metric_difference':delta,'maximum_step_difference':step_delta,'paths_identical':True})
print(json.dumps(rows))
'''
    replay=subprocess.run([sys.executable,'-c',code],cwd=root,env=ENV,capture_output=True,text=True,check=True)
    return {'aggregate_outputs_byte_identical':same,'executed_checks':results,'full_day_policy_replay':json.loads(replay.stdout.strip().splitlines()[-1]),'full_retraining_repeated_during_package_qa':False,'linux_execution_performed':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reproduce',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parent
    report={'status':'PASS',**integrity(root)}
    if args.reproduce:
        report.update(reproduce(root))
    print(json.dumps(report,indent=2))
