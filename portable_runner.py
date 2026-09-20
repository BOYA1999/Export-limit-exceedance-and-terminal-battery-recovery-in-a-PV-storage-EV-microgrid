from pathlib import Path
import argparse
import ast
import copy
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import causal_run as run

def baseline(stage,controller):
    import review_baselines as baseline
    native_hashes=baseline.hashes
    baseline.hashes=lambda:{key.replace('/',chr(92)):value for key,value in native_hashes().items()}
    if stage=='test':
        for name in [controller] if controller else baseline.CONTROLLERS:
            baseline.test(name)
    else:
        getattr(baseline,stage)()

def synthetic_bounds():
    source=ROOT/'src/bounds_regression.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='synthetic')
    namespace={'run':run,'core':run.core,'copy':copy,'np':np,'pd':pd}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
    rows=namespace['synthetic']()
    assert len(rows)==64 and sum(row['result']=='explicit_infeasible' for row in rows)==16
    print(json.dumps({'status':'PASS','synthetic_cases':len(rows),'explicit_infeasible_cases':16}))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('task',choices=['baseline','synthetic-bounds'])
    parser.add_argument('stage',nargs='?',choices=['smoke','validate','test','verify'])
    parser.add_argument('--controller',choices=['Rule+QP','MPC-H24+QP','NetLoadRule+QP','MPC-H24-FRR+QP'])
    args=parser.parse_args()
    if args.task=='baseline' and args.stage is None:
        parser.error('baseline requires a stage')
    if args.task=='synthetic-bounds':
        synthetic_bounds()
    else:
        baseline(args.stage,args.controller)
