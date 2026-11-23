"""Sequential, resumable irace -> 6960 final runs -> summaries. One CPU thread."""
import argparse
import csv
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from experiments.research import METHODS,source_hash,run
from experiments.prepare import inventory,write_json
from data.instance_loader import load_instance


def parameters(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        values={k:float(v) for k,v in next(csv.DictReader(f)).items() if v and not k.startswith('.')}
    if any(k.startswith('w') and len(k)==2 and k[1].isdigit() for k in values):
        values['weights']=[values.pop(f'w{i}') for i in range(7)]
    return values


def main():
    p=argparse.ArgumentParser();p.add_argument('--rscript',required=True);p.add_argument('--rlib',required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--kappa',type=int,default=2000);args=p.parse_args()
    if args.kappa<5:p.error('kappa must be >=5 to cover every calibrated population')
    digest=source_hash()
    out=ROOT/'experiments/records'/(('pipeline_smoke_'+digest[:10]) if args.smoke else 'production')
    out.mkdir(parents=True,exist_ok=True)
    rows=inventory()
    frozen=dict(source_hash=digest,instances=rows,methods=list(METHODS),kappa=args.kappa,
                repetitions=30,irace_max_experiments=500,horizon_minutes=480,
                test_seeds=list(range(30000,30030)),smoke=args.smoke)
    manifest=out/'frozen_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=frozen:
        raise RuntimeError('Frozen protocol differs; refuse to mix records. Archive this production folder first.')
    write_json(manifest,frozen)
    status=dict(stage='starting',pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),source_hash=digest)
    def update(**kw):
        status.update(kw);status['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(out/'status.json',status)
    try:
        selected={}
        for method in (('qlma',) if args.smoke else METHODS):
            location=out/'calibration'/method
            if not (location/'completed.txt').exists():
                update(stage='calibration',method=method)
                location.mkdir(parents=True,exist_ok=True)
                with (location/'console.log').open('a',encoding='utf-8') as log:
                    env={**os.environ,'OPTIEMA_EXPECTED_SOURCE_HASH':digest}
                    subprocess.run([args.rscript,'--vanilla',str(ROOT/'experiments/calibrate.R'),method,
                        sys.executable,str(ROOT),args.rlib,str(location),'80' if args.smoke else '500',
                        '5' if args.smoke else str(args.kappa)],stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
            selected[method]=parameters(location/'selected.csv')
        write_json(out/'selected_parameters.json',selected)
        if args.smoke:
            # Canary uses pilot cases only; never inspect held-out test performance.
            jobs=[r for r in rows if r['phase']=='pilot'];methods=('qlma',);seeds=[20001];kappa=5
        else:
            jobs=[r for r in rows if r['phase']=='test'];methods=METHODS;seeds=range(30000,30030);kappa=args.kappa
        total=len(jobs)*len(methods)*len(seeds);done=0
        for row in jobs:
            ctx=load_instance(ROOT/row['path'])
            for seed in seeds:
                order=list(methods);random.Random(f"{row['instance_id']}:{seed}").shuffle(order)
                for method in order:
                    if source_hash()!=digest:raise RuntimeError('Source changed during execution; stopping to preserve reproducibility')
                    path=out/'runs'/f"{ctx.instance_id}__{method}__{seed}.json"
                    if path.exists():
                        prior=json.loads(path.read_text(encoding='utf-8'))
                        if prior.get('source_hash')!=digest:raise RuntimeError('Incompatible saved run')
                    else:
                        update(stage='final' if not args.smoke else 'canary',instance=ctx.instance_id,
                               method=method,seed=seed,completed=done,total=total)
                        r=run(ctx,method,seed,kappa*len(ctx.jobs),selected[method])
                        r.update(phase='canary' if args.smoke else 'test',source_hash=digest,
                                 usable_for_final=not args.smoke and r['status']=='completed')
                        write_json(path,r)
                    done+=1
        update(stage='summarizing',completed=done,total=total)
        from experiments.summarize import summarize
        summarize(out)
        update(stage='complete',completed=done,total=total)
    except BaseException as exc:
        update(stage='failed',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
