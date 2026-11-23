"""Compare two affordable budgets only on reserved pilot instances."""
import json
from experiments.prepare import inventory,write_json
from experiments.research import ROOT,METHODS,run,source_hash
from data.instance_loader import load_instance

out=ROOT/'experiments/records/budget_pilot';rows=[]
for row in inventory():
    if row['phase']!='pilot':continue
    ctx=load_instance(ROOT/row['path'])
    for method in METHODS:
        for kappa in (100,200):
            r=run(ctx,method,20002,kappa*len(ctx.jobs))
            r.update(phase='budget_pilot',kappa=kappa,source_hash=source_hash(),usable_for_final=False)
            write_json(out/f'{ctx.instance_id}__{method}__{kappa}.json',r);rows.append(r)
            print(ctx.instance_id,method,kappa,r['status'],flush=True)
comparisons=[]
for r in rows:
    if r['kappa']!=100:continue
    b=next(s for s in rows if s['kappa']==200 and s['method']==r['method'] and s['instance_id']==r['instance_id'])
    comparisons.append(dict(instance=r['instance_id'],method=r['method'],f100=r['objective'],f200=b['objective'],
        gain_percent=100*(r['objective']-b['objective'])/r['objective'],
        seconds100=r['wall_seconds'],seconds200=b['wall_seconds']))
summary=dict(runs=len(rows),failed=sum(r['status']!='completed' for r in rows),comparisons=comparisons,
    limitation='one seed, two reserved pilot instances; not final performance or a guarantee of convergence')
write_json(out/'summary.json',summary);print(json.dumps(summary))
