"""Post-hoc sign sensitivity for saved runs; does not execute algorithms."""
import argparse
import json
import math
from pathlib import Path


def sign_p(differences):
    nonzero=[d for d in differences if d!=0]
    n=len(nonzero)
    if not n:return 1.0
    tail=min(sum(d<0 for d in nonzero),sum(d>0 for d in nonzero))
    return min(1.0,2*sum(math.comb(n,k) for k in range(tail+1))/2**n)


def holm(rows):
    for family in ('baseline','ablation'):
        ordered=sorted([r for r in rows if r['family']==family],key=lambda r:r['p'])
        previous=0.0
        for i,row in enumerate(ordered):
            previous=max(previous,(len(ordered)-i)*row['p']);row['holm_p']=min(1.0,previous)


def analyze(root):
    records=[json.loads(p.read_text(encoding='utf-8')) for p in (root/'runs').glob('*.json')]
    assert len(records)==6960 and all(r['status']=='completed' for r in records)
    values={(r['instance_id'],r['method'],r['seed']):r['objective'] for r in records}
    instances=sorted({r['instance_id'] for r in records});local=[];global_rows=[]
    for method in ('ga','dpso','sboa','dmshoa','fixed','uniform','no_diversification'):
        family='baseline' if method in ('ga','dpso','sboa','dmshoa') else 'ablation';means=[]
        for instance in instances:
            diff=[values[instance,'qlma',s]-values[instance,method,s] for s in range(30000,30030)]
            means.append(math.fsum(diff)/len(diff))
            local.append(dict(instance=instance,method=method,p=sign_p(diff),family=family))
        global_rows.append(dict(method=method,wins=sum(d<0 for d in means),ties=sum(d==0 for d in means),
            losses=sum(d>0 for d in means),p=sign_p(means),family=family))
    holm(local);holm(global_rows)
    return dict(global_sign=global_rows,local_sign=local,
                status='post-hoc sensitivity, not preregistered; signs of paired differences and instance means')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('production',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();result=analyze(args.production);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['global_sign']))
