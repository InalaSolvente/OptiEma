"""Descriptive outputs and predeclared paired inference; no invented measurements."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy import stats
from experiments.prepare import write_json


def holm(values):
    values=np.asarray(values,dtype=float);order=np.argsort(values)
    adjusted=np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values))))
    result=np.empty(len(values));result[order]=np.minimum(1,adjusted)
    return result.tolist()


def effect(d,seed=715):
    d=np.asarray(d);i,j=np.triu_indices(len(d))
    hl=float(np.median((d[i]+d[j])/2))
    samples=np.random.default_rng(seed).choice(d,(2000,len(d)),replace=True)
    bootstrap=np.median((samples[:,i]+samples[:,j])/2,axis=1)
    lo,hi=np.quantile(bootstrap,[.025,.975])
    return hl,float(lo),float(hi)


def csv_write(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize(root):
    root=Path(root);records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((root/'runs').glob('*.json'))]
    if not records:raise ValueError('No measured runs to summarize')
    manifest=json.loads((root/'frozen_manifest.json').read_text(encoding='utf-8'))
    if any(r['source_hash']!=manifest['source_hash'] for r in records):raise ValueError('mixed source versions')
    out=root/'analysis';out.mkdir(exist_ok=True)
    instances=sorted({r['instance_id'] for r in records});methods=sorted({r['method'] for r in records})
    groups={(i,m):[r for r in records if r['instance_id']==i and r['method']==m] for i in instances for m in methods}
    summary=[]
    for instance in instances:
        valid=[r['objective'] for r in records if r['instance_id']==instance and r['status']=='completed']
        reference=min(valid) if valid else None
        for method in methods:
            rows=groups[instance,method];good=[r for r in rows if r['status']=='completed']
            values=np.array([r['objective'] for r in good]);metrics={}
            for name,vals in [('f',values),('makespan',[r['result']['makespan'] for r in good]),
                             ('wait_sum',[r['result']['wait_sum'] for r in good]),
                             ('wait_max',[r['result']['wait_max'] for r in good]),
                             ('imbalance',[r['result']['imbalance'] for r in good]),
                             ('cpu_seconds',[r['cpu_seconds'] for r in good]),('wall_seconds',[r['wall_seconds'] for r in good])]:
                metrics.update({name+'_mean':float(np.mean(vals)) if len(vals) else None,
                    name+'_median':float(np.median(vals)) if len(vals) else None,
                    name+'_sd':float(np.std(vals,ddof=1)) if len(vals)>1 else None})
            summary.append(dict(instance=instance,method=method,observed=len(rows),valid=len(good),
                failed=len(rows)-len(good),empirical_reference=reference,
                rpd_mean=float(np.mean(100*(values-reference)/reference)) if good and reference>0 else None,**metrics))
    csv_write(out/'by_instance.csv',summary)
    aggregate=[]
    for method in methods:
        rows=[r for r in summary if r['method']==method]
        means=[r['rpd_mean'] for r in rows if r['rpd_mean'] is not None]
        aggregate.append(dict(method=method,instances=len(rows),runs=sum(r['observed'] for r in rows),
            failed=sum(r['failed'] for r in rows),mean_rpd=float(np.mean(means)) if means else None))
    csv_write(out/'aggregate.csv',aggregate)
    tests=[]
    if not manifest['smoke']:
        for instance in instances:
            control={r['seed']:r for r in groups[instance,'qlma'] if r['status']=='completed'}
            for method in methods:
                if method=='qlma':continue
                other={r['seed']:r for r in groups[instance,method] if r['status']=='completed'}
                paired=sorted(control.keys()&other.keys());d=[control[s]['objective']-other[s]['objective'] for s in paired]
                if not d:continue
                p=1. if np.allclose(d,0,rtol=0,atol=1e-12) else float(stats.wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='auto').pvalue)
                hl,lo,hi=effect(d)
                tests.append(dict(instance=instance,comparator=method,pairs=len(d),p=p,
                    family='baselines' if method in ('ga','dpso','sboa','dmshoa') else 'ablation',
                    hodges_lehmann=hl,bootstrap_hl_ci_low=lo,bootstrap_hl_ci_high=hi,
                    wins=sum(v<0 for v in d),ties=sum(v==0 for v in d),losses=sum(v>0 for v in d)))
        for family in ('baselines','ablation'):
            ix=[i for i,r in enumerate(tests) if r['family']==family]
            for k,padj in zip(ix,holm([tests[i]['p'] for i in ix])):tests[k]['holm_p']=padj
        csv_write(out/'paired_tests.csv',tests)
        five=('qlma','ga','dpso','sboa','dmshoa')
        blocks=[i for i in instances if all(len(groups[i,m])==30 and all(r['status']=='completed' for r in groups[i,m]) for m in five)]
        global_result=dict(complete_blocks=len(blocks),excluded_blocks=len(instances)-len(blocks),methods=five)
        if len(blocks)>=3:
            matrix=np.array([[np.mean([r['objective'] for r in groups[i,m]]) for m in five] for i in blocks])
            if np.all(np.ptp(matrix,axis=1)==0):stat,p=0.,1.
            else:
                f=stats.friedmanchisquare(*matrix.T);stat,p=float(f.statistic),float(f.pvalue)
            n,k=matrix.shape;den=n*(k-1)-stat
            iman=(n-1)*stat/den if den>0 else None
            global_result.update(friedman_stat=stat,friedman_p=p,iman_davenport_f=iman,
                iman_davenport_p=float(stats.f.sf(iman,k-1,(k-1)*(n-1))) if iman is not None else None)
            post=[]
            for k,method in enumerate(five[1:],1):
                d=matrix[:,0]-matrix[:,k]
                p=1. if np.allclose(d,0,rtol=0,atol=1e-12) else float(stats.wilcoxon(d,method='auto').pvalue)
                post.append(dict(comparator=method,p=p))
            for row,adj in zip(post,holm([r['p'] for r in post])):row['holm_p']=adj
            global_result['posthoc_paired_instance_means']=post
        write_json(out/'global_tests.json',global_result)
    write_json(out/'analysis_metadata.json',dict(runs=len(records),failures=sum(r['status']!='completed' for r in records),
        smoke=manifest['smoke'],source_hash=manifest['source_hash'],
        interval_method='2000-resample percentile bootstrap of paired Hodges-Lehmann; fixed seed 715',
        wilcoxon_assumption='symmetric independent paired differences; assess before inferential conclusions',
        multiplicity='Holm over all instance/baseline comparisons, separate ablation family',
        interpretation='descriptive outputs; no automatic assertion of superiority; failures retained'))
    if not manifest['smoke']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        values=[[r['rpd_mean'] for r in summary if r['method']==m and r['rpd_mean'] is not None] for m in methods]
        fig,ax=plt.subplots(figsize=(10,5))
        ax.boxplot(values,tick_labels=methods,showmeans=True)
        ax.set(ylabel='RPD medio por instancia (%)',title='Comparación en instancias de prueba; referencia empírica')
        ax.tick_params(axis='x',rotation=25);fig.tight_layout()
        fig.savefig(out/'rpd_by_instance.png',dpi=180);fig.savefig(out/'rpd_by_instance.svg');plt.close(fig)
        for instance in ('HOSP-12R-15-01','HOSP-12R-30-01','HOSP-12R-60-01'):
            if instance not in instances:continue
            fig,ax=plt.subplots(figsize=(10,5))
            for method in methods:
                runs=[r for r in groups[instance,method] if r['status']=='completed']
                if not runs:continue
                grid=np.linspace(1,runs[0]['budget'],101);curves=[]
                for r in runs:
                    points=[p for p in r['curve'] if p['best_f'] is not None]
                    indices=np.searchsorted([p['evaluation'] for p in points],grid,side='right')-1
                    vals=np.array([p['best_f'] for p in points]);curve=vals[np.maximum(0,indices)].astype(float)
                    curve[indices<0]=np.nan;curves.append(curve)
                low,mid,high=np.nanquantile(curves,[.25,.5,.75],axis=0)
                line,=ax.plot(grid,mid,label=method);ax.fill_between(grid,low,high,color=line.get_color(),alpha=.1)
            ax.set(xlabel='Evaluaciones',ylabel='Mejor función objetivo',title=instance+'; mediana y rango intercuartílico')
            ax.legend(fontsize=8,ncol=2);fig.tight_layout()
            fig.savefig(out/(instance+'_convergence.png'),dpi=180);plt.close(fig)
