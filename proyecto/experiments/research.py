"""Common-budget research runs. Historical diagnostic records remain untouched."""
import argparse
from contextlib import redirect_stdout
import copy
from dataclasses import asdict
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
from data.instance_loader import load_instance
from simulation.research_scheduler import evaluate,validate
from experiments.prepare import inventory,write_json,BudgetStop

METHODS=('ga','dpso','sboa','dmshoa','fixed','uniform','qlma','no_diversification')
HORIZON=480.0


class Evaluator:
    def __init__(self,context,budget):
        self.context=context;self.budget=budget;self.count=0;self.rejected=0
        self.best=math.inf;self.best_solution=None;self.best_result=None;self.last_result=None
        self.trace=[];self.detail_cache={}

    def __call__(self,solution):
        if self.count>=self.budget:raise BudgetStop()
        self.count+=1
        try:
            result=evaluate(self.context,solution,HORIZON)
        except (ValueError,KeyError,TypeError,IndexError):
            result=None;self.rejected+=1
        self.last_result=result
        f=result.f if result else math.inf
        improved=f<self.best
        if improved:
            self.best=f;self.best_solution=copy.deepcopy(solution);self.best_result=result
        # Fixed evaluation grid plus every improvement, to bound log sizes.
        if self.count in (1,self.budget) or self.count%max(1,self.budget//100)==0 or improved:
            point=dict(evaluation=self.count,best_f=self.best if math.isfinite(self.best) else None)
            if not self.trace or point!=self.trace[-1]:self.trace.append(point)
        return f


def source_hash():
    h=hashlib.sha256()
    for folder in ('algorithms','simulation','experiments','config'):
        for p in sorted(p for p in (ROOT/folder).iterdir() if p.suffix in ('.py','.R','.yaml')):
            h.update(p.relative_to(ROOT).as_posix().encode());h.update(p.read_bytes())
    return h.hexdigest()


def run(context,method,seed,budget,params=None):
    params=params or {};ev=Evaluator(context,budget);diagnostics={};error=None
    start=time.perf_counter();cpu=time.process_time()
    try:
        if method in METHODS[4:]:
            from algorithms.qlma import optimize
            diagnostics=optimize(context,seed,ev,params,method)
        else:
            import config.config as cfg
            module=importlib.import_module('algorithms.'+method)
            target='_instance_fitness' if method=='dmshoa' else 'calculate_schedule_fitness'
            old=getattr(module,target);changed=[]
            pop=int(params.get('population',30))
            if pop<3:raise ValueError('population must be >=3')
            controls={
              'ga':dict(POPULATION_SIZE_GA=pop,MAX_GENERATIONS=math.ceil(budget/max(1,2*pop-2))+2,
                        CROSSOVER_PROBABILITY=params.get('crossover',.8),MUTATION_PROBABILITY=params.get('mutation',.3)),
              'dpso':dict(SWARM_SIZE_DPSO=pop,MAX_ITERATIONS_DPSO=math.ceil(budget/pop)+2,
                         W_DPSO=params.get('w',.7),C1_DPSO=params.get('c1',1.5),C2_DPSO=params.get('c2',1.5)),
              'sboa':dict(SBOA_POP_SIZE=pop,SBOA_MAX_ITER=math.ceil(budget/(2*pop))+2),
              'dmshoa':dict(MSHOA_POP_SIZE=pop,MAX_ITERATIONS_MSHOA=math.ceil(budget/pop)+1,MSHOA_K=params.get('k',.3))
            }[method]
            for obj in (cfg,module):
                for key,value in {**controls,'VERBOSE_MODE':False}.items():
                    if hasattr(obj,key):changed.append((obj,key,getattr(obj,key)));setattr(obj,key,value)

            def fitness(solution,data=None,return_details=False):
                if method=='dmshoa':solution=data
                # Diagnostic requests reuse a previously evaluated calendar;
                # candidate submissions, including duplicates, still cost one.
                key=repr(solution)
                if return_details and solution==ev.best_solution:
                    return ev.best,ev.best_result.makespan,ev.best_result.entries
                if return_details and key in ev.detail_cache:return ev.detail_cache[key]
                f=ev(solution)
                result=ev.last_result
                value=(f,result.makespan if result else math.inf,result.entries if result else [])
                # Only a bounded cache is needed for native immediate detail requests.
                if len(ev.detail_cache)>2*pop:ev.detail_cache.pop(next(iter(ev.detail_cache)))
                ev.detail_cache[key]=value
                return value if return_details else f

            setattr(module,target,fitness)
            try:
                with redirect_stdout(io.StringIO()):module.run(context,seed)
            finally:
                setattr(module,target,old)
                for obj,key,value in reversed(changed):setattr(obj,key,value)
    except BudgetStop:
        pass
    except Exception as exc:
        error=repr(exc)
    wall=time.perf_counter()-start;cpu=time.process_time()-cpu
    violations=validate(context,ev.best_result) if ev.best_result else ['no feasible solution']
    if ev.count!=budget:error=error or f'budget not exhausted: {ev.count}/{budget}'
    result=asdict(ev.best_result) if ev.best_result else None
    return dict(instance_id=context.instance_id,instance_digest=context.digest,method=method,
        seed=seed,parameters=params,budget=budget,evaluations=ev.count,rejected=ev.rejected,
        status='completed' if not error and not violations else 'failed',error=error,
        validation_errors=violations,objective=ev.best if math.isfinite(ev.best) else None,
        wall_seconds=wall,cpu_seconds=cpu,result=result,solution=ev.best_solution,
        curve=ev.trace,operator_diagnostics=diagnostics,horizon_minutes=HORIZON)


def pilot(budget=300):
    records=[];out=ROOT/'experiments/records/research_pilot'
    for row in inventory():
        if row['phase'] not in ('validation','pilot'):continue
        ctx=load_instance(ROOT/row['path'])
        for method in METHODS:
            r=run(ctx,method,20000,budget);r.update(phase='pilot',source_hash=source_hash(),usable_for_final=False)
            write_json(out/f"{ctx.instance_id}__{method}.json",r);records.append(r)
            print(ctx.instance_id,method,r['status'],r['error'] or '',flush=True)
    summary=dict(runs=len(records),failed=sum(r['status']!='completed' for r in records),
        evaluations=sum(r['evaluations'] for r in records),wall_seconds=sum(r['wall_seconds'] for r in records),
        source_hash=source_hash(),horizon_minutes=HORIZON,usable_for_final=False)
    write_json(out/'summary.json',summary);print(json.dumps(summary));return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');parser.add_argument('--budget',type=int,default=300)
    parser.add_argument('--target');parser.add_argument('--instance');parser.add_argument('--seed',type=int);parser.add_argument('--parameters')
    parser.add_argument('--kappa',type=int,default=2000);parser.add_argument('--record-dir')
    args=parser.parse_args()
    if args.pilot:
        sys.exit(bool(pilot(args.budget)['failed']))
    if args.target:
        if os.environ.get('OPTIEMA_EXPECTED_SOURCE_HASH',source_hash())!=source_hash():
            raise RuntimeError('Source changed during calibration')
        import csv
        with open(args.parameters,newline='',encoding='utf-8-sig') as f:params={k:float(v) for k,v in next(csv.DictReader(f)).items() if v and not k.startswith('.')}
        weights=[params.pop('w'+str(i),1.) for i in range(7)]
        if args.target=='fixed':params['weights']=weights
        ctx=load_instance(args.instance)
        r=run(ctx,args.target,args.seed,args.kappa*len(ctx.jobs),params)
        # Normalize using a fixed constructive solution on the same calibration instance.
        from algorithms.qlma import random_solution
        import random
        reference=evaluate(ctx,random_solution(ctx,random.Random(0))).f
        cost=r['objective']/reference if r['status']=='completed' else 1e12
        log=Path(args.record_dir) if args.record_dir else ROOT/'experiments/records/calibration_calls'
        log.mkdir(parents=True,exist_ok=True)
        r['phase']='calibration';r['source_hash']=source_hash()
        write_json(log/f'{args.target}_{time.time_ns()}.json',r)
        print(cost)
