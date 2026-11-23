"""QL-MA and A/B/D ablations; all neighbors use the shared budget evaluator."""
import copy
import random
from math import isfinite
from simulation.result_model import SolverOutput


DEFAULTS=dict(population=50,alpha=.15,discount=.85,epsilon=.35,tau=50.,weights=[1.]*7)


def random_solution(context,rng):
    seq=[j.job_id for j in context.jobs];rng.shuffle(seq);assign={};load={r:0. for r in context.rooms}
    for job in context.jobs:
        a,b=job.operations;common=sorted(set(a.eligible_rooms)&set(b.eligible_rooms))
        if common and rng.random()<.5:
            room=min(common,key=lambda r:(load[r],r));assign[job.job_id]={1:room,2:room}
            load[room]+=a.duration+b.duration
        else:
            assign[job.job_id]={}
            for op in job.operations:
                room=min(op.eligible_rooms,key=lambda r:(load[r],r))
                assign[job.job_id][op.operation_id]=room;load[room]+=op.duration
    return dict(job_sequence_base=seq,room_assignment=assign)


def state(ev,stagnant,tau):
    phase=0 if ev.count<.33*ev.budget else 1 if ev.count<.66*ev.budget else 2
    theta=0 if stagnant<3 else 1 if stagnant<7 else 2
    best=ev.best_result
    pw=.5*best.wait_sum+1.4*best.wait_max if best else 0
    pb=100*best.imbalance if best else 0
    omega=0 if max(pw,pb)<tau else 1 if pw>=pb else 2
    return 9*phase+3*theta+omega


def neighbors(action,parent,elite,result,context,rng):
    """Deterministic critical ties; exhaustive 2-opt and greedy critical reinsertion."""
    x=copy.deepcopy(parent);seq=x['job_sequence_base'];n=len(seq)
    byjob={j.job_id:j for j in context.jobs}
    surgery=[e for e in result.entries if e['Operation']==2] if result else []
    critical=sorted(surgery,key=lambda e:(-e['Wait'],-e['ProcessingEnd'],e['Job']))
    jid=critical[0]['Job'] if critical else seq[0]
    if action==0:
        i=seq.index(jid);others=[seq.index(e['Job']) for e in critical[1:] if e['Personnel']==critical[0]['Personnel']]
        j=others[0] if others else rng.randrange(n);seq[i],seq[j]=seq[j],seq[i]
    elif action==1:
        late=max(surgery,key=lambda e:(e['ProcessingEnd'],-e['Job']))['Job'] if surgery else seq[-1]
        i=seq.index(late);seq.insert(rng.randrange(i+1),seq.pop(i))
    elif action==2:
        loads=result.loads if result else {r:0 for r in context.rooms}
        pairs=[(loads[x['room_assignment'][j.job_id][op.operation_id]],j.job_id,op) for j in context.jobs for op in j.operations]
        _,j,op=max(pairs,key=lambda p:(p[0],-p[1],p[2].operation_id))
        x['room_assignment'][j][op.operation_id]=min(op.eligible_rooms,key=lambda r:(loads[r],r))
    elif action==3:
        choices=[j for j in context.jobs if set(j.operations[0].eligible_rooms)&set(j.operations[1].eligible_rooms)]
        if choices:
            job=next((j for j in choices if j.job_id==jid),rng.choice(choices))
            room=rng.choice(sorted(set(job.operations[0].eligible_rooms)&set(job.operations[1].eligible_rooms)))
            x['room_assignment'][job.job_id]={1:room,2:room}
    elif action==4:
        i,j=sorted(rng.sample(range(n),2));j+=1
        keep=seq[i:j];remaining=[v for v in elite['job_sequence_base'] if v not in keep]
        child=[None]*n;child[i:j]=keep
        indices=list(range(j,n))+list(range(i))
        # OX1 scans the second parent cyclically from the end of the segment.
        scan=elite['job_sequence_base'][j:]+elite['job_sequence_base'][:j]
        for k,v in zip(indices,[v for v in scan if v not in keep]):child[k]=v
        x['job_sequence_base']=child
        for v in child:
            if v not in keep:x['room_assignment'][v]=copy.deepcopy(elite['room_assignment'][v])
    elif action==5:
        for i in range(n-1):
            for j in range(i+2,n+1):
                y=copy.deepcopy(parent);y['job_sequence_base'][i:j]=reversed(seq[i:j]);yield y
        return
    else:
        # One most-blocked surgery is ruined; exhaustive insertion is its greedy recreation.
        base=[v for v in seq if v!=jid]
        for i in range(n):
            y=copy.deepcopy(parent);y['job_sequence_base']=base[:i]+[jid]+base[i:];yield y
        return
    yield x


def optimize(context,seed,ev,params=None,variant='qlma'):
    p={**DEFAULTS,**(params or {})};rng=random.Random(int(seed))
    size=int(p['population'])
    if size<3 or ev.budget<size:raise ValueError('budget must cover a population of at least 3')
    population=[];q=[[0.]*7 for _ in range(27)];stagnant=0;actions=[0]*7
    costs=[0]*7;wins=[0]*7;history=[]
    for _ in range(size):
        x=random_solution(context,rng);v=ev(x);population.append((v,x,ev.last_result))
    while ev.count<ev.budget:
        population.sort(key=lambda t:t[0]);new=population[:2];improved=False
        while len(new)<size and ev.count<ev.budget:
            parent=min(rng.choices(population,k=3),key=lambda t:t[0]);z=state(ev,stagnant,p['tau'])
            epsilon=max(.05,p['epsilon']*(1-ev.count/ev.budget))
            if variant=='fixed':action=rng.choices(range(7),weights=p['weights'])[0]
            elif variant=='uniform' or rng.random()<epsilon:action=rng.randrange(7)
            else:
                bestq=max(q[z]);action=rng.choice([a for a in range(7) if q[z][a]==bestq])
            before=ev.best;start=ev.count;candidate=(float('inf'),parent[1],None)
            for x in neighbors(action,parent[1],ev.best_solution or parent[1],parent[2],context,rng):
                if ev.count==ev.budget:break
                v=ev(x)
                if v<candidate[0]:candidate=(v,x,ev.last_result)
            delta=max(0,parent[2].makespan-candidate[2].makespan) if parent[2] and candidate[2] else 0
            v=candidate[0]
            reward=-5 if not isfinite(v) else 10+2*delta if v<before else 3+delta if v<parent[0] else .5 if v==parent[0] else -1
            actions[action]+=1;costs[action]+=ev.count-start;wins[action]+=v<before
            improved|=ev.best<before
            new.append(candidate if v<parent[0] else parent)
            if len(new)==size:
                population=new;stagnant=0 if improved else stagnant+1
                if stagnant>=10 and variant!='no_diversification':
                    population.sort(key=lambda t:t[0])
                    for i in range(int(.3*(size-2))):
                        if ev.count==ev.budget:break
                        x=random_solution(context,rng);v=ev(x);population[-1-i]=(v,x,ev.last_result)
                    stagnant=0
            nextz=state(ev,stagnant,p['tau'])
            future=0 if ev.count==ev.budget else p['discount']*max(q[nextz])
            if variant not in ('uniform','fixed'):q[z][action]+=p['alpha']*(reward+future-q[z][action])
        history.append(ev.best)
    return dict(q=q,actions=actions,evaluations=costs,improvements=wins,generations=len(history))


def run(context,seed,on_iteration=None):
    from experiments.research import Evaluator
    ev=Evaluator(context,2000*len(context.jobs))
    optimize(context,seed,ev)
    if ev.best_solution is None:raise RuntimeError('no feasible solution')
    if on_iteration:on_iteration(algo_step=ev.count,best_fitness=ev.best,best_makespan=ev.best_result.makespan)
    return SolverOutput(ev.best,ev.best_solution,tuple(p['best_f'] for p in ev.trace if p['best_f'] is not None))
