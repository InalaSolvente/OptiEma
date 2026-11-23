"""Report model v1: joint room reservation, staff during processing, no buffer.

H=480 minutes is a declared experimental normalization, not patient data.
Transition and setup combine as max(t,s) in induction and t+s in surgery,
following the course timing convention. max_wait metadata is reported, not a
hard constraint: the report penalizes waiting instead of imposing a deadline.
"""
from dataclasses import dataclass
from math import isfinite, sqrt


@dataclass
class Evaluation:
    f: float
    makespan: float
    wait_sum: float
    wait_max: float
    imbalance: float
    entries: list
    loads: dict
    wait_limit_exceedances: int


def evaluate(context, solution, horizon=480.0):
    if horizon <= 0 or not isfinite(horizon):
        raise ValueError('horizon must be finite and positive')
    if hasattr(solution, 'to_dict'):
        solution = solution.to_dict()
    seq = list(solution['job_sequence_base'])
    jobs = {j.job_id:j for j in context.jobs}
    # Only the exact historical doubled representation is normalized.
    if len(seq)==2*len(jobs) and seq[:len(jobs)]==seq[len(jobs):]:
        seq=seq[:len(jobs)]
    if len(seq)!=len(jobs) or set(seq)!=set(jobs):
        raise ValueError('candidate is not a permutation')
    assignments=solution['room_assignment']
    if set(assignments)!=set(jobs):
        raise ValueError('incomplete assignment')
    release={r:0.0 for r in context.rooms}
    loads=release.copy()
    people={p:0.0 for _,ps in context.personnel_by_operation for p in ps}
    entries=[]; waits=[]; starts=[]; exceedances=0
    for jid in seq:
        job=jobs[jid]; a,b=job.operations
        r1,r2=(assignments[jid][k] for k in (1,2))
        if r1 not in a.eligible_rooms or r2 not in b.eligible_rooms:
            raise ValueError('ineligible room')
        explicit=solution.get('personnel_assignment') or {}
        selected=explicit.get(jid,{})
        q1=selected.get(1) or min(a.eligible_personnel,key=lambda p:(people[p],p))
        q2=selected.get(2) or min(b.eligible_personnel,key=lambda p:(people[p],p))
        if q1 not in a.eligible_personnel or q2 not in b.eligible_personnel:
            raise ValueError('ineligible personnel')
        t1=max(a.setup,a.transition);t2=b.setup+b.transition
        begin1=max(0.0,release[r1],people[q1]-t1)
        s1=begin1+t1;c1=s1+a.duration
        s2=max(c1+t2 if r1==r2 else c1,people[q2],release[r2]+t2)
        begin2=s2-t2;c2=s2+b.duration;end2=c2+b.cleanup
        end1=end2 if r1==r2 else s2+a.cleanup
        if r1==r2:
            loads[r1]+=end2-begin1
        else:
            loads[r1]+=end1-begin1;loads[r2]+=end2-begin2
        release[r1]=end1;release[r2]=end2
        people[q1]=c1;people[q2]=c2
        wait=s2-c1;waits.append(wait);starts.extend((s1,s2))
        exceedances+=wait>b.max_wait+1e-9
        for k,r,q,begin,start,end,finish,op in [(1,r1,q1,begin1,s1,c1,end1,a),(2,r2,q2,begin2,s2,c2,end2,b)]:
            entries.append(dict(Job=jid,Operation=k,Resource=r,Personnel=q,Start=begin,
                ActualStart=start,ProcessingEnd=end,Finish=finish,SetupUsed=op.setup,
                TransitionUsed=op.transition,CleanupUsed=op.cleanup,Wait=wait if k==2 else 0.0))
    makespan=max(release.values(),default=0.0)
    u=[v/horizon for v in loads.values()];mean=sum(u)/len(u)
    psi=sqrt(sum((v-mean)**2 for v in u)/len(u))+sum(v==0 for v in u)/len(u)
    f=makespan+1e-6*sum(starts)+.5*sum(waits)+1.4*max(waits,default=0.0)+100*psi
    return Evaluation(f,makespan,sum(waits),max(waits,default=0.0),psi,entries,loads,exceedances)


def validate(context, result):
    """Independent interval/resource checks; never call the decoder here."""
    errors=[];items={};rooms=[];people=[]
    for e in result.entries:
        key=(e['Job'],e['Operation'])
        if key in items:errors.append('duplicate operation')
        items[key]=e
    for job in context.jobs:
        if any((job.job_id,k) not in items for k in (1,2)):
            errors.append('missing operation');continue
        a,b=(items[job.job_id,k] for k in (1,2))
        if b['ActualStart'] < a['ProcessingEnd']-1e-8:errors.append('precedence')
        if a['Resource']==b['Resource']:
            rooms.append((a['Resource'],a['Start'],b['Finish'],job.job_id))
            if b['Start'] < a['ProcessingEnd']-1e-8:errors.append('same-room preparation')
        else:
            rooms.extend([(a['Resource'],a['Start'],b['ActualStart']+job.operations[0].cleanup,job.job_id),
                          (b['Resource'],b['Start'],b['Finish'],job.job_id)])
        for op,e in zip(job.operations,(a,b)):
            if e['Resource'] not in op.eligible_rooms or e['Personnel'] not in op.eligible_personnel:errors.append('eligibility')
            if abs(e['ProcessingEnd']-e['ActualStart']-op.duration)>1e-8:errors.append('duration')
            prep=max(op.setup,op.transition) if op.operation_id==1 else op.setup+op.transition
            if abs(e['ActualStart']-e['Start']-prep)>1e-8:errors.append('preparation')
            if e['Start'] < -1e-8 or e['ActualStart']<e['Start']:errors.append('negative or reversed time')
            people.append((e['Personnel'],e['ActualStart'],e['ProcessingEnd'],(job.job_id,op.operation_id)))
        expected_first=b['Finish'] if a['Resource']==b['Resource'] else b['ActualStart']+job.operations[0].cleanup
        if abs(a['Finish']-expected_first)>1e-8 or abs(b['Finish']-b['ProcessingEnd']-job.operations[1].cleanup)>1e-8:
            errors.append('cleanup/release')
    for label,intervals in [('room',rooms),('personnel',people)]:
        for i,a in enumerate(intervals):
            for b in intervals[i+1:]:
                if a[0]==b[0] and a[3]!=b[3] and max(a[1],b[1])<min(a[2],b[2])-1e-8:
                    errors.append(f'{label} overlap: {a[3]}/{b[3]}')
    if len(items)!=2*len(context.jobs):errors.append('unexpected operation count')
    end=max((t[2] for t in rooms),default=0)
    if abs(result.makespan-end)>1e-8:errors.append('makespan')
    actual_load={r:sum(t[2]-t[1] for t in rooms if t[0]==r) for r in context.rooms}
    if any(abs(actual_load[r]-result.loads[r])>1e-8 for r in actual_load):errors.append('room load')
    return errors
