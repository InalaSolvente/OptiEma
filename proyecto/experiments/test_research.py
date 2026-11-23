from dataclasses import replace
from copy import deepcopy
import random
import pytest
from data.instance_model import InstanceContext,Job,Operation
from data.instance_loader import load_instance
from simulation.research_scheduler import evaluate,validate
from algorithms.qlma import random_solution,neighbors,optimize
from experiments.research import Evaluator,ROOT,METHODS,run


def example():
    def ops(su,dur):return (Operation(1,20,0,0,0,0,('R1','R2'),('A',)),Operation(2,dur,0,0,10,100,('R1','R2'),(su,)))
    return InstanceContext(1,'example','test','synthetic',0,('R1','R2'),((1,('A',)),(2,('S1','S2'))),
        (Job(1,'',ops('S1',40)),Job(2,'',ops('S1',30)),Job(3,'',ops('S2',30))),'0'*64)


def test_hand_calculated_blocking_and_cleanup():
    c=example();x=dict(job_sequence_base=[1,2,3],room_assignment={1:{1:'R1',2:'R1'},2:{1:'R2',2:'R2'},3:{1:'R1',2:'R1'}})
    r=evaluate(c,x)
    assert r.makespan==130 and r.wait_sum==20 and r.wait_max==20
    assert r.loads=={'R1':130,'R2':80}
    assert r.f==pytest.approx(130+.00026+.5*20+1.4*20+100*(25/480))
    assert not validate(c,r)
    # A conflicting room interval must be caught independently of the decoder.
    bad=deepcopy(r);bad.entries[-2]['Start']=1
    assert validate(c,bad)


def test_first_room_cleaning_may_define_makespan():
    c=example();job=c.jobs[0];a,b=job.operations
    job=replace(job,operations=(replace(a,cleanup=100),replace(b,duration=1,cleanup=0)))
    c=replace(c,jobs=(job,));x=dict(job_sequence_base=[1],room_assignment={1:{1:'R1',2:'R2'}})
    r=evaluate(c,x);assert r.makespan==120;assert not validate(c,r)


def test_exact_permutation_and_staff_eligibility():
    c=example();x=random_solution(c,random.Random(1));x['job_sequence_base']=[1,1,2]
    with pytest.raises(ValueError):evaluate(c,x)
    x=random_solution(c,random.Random(1));x['personnel_assignment']={1:{1:'missing'}}
    with pytest.raises(ValueError):evaluate(c,x)


def test_ga_crossover_and_mutation_leave_single_valid_permutation():
    from algorithms import ga
    c=example();ga.PABELLONES=list(c.rooms)
    for seed in range(20):
        random.seed(seed);a=ga.create_individual([1,2,3]);b=ga.create_individual([1,2,3])
        for child in ga.crossover(a,b):
            x=ga.mutate(child)
            assert len(x['job_sequence_base'])==3
            assert set(x['job_sequence_base'])=={1,2,3}


def test_holm_and_paired_effect_are_not_p_values_without_observations():
    from experiments.summarize import holm,effect
    assert holm([.01,.04,.03])==pytest.approx([.03,.06,.06])
    assert effect([0,0,0])==(0.,0.,0.)
    assert effect([-2,-2,-2])==(-2.,-2.,-2.)


@pytest.mark.parametrize('action',range(7))
def test_operators_preserve_permutation_and_eligibility(action):
    c=example();rng=random.Random(2);x=random_solution(c,rng);r=evaluate(c,x)
    children=list(neighbors(action,x,x,r,c,rng));assert children
    for child in children:assert not validate(c,evaluate(c,child))


@pytest.mark.parametrize('method',METHODS)
def test_all_methods_share_validated_objective_and_exact_budget(method):
    c=load_instance(ROOT/'instances/didactic/HOSP-DIDACT-03-01.yaml')
    a=run(c,method,20000,123);b=run(c,method,20000,123)
    assert a['status']=='completed',(method,a['error'],a['validation_errors'])
    assert a['evaluations']==123
    assert a['objective']==b['objective'] and a['solution']==b['solution']
    assert evaluate(c,a['solution']).f==a['objective']
    assert a['curve'][-1]['best_f']==a['objective']
    if method in METHODS[4:]:assert len(a['operator_diagnostics']['q'])==27
