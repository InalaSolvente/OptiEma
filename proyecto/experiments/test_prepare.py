from dataclasses import replace
from experiments.prepare import inventory, inspect_calendar, pilot
from data.instance_loader import load_instance


def test_bank_and_no_partition_leak():
    rows = inventory()
    assert len(rows) == 37
    assert sum(r['phase']=='test' for r in rows)==29
    assert len(set(r['digest'] for r in rows))==37


def test_budget_last_improvement_and_records(tmp_path):
    row = inventory()[0]
    record = pilot(row, 'ga', 20000, 7, tmp_path)
    assert record['error'] is None
    assert record['evaluations']==7
    finite = [p['best_f'] for p in record['curve'] if p['best_f'] is not None]
    assert finite == sorted(finite, reverse=True)
    assert record['best_f'] == finite[-1]
    assert len(list(tmp_path.glob('*.json')))==1
    assert record['usable_for_final'] is False


def test_validator_detects_missing_operations():
    row = inventory()[0]
    from experiments.prepare import ROOT
    assert inspect_calendar(load_instance(ROOT/row['path']),[])==['missing operations']


def test_validator_detects_room_reused_during_blocking():
    from data.instance_model import InstanceContext, Job, Operation
    ops = tuple(Operation(k,10,0,0,0,100,('R1','R2'),('A1','A2') if k==1 else ('S1','S2')) for k in (1,2))
    ctx = InstanceContext(1,'fixture','fixture','synthetic',0,('R1','R2'),(),
                          (Job(1,'',ops),Job(2,'',ops)),'0'*64)
    def entry(j,k,r,p,start):
        return dict(Job=j,Operation=k,Resource=r,Personnel=p,Start=start,
                    ProcessingEnd=start+10,Finish=start+10)
    entries=[entry(1,1,'R1','A1',0),entry(1,2,'R2','S1',20),
             entry(2,1,'R1','A2',12),entry(2,2,'R1','S2',22)]
    assert any('room holding' in e for e in inspect_calendar(ctx,entries))
    assert not inspect_calendar(replace(ctx,jobs=ctx.jobs[:1]),entries[:2])
