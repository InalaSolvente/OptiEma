"""Inventory the course bank and collect explicitly provisional baseline pilots."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
from data.instance_loader import load_instance
from simulation.scheduler import _fitness_for_context


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def inventory():
    rows = []
    for path in sorted((ROOT / 'instances').rglob('HOSP-*.yaml')):
        ctx = load_instance(path)
        phase = 'test'
        if 'DIDACT' in ctx.instance_id:
            phase = 'validation'
        elif ctx.instance_id in ('HOSP-STD-20-01', 'HOSP-12R-40-01'):
            phase = 'calibration'
        elif ctx.instance_id in ('HOSP-STD-20-02', 'HOSP-12R-40-02'):
            phase = 'pilot'
        rows.append(dict(instance_id=ctx.instance_id, path=path.relative_to(ROOT).as_posix(),
                         family=ctx.family, n=len(ctx.jobs), rooms=len(ctx.rooms),
                         digest=ctx.digest, file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), phase=phase))
    assert len({r['digest'] for r in rows}) == len(rows), 'Duplicate instances across partitions'
    assert len({r['instance_id'] for r in rows}) == len(rows)
    return rows


def inspect_calendar(ctx, entries):
    """Check the no-buffer room holding convention specified in the report."""
    errors = []
    ops = {(j.job_id, o.operation_id): o for j in ctx.jobs for o in j.operations}
    by_job = {}
    intervals = []
    personnel = []
    for e in entries:
        key = e['Job'], e['Operation']
        if key in by_job:
            errors.append(f'duplicate operation {key}')
        by_job[key] = e
        op = ops[key]
        start = e['Start'] + (max(op.setup, op.transition) if key[1] == 1 else op.setup + op.transition)
        if e['Resource'] not in op.eligible_rooms or e['Personnel'] not in op.eligible_personnel:
            errors.append(f'ineligible resource {key}')
        if abs(e['ProcessingEnd'] - start - op.duration) > 1e-7:
            errors.append(f'duration {key}')
        personnel.append((e['Personnel'], start, e['ProcessingEnd'], key))
    if set(by_job) != set(ops):
        return ['missing operations']
    for job in ctx.jobs:
        a, b = by_job[job.job_id, 1], by_job[job.job_id, 2]
        op = ops[job.job_id, 2]
        surgery_start = b['Start'] + op.setup + op.transition
        if surgery_start < a['ProcessingEnd'] - 1e-7:
            errors.append(f'precedence {job.job_id}')
        if a['Resource'] == b['Resource']:
            intervals.append((a['Resource'], a['Start'], b['Finish'], job.job_id))
        else:
            intervals.extend([(a['Resource'], a['Start'], surgery_start + ops[job.job_id, 1].cleanup, job.job_id),
                              (b['Resource'], b['Start'], b['Finish'], job.job_id)])
    for label, items in [('room holding', intervals), ('personnel', personnel)]:
        for i, a in enumerate(items):
            for b in items[i+1:]:
                if a[0] == b[0] and a[3] != b[3] and max(a[1], b[1]) < min(a[2], b[2]) - 1e-7:
                    errors.append(f'{label}: {a[3]} overlaps {b[3]} in {a[0]}')
    return sorted(set(errors))


class BudgetStop(Exception):
    pass


def pilot(row, method, seed, budget, out):
    ctx = load_instance(ROOT / row['path'])
    module = importlib.import_module('algorithms.' + method)
    count, simulations, rejects, best, best_solution, best_entries = 0, 0, 0, math.inf, None, []
    cache, trace = {}, []
    target = '_instance_fitness' if method == 'dmshoa' else 'calculate_schedule_fitness'
    original = getattr(module, target)

    def evaluate(solution, data=None, return_details=False):
        nonlocal count, simulations, rejects, best, best_solution, best_entries
        if method == 'dmshoa':
            solution = data
        if hasattr(solution, 'to_dict'):
            solution = solution.to_dict()
        key = json.dumps(solution, sort_keys=True)
        if return_details and key in cache:
            return cache[key]
        if count >= budget:
            raise BudgetStop()
        count += 1
        simulations += 1
        f, makespan, entries = _fitness_for_context(ctx, solution, return_details=True)
        if not math.isfinite(f):
            rejects += 1
        if f < best:
            best, best_solution, best_entries = f, copy.deepcopy(solution), entries
        cache[key] = (f, makespan, entries)
        trace.append({'evaluation': count, 'best_f': best if math.isfinite(best) else None})
        return (f, makespan, entries) if return_details else f

    setattr(module, target, evaluate)
    # All baselines receive the same objective here, including native dMShOA
    # which otherwise uses the typed scheduler's makespan alone.
    import config.config as config
    old_verbose = getattr(module, 'VERBOSE_MODE', None)
    module.VERBOSE_MODE = False
    config.VERBOSE_MODE = False
    started, cpu = time.perf_counter(), time.process_time()
    error = None
    try:
        module.run(ctx, seed)
    except BudgetStop:
        pass
    except Exception as exc:
        error = repr(exc)
    finally:
        setattr(module, target, original)
        if old_verbose is not None:
            module.VERBOSE_MODE = old_verbose
    seconds, cpu_seconds = time.perf_counter() - started, time.process_time() - cpu
    issues = inspect_calendar(ctx, best_entries) if best_solution is not None else ['no feasible candidate']
    record = dict(instance_id=ctx.instance_id, instance_digest=ctx.digest, method=method,
                  seed=seed, phase='diagnostic_pilot', usable_for_final=False,
                  requested_budget=budget, evaluations=count, simulator_calls=simulations,
                  rejected_candidates=rejects, best_f=best if math.isfinite(best) else None,
                  wall_seconds=seconds, cpu_seconds=cpu_seconds, error=error,
                  report_model_violations=issues, curve=trace, solution=best_solution, calendar=best_entries)
    write_json(out / f'{ctx.instance_id}__{method}__{seed}.json', record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--budget', type=int, default=120)
    args = parser.parse_args()
    if args.budget < 1:
        parser.error('budget must be positive')
    rows = inventory()
    out = ROOT / 'experiments' / 'records'
    out.mkdir(parents=True, exist_ok=True)
    with (out/'instances.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    commit = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    deps = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (out/'environment.lock.txt').write_text(deps, encoding='utf-8')
    write_json(out/'provenance.json', dict(repository='https://github.com/Saicooh/OII464_Hospitales',
        commit=commit, python=sys.version, platform=platform.platform(), processor=platform.processor(),
        cpu_count=os.cpu_count(), files=len(rows), partitions=dict(Counter(r['phase'] for r in rows)),
        experiment_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seeds={'calibration':list(range(10000,10030)), 'pilot':[20000], 'test':list(range(30000,30030))},
        final_status='BLOCKED: QL-MA, calibration and evaluator alignment not yet implemented',
        pilot_status='diagnostic only; not final metrics'))
    if args.pilot:
        results = []
        for row in rows:
            # Do not inspect held-out test performance during diagnosis.
            if row['phase'] not in ('pilot', 'validation'):
                continue
            for method in ('ga','dpso','sboa','dmshoa'):
                result = pilot(row, method, 20000, args.budget, out/'pilot')
                results.append(result)
                print(row['instance_id'], method, result['evaluations'], 'checks',len(result['report_model_violations']), flush=True)
        summary = dict(runs=len(results), errors=sum(r['error'] is not None for r in results),
            model_violations=sum(bool(r['report_model_violations']) for r in results),
            budget_exhausted=sum(r['evaluations']==args.budget for r in results),
            seconds=sum(r['wall_seconds'] for r in results), usable_for_final=False)
        write_json(out/'pilot_summary.json', summary)
        print(json.dumps(summary))
    else:
        print(json.dumps(dict(instances=len(rows), partitions=dict(Counter(r['phase'] for r in rows)))))


if __name__ == '__main__':
    main()
