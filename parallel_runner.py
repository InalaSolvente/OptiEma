"""Parallel coordinator; deliberately outside the frozen algorithm source tree.

Changes scheduling only. Four independent processes, each with one math thread.
An in-flight serial calibration can be adopted without restarting R.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import traceback

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'


def process_table():
    class Entry(ctypes.Structure):
        _fields_=[('dwSize',wintypes.DWORD),('cntUsage',wintypes.DWORD),
            ('pid',wintypes.DWORD),('heap',ctypes.c_size_t),('module',wintypes.DWORD),
            ('threads',wintypes.DWORD),('parent',wintypes.DWORD),('priority',wintypes.LONG),
            ('flags',wintypes.DWORD),('exe',wintypes.WCHAR*260)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
    kernel.Process32FirstW.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)]
    kernel.Process32NextW.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    snapshot=kernel.CreateToolhelp32Snapshot(2,0)
    if snapshot==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    result={};entry=Entry();entry.dwSize=ctypes.sizeof(entry)
    try:
        ok=kernel.Process32FirstW(snapshot,ctypes.byref(entry))
        while ok:
            result[entry.pid]=(entry.parent,entry.exe)
            ok=kernel.Process32NextW(snapshot,ctypes.byref(entry))
    finally:kernel.CloseHandle(snapshot)
    return result


def descendants(table,pid):
    found={pid}
    while True:
        added={p for p,(parent,_) in table.items() if parent in found}
        if added<=found:return found-{pid}
        found|=added


def stop_serial_parent(status):
    """Terminate only the verified Python coordinator; leave its R child alive."""
    table=process_table();pid=int(status['pid'])
    if pid not in table or table[pid][1].lower()!='python.exe':
        raise RuntimeError('Serial coordinator identity changed; refusing takeover')
    children=descendants(table,pid)
    r_children=[p for p in children if table[p][1].lower()=='rscript.exe']
    if not r_children:raise RuntimeError('Cannot identify active R child; refusing takeover')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    kernel.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x1000|1,False,pid)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    try:
        creation,exit_time,kern,user=(wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle,*map(ctypes.byref,(creation,exit_time,kern,user))):
            raise ctypes.WinError(ctypes.get_last_error())
        born=((creation.dwHighDateTime<<32)+creation.dwLowDateTime)/1e7-11644473600
        started=datetime.fromisoformat(status['started_utc']).timestamp()
        if not 0<=started-born<=15:raise RuntimeError('PID birth time differs; refusing takeover')
        if not kernel.TerminateProcess(handle,0):raise ctypes.WinError(ctypes.get_last_error())
    finally:kernel.CloseHandle(handle)
    return r_children


def init_worker(project):
    global PROJECT
    PROJECT=Path(project)
    os.chdir(PROJECT);sys.path.insert(0,str(PROJECT))


def final_job(job):
    from experiments.research import source_hash,run
    from experiments.prepare import write_json
    from data.instance_loader import load_instance
    row,method,seed,params,kappa,digest,out,execution=job
    if source_hash()!=digest:raise RuntimeError('Frozen algorithm sources changed')
    ctx=load_instance(PROJECT/row['path'])
    r=run(ctx,method,seed,kappa*len(ctx.jobs),params)
    r.update(phase='test',source_hash=digest,usable_for_final=r['status']=='completed',execution=execution)
    write_json(Path(out)/f'{ctx.instance_id}__{method}__{seed}.json',r)
    return r['status'],ctx.instance_id,method


def self_test(project,workers):
    init_worker(project)
    from experiments.research import METHODS,run,source_hash
    from experiments.prepare import inventory
    from data.instance_loader import load_instance
    row=next(r for r in inventory() if r['phase']=='validation')
    context=load_instance(PROJECT/row['path']);digest=source_hash()
    originals={m:run(context,m,20003,123) for m in METHODS}
    with tempfile.TemporaryDirectory(dir=PROJECT.parent,prefix='parallel-check-') as tmp:
        jobs=[(row,m,20003,{},123//row['n'],digest,tmp,{'workers':workers,'threads_per_worker':1}) for m in METHODS]
        # Use a divisible exact budget for the same comparison on both paths.
        budget=jobs[0][4]*row['n']
        originals={m:run(context,m,20003,budget) for m in METHODS}
        with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(project,)) as pool:
            completed=list(pool.map(final_job,jobs))
        assert all(s=='completed' for s,_,_ in completed),completed
        for m in METHODS:
            r=json.loads((Path(tmp)/f"{context.instance_id}__{m}__20003.json").read_text())
            assert r['status']=='completed' and r['evaluations']==budget
            assert r['objective']==originals[m]['objective'],m
            assert r['solution']==json.loads(json.dumps(originals[m]['solution'])),m
            assert r['result']==originals[m]['result'],m
    assert source_hash()==digest
    print(json.dumps({'parallel_methods_verified':8,'workers':workers,'algorithm_hash_unchanged':digest}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--self-test',action='store_true');p.add_argument('--takeover',action='store_true')
    args=p.parse_args();project=str(Path(args.project).resolve());init_worker(project)
    if not 1<=args.workers<=min(4,os.cpu_count() or 1):p.error('workers must be between 1 and 4')
    if args.self_test:return self_test(project,args.workers)
    from experiments.research import source_hash,METHODS
    from experiments.prepare import write_json
    from experiments.pipeline import parameters
    out=PROJECT/'experiments/records/production'
    frozen=json.loads((out/'frozen_manifest.json').read_text())
    digest=frozen['source_hash']
    if source_hash()!=digest:raise RuntimeError('Frozen algorithm sources changed')
    lock=out/'parallel.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    previous=json.loads((out/'status.json').read_text())
    adopted=None;r_children=[]
    execution=dict(mode='process_parallel',workers=args.workers,threads_per_worker=1,
                   coordinator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   wall_time_condition='concurrent runs; distinguish from sequential pilot timings')
    state=dict(pid=os.getpid(),started_utc=previous['started_utc'],source_hash=digest,execution=execution)
    def update(**kw):
        state.update(kw);state['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(out/'status.json',state)
    try:
        if args.takeover:
            if previous['stage']!='calibration':raise RuntimeError('Takeover only supported during calibration')
            adopted=previous['method']
            if adopted not in METHODS:raise RuntimeError('Unexpected serial method')
            r_children=stop_serial_parent(previous)
        else:
            if previous.get('stage') not in ('failed','complete'):
                raise RuntimeError('An active coordinator requires explicit takeover')
        write_json(out/'parallel_execution.json',dict(previous_status=previous,adopted_method=adopted,
            preserved_R_pids=r_children,started_utc=datetime.now(timezone.utc).isoformat(),**execution))
        runtime=json.loads((PROJECT/'experiments/records/runtime_manifest.json').read_text())['runtime_paths']
        def calibrate(method):
            location=out/'calibration'/method
            if (location/'completed.txt').exists():return method
            if method==adopted:
                while not (location/'completed.txt').exists():
                    table=process_table()
                    if not any(pid in table for pid in r_children):
                        # Allow the completion marker's final flush after R exits.
                        time.sleep(2)
                        if not (location/'completed.txt').exists():raise RuntimeError('Adopted R ended without completion marker')
                        break
                    time.sleep(2)
                return method
            location.mkdir(parents=True,exist_ok=True)
            with (location/'console.log').open('a',encoding='utf-8') as log:
                subprocess.run([runtime['Rscript'],'--vanilla',str(PROJECT/'experiments/calibrate.R'),method,
                    sys.executable,str(PROJECT),runtime['Rlib'],str(location),
                    str(frozen['irace_max_experiments']),str(frozen['kappa'])],
                    stdout=log,stderr=subprocess.STDOUT,check=True,
                    env={**os.environ,'OPTIEMA_EXPECTED_SOURCE_HASH':digest})
            return method
        pending=[m for m in METHODS if not (out/'calibration'/m/'completed.txt').exists()]
        # Adopted R occupies one slot; start it first before scheduling new work.
        if adopted in pending:pending.remove(adopted);pending.insert(0,adopted)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(calibrate,m):m for m in pending}
            while futures:
                done,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED)
                for f in done:f.result();del futures[f]
                complete=[m for m in METHODS if (out/'calibration'/m/'completed.txt').exists()]
                active=[m for m in METHODS if m not in complete and (out/'calibration'/m/'console.log').exists()]
                counts={m:len(list((out/'calibration'/m/'calls').glob('*.json'))) for m in METHODS}
                update(stage='calibration',methods=active,method=', '.join(active),
                       completed_methods=complete,calibration_calls=counts,workers=args.workers)
        if source_hash()!=digest:raise RuntimeError('Frozen algorithm sources changed')
        selected={m:parameters(out/'calibration'/m/'selected.csv') for m in METHODS}
        write_json(out/'selected_parameters.json',selected)
        jobs=[];already=0
        for row in frozen['instances']:
            if row['phase']!='test':continue
            for seed in frozen['test_seeds']:
                order=list(METHODS);random.Random(f"{row['instance_id']}:{seed}").shuffle(order)
                for method in order:
                    path=out/'runs'/f"{row['instance_id']}__{method}__{seed}.json"
                    if path.exists():
                        prior=json.loads(path.read_text())
                        if prior.get('source_hash')!=digest:raise RuntimeError('Incompatible saved run')
                        already+=1
                    else:jobs.append((row,method,seed,selected[method],frozen['kappa'],digest,str(out/'runs'),execution))
        total=already+len(jobs);completed=already;failed=0
        update(stage='final',completed=completed,total=total,workers=args.workers)
        with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=(project,)) as pool:
            futures={pool.submit(final_job,j) for j in jobs}
            while futures:
                done,futures=wait(futures,timeout=5,return_when=FIRST_COMPLETED)
                for f in done:
                    status,instance,method=f.result();completed+=1;failed+=status!='completed'
                update(stage='final',completed=completed,total=total,failed=failed)
        update(stage='summarizing',completed=completed,total=total)
        from experiments.summarize import summarize
        summarize(out);update(stage='complete',completed=completed,total=total,failed=failed)
    except BaseException as exc:
        update(stage='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
