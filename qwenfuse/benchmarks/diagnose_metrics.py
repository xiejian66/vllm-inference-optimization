"""Focused diagnostics. Run one action at a time; preserve original evidence."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import fcntl

SNAP = None

def launch(job, label, variant, worker, out, monitor=False):
    paths=json.loads((SNAP/'paths.json').read_text())
    d=out/label;d.mkdir()
    jf=d/'job.json';jf.write_text(json.dumps(job,indent=2))
    env=os.environ.copy()
    env.update(PYTHONPATH=paths['overlays'][variant]+':'+str(SNAP/'code'),
               VLLM_WORKER_MULTIPROC_METHOD='spawn',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONHASHSEED='0',
               VLLM_CACHE_ROOT=str(out/'cache'/variant/'vllm'),
               TORCHINDUCTOR_CACHE_DIR=str(out/'cache'/variant/'inductor'),
               TRITON_CACHE_DIR=str(out/'cache'/variant/'triton'))
    args=[sys.executable,'-u',str(worker),'--job',str(jf),
          '--library',paths['libraries'][variant],'--out',str(d)]
    if job['kind']=='model':args+=['--overlay',paths['overlays'][variant]]
    (d/'command.json').write_text(json.dumps(args))
    print('START',label, 'LOG',d/'run.log',flush=True)
    telemetry=None
    with (d/'run.log').open('w') as log, (d/'gpu-timeline.csv').open('w') as gpu:
        if monitor:
            telemetry=subprocess.Popen(['nvidia-smi',
                '--query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used',
                '--format=csv','--loop-ms=500'],stdout=gpu,stderr=subprocess.STDOUT)
        start=time.time()
        try:subprocess.run(args,env=env,cwd=paths['overlays'][variant],stdout=log,stderr=subprocess.STDOUT,check=True)
        finally:
            if telemetry is not None:
                telemetry.terminate();telemetry.wait()
    (d/'status.json').write_text(json.dumps(dict(state='PASS',elapsed_s=time.time()-start)))
    print('PASS',label,flush=True)
    return d

def inspect(out):
    lines=[]
    for name in ('random8-seed20260926-r1-S1','random8-seed20260926-r2-S1'):
        lines.append('\n'+name)
        for f in sorted((SNAP/'stages'/name).glob('b64-*.json')):
            x=json.loads(f.read_text())
            lines.append(json.dumps(dict(sample=f.name,seconds=x['seconds'],
                gpu_before=x['gpu_before'],gpu_after=x['gpu_after']),ensure_ascii=False))
    text='\n'.join(lines);(out/'old-evidence.txt').write_text(text);print(text)

def numerical(out):
    original=(SNAP/'code/base_model_worker.py').read_text()
    anchor="fusion=variant in ('B4','C4','JOINT')"
    assert original.count(anchor)==1
    off=out/'model_fusion_off.py';off.write_text(original.replace(anchor,'fusion=False'))
    import numpy as np
    sys.path.insert(0,str(SNAP/'code'))
    from report import numeric
    records={}
    # B4 OFF distinguishes B's native changes from C's register-preload changes.
    for label,v,worker in (
        ('A4','A4',SNAP/'code/model_worker.py'),
        ('B4_OFF','B4',off),('C4_OFF','C4',off),
        ('C4_ON','C4',SNAP/'code/model_worker.py')):
        j=dict(id=label,kind='model',variant=v,scenario='operator',seed=191840815,
               verify=True,check_only=True,samples=1)
        records[label]=launch(j,label,v,worker,out)
    result={}
    for a,b in (('A4','B4_OFF'),('B4_OFF','C4_OFF'),('C4_OFF','C4_ON'),('A4','C4_ON')):
        assert json.loads((records[a]/'logits-inputs.json').read_text())==json.loads((records[b]/'logits-inputs.json').read_text())
        result[a+' -> '+b]=numeric(np.load(records[a]/'logits.npy'),np.load(records[b]/'logits.npy'))
    (out/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    print('Diagnostic localization only: differences need explanation; no accuracy tolerance inferred.')

def slow(out):
    inspect(out)
    for r in (1,2):
        j=dict(id=f'S1-repeat{r}',kind='model',variant='S1',scenario='random8',
               seed=20260926,verify=False,samples=3,purpose='diagnostic')
        d=launch(j,j['id'],'S1',SNAP/'code/model_worker.py',out,monitor=True)
        for f in sorted(d.glob('b64-*.json')):
            x=json.loads(f.read_text());print(f.name,x['seconds'],flush=True)

def micro(out):
    # Patch the benchmark only, reuse fixed model geometry from kernel_worker.
    bench=r'''
import json, statistics, time
import torch
import operator_support as op
def measured(a):
    cs=[op.Case(op.spec(a.tokens),a.mode,seed=123+i) for i in range(a.unroll)]
    def reset():
        for c in cs:c.reset()
    def run():
        for c in cs:c.run()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):reset();run()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True,external=True)
    end=torch.cuda.Event(enable_timing=True,external=True)
    g=torch.cuda.CUDAGraph()
    # Resets are graph nodes before start, excluded from the timed interval.
    # Independent input per invocation; every replay restores original inputs.
    with torch.cuda.graph(g):
        reset();start.record();run();end.record()
    def sample():
        g.replay();torch.cuda.synchronize()
        return start.elapsed_time(end)*1000/a.unroll
    warm=[];deadline=time.perf_counter()+3.0
    while time.perf_counter()<deadline:warm.append(sample())
    values=[sample() for _ in range(a.samples)]
    assert all(x>0 for x in values)
    for c in (cs[0],cs[-1]):c.check()
    op.dump(a.out,dict(mode=a.mode,tokens=a.tokens,unroll=a.unroll,samples_us=values,
        median_us=statistics.median(values),std_us=statistics.stdev(values),
        warmup_last_us=warm[-30:],qkv_stride=list(cs[0].x.stride()),
        kv_stride=list(cs[0].kc.stride()),
        note='Graph-internal timing events; captured reset outside timed interval; 3s time-based warmup. Cache state differs from legacy harness; compare A/B/C only within this method.'))
op.bench=measured
import kernel_worker
kernel_worker.main()
'''
    worker=out/'graph_internal_events.py';worker.write_text(bench)
    lines=['round tokens A4_us B4_us C4_us A4_C4_pct B4_C4_pct']
    for r,order in enumerate((('A4','B4','C4'),('C4','B4','A4'),('B4','A4','C4')),1):
        for t in (1,8,32,64,128,256):
            values={}
            for v in order:
                label=f'r{r}-t{t}-{v}'
                d=launch(dict(id=label,kind='micro',variant=v,tokens=t,unroll=128,samples=100),
                         label,v,worker,out,monitor=True)
                values[v]=json.loads((d/'result.json').read_text())['median_us']
            a,b,c=[values[v] for v in ('A4','B4','C4')]
            lines.append(f'{r} {t} {a:.3f} {b:.3f} {c:.3f} {(1-c/a)*100:+.2f} {(1-c/b)*100:+.2f}')
            print(lines[-1],flush=True)
    (out/'summary.txt').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['inspect','numeric','slow','micro'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True,help='Completed run containing code, overlays and paths.json')
    a=p.parse_args()
    global SNAP
    SNAP=a.snapshot.expanduser().resolve()
    if not (SNAP/'paths.json').is_file():
        p.error('snapshot must be a full run, not the trimmed published results')
    work=Path(os.environ.get('QWENFUSE_WORKSPACE','.qwenfuse-work')).expanduser().resolve()
    (work/'reports').mkdir(parents=True,exist_ok=True)
    lock=(work/'reports/qwenfuse-formal.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    a.out.mkdir(parents=True,exist_ok=False)
    if a.action!='inspect':
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
        assert not apps.strip(), 'Other GPU processes active: '+apps
        assert shutil.disk_usage(a.out).free>3*1024**3,'Need 3 GiB free'
    {'inspect':inspect,'numeric':numerical,'slow':slow,'micro':micro}[a.action](a.out)
    print('DONE',a.out)

if __name__=='__main__':main()
