import hashlib
import json
import random
import statistics
import subprocess
from pathlib import Path

from runtime_config import ROOT, MODEL, REPOS
TOKENS = [1, 8, 32, 64, 128, 256]

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)

def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def gpu():
    p = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu',
                        '--format=csv,noheader'], capture_output=True, text=True)
    return dict(returncode=p.returncode, csv=p.stdout.strip(), error=p.stderr.strip())

def stats(values):
    a = sorted(values)
    if not a:
        raise ValueError('Empty measurement set')
    def percentile(p):
        x=(len(a)-1)*p; i=int(x); j=min(i+1,len(a)-1)
        return a[i]+(a[j]-a[i])*(x-i)
    return dict(n=len(a), median=statistics.median(a),
                std=statistics.stdev(a) if len(a)>1 else None,
                min=a[0], max=a[-1], p10=percentile(.1), p90=percentile(.9))

def workload(scenario, batch=32, input_len=512, output_len=128):
    rng=random.Random(54321)
    def seq(n): return [rng.randrange(3000,30000) for _ in range(n)]
    if scenario == 'operator':
        w=dict(prompts=[[100+i]+seq(input_len-1) for i in range(batch)],
               prefixes=[], labels=[dict(kind='independent',family=i,occurrence=0) for i in range(batch)],
               output_len=output_len)
    elif scenario in ('mixed8','mixed16','cold8'):
        prefixes=[[1000+i]+seq(2047) for i in range(16)]
        cold=[[2000+i]+seq(2175) for i in range(32)]
        if scenario=='cold8':
            prompts=cold+[[2032+i]+seq(2175) for i in range(32)]
            labels=[dict(kind='cold',family=i,occurrence=0) for i in range(64)]
            prefixes=[]
        else:
            warm=[prefixes[i]+seq(128) for occurrence in range(2) for i in range(16)]
            prompts=cold+warm
            labels=[dict(kind='cold',family=i,occurrence=0) for i in range(32)]
            labels += [dict(kind='warm',family=i,occurrence=occurrence)
                       for occurrence in range(2) for i in range(16)]
        w=dict(prompts=prompts,prefixes=prefixes,labels=labels,output_len=256)
    else:
        raise ValueError(scenario)
    w['sha256']=hashlib.sha256(json.dumps(w,sort_keys=True).encode()).hexdigest()
    return w

def plan(rounds=3, samples=3, micro_samples=30, unroll=128):
    jobs=[]
    for variant in ('A4','B4','C4'):
        jobs.append(dict(kind='correctness',variant=variant,id='check-'+variant))
        jobs.append(dict(kind='trace',variant=variant,id='kernel-path-'+variant))
    orders=[('A4','B4','C4'),('B4','C4','A4'),('C4','A4','B4')]
    for r in range(rounds):
        for v in orders[r%3]:
            for t in TOKENS:
                jobs.append(dict(kind='micro',variant=v,tokens=t,round=r+1,
                    samples=micro_samples,unroll=unroll,id=f'micro-r{r+1}-{v}-t{t}'))
    for r in range(rounds):
        for v in orders[r%3]:
            jobs.append(dict(kind='model',variant=v,scenario='operator',round=r+1,
                samples=samples,verify=r==0,id=f'e2e-op-r{r+1}-{v}'))
    scenarios=['mixed8','mixed16','cold8']
    for r in range(rounds):
        for s in scenarios[r%3:]+scenarios[:r%3]:
            for v in (('S0','S1') if r%2==0 else ('S1','S0')):
                jobs.append(dict(kind='model',variant=v,scenario=s,round=r+1,
                    samples=samples,verify=r==0 and s=='mixed8',id=f'e2e-sched-r{r+1}-{s}-{v}'))
    return jobs
