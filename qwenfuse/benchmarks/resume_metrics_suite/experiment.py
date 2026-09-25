"""Predeclared workloads and comparisons; no GPU imports."""
import hashlib
import json
import random
from collections import Counter
from common import workload

SEEDS = [191840815, 20260925, 20260926]

def make_workload(scenario, batch=32, input_len=512, output_len=128, *, seed):
    base = {'random8': 'mixed8', 'random16': 'mixed16'}.get(scenario, scenario)
    w = workload(base, batch, input_len, output_len)
    if scenario.startswith('random'):
        pairs = list(zip(w['prompts'], w['labels']))
        random.Random(seed).shuffle(pairs)
        w['prompts'] = [p for p, _ in pairs]
        w['labels'] = [dict(label) for _, label in pairs]
        seen = Counter()
        for label in w['labels']:
            if label['kind'] == 'warm':
                label['occurrence'] = seen[label['family']]
                seen[label['family']] += 1
        w.pop('sha256')
        w['shuffle_seed'] = seed
        w['sha256'] = hashlib.sha256(json.dumps(w, sort_keys=True).encode()).hexdigest()
    return w

def plan():
    jobs = []
    def add(**j):
        jobs.append(j)
    # Two fresh model processes per configuration; fixed teacher-forced histories.
    for repeat, order in enumerate((('A4','C4','S0','S1','JOINT'),
                                    ('JOINT','S1','S0','C4','A4')), 1):
        for v in order:
            add(id=f'numeric-r{repeat}-{v}', kind='model', variant=v,
                scenario='operator' if v in ('A4','C4') else 'random8',
                repeat=repeat, seed=SEEDS[0], verify=True, check_only=True,
                samples=1, purpose='numeric')
    for r, order in enumerate((('A4','B4','C4'),('B4','C4','A4'),('C4','A4','B4')), 1):
        for t in (1,8,32,64,128,256):
            for v in order:
                add(id=f'micro-r{r}-{v}-t{t}', kind='micro', variant=v,
                    tokens=t, round=r, samples=100, unroll=128, purpose='micro')
    for r, order in enumerate((('A4','C4'),('C4','A4')), 1):
        for v in order:
            add(id=f'anomaly-r{r}-{v}',kind='model',variant=v,scenario='operator',
                round=r,seed=SEEDS[0],verify=False,samples=3,batches=[8],purpose='anomaly')
    for seed in SEEDS:
        for r, order in enumerate((('S0','S1','JOINT'),('JOINT','S1','S0')), 1):
            for v in order:
                add(id=f'random8-seed{seed}-r{r}-{v}',kind='model',variant=v,
                    scenario='random8',seed=seed,round=r,verify=False,samples=3,purpose='main')
    for scene in ('random16','cold8'):
        for r, order in enumerate((('S0','S1'),('S1','S0')), 1):
            for v in order:
                add(id=f'{scene}-r{r}-{v}',kind='model',variant=v,scenario=scene,
                    seed=SEEDS[0],round=r,verify=False,samples=3,purpose='control')
    return jobs
