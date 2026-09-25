import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from common import dump

def numeric(a,b):
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    a=a.astype(np.float64); b=b.astype(np.float64)
    delta=b-a
    return dict(exact=bool(np.array_equal(a,b)),max_abs=float(np.abs(delta).max()),
                mean_abs=float(np.abs(delta).mean()),
                relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(a),1e-30)),
                argmax_agreement=float(np.mean(a.argmax(-1)==b.argmax(-1))))

def main():
    root=Path(sys.argv[1]);jobs=json.loads((root/'plan.json').read_text())
    micro=defaultdict(dict); models=defaultdict(dict); logits={}; inputs={}; drift=[]
    for j in jobs:
        d=root/'stages'/j['id']
        assert json.loads((d/'status.json').read_text())['state']=='PASS',j['id']
        x=json.loads((d/'result.json').read_text()); v=j['variant']
        if j['purpose']=='numeric':
            logits[v,j['repeat']]=np.load(d/'logits.npy')
            inputs[v,j['repeat']]=json.loads((d/'logits-inputs.json').read_text())
        elif j['kind']=='micro': micro[j['round'],j['tokens']][v]=x
        else:
            for row in x['results']:
                models[j['purpose'],j['scenario'],j['seed'],j['round'],row['batch']][v]=row
    lines=[
        'A4/B4/C4 micro and anomaly force heads/warp=4.',
        'S0/S1: official native library, auto warp; JOINT: C4 native forced4 + final risk scheduler.',
        'All performance uses unprofiled timing, after warmup. Burst workload, not sustained serving.',
        'Positive reduction means lower latency; throughput gain is reported separately.',
        '', 'MICRO: round tokens A4_us B4_us C4_us A4_B4% B4_C4% A4_C4%']
    for (r,t),g in sorted(micro.items()):
        a,b,c=[g[v]['median_us'] for v in ('A4','B4','C4')]
        lines.append(f'{r} {t} {a:.3f} {b:.3f} {c:.3f} {(1-b/a)*100:+.2f} {(1-c/b)*100:+.2f} {(1-c/a)*100:+.2f}')
    for t in sorted({t for r,t in micro}):
        for v in ('A4','B4','C4'):
            vals=[g[v]['median_us'] for (r,n),g in micro.items() if n==t]
            spread=(max(vals)-min(vals))/min(vals)
            if spread>.15: drift.append(dict(kind='micro',tokens=t,variant=v,spread=spread))
    rows=[]
    model_rounds=defaultdict(list)
    lines+=['','END-TO-END: purpose scenario seed round batch variant median_s tok_s warm_cached warm_TTFT cold_TTFT']
    for key,g in sorted(models.items()):
        purpose,scene,seed,r,batch=key
        assert len({x['workload_sha256'] for x in g.values()})==1, key
        for v,x in g.items():
            model_rounds[purpose,scene,seed,batch,v].append(x['latency']['median'])
            lines.append(f"{purpose} {scene} {seed} {r} {batch} {v} {x['latency']['median']:.6f} "
                         f"{x['throughput']['median']:.2f} {x.get('warm_cached_by_sample',[])} "
                         f"{x.get('warm',{}).get('ttft_s',{}).get('median')} "
                         f"{x.get('cold',{}).get('ttft_s',{}).get('median')}")
            if (x['latency']['max']-x['latency']['min'])/x['latency']['min']>.15:
                drift.append(dict(kind='model-within-stage',key=key,variant=v))
        pairs=[('A4','C4')] if purpose=='anomaly' else [('S0','S1')]
        if purpose=='main': pairs += [('S0','JOINT'),('S1','JOINT')]
        for av,bv in pairs:
            a,b=g[av],g[bv]
            row=dict(purpose=purpose,scenario=scene,seed=seed,round=r,batch=batch,
                     baseline=av,candidate=bv,baseline_s=a['latency']['median'],candidate_s=b['latency']['median'],
                     latency_reduction_pct=(1-b['latency']['median']/a['latency']['median'])*100,
                     throughput_gain_pct=(b['throughput']['median']/a['throughput']['median']-1)*100)
            rows.append(row)
            lines.append(f"  {av}->{bv}: latency {row['latency_reduction_pct']:+.2f}% throughput {row['throughput_gain_pct']:+.2f}%")
    for key,vals in model_rounds.items():
        spread=(max(vals)-min(vals))/min(vals)
        if spread>.15: drift.append(dict(kind='model-cross-round',key=key,spread=spread))
    if rows:
        with (root/'paired-performance.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups=defaultdict(list)
    for row in rows: groups[row['purpose'],row['scenario'],row['baseline'],row['candidate']].append(row)
    lines+=['','PAIRED SUMMARY (all predeclared seeds, no best-seed selection):']
    for key,rs in groups.items():
        vals=[r['latency_reduction_pct'] for r in rs]
        gains=[r['throughput_gain_pct'] for r in rs]
        lines.append(f'{key}: n={len(rs)}, latency median={statistics.median(vals):+.2f}%, '
                     f'range=[{min(vals):+.2f},{max(vals):+.2f}]%, faster={sum(x>0 for x in vals)}/{len(vals)}, '
                     f'throughput median={statistics.median(gains):+.2f}%')
    comparisons={}
    def compare(name,left,right):
        assert inputs[left]==inputs[right],name
        comparisons[name]=numeric(logits[left],logits[right])
    for v in ('A4','C4','S0','S1','JOINT'):
        compare(v+' repeat', (v,1),(v,2))
    for r in (1,2):
        for a,b in (('A4','C4'),('S0','S1'),('S0','JOINT')):
            compare(f'{a}->{b} repeat{r}',(a,r),(b,r))
    dump(root/'numerical-review.json',comparisons)
    dump(root/'timing-review.json',drift)
    dump(root/'paired-performance.json',rows)
    lines+=['','NUMERICAL REVIEW:',json.dumps(comparisons,indent=2),
            '', 'LIMITS:',
            'No model error tolerance preapproved. Exact differences require explanation, not automatic acceptance.',
            'Fixed-history finite logits are not model task-accuracy validation.',
            'Cold/warm TTFT, TPOT, queue median/max and preemptions are in every stage/result.json.',
            '15% spread is a screening flag, not a significance test.',
            f'Timing screening flags: {len(drift)}',
            'No P99, memory reduction, maximum concurrency, or sustained-arrival fairness claims.',
            'No NCU collection in this script. Existing counter evidence remains separate.']
    (root/'summary.txt').write_text('\n'.join(lines)+'\n')
    print('REPORT:',root/'summary.txt')

if __name__=='__main__':main()
