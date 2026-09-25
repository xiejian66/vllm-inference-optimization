import argparse
import json
from collections import Counter
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
for d in sorted(a.root.iterdir()):
    if not d.is_dir():continue
    print('\n===',d.name,'===')
    summary=d/'summary.json'
    if not summary.exists():
        print('INCOMPLETE: inspect',d/'run.log');continue
    s=json.loads(summary.read_text())
    print('hot_cached_tokens=',s['hot_cached_tokens'],'preemptions=',s['preemption_events'])
    events=[json.loads(l) for l in (d/'events.jsonl').read_text().splitlines()]
    attempts=[e for e in events if e['event']=='reorder']
    queries=[e for e in attempts if e['policy_queries']]
    changes=[e for e in attempts if e['changed']]
    print('reorder calls with >=2 waiting:',len(attempts))
    print('guard reasons:',dict(Counter(e['reason'] for e in attempts)))
    print('policy cache probes:',sum(len(e['policy_queries']) for e in attempts))
    print('lookup calls without reorder:',sum(not e['changed'] for e in queries))
    print('actual reorder count:',len(changes))
    losses=[e for e in events if e['event']=='allocation' and e['lost_reusable_prefix_tokens']]
    print('allocations losing reusable hot prefixes:',len(losses))
    if losses:
        e=losses[0]
        print('FIRST LOSS:',json.dumps({k:e[k] for k in ('step','seq','request_tag','usage_before','lost_reusable_prefix_tokens','evicted_block_ids')},ensure_ascii=False))
    if changes:
        e=changes[0]
        print('FIRST REORDER: step=',e['step'],'usage=',e['usage'],
              'remaining waiting hot prefix tokens=',sum(e['hot_before'].values()))
        print('cumulative reusable-prefix loss before first reorder=',
              sum(sum(x['lost_reusable_prefix_tokens'].values()) for x in losses if x['seq']<e['seq']))
    else:print('No effective reorder observed.')
    print('Note: diagnostic probes are excluded from policy probe counters. Lost prefix length is not physical bytes evicted.')
