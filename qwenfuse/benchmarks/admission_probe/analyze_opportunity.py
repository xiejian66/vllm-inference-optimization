import argparse
import json
from collections import Counter
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
assert (a.run/'summary.json').exists(), 'Run incomplete; inspect run.log'
s=json.loads((a.run/'summary.json').read_text())
events=[json.loads(line) for line in (a.run/'events.jsonl').read_text().splitlines()]
opp={e['step']:e for e in events if e['event']=='admission_opportunity'}
end={e['step']:e for e in events if e['event']=='admission_outcome'}
lookups=[e for e in events if e['event']=='reorder' and e['policy_queries']]
blocked=[e for e in lookups if opp[e['step']]['blocked_by']]
queries=sum(len(e['policy_queries']) for e in lookups)
wasted=sum(len(e['policy_queries']) for e in blocked)
print('热请求实际缓存命中:',s['hot_cached_tokens'])
print('抢占:',s['preemption_events'])
print('查询轮次:',len(lookups),'策略查询总次数:',queries)
print('硬性准入条件不满足但仍查询的轮次:',len(blocked))
print('这些轮次的查询次数:',wasted)
print('这些查询占比:',f'{100*wasted/queries:.2f}%' if queries else 'N/A')
print('阻挡原因（允许重叠）:',dict(Counter(r for e in blocked for r in opp[e['step']]['blocked_by'])))
print('这些轮次仍有新请求准入:',sum(bool(end[e['step']]['newly_admitted']) for e in blocked))
print('硬性条件满足、查询后仍无新请求准入的轮次:',sum(not opp[e['step']]['blocked_by'] and not end[e['step']]['newly_admitted'] for e in lookups))
print('\n前5个被阻挡的查询轮次:')
for e in blocked[:5]:
 o=opp[e['step']]
 print(json.dumps(dict(step=e['step'],**o['context'],reasons=o['blocked_by'],
                      queries=len(e['policy_queries']),reordered=e['changed'],
                      newly_admitted=end[e['step']]['newly_admitted']),ensure_ascii=False))
print('\n注意：这是带诊断的因果检查，不是CPU耗时测量；条件通过不保证KV分配成功。')
