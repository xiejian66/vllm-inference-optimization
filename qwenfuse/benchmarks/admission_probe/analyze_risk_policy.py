import json
import sys
from pathlib import Path
from collections import Counter
root=Path(sys.argv[1])
for name in ('4g-ON','8g-ON'):
    p=root/name
    print('\n===',name,'===')
    if not (p/'summary.json').exists():
        print('No completed summary; inspect status.txt/run.log');continue
    s=json.loads((p/'summary.json').read_text())
    print(json.dumps(s,ensure_ascii=False,indent=2))
    es=[json.loads(x) for x in (p/'events.jsonl').read_text().splitlines()]
    decisions=[e for e in es if e.get('event')=='risk_decision']
    print('Risk decisions:',dict(Counter(e.get('full_risk',{}).get('reason','unsupported') for e in decisions)))
    print('Risk-triggered ranking attempts:',sum(e.get('event')=='risk_reorder' for e in es))
    print('Head changes:',sum(bool(e.get('changed_head')) for e in es if e.get('event')=='risk_reorder'))
    print('Policy cache probes:',sum(len(e.get('policy_queries',[])) for e in es))
    losses=[e for e in es if e.get('lost_reusable_prefix_tokens')]
    print('Allocations losing reusable hot prefixes:',len(losses))
    if losses:
        e=losses[0]
        print('First loss:',{k:e[k] for k in ('step','request_tag','lost_reusable_prefix_tokens')})
    for e in decisions:
        if e.get('request_tag')==2005:
            print('Cold 2005:',{k:e[k] for k in ('step','current_new_blocks','full_prompt_new_blocks','full_risk')})
print('\nDiagnostic run: no timing claims; no model-logit equivalence assertion.')
