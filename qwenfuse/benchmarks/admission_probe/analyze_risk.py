import json
import sys
from collections import Counter
from pathlib import Path

root=Path(sys.argv[1])
for name in ('4g-ON','8g-ON'):
    directory=root/name
    print('\n===',name,'===')
    if not (directory/'summary.json').exists():
        print('No completed summary; inspect status.txt and run.log.'); continue
    summary=json.loads((directory/'summary.json').read_text())
    print('Workload SHA256:',summary['workload_sha256'])
    print('hot_cached_tokens:',summary['hot_cached_tokens'],
          'preemptions:',summary['preemption_events'])
    events=[json.loads(x) for x in (directory/'events.jsonl').read_text().splitlines()]
    # Mixed burst only; exclude seed-prefix warmup.
    risk=[e for e in events if e.get('event')=='admission_risk'
          and e.get('prompt_tokens')==2176 and e.get('supported')]
    unsupported=[e for e in events if e.get('event')=='admission_risk'
                 and not e.get('supported')]
    print('Unsupported attempts:',len(unsupported))
    print('Full-prompt risk counts (allocation attempts):',
          dict(Counter(e['full_risk']['reason'] for e in risk)))
    for e in risk:
        if e.get('request_tag')==2005:
            print('Cold 2005:',json.dumps({k:e[k] for k in (
                'step','seq','usage_before','cached_tokens','current_new_blocks',
                'full_prompt_new_blocks','current_risk','full_risk')}))
    losses=[e for e in events if e.get('event')=='allocation'
            and e.get('lost_reusable_prefix_tokens')]
    if losses:
        first=losses[0]
        earlier=[e for e in risk if e['request']==first['request']
                 and e['seq']<first['seq']]
        print('First actual loss:',json.dumps({k:first[k] for k in (
            'step','seq','request_tag','lost_reusable_prefix_tokens')}))
        print('Earlier risk for that request:',
              [(e['step'],e['full_risk']['reason']) for e in earlier])
    else:
        print('No observed loss of waiting hot prefixes.')
print('\nDiagnostic only: risk means possible cache exposure, not guaranteed hot-prefix loss.')
