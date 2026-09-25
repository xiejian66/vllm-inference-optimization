import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

from runtime_config import ROOT, MODEL, REPOS
REPO = REPOS["S1"]
HERE = Path(__file__).resolve().parent

def worker(a):
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    assert Path(vllm.__file__).resolve().is_relative_to(a.out / 'overlay')
    comp = dict(mode=3, compile_ranges_endpoints=[256,4096],
                cudagraph_mode='PIECEWISE', cudagraph_capture_sizes=[1,2,4,8],
                use_inductor_graph_partition=True, pass_config=dict(
                    enable_qk_norm_rope_fusion=True,
                    fuse_qk_norm_rope_kvcache=True,
                    fuse_rope_kvcache=False, rope_kvcache_fusion_max_token_num=256))
    settings = dict(model=str(MODEL), dtype='bfloat16',
        quantization=None, kv_cache_dtype='auto', tensor_parallel_size=1,
        kv_cache_memory_bytes=a.kv_gib * 1024**3,
        max_model_len=4096, max_num_seqs=8, max_num_batched_tokens=2048,
        block_size=16, enable_prefix_caching=True, enable_chunked_prefill=True,
        scheduling_policy='fcfs', async_scheduling=False,
        cache_aware_admission_window=0 if a.mode=='OFF' else 16,
        cache_aware_admission_threshold=a.threshold,
        scheduler_cls='risk_policy_scheduler.RiskPolicyScheduler',
        attention_config=dict(backend='FLASH_ATTN', flash_attn_version=2),
        compilation_config=comp, enforce_eager=False, seed=42,
        skip_tokenizer_init=True, generation_config='vllm')
    (a.out/'settings.json').write_text(json.dumps(settings, indent=2))
    llm = LLM(**settings)
    config = llm.llm_engine.vllm_config
    assert config.model_config.dtype == torch.bfloat16
    assert config.cache_config.kv_cache_memory_bytes == a.kv_gib * 1024**3
    rng = random.Random(54321)
    def tokens(n): return [rng.randrange(3000,30000) for _ in range(n)]
    prefixes = [[1000+i]+tokens(2047) for i in range(8)]
    warm = [p+tokens(128) for p in prefixes]
    cold = [[2000+i]+tokens(2175) for i in range(8)]
    workload = dict(prefixes=prefixes, mixed=cold+warm,
                    labels=[f'cold-{i}' for i in range(8)]+[f'warm-{i}' for i in range(8)])
    data = json.dumps(workload, sort_keys=True).encode()
    (a.out/'workload.json').write_bytes(data)
    print('WORKLOAD SHA256:', hashlib.sha256(data).hexdigest(), flush=True)
    def generate(ps, n):
        return llm.generate([{'prompt_token_ids':p} for p in ps],
            SamplingParams(temperature=0, max_tokens=n, ignore_eos=True, detokenize=False),
            use_tqdm=False)
    seeded = generate(prefixes,1)
    assert len(seeded)==8 and all(len(x.outputs[0].token_ids)==1 for x in seeded)
    print('PREFIX WARMUP PASS', flush=True)
    outputs = generate(cold+warm,256)
    assert len(outputs)==16
    rows=[]
    for label, prompt, r in zip(workload['labels'],cold+warm,outputs):
        assert list(r.prompt_token_ids)==prompt, 'Unexpected output ordering'
        assert r.finished and len(r.outputs[0].token_ids)==256
        rows.append(dict(label=label,id=r.request_id,cached_tokens=r.num_cached_tokens,
                         output_tokens=len(r.outputs[0].token_ids)))
    (a.out/'requests.json').write_text(json.dumps(rows,indent=2))
    events=[json.loads(s) for s in (a.out/'events.jsonl').read_text().splitlines()]
    hot=[r for r in rows if r['label'].startswith('warm')]
    known=all(r['cached_tokens'] is not None for r in hot)
    summary=dict(trigger='prompt_risk' if a.mode=='ON' else 'disabled', mode=a.mode, kv_gib=a.kv_gib, configured_legacy_threshold=a.threshold, workload_sha256=hashlib.sha256(data).hexdigest(),
                 all_requests_completed=True,
                 hot_cached_tokens=sum(r['cached_tokens'] for r in hot) if known else None,
                 hot_max_expected_cached_tokens=8*2048,
                 reorder_events=sum(e.get('changed',False) for e in events),
                 preemption_events=sum(len(e.get('preempted',[])) for e in events),
                 note='Diagnostic burst, not sustained-serving performance or model accuracy validation; C4 configured, kernel path not profiled in this run.')
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['OFF','ON'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--kv-gib',type=int,default=4)
    p.add_argument('--threshold',type=float,default=0.5)
    p.add_argument('--worker',action='store_true')
    a=p.parse_args();a.out=a.out.resolve()
    assert 0 <= a.threshold <= 1
    if a.worker:
        worker(a);return
    a.out.mkdir(parents=True,exist_ok=False)
    overlay=a.out/'overlay'
    shutil.copytree(REPO/'vllm',overlay/'vllm',symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__','*.so'))
    for base in [REPOS['A4']/'vllm',REPO/'vllm']:
        for src in base.rglob('*.so'):
            dst=overlay/'vllm'/src.relative_to(base)
            dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists() or dst.is_symlink(): dst.unlink()
            dst.symlink_to(src.resolve())
    f=overlay/'vllm/compilation/passes/fusion/qk_norm_rope_fusion.py'
    s=f.read_text();assert 'forced_token_heads_per_warp=-1' in s
    f.write_text(s.replace('forced_token_heads_per_warp=-1','forced_token_heads_per_warp=4'))
    f=overlay/'vllm/_custom_ops.py';s=f.read_text()
    begin=s.index('def fused_qk_norm_rope_kvcache(');end=s.index('\ndef ',begin+1)
    part=s[begin:end];assert part.count('forced_token_heads_per_warp: int = -1')==1
    f.write_text(s[:begin]+part.replace('forced_token_heads_per_warp: int = -1','forced_token_heads_per_warp: int = 4')+s[end:])
    ast.parse(f.read_text())
    from patch_risk_policy import patch_scheduler
    patch_scheduler(overlay/'vllm/v1/core/sched/scheduler.py')
    diagnostic_dir=a.out/'diagnostic-code'
    diagnostic_dir.mkdir()
    for name in ('risk_policy_scheduler.py', 'risk_scheduler.py',
                 'threshold_scheduler.py', 'prompt_risk_v1.py', 'patch_risk_policy.py'):
        shutil.copy2(HERE/name, diagnostic_dir/name)
    env=os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join([str(overlay),str(diagnostic_dir),str(HERE)]),
               ADMISSION_EVENTS=str(a.out/'events.jsonl'),VLLM_WORKER_MULTIPROC_METHOD='spawn',
               VLLM_CACHE_ROOT=str(a.out/'vllm-cache'))
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',
         '--mode',a.mode,'--out',str(a.out),'--kv-gib',str(a.kv_gib),
         '--threshold',str(a.threshold)]
    with (a.out/'run.log').open('w') as log:
        proc=subprocess.Popen(cmd,env=env,cwd=overlay,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True)
        for line in proc.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        code=proc.wait()
    (a.out/'status.txt').write_text('COMPLETED\n' if code==0 else f'FAILED exit={code}\n')
    if code: raise SystemExit(code)
if __name__=='__main__':main()
