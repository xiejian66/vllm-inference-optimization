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
import time
import statistics

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
        disable_log_stats=False,
        attention_config=dict(backend='FLASH_ATTN', flash_attn_version=2),
        compilation_config=comp, enforce_eager=False, seed=42,
        skip_tokenizer_init=True, generation_config='vllm')
    if a.variant == 'RISK':
        settings['scheduler_cls']='risk_perf_scheduler.RiskPerfScheduler'
    (a.out/'settings.json').write_text(json.dumps(settings, indent=2))
    llm = LLM(**settings)
    config = llm.llm_engine.vllm_config
    assert config.model_config.dtype == torch.bfloat16
    assert config.cache_config.kv_cache_memory_bytes == a.kv_gib * 1024**3
    assert config.scheduler_config.cache_aware_admission_threshold == a.threshold
    assert config.scheduler_config.cache_aware_admission_window == (0 if a.mode == "OFF" else 16)
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
    def trial(index):
        assert llm.reset_prefix_cache(), 'Prefix-cache reset failed'
        seeded = generate(prefixes, 1)
        assert len(seeded) == 8
        gpu_before = gpu_snapshot()
        # generate() returns only after all requests finish; includes submission,
        # queueing and CPU/GPU execution. No profiler or per-step logging.
        begin = time.perf_counter()
        outputs = generate(cold+warm, 256)
        elapsed = time.perf_counter() - begin
        gpu_after = gpu_snapshot()
        assert len(outputs) == 16
        rows = []
        for label, prompt, r in zip(workload['labels'], cold+warm, outputs):
            assert list(r.prompt_token_ids) == prompt
            assert r.finished and len(r.outputs[0].token_ids) == 256
            m = r.metrics
            assert m is not None, 'Request metrics unavailable; no fabricated TTFT'
            ttft = m.first_token_latency
            assert ttft > 0 and m.last_token_ts >= m.first_token_ts > 0
            rows.append(dict(label=label, cached_tokens=r.num_cached_tokens,
                ttft_s=ttft, tpot_s=(m.last_token_ts-m.first_token_ts)/255,
                preemptions=m.num_preemptions, output_tokens=256))
        result = dict(sample=index, batch_s=elapsed, output_tok_s=4096/elapsed,
                      requests=rows, gpu_before=gpu_before, gpu_after=gpu_after)
        (a.out/f'sample-{index}.json').write_text(json.dumps(result,indent=2))
        print(f'SAMPLE {index}: batch={elapsed:.6f}s output={4096/elapsed:.2f}tok/s',flush=True)
        return result
    trial('warmup')  # Full workload warmup; discarded. Cache reset before every trial.
    samples = [trial(i) for i in range(a.samples)]
    durations = [r['batch_s'] for r in samples]
    summary = dict(variant=a.variant,trigger="prompt_risk" if a.variant=="RISK" else "usage_threshold",mode=a.mode,kv_gib=a.kv_gib,samples=a.samples,threshold=a.threshold,
        workload_sha256=hashlib.sha256(data).hexdigest(),
        batch_median_s=statistics.median(durations),
        batch_std_s=statistics.stdev(durations) if len(durations)>1 else 0,
        output_tok_s=4096/statistics.median(durations),
        per_sample_hot_cached_tokens=[sum(q['cached_tokens'] for q in r['requests']
            if q['label'].startswith('warm')) for r in samples],
        preemptions=sum(q['preemptions'] for r in samples for q in r['requests']),
        note='Burst workload; threshold applies only to THRESHOLD variant; synchronous scheduler, fixed C4 configuration; not sustained-server throughput. TTFT uses frontend latency; TPOT uses engine monotonic timestamps.')
    for kind in ('warm','cold'):
        rows=[q for r in samples for q in r['requests'] if q['label'].startswith(kind)]
        summary[kind+'_ttft_median_s']=statistics.median(q['ttft_s'] for q in rows)
        summary[kind+'_ttft_max_s']=max(q['ttft_s'] for q in rows)
        summary[kind+'_tpot_median_s']=statistics.median(q['tpot_s'] for q in rows)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

def gpu_snapshot():
    p=subprocess.run(['nvidia-smi','--query-gpu=name,temperature.gpu,power.draw,clocks.sm,memory.used',
                      '--format=csv,noheader'],capture_output=True,text=True)
    return dict(returncode=p.returncode,output=p.stdout.strip(),error=p.stderr.strip())

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['OFF','ON'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--kv-gib',type=int,default=4)
    p.add_argument('--samples',type=int,default=5)
    p.add_argument('--threshold',type=float,default=0.0)
    p.add_argument('--variant',choices=['THRESHOLD','RISK'],required=True)
    p.add_argument('--worker',action='store_true')
    a=p.parse_args();a.out=a.out.resolve()
    assert 0.0 <= a.threshold <= 1.0
    assert a.samples > 0
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
    if a.variant == 'RISK':
        patch_scheduler(overlay/'vllm/v1/core/sched/scheduler.py')
    support=a.out/'support-code'
    support.mkdir()
    for name in ('risk_perf_scheduler.py','risk_scheduler.py',
                 'threshold_scheduler.py','prompt_risk_v1.py','patch_risk_policy.py'):
        shutil.copy2(HERE/name,support/name)
    env=os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join([str(overlay),str(support),str(HERE)]),
               ADMISSION_EVENTS=str(a.out/'events.jsonl'),VLLM_WORKER_MULTIPROC_METHOD='spawn',
               VLLM_CACHE_ROOT=str(a.out/'vllm-cache'))
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',
         '--mode',a.mode,'--out',str(a.out),'--kv-gib',str(a.kv_gib),
         '--samples',str(a.samples),'--threshold',str(a.threshold),'--variant',a.variant]
    with (a.out/'run.log').open('w') as log:
        proc=subprocess.Popen(cmd,env=env,cwd=overlay,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True)
        for line in proc.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        code=proc.wait()
    (a.out/'status.txt').write_text('COMPLETED\n' if code==0 else f'FAILED exit={code}\n')
    if code: raise SystemExit(code)
if __name__=='__main__':main()
