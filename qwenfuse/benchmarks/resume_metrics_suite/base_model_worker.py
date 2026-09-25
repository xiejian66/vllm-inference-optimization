"""Serial suite child: actual model path check separate from unprofiled timing."""
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import time
from collections import Counter
from common import MODEL, dump, gpu, sha, stats, workload

def trace_names(root):
    names=[]
    for f in root.rglob('*'):
        if f.name.endswith('.json.gz'):
            with gzip.open(f,'rt') as h:data=json.load(h)
        elif f.suffix=='.json':
            data=json.loads(f.read_text())
        else:continue
        if isinstance(data,dict):
            names.extend(e['name'] for e in data.get('traceEvents',[]) if e.get('cat')=='kernel')
    if not names:raise RuntimeError('No CUDA trace events; path validation incomplete')
    return names

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--overlay',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    a=p.parse_args();j=json.loads(a.job.read_text());a.out.mkdir(parents=True,exist_ok=True)
    import numpy as np
    import torch
    import vllm
    import vllm._C_stable_libtorch as native
    from vllm import LLM, SamplingParams
    assert Path(vllm.__file__).resolve().is_relative_to(a.overlay.resolve())
    assert Path(native.__file__).resolve()==a.library.resolve()
    variant=j['variant']; operator=j['scenario']=='operator'; fusion=variant in ('B4','C4','JOINT')
    comp=dict(mode=3,compile_ranges_endpoints=[256,4096],cudagraph_mode='PIECEWISE',
              cudagraph_capture_sizes=[1,2,4,8,16,32] if operator else [1,2,4,8,16],
              use_inductor_graph_partition=True,pass_config=dict(
                  enable_qk_norm_rope_fusion=True,fuse_qk_norm_rope_kvcache=fusion,
                  fuse_rope_kvcache=False,rope_kvcache_fusion_max_token_num=256))
    kv_gib=16 if j['scenario'] in ('mixed16','random16') else 8
    settings=dict(model=str(MODEL),dtype='bfloat16',quantization=None,kv_cache_dtype='auto',
        tensor_parallel_size=1,seed=42,max_model_len=4096,
        max_num_seqs=32 if operator else 16,max_num_batched_tokens=2048,
        block_size=16,kv_cache_memory_bytes=kv_gib*1024**3,
        enable_prefix_caching=not operator,enable_chunked_prefill=True,
        async_scheduling=False,scheduling_policy='fcfs',enforce_eager=False,
        skip_tokenizer_init=True,generation_config='vllm',disable_log_stats=False,
        attention_config=dict(backend='FLASH_ATTN',flash_attn_version=2),compilation_config=comp)
    if variant in ('S1','JOINT'):
        settings.update(cache_aware_admission_window=16,cache_aware_admission_threshold=.5,
                        scheduler_cls='risk_perf_scheduler.RiskPerfScheduler')
    if j.get('verify'):
        settings.update(logprobs_mode='raw_logits',max_logprobs=-1,
            profiler_config=dict(profiler='torch',torch_profiler_dir=str(a.out/'trace'),
                                 torch_profiler_with_stack=False))
    dump(a.out/'settings.json',settings)
    llm=LLM(**settings)
    config=llm.llm_engine.vllm_config
    assert config.model_config.dtype==torch.bfloat16
    assert config.cache_config.cache_dtype in ('auto','bfloat16')
    assert config.cache_config.kv_cache_memory_bytes==kv_gib*1024**3

    def generate(prompts,count,logits=False):
        params=dict(temperature=0,max_tokens=count,ignore_eos=True,detokenize=False)
        if logits:params['logprobs']=-1
        out=llm.generate([dict(prompt_token_ids=p) for p in prompts],SamplingParams(**params),use_tqdm=False)
        assert len(out)==len(prompts)
        for p,r in zip(prompts,out):
            assert list(r.prompt_token_ids)==p
            assert r.finished and len(r.outputs[0].token_ids)==count
        return out

    if j.get('verify'):
        if not operator: assert llm.reset_prefix_cache()
        # 8*16=128 tokens keeps this verification prefill inside fusion gate <=256.
        path_prompts=workload('operator',batch=8,input_len=16)['prompts']
        generate(path_prompts,4)
        if not operator:assert llm.reset_prefix_cache()
        llm.start_profile()
        try:generate(path_prompts,4)
        finally:llm.stop_profile()
        names=trace_names(a.out/'trace')
        target=[n for n in names if 'fusedQKNormRopeKernel' in n]
        assert target,'QK Norm/RoPE kernel missing'
        cache_calls=sum('reshape_and_cache_flash_kernel' in n for n in names)
        if operator or variant=='JOINT':
            assert any('NTokenHeads' in n and re.search(r',\s*4\s*[,>]',n) for n in target)
        if fusion:
            assert any('NTokenHeads' in n and re.search(r',\s*4,\s*true\s*>',n) for n in target)
            assert cache_calls==0,'Independent KV writes remain in verification workload'
        else:
            assert cache_calls>0,'Official independent KV path missing'
        dump(a.out/'kernel-path.json',dict(targets=dict(Counter(target)),cache_calls=cache_calls,
             warp_policy='forced4' if operator or variant=='JOINT' else 'official_auto',all_kernel_count=len(names)))
        # Fixed-history logits, not comparisons after sampled histories diverge.
        forced=workload('operator',batch=2,input_len=128)['prompts']
        cfg=json.loads((MODEL/'config.json').read_text());vocab=cfg['vocab_size']
        rows=[]
        for step in range(2):
            if not operator:assert llm.reset_prefix_cache()
            out=generate([p+[777]*step for p in forced],1,logits=True)
            rows.append([[r.outputs[0].logprobs[0][t].logprob for t in range(vocab)] for r in out])
        values=np.asarray(rows,dtype=np.float32);assert np.isfinite(values).all()
        np.save(a.out/'logits.npy',values)
        dump(a.out/'logits-inputs.json',dict(prompts=forced,suffix=777,steps=2))

    if j.get('check_only'):
        dump(a.out/'result.json',dict(job=j,check_only=True,settings=settings,
             native=str(a.library),native_sha256=sha(a.library)))
        print('PATH_AND_FINITE_LOGITS_PASS',variant,flush=True)
        return

    # Profile/logits operations are followed by a separate full-workload warmup.
    summaries=[]
    for batch in j.get('batches', [1,8,32] if operator else [64]):
        w=workload(j['scenario'],batch=batch)
        dump(a.out/f'workload-b{batch}.json',w)
        def trial(sample):
            if not operator:
                assert llm.reset_prefix_cache(),'Prefix cache reset failed'
                if w['prefixes']:generate(w['prefixes'],1)
            before=gpu();start=time.perf_counter()
            outputs=generate(w['prompts'],w['output_len'])
            elapsed=time.perf_counter()-start;after=gpu()
            requests=[]
            for label,r in zip(w['labels'],outputs):
                m=r.metrics
                assert m is not None and m.first_token_latency>0
                assert not m.is_corrupted,'vLLM reported corrupted request/logits'
                assert m.last_token_ts>=m.first_token_ts>0
                assert m.last_token_ts>=m.scheduled_ts>=m.queued_ts>0
                q=dict(**label,cached_tokens=r.num_cached_tokens,ttft_s=m.first_token_latency,
                    tpot_s=(m.last_token_ts-m.first_token_ts)/(w['output_len']-1),
                    engine_residence_s=m.last_token_ts-m.queued_ts,
                    queue_s=m.scheduled_ts-m.queued_ts,
                    preemptions=m.num_preemptions,output_tokens=list(r.outputs[0].token_ids))
                requests.append(q)
            if j['scenario']=='cold8':
                assert all(r['cached_tokens']==0 for r in requests),'Unexpected cold-workload prefix hit'
            result=dict(sample=sample,batch=batch,seconds=elapsed,
                output_tok_s=len(outputs)*w['output_len']/elapsed,workload_sha256=w['sha256'],
                requests=requests,gpu_before=before,gpu_after=after)
            dump(a.out/f'b{batch}-{sample}.json',result)
            print('SAMPLE',variant,j['scenario'],batch,sample,elapsed,flush=True)
            return result
        trial('warmup')
        results=[trial(i) for i in range(j['samples'])]
        metrics=dict(batch=batch,workload_sha256=w['sha256'],latency=stats([r['seconds'] for r in results]),
            throughput=stats([r['output_tok_s'] for r in results]),preemptions=sum(q['preemptions'] for r in results for q in r['requests']))
        for kind in ('independent','cold','warm'):
            qs=[q for r in results for q in r['requests'] if q['kind']==kind]
            if qs:
                metrics[kind]={key:stats([q[key] for q in qs]) for key in ('ttft_s','tpot_s','engine_residence_s','queue_s')}
        if not operator:
            metrics['warm_cached_by_sample']=[sum(q['cached_tokens'] for q in r['requests'] if q['kind']=='warm') for r in results]
            metrics['warm_cached_by_occurrence']={str(o):[sum(q['cached_tokens'] for q in r['requests'] if q['kind']=='warm' and q['occurrence']==o) for r in results] for o in (0,1)}
        summaries.append(metrics)
    dump(a.out/'result.json',dict(job=j,results=summaries,native=str(a.library),native_sha256=sha(a.library),
        torch=torch.__version__,cuda=torch.version.cuda,settings=settings,
        note='Burst completion wall time; not sustained service throughput. Logits assessed separately. Prefix warmup excluded.'))

if __name__=='__main__':main()
