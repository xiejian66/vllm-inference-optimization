"""Frozen-source sequential experiments. Default: preflight/plan; --execute starts jobs."""
import argparse
import ast
import difflib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from common import ROOT, MODEL, REPOS, TOKENS, dump, sha, gpu, plan

HERE=Path(__file__).resolve().parent
NATIVE='_C_stable_libtorch.abi3.so'

def replace_once(s, old, new):
    if s.count(old)!=1:raise RuntimeError(f'Expected one patch anchor ({s.count(old)} found): {old[:70]}')
    return s.replace(old,new)

def force_four(overlay, fused):
    p=overlay/'vllm/compilation/passes/fusion/qk_norm_rope_fusion.py'
    s=replace_once(p.read_text(),'forced_token_heads_per_warp=-1','forced_token_heads_per_warp=4')
    ast.parse(s);p.write_text(s)
    if fused:
        p=overlay/'vllm/_custom_ops.py';s=p.read_text()
        start=s.index('def fused_qk_norm_rope_kvcache(');end=s.index('\ndef ',start+1)
        part=replace_once(s[start:end],'forced_token_heads_per_warp: int = -1','forced_token_heads_per_warp: int = 4')
        s=s[:start]+part+s[end:];ast.parse(s);p.write_text(s)

def preflight():
    for label, repo in REPOS.items():
        if not (repo/'vllm'/NATIVE).is_file():
            raise RuntimeError(f'{label}: missing compiled extension in {repo}; prepare and build variants first')
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert torch.cuda.device_count()==1,'This suite is single-GPU'
    assert torch.cuda.get_device_capability()==(8,0),'Expected current A100; reassess on another GPU'
    c=json.loads((MODEL/'config.json').read_text())
    assert (c['num_hidden_layers'],c['num_attention_heads'],c['num_key_value_heads'],c['head_dim'])==(36,32,8,128)
    idx=MODEL/'model.safetensors.index.json'
    if idx.exists():
        for file in set(json.loads(idx.read_text())['weight_map'].values()):assert (MODEL/file).is_file()
    else:assert list(MODEL.glob('*.safetensors'))
    for repo in set(REPOS.values()):
        assert (repo/'vllm'/NATIVE).is_file(),str(repo)
        head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        assert head=='27757dde020ecda4f9b0e2c5ca2df1c29badc82f',(repo,head)
    subprocess.run(['git','-C',str(REPOS['A4']),'diff','--exit-code','HEAD','--','vllm','csrc'],
                   stdout=subprocess.DEVNULL,check=True)
    assert shutil.disk_usage(ROOT).free>8*1024**3,'Need at least 8 GiB free for run artifacts/caches'
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                              '--format=csv,noheader'],capture_output=True,text=True,check=True)
    # CUDA environment check creates this process's own context; others must be absent.
    other=[x for x in processes.stdout.splitlines() if x.strip() and x.split(',')[0].strip()!=str(os.getpid())]
    assert not other, f'Other GPU processes: {other}'
    return dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
                gpu=gpu(),free_disk=shutil.disk_usage(ROOT).free,model_config=c)

def freeze(out):
    # Copy Python/source, copy each distinct main native library once. Auxiliary .so
    # remains linked but is hashed and checked before/after every stage.
    overlays={}; auxiliaries={}; libraries={}
    for v,repo in REPOS.items():
        dst=out/'overlays'/v
        shutil.copytree(repo/'vllm',dst/'vllm',symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__','*.so'))
        for base in (REPOS['A4']/'vllm',repo/'vllm'):
            for so in base.rglob('*.so'):
                if so.name==NATIVE:continue
                source=so.resolve();target=dst/'vllm'/so.relative_to(base)
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists() or target.is_symlink():target.unlink()
                target.symlink_to(source)
                auxiliaries[str(source)]=None
        lib_variant='A4' if v in ('A4','S0','S1') else v
        lib_source=REPOS[lib_variant]/'vllm'/NATIVE
        lib_dst=out/'binaries'/lib_variant/NATIVE
        if not lib_dst.exists():
            lib_dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(lib_source,lib_dst)
        target=dst/'vllm'/NATIVE
        if target.exists() or target.is_symlink():target.unlink()
        target.symlink_to(lib_dst)
        libraries[v]=str(lib_dst)
        if v in ('A4','B4','C4'):force_four(dst,v!='A4')
        if v=='S1':
            from patch_risk_policy import patch_scheduler
            patch_scheduler(dst/'vllm/v1/core/sched/scheduler.py')
        overlays[v]=str(dst)
        code=repo/'csrc/libtorch_stable/fused_qknorm_rope_kernel.cu'
        target_source=out/'sources'/v
        target_source.mkdir(parents=True,exist_ok=True)
        shutil.copy2(code,target_source/code.name)
        diff=subprocess.check_output(['git','-C',str(repo),'diff','HEAD','--binary'])
        (target_source/'source.patch').write_bytes(diff)
    for k in auxiliaries:auxiliaries[k]=sha(k)
    for v,lib in libraries.items():auxiliaries[lib]=sha(lib)
    dump(out/'native-hashes.json',auxiliaries)
    dump(out/'paths.json',dict(overlays=overlays,libraries=libraries))
    a=(out/'sources/B4/fused_qknorm_rope_kernel.cu').read_text().splitlines(True)
    b=(out/'sources/C4/fused_qknorm_rope_kernel.cu').read_text().splitlines(True)
    (out/'sources/B4-C4.diff').write_text(''.join(difflib.unified_diff(a,b,fromfile='B4',tofile='C4')))
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(out/'packages.txt').open('w'),check=True)
    return overlays,libraries

def check_binaries(out):
    for p,h in json.loads((out/'native-hashes.json').read_text()).items():
        if sha(p)!=h:raise RuntimeError(f'Native binary changed during experiment: {p}')

def compare_operator_checks(out):
    rows={v:json.loads((out/'stages'/('check-'+v)/'result.json').read_text()) for v in ('A4','B4','C4')}
    assert len({len(x) for x in rows.values()})==1
    for i,a in enumerate(rows['A4']):
        for v in ('B4','C4'):
            b=rows[v][i]
            assert (a['spec'],a['execution'])==(b['spec'],b['execution'])
            for k in ('Q','K','KV'):
                assert a['result'][k]==b['result'][k],f'{v} differs from official A4 at {i}: {k}'
    dump(out/'operator-correctness.json',dict(exact=True,comparisons=len(rows['A4']),scope='Qwen3 full NeoX BF16/FP16'))

def stage(out, job, overlays, libraries):
    if shutil.disk_usage(ROOT).free<3*1024**3:raise RuntimeError('Less than 3 GiB free; stopping without deleting results')
    check_binaries(out)
    v=job['variant'];dest=out/'stages'/job['id'];dest.mkdir(parents=True,exist_ok=False)
    dump(dest/'job.json',job)
    env=os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join((overlays[v],str(out/'code'))),
        VLLM_WORKER_MULTIPROC_METHOD='spawn',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        PYTHONHASHSEED='0',VLLM_CACHE_ROOT=str(out/'cache'/v/'vllm'),
        TORCHINDUCTOR_CACHE_DIR=str(out/'cache'/v/'inductor'),TRITON_CACHE_DIR=str(out/'cache'/v/'triton'))
    env.pop('ADMISSION_EVENTS',None)
    worker='model_worker.py' if job['kind']=='model' else 'kernel_worker.py'
    cmd=[sys.executable,'-u',str(out/'code'/worker),'--job',str(dest/'job.json'),
         '--out',str(dest),'--library',libraries[v]]
    if job['kind']=='model':cmd+=['--overlay',overlays[v]]
    dump(dest/'command.json',cmd)
    print('START',job['id'],'log:',dest/'run.log',flush=True)
    dump(out/'status.json',dict(state='RUNNING',stage=job['id'],pid=os.getpid(),updated=time.time()))
    with (dest/'run.log').open('w') as log:
        process=subprocess.Popen(cmd,env=env,cwd=overlays[v],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait()
        except BaseException:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
    if code:
        # Terminate surviving descendants of THIS failed stage, never unrelated jobs.
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        raise RuntimeError(f'{job["id"]} exited {code}; inspect {dest / "run.log"}')
    check_binaries(out)
    dump(dest/'status.json',dict(state='PASS',finished=time.time()))
    print('DONE',job['id'],flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute',action='store_true',help='Without this flag only plan/preflight runs')
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--samples',type=int,default=3)
    p.add_argument('--micro-samples',type=int,default=30);p.add_argument('--unroll',type=int,default=128)
    a=p.parse_args();a.out=a.out.resolve()
    assert a.rounds>=1 and a.samples>=1 and a.micro_samples>=3 and a.unroll>=1
    jobs=plan(a.rounds,a.samples,a.micro_samples,a.unroll)
    a.out.mkdir(parents=True,exist_ok=False)
    lock=(ROOT/'reports/qwenfuse-formal.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        dump(a.out/'plan.json',jobs)
        # Check CUDA in a child so orchestration holds no GPU context during trials.
        cmd=[sys.executable,'-c',
             'import run_suite; from common import dump; dump('+repr(str(a.out/'environment.json'))+',run_suite.preflight())']
        subprocess.run(cmd,cwd=HERE,check=True)
        if not a.execute:
            dump(a.out/'status.json',dict(state='PLANNED',jobs=len(jobs)))
            print('PLAN PASS:',len(jobs),'serial jobs. Use --execute with a NEW output directory.');return
        shutil.copytree(HERE,a.out/'code',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
        overlays,libraries=freeze(a.out)
        dump(a.out/'code-hashes.json',{str(p.relative_to(a.out)):sha(p) for p in (a.out/'code').rglob('*.py')})
        checked=False
        for i,j in enumerate(jobs):
            if not checked and j['kind']=='micro':
                compare_operator_checks(a.out);checked=True
            stage(a.out,j,overlays,libraries)
        subprocess.run([sys.executable,str(a.out/'code/summarize.py'),str(a.out)],check=True)
        dump(a.out/'status.json',dict(state='COMPLETED_REVIEW_REQUIRED',jobs=len(jobs),
             note='Read summary and model-logits comparison; completion is not automatic model-quality approval.'))
    except BaseException as e:
        dump(a.out/'status.json',dict(state='INTERRUPTED' if isinstance(e,KeyboardInterrupt) else 'FAILED',error=repr(e)))
        raise

if __name__=='__main__':main()
