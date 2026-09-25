import argparse
import json
from pathlib import Path
import torch
import operator_support as op
from common import MODEL, dump, gpu, sha

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--job',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); j=json.loads(a.job.read_text()); a.out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((MODEL/'config.json').read_text())
    assert (cfg['num_attention_heads'],cfg['num_key_value_heads'],cfg['head_dim'])==(32,8,128)
    torch.ops.load_library(str(a.library))
    mode=j['variant']; before=gpu()
    # Scope: Qwen3 full NeoX, head_dim=128; FP16 is compatibility, BF16 performance.
    def cases(quick=False):
        return [op.spec(t,slot=slot,dtype=dtype,cache_dtype=dtype,
                        pos=('zero','max','repeated','random')[i%4])
                for dtype in ('bfloat16','float16')
                for i,t in enumerate((1,2,5,8,17,32,64,128,256))
                for slot in ('sequential','random','partial','invalid','short','empty')]
    op.cases=cases
    # Correctness / bench functions use Case symbol dynamically.
    Base=op.Case
    class Case(Base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            assert cfg['rms_norm_eps']==1e-6
            r=self.s['rot']; theta=cfg.get('rope_theta',1000000.)
            inv=float(theta)**(-torch.arange(0,r,2,device='cuda').float()/r)
            angle=torch.arange(4096,device='cuda').float()[:,None]*inv
            self.cs=torch.cat((angle.cos(),angle.sin()),-1).to(getattr(torch,self.s['cache_dtype']))
            # Match the model layout previously captured on this pinned baseline.
            assert self.x.stride()==((32+2*8)*128,1)
            assert self.kc.stride()==(32768,2048,256,1)
        def check(self):
            if self.mode!='A4' and self.s['write']:
                assert torch.equal(self.x,self.seed),'Fusion changed QKV input'
            return super().check()
    op.Case=Case
    ns=argparse.Namespace(mode=mode,out=a.out/'result.json',quick=False,
        tokens=j.get('tokens',32),unroll=j.get('unroll',128),samples=j.get('samples',30),
        large=False,execution='graph')
    if j['kind']=='correctness':op.correctness(ns)
    elif j['kind']=='trace':
        op.trace(ns)
        names=json.loads(ns.out.read_text())['kernels']
        target=[x for x in names if 'fusedQKNormRopeKernelNTokenHeads' in x]
        import re
        if mode=='A4':
            assert any('reshape_and_cache_flash_kernel' in x for x in names)
            assert not any(re.search(r',\s*4,\s*true\s*>',x) for x in target)
        else:
            assert any(re.search(r',\s*4,\s*true\s*>',x) for x in target)
            assert not any('reshape_and_cache_flash_kernel' in x for x in names)
    else:
        from stable_micro import measured
        measured(ns)
    dump(a.out/'metadata.json',dict(library=str(a.library),sha256=sha(a.library),
         torch=torch.__version__,cuda=torch.version.cuda,gpu_before=before,gpu_after=gpu(),
         rope_theta=cfg.get('rope_theta'),heads_per_warp=4,
         quality_scope='Full NeoX Qwen3 geometry; not all upstream supported shapes'))
    print('PASS',j['id'],flush=True)

if __name__=='__main__':main()
