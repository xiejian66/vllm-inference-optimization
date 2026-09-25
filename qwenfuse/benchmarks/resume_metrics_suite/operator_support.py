"""A4/C4 operator correctness, graph replay, regression and microbench workers."""
import argparse, hashlib, json, statistics, time
from pathlib import Path
import torch
from runtime_config import ROOT, REPOS

def dump(p,x):
    Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False))
def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def load(mode):
    repo=REPOS[mode]
    lib=repo/'vllm/_C_stable_libtorch.abi3.so'
    torch.ops.load_library(str(lib))
    return lib

def spec(t=8,**kw):
    s=dict(t=t,nq=32,nk=8,h=128,rot=128,dtype='bfloat16',cache_dtype='bfloat16',
           neox=True,hpw=4,slot='random',pos='random',layout='actual',write=True)
    s.update(kw);return s

def cases(quick=False):
    result=[]
    for t in ([1,8,17,32] if quick else [1,2,5,8,17,32,64,128,256]):
        for sl in (['random','short'] if quick else ['sequential','random','partial','invalid','short','empty']):
            result.append(spec(t,slot=sl,pos=['zero','max','repeated','random'][len(result)%4]))
    if not quick:
        for dtype in ['bfloat16','float16']:
            for h in [64,128,256]:
                for neox in [True,False]:
                    for hpw in [2,4,8]:
                        result.append(spec(5,nq=7,nk=3,h=h,rot=h,dtype=dtype,
                                           cache_dtype='float32',neox=neox,hpw=hpw,
                                           slot='partial',layout='head_major'))
        for write in [True,False]:
            for neox in [True,False]:
                for hpw in [1,2,4,8]:
                    result.append(spec(17,rot=64,neox=neox,hpw=hpw,write=write))
    return result

class Case:
    def __init__(self,s,mode,seed=42,large=False):
        self.s=s;self.mode=mode
        torch.manual_seed(seed);t,nq,nk,h=(s[k] for k in ('t','nq','nk','h'))
        dtype=getattr(torch,s['dtype']);cd=getattr(torch,s['cache_dtype']);dev='cuda'
        self.seed=torch.randn(t,(nq+2*nk)*h,device=dev,dtype=dtype)
        self.x=self.seed.clone();self.q=torch.empty(t,nq,h,device=dev,dtype=dtype)
        self.k=torch.empty(t,nk,h,device=dev,dtype=dtype)
        self.qw=(1+.1*torch.randn(h,device=dev)).to(dtype)
        self.kw=(1+.1*torch.randn(h,device=dev)).to(dtype)
        r=s['rot'];inv=10000.**(-torch.arange(0,r,2,device=dev).float()/r)
        angle=torch.arange(4096,device=dev).float()[:,None]*inv
        self.cs=torch.cat((angle.cos(),angle.sin()),-1).to(cd)
        self.pos=torch.randint(0,4096,(t,),device=dev)
        if s['pos']!='random':self.pos.fill_({'zero':0,'max':4095,'repeated':131}[s['pos']])
        blocks=max((t+15)//16+2,1024 if large else 0)
        shape=(blocks,16,nk,2*h) if s['layout']=='actual' else (blocks,nk,16,2*h)
        self.initial=torch.randn(shape,device=dev,dtype=dtype)
        self.kv=self.initial.clone()
        view=self.kv if s['layout']=='actual' else self.kv.transpose(1,2)
        self.kc,self.vc=view.split(h,-1)
        n=0 if s['slot']=='empty' else max(0,t-3) if s['slot']=='short' else t
        sl=torch.randperm(blocks*16,device=dev)[:n]
        if s['slot']=='sequential':sl=torch.arange(n,device=dev)
        if s['slot']=='partial':sl[::2]=-1
        if s['slot']=='invalid':sl.fill_(-1)
        self.slots=sl;self.one=torch.ones((),device=dev)
        self.qr,self.kr,self.vr=(v.reshape(t,heads,h) for v,heads in zip(
            self.x.split([nq*h,nk*h,nk*h],-1),[nq,nk,nk]))
    def reset(self):self.x.copy_(self.seed);self.kv.copy_(self.initial)
    def run(self):
        s=self.s;nq,nk,h=(s[k] for k in ('nq','nk','h'))
        if self.mode=='A4' or not s['write']:
            torch.ops._C.fused_qk_norm_rope(self.x,nq,nk,nk,h,1e-6,self.qw,self.kw,
                self.cs,s['neox'],self.pos,s['hpw'])
            if s['write'] and self.slots.numel():
                n=self.slots.numel()
                torch.ops._C_cache_ops.reshape_and_cache_flash(self.kr[:n],self.vr[:n],self.kc,self.vc,
                    self.slots,'auto',self.one,self.one)
        else:
            torch.ops._C.fused_qk_norm_rope_kvcache(self.x,self.q,self.k,nq,nk,nk,h,1e-6,
                self.qw,self.kw,self.cs,s['neox'],self.pos,self.kc,self.vc,self.slots,s['hpw'])
    def outputs(self):
        q,k=(self.qr,self.kr) if self.mode=='A4' or not self.s['write'] else (self.q,self.k)
        return q,k,self.kv
    def check(self):
        s=self.s;q,k,kv=self.outputs()
        assert torch.isfinite(q).all() and torch.isfinite(k).all()
        if self.mode=='C4' and s['write']: assert torch.equal(self.x,self.seed),'C changed QKV'
        # Independent FP32 expression: normalization and RoPE, only final output cast.
        # Differential A/C is strict; tolerance here allows FP32 reduction/FMA variation.
        nq,nk,h=(s[a] for a in ('nq','nk','h'));r=s['rot']
        def ref(x,w):
            z=x.float();z=z*(torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-6)*w.float())
            co,si=self.cs[self.pos].float().chunk(2,-1);co=co[:,None,:];si=si[:,None,:]
            out=z.clone()
            if s['neox']:
                u,v=z[...,:r//2],z[...,r//2:r]
                out[...,:r//2]=u*co-v*si;out[...,r//2:r]=v*co+u*si
            else:
                u,v=z[...,:r:2],z[...,1:r:2]
                out[...,:r:2]=u*co-v*si;out[...,1:r:2]=v*co+u*si
            return out.to(x.dtype)
        qr=ref(self.seed[:,:nq*h].reshape(s['t'],nq,h),self.qw)
        kr=ref(self.seed[:,nq*h:(nq+nk)*h].reshape(s['t'],nk,h),self.kw)
        # ~2 BF16 ULP at magnitude 1; absolute term protects near-zero cancellation.
        tol=.016 if s['dtype']=='bfloat16' else .002
        torch.testing.assert_close(q,qr,rtol=tol,atol=tol)
        torch.testing.assert_close(k,kr,rtol=tol,atol=tol)
        expected=self.initial.clone();ev=expected if s['layout']=='actual' else expected.transpose(1,2)
        ek,evv=ev.split(h,-1)
        if s['write']:
            valid=self.slots>=0;sl=self.slots[valid];ids=torch.arange(self.slots.numel(),device='cuda')[valid]
            ek[sl//16,sl%16]=k[ids]
            evv[sl//16,sl%16]=self.seed[:,(nq+nk)*h:].reshape(s['t'],nk,h)[ids]
        assert torch.equal(kv,expected),'Unexpected KV writes (includes untouched locations)'
        return dict(Q=digest(q),K=digest(k),KV=digest(kv),
            ref_q_max_abs=float((q.float()-qr.float()).abs().max()),
            ref_k_max_abs=float((k.float()-kr.float()).abs().max()))

def correctness(a):
    rows=[]
    for i,s in enumerate(cases(a.quick)):
        c=Case(s,a.mode,seed=42+i);c.run();torch.cuda.synchronize()
        check=c.check()
        rows.append(dict(case=i,spec=s,execution='eager',result=check))
        # Fixed regression: retain real values for the previously failing FP16 case.
        if not a.quick and i==76:
            torch.save(dict(spec=s,seed=42+i,input=c.seed.cpu(),
                       **{name:t.cpu().clone() for name,t in zip(['Q','K','KV'],c.outputs())}),
                       str(a.out)+'.case76.pt')
        if i < (8 if a.quick else 54):
            c.reset();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream): c.run()
            torch.cuda.current_stream().wait_stream(stream);c.reset();torch.cuda.synchronize()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):c.run()
            for rep in range(3):
                c.seed.normal_();c.pos.copy_(torch.randint(0,4096,c.pos.shape,device='cuda'))
                if rep==1:c.slots[::2]=-1
                if rep==2:c.slots.fill_(-1)
                c.reset();g.replay();torch.cuda.synchronize()
                rows.append(dict(case=i,spec=s,execution=f'graph{rep}',result=c.check()))
            del g
        del c
        if i%10==0: print('CHECK',a.mode,i,flush=True)
    dump(a.out,rows)

def bench(a):
    s=spec(a.tokens);count=a.unroll
    # Each invocation has independent inputs; cache size sensitivity is explicit.
    cs=[Case(s,a.mode,seed=123+i,large=a.large) for i in range(count)]
    def reset():
        for c in cs:c.reset()
    def run():
        for c in cs:c.run()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):reset();run()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    g=None
    if a.execution=='graph':
        reset();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):run()
        execute=g.replay
    else:execute=run
    for _ in range(20):reset();execute()
    torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    values=[]
    for _ in range(a.samples):
        reset();start.record();execute();end.record();end.synchronize()
        values.append(start.elapsed_time(end)*1000/count)
    dump(a.out,dict(mode=a.mode,tokens=a.tokens,execution=a.execution,unroll=count,
        large=a.large,samples_us=values,median_us=statistics.median(values),std_us=statistics.stdev(values),
        spec=s,qkv_stride=list(cs[0].x.stride()),kv_stride=list(cs[0].kc.stride()),
        cache_bytes_per_call=cs[0].kv.numel()*cs[0].kv.element_size(),
        note='GPU event path time; eager includes host submission gaps. Reset outside timing influences cache state.'))

def trace(a):
    from torch.profiler import profile,ProfilerActivity
    c=Case(spec(32),a.mode);c.run();c.reset();torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:
        c.run();torch.cuda.synchronize()
    p.export_chrome_trace(str(a.out)+'.trace.json')
    names=[e.name for e in p.events() if e.device_type==torch.autograd.DeviceType.CUDA]
    import re
    assert any('NTokenHeads' in n and re.search(r',\s*4\s*[,>]', n) for n in names),names
    dump(a.out,dict(kernels=names))

def main():
    p=argparse.ArgumentParser();p.add_argument('task',choices=['correctness','bench','trace','probe'])
    p.add_argument('--mode',choices=['A4','C4'],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--quick',action='store_true');p.add_argument('--tokens',type=int,default=8)
    p.add_argument('--unroll',type=int,default=128);p.add_argument('--samples',type=int,default=50)
    p.add_argument('--large',action='store_true');p.add_argument('--execution',default='graph',choices=['graph','eager'])
    a=p.parse_args();lib=load(a.mode)
    if a.task=='correctness':correctness(a)
    elif a.task=='bench':bench(a)
    elif a.task=='trace':trace(a)
    else:
        c=Case(spec(32),a.mode);c.run();torch.cuda.synchronize();dump(a.out,c.check())
    print('PASS',a.task,a.mode,lib,flush=True)
if __name__=='__main__':main()
