
import json, statistics, time
import torch
import operator_support as op
def measured(a):
    cs=[op.Case(op.spec(a.tokens),a.mode,seed=123+i) for i in range(a.unroll)]
    def reset():
        for c in cs:c.reset()
    def run():
        for c in cs:c.run()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):reset();run()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True,external=True)
    end=torch.cuda.Event(enable_timing=True,external=True)
    g=torch.cuda.CUDAGraph()
    # Resets are graph nodes before start, excluded from the timed interval.
    # Independent input per invocation; every replay restores original inputs.
    with torch.cuda.graph(g):
        reset();start.record();run();end.record()
    def sample():
        g.replay();torch.cuda.synchronize()
        return start.elapsed_time(end)*1000/a.unroll
    warm=[];deadline=time.perf_counter()+3.0
    while time.perf_counter()<deadline:warm.append(sample())
    values=[sample() for _ in range(a.samples)]
    assert all(x>0 for x in values)
    for c in (cs[0],cs[-1]):c.check()
    op.dump(a.out,dict(mode=a.mode,tokens=a.tokens,unroll=a.unroll,samples_us=values,
        median_us=statistics.median(values),std_us=statistics.stdev(values),
        warmup_last_us=warm[-30:],qkv_stride=list(cs[0].x.stride()),
        kv_stride=list(cs[0].kc.stride()),
        note='Graph-internal timing events; captured reset outside timed interval; 3s time-based warmup. Cache state differs from legacy harness; compare A/B/C only within this method.'))
