"""Diagnostic logging only; not for performance measurement."""
import json
import os
from pathlib import Path
from vllm.v1.core.sched.scheduler import Scheduler

class ProbeScheduler(Scheduler):
    def _record(self, row):
        with Path(os.environ['ADMISSION_EVENTS']).open('a') as f:
            f.write(json.dumps(row) + '\n')

    def _reorder_waiting_by_cached_prefix(self):
        before = [r.request_id for r in self.waiting]
        usage = self.kv_cache_manager.usage
        super()._reorder_waiting_by_cached_prefix()
        after = [r.request_id for r in self.waiting]
        if len(before) >= 2:
            self._record(dict(event='reorder', before=before, after=after,
                              changed=before != after, usage=usage))

    def schedule(self, throttle_prefills: bool = False):
        waiting = [r.request_id for r in self.waiting]
        usage = self.kv_cache_manager.usage
        out = super().schedule(throttle_prefills=throttle_prefills)
        admitted = [dict(id=r.req_id, cached_at_admission=r.num_computed_tokens)
                    for r in out.scheduled_new_reqs]
        preempted = list(out.preempted_req_ids or [])
        if waiting or admitted or preempted:
            self._record(dict(event='schedule', waiting=waiting, usage=usage,
                              admitted=admitted, preempted=preempted))
        return out
