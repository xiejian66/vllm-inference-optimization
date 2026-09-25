"""Read-only diagnostics around the unchanged admission policy; not a benchmark."""
import itertools
import json
import os
from pathlib import Path
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

class ThresholdScheduler(Scheduler):
    def _record(self, **row):
        self._diag_seq = getattr(self, '_diag_seq', 0) + 1
        row.update(seq=self._diag_seq, step=getattr(self, '_diag_step', 0))
        with Path(os.environ['ADMISSION_EVENTS']).open('a') as f:
            f.write(json.dumps(row)+'\n')

    def _hot_waiting(self):
        return [r for r in self.waiting
                if r.status == RequestStatus.WAITING and not r.num_computed_tokens
                and r.prompt_token_ids and len(r.prompt_token_ids) == 2176
                and 1000 <= r.prompt_token_ids[0] < 1008]

    def _snapshot(self, requests):
        return {r.request_id: min(2048, self.kv_cache_manager.get_num_cached_tokens(r))
                for r in requests}

    def _reorder_waiting_by_cached_prefix(self):
        before = [r.request_id for r in self.waiting]
        usage = self.kv_cache_manager.usage
        head = list(itertools.islice(self.waiting, self.cache_aware_window))
        eligible = [r for r in head if r.status == RequestStatus.WAITING
                    and not r.num_computed_tokens]
        if len(before) < 2:
            reason = 'fewer_than_two_waiting'
        elif usage < self.cache_aware_threshold:
            reason = 'below_threshold'
        elif len(eligible) < 2:
            reason = 'fewer_than_two_candidates'
        else:
            reason = 'lookup'
        hot_before = self._snapshot(self._hot_waiting())
        calls = []
        manager = self.kv_cache_manager
        original = manager.get_num_cached_tokens
        def counted(request):
            hit = original(request)
            calls.append(dict(id=request.request_id, cached=hit))
            return hit
        manager.get_num_cached_tokens = counted
        try:
            super()._reorder_waiting_by_cached_prefix()
        finally:
            manager.get_num_cached_tokens = original
        after = [r.request_id for r in self.waiting]
        if len(before) >= 2:
            self._record(event='reorder', before=before, after=after,
                         changed=before != after, usage=usage,
                         threshold=self.cache_aware_threshold, reason=reason,
                         policy_queries=calls, hot_before=hot_before)

    def schedule(self, throttle_prefills: bool = False):
        self._diag_step = getattr(self, '_diag_step', 0) + 1
        manager = self.kv_cache_manager
        pool = manager.block_pool
        original_alloc = manager.allocate_slots
        original_evict = pool._maybe_evict_cached_block
        evictions = []
        def evict(block):
            result = original_evict(block)
            if result:
                evictions.append(block.block_id)
            return result
        def allocate(*args, **kwargs):
            request = args[0] if args else kwargs['request']
            watchers = self._hot_waiting()
            before = self._snapshot(watchers)
            usage = manager.usage
            free = pool.get_num_free_blocks()
            evict_begin = len(evictions)
            result = original_alloc(*args, **kwargs)
            after = self._snapshot(watchers)
            losses = {rid: before[rid]-after[rid] for rid in before
                      if before[rid] > after[rid]}
            if watchers:
                self._record(event='allocation', request=request.request_id,
                    request_tag=request.prompt_token_ids[0] if request.prompt_token_ids else None,
                    success=result is not None, usage_before=usage,
                    free_blocks_before=free, free_blocks_after=pool.get_num_free_blocks(),
                    hot_before=before, hot_after=after, lost_reusable_prefix_tokens=losses,
                    evicted_block_ids=evictions[evict_begin:])
            return result
        manager.allocate_slots = allocate
        pool._maybe_evict_cached_block = evict
        try:
            output = super().schedule(throttle_prefills=throttle_prefills)
        finally:
            manager.allocate_slots = original_alloc
            pool._maybe_evict_cached_block = original_evict
        admitted = [dict(id=r.req_id, cached_at_admission=r.num_computed_tokens)
                    for r in output.scheduled_new_reqs]
        preempted = list(output.preempted_req_ids or [])
        if admitted or preempted:
            self._record(event='schedule', admitted=admitted, preempted=preempted)
        return output
