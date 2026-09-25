"""Observe fresh admissions only. Never change allocation, ordering or threshold."""
import inspect
from dataclasses import asdict
from threshold_scheduler import ThresholdScheduler
from prompt_risk_v1 import inspect_risk, full_prompt_new_blocks
from vllm.v1.request import RequestStatus


def predict_fresh(request, params, queue, block_size):
    if request.status != RequestStatus.WAITING or request.num_computed_tokens != 0:
        return None
    if any(params.get(k, 0) for k in (
        'num_lookahead_tokens', 'num_external_computed_tokens', 'num_encoder_tokens'
    )) or params.get('delay_cache_blocks', False):
        return dict(supported=False, reason='outside_single_group_local_prefill_scope')
    computed = params.get('new_computed_blocks')
    groups = computed.blocks if computed is not None else ([],)
    if len(groups) != 1:
        return dict(supported=False, reason='multiple_cache_groups')
    hits = groups[0]
    if any(b.is_null for b in hits):
        return dict(supported=False, reason='null_hit_block')
    hit_ids = {b.block_id for b in hits}
    local = params.get('num_new_computed_tokens', 0)
    if local % block_size or len(hit_ids) * block_size != local:
        return dict(supported=False, reason='partial_or_nonstandard_prefix_hit')
    step_demand = full_prompt_new_blocks(local + params['num_new_tokens'],
                                        block_size, len(hit_ids))
    full_demand = full_prompt_new_blocks(request.num_prompt_tokens,
                                        block_size, len(hit_ids))
    return dict(supported=True, prompt_tokens=request.num_prompt_tokens,
                cached_tokens=local, current_new_blocks=step_demand,
                full_prompt_new_blocks=full_demand,
                current_risk=asdict(inspect_risk(queue, step_demand, hit_ids)),
                full_risk=asdict(inspect_risk(queue, full_demand, hit_ids)))


class RiskScheduler(ThresholdScheduler):
    def schedule(self, throttle_prefills=False):
        manager = self.kv_cache_manager
        original = manager.allocate_slots
        signature = inspect.signature(original)
        def observe(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            params = bound.arguments
            request = params['request']
            risk = predict_fresh(request, params, manager.block_pool.free_block_queue,
                                 self.block_size)
            if risk is not None:
                self._record(event='admission_risk', request=request.request_id,
                             request_tag=request.prompt_token_ids[0],
                             usage_before=manager.usage, **risk)
            return original(*args, **kwargs)
        manager.allocate_slots = observe
        try:
            return super().schedule(throttle_prefills=throttle_prefills)
        finally:
            manager.allocate_slots = original
