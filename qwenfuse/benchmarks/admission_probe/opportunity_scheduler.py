"""Observe admission guards without skipping or changing the upstream policy."""
from threshold_scheduler import ThresholdScheduler

class OpportunityScheduler(ThresholdScheduler):
    def _reorder_waiting_by_cached_prefix(self):
        ctx = self._admission_diag_context.copy()
        reasons = []
        if ctx['token_budget'] <= 0:
            reasons.append('token_budget_exhausted')
        if ctx['input_budget'] <= ctx['draft_slots']:
            reasons.append('input_budget_exhausted')
        if ctx['occupied_slots'] >= ctx['max_active']:
            reasons.append('active_slots_full')
        self._record(event='admission_opportunity', context=ctx,
                     blocked_by=reasons, necessary_guards_pass=not reasons)
        # Always execute the original code, even if these guards fail.
        return super()._reorder_waiting_by_cached_prefix()

    def schedule(self, throttle_prefills: bool = False):
        out = super().schedule(throttle_prefills=throttle_prefills)
        self._record(event='admission_outcome',
                     newly_admitted=[r.req_id for r in out.scheduled_new_reqs],
                     scheduled_tokens=out.total_num_scheduled_tokens)
        return out
