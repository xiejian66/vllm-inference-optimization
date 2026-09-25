import ast
from pathlib import Path

def patch_scheduler(path):
    path=Path(path)
    s=path.read_text()
    edits=[(
        '                self.cache_aware_window\n                and not defer_prefills',
        '                self.cache_aware_window\n'
        '                and not getattr(self, "_qwenfuse_risk_enabled", False)\n'
        '                and not defer_prefills'),(
        '            step_skipped_waiting = create_request_queue(self.policy)\n',
        '            risk_reorder_checked = False\n'
        '            step_skipped_waiting = create_request_queue(self.policy)\n'),(
        '                new_blocks = self.kv_cache_manager.allocate_slots(\n'
        '                    request,\n                    num_tokens_past_hit,',
        '''                if (
                    getattr(self, "_qwenfuse_risk_enabled", False)
                    and not risk_reorder_checked
                    and request_queue is self.waiting
                    and request.status == RequestStatus.WAITING
                    and request.num_computed_tokens == 0
                    and len(self.waiting) >= 2
                ):
                    # One ranking attempt per admission position, including retry.
                    risk_reorder_checked = True
                    if self._qwenfuse_risk_reorder(
                        request, num_tokens_past_hit,
                        num_new_local_computed_tokens, new_computed_blocks,
                    ):
                        # No blocks allocated yet: recompute the new head's inputs.
                        continue

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_tokens_past_hit,'''),(
        '                request = request_queue.pop_request()\n                if load_kv_async:',
        '                request = request_queue.pop_request()\n'
        '                risk_reorder_checked = False\n                if load_kv_async:')]
    for old,new in edits:
        if s.count(old)!=1:
            raise RuntimeError(f'Patch anchor count {s.count(old)}: {old[:90]}')
        s=s.replace(old,new)
    ast.parse(s)
    path.write_text(s)
