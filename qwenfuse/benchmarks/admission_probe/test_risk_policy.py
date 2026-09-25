from unittest.mock import Mock
import pytest
from tests.v1.core.test_scheduler import _setup_cache_aware
from vllm.v1.request import RequestStatus
from risk_policy_scheduler import RiskPolicyScheduler


def setup(blocks=5, window=4):
    s,w,c=_setup_cache_aware(num_tokens=48,num_blocks=blocks,block_size=16,
        cache_aware_admission_window=window,cache_aware_admission_threshold=0.5,
        max_num_active_seqs=1)
    s.__class__=RiskPolicyScheduler
    s.enable_risk_policy()
    s.records=[]
    s._record=lambda **row:s.records.append(row)
    s.add_request(c);s.add_request(w)
    return s,w,c


def test_risk_reorders_and_recomputes_hit():
    s,w,c=setup()
    assert s.kv_cache_manager.usage < 0.5
    original=s.kv_cache_manager.allocate_slots
    s.kv_cache_manager.allocate_slots=Mock(wraps=original)
    out=s.schedule()
    assert [r.req_id for r in out.scheduled_new_reqs]==['warm']
    assert out.scheduled_new_reqs[0].num_computed_tokens==32
    calls=s.kv_cache_manager.allocate_slots.call_args_list
    assert len(calls)==1 and calls[0].args[0] is w
    assert calls[0].kwargs['num_new_computed_tokens']==32
    assert len([r for r in s.records if r['event']=='risk_reorder'])==1
    assert s.cache_aware_threshold==0.5


def test_no_risk_keeps_fcfs_without_policy_probes():
    s,w,c=setup(100)
    s.kv_cache_manager.get_num_cached_tokens=Mock(wraps=s.kv_cache_manager.get_num_cached_tokens)
    out=s.schedule()
    assert [r.req_id for r in out.scheduled_new_reqs]==['cold']
    assert not [r for r in s.records if r['event']=='risk_reorder']
    # Diagnostic snapshots may call the readonly lookup; no policy ranking occurred.
    assert not [r for r in s.records if r['event']=='reorder']


def test_streaming_slot_full_skips_risk():
    s,w,c=setup()
    s.num_waiting_for_streaming_input=1
    out=s.schedule()
    assert not out.scheduled_new_reqs
    assert not [r for r in s.records if r['event']=='risk_decision']
    s.num_waiting_for_streaming_input=0
    assert s.schedule().scheduled_new_reqs[0].req_id=='warm'


def test_disabled_keeps_fcfs():
    s,w,c=setup(window=0)
    assert s.schedule().scheduled_new_reqs[0].req_id=='cold'
    assert not [r for r in s.records if r['event']=='risk_decision']


def test_unchanged_ranking_terminates():
    s,w,c=setup()
    # Simulate a window whose ranking does not change, despite risk.
    s._reorder_waiting_by_cached_prefix=Mock()
    out=s.schedule()
    assert out.scheduled_new_reqs[0].req_id=='cold'
    assert s._reorder_waiting_by_cached_prefix.call_count==1


def test_threshold_restored_on_exception():
    s,w,c=setup()
    s._reorder_waiting_by_cached_prefix=Mock(side_effect=RuntimeError('injected'))
    with pytest.raises(RuntimeError,match='injected'):
        s.schedule()
    assert s.cache_aware_threshold==0.5
    assert list(s.waiting)==[c,w]
    assert c.status==RequestStatus.WAITING


def test_future_chunk_risk_reorders_before_first_allocation():
    s,w,c=setup(blocks=6)
    s.max_num_scheduled_tokens=32
    out=s.schedule()
    decision=next(r for r in s.records if r['event']=='risk_decision')
    assert decision['current_new_blocks']==2
    assert decision['full_prompt_new_blocks']==3
    assert decision['current_risk']['reason']=='no_exposure_in_snapshot'
    assert decision['full_risk']['reason']=='cached_block_exposed'
    assert out.scheduled_new_reqs[0].req_id=='warm'
    assert out.scheduled_new_reqs[0].num_computed_tokens==32
