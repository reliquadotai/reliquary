"""Final authority admission is shared by the gradient gate and accounting."""
from __future__ import annotations

import time

import pytest
import torch

from reliquary.trainer.async_trainer import DrawGate, training_group
from reliquary.validator import training, verifier
from reliquary.validator.training import TrainingAdmissionError
from tests.unit.test_rl_async_trainer import (
    Book, _Model, _GUMBEL, grads_of, loop, submitted, ROLLOUTS, N_TOK,
)
from reliquary.validator.rl_buffer import TrainingBuffer
from tests.unit.test_rl_submit import book as book
from tests.unit.test_rl_submit import small_cap as small_cap
from tests.unit.test_rl_submit import service, lease, group, text


@pytest.fixture(autouse=True)
def training_state(monkeypatch):
    monkeypatch.setattr(training, 'KL_BETA', 0.0)
    monkeypatch.setattr('reliquary.constants.TRAINING_LOSS', 'substitution')
    monkeypatch.setattr(verifier, 'draw_settings', lambda: _GUMBEL)
    training.reset_training_state()
    yield
    training.reset_training_state()


def receipt(lease_id, accepted=True):
    return {'lease_id': lease_id, 'admission_id': 'receipt-' + lease_id,
            'accepted': accepted, 'expires_at': time.time() + 3600}


def test_authority_rejection_backs_out_its_gradient_before_commit():
    model = _Model()
    good = training_group(submitted(model, 'good'), slot0=0, index=0)
    rejected = training_group(submitted(model, 'rejected'), slot0=ROLLOUTS, index=1)
    scale = 1.0 / (2 * ROLLOUTS * N_TOK)
    plan = [(good, [1.0,-1.0,1.0,-1.0], scale), (rejected, [1.0,-1.0,1.0,-1.0], scale)]
    gate = DrawGate([[0,1,2,3],[4,5,6,7]], lease_ids=['good','rejected'], require_admission=True,
                    admission_check=lambda gate: [receipt('good'),receipt('rejected', False)])
    actual = grads_of(model, plan, gate)
    expected = grads_of(model, plan[:1], DrawGate([[0,1,2,3]]))
    assert gate.accepted == [True, False] and gate.undone == 1
    assert all(torch.allclose(a,b,atol=3e-4) for a,b in zip(actual,expected))
    gate.check_commit()


@pytest.mark.parametrize('answer', ['error','expired','wrong_lease','wrong_type','missing_id'])
def test_unavailable_or_malformed_authority_never_updates_weights(answer):
    model, state = _Model(), Book()
    buffer = TrainingBuffer(state)
    buffer.push(submitted(model, 'g'))
    trainer, subs, publisher = loop(model, state, buffer, batch=1)
    before = [p.detach().clone() for p in model.parameters()]
    def admit(lease_id, substitutions):
        if answer == 'error':
            raise RuntimeError('authority unavailable')
        out = receipt(lease_id)
        if answer == 'expired': out['expires_at'] = 0
        if answer == 'wrong_lease': out['lease_id'] = 'another'
        if answer == 'wrong_type': out['accepted'] = 'true'
        if answer == 'missing_id': out.pop('admission_id')
        return out
    subs.admit_training = admit
    try:
        with pytest.raises(TrainingAdmissionError): trainer.run_once()
        assert all(torch.equal(a,b) for a,b in zip(before,model.parameters()))
        assert buffer.depth() == {'math': 1} and subs.reports == {} and publisher.versions == []
    finally: trainer.close()


def test_all_rejected_admission_skips_optimizer_and_reports_final_set():
    model, state = _Model(), Book()
    buffer = TrainingBuffer(state)
    buffer.push(submitted(model, 'g'))
    trainer, subs, publisher = loop(model, state, buffer, batch=1)
    subs.admit_training = lambda lease_id, substitutions: receipt(lease_id, False)
    before = [p.detach().clone() for p in model.parameters()]
    try:
        assert trainer.run_once() == 'skipped'
        assert all(torch.equal(a,b) for a,b in zip(before,model.parameters()))
        assert subs.reports['g']['accepted'] is False
        assert subs.reports['g']['admission_id'] == 'receipt-g'
        assert trainer.last_step['async/groups_accepted'] == 0 and publisher.versions == []
    finally: trainer.close()


def test_commit_rechecks_receipt_expiry():
    gate = DrawGate([[0]], lease_ids=['g'], require_admission=True,
                    admission_check=lambda gate: [receipt('g')])
    gate.accepted = [True]
    gate._finalize_admission(None)
    gate.admission_expires_at = time.time() - 1
    with pytest.raises(TrainingAdmissionError, match='expired'): gate.check_commit()


def test_reward_decision_is_finalized_once_and_bound_to_owner(book, monkeypatch):
    from reliquary.validator.rl_trainer_owner import TrainerOwner
    owner_epoch = TrainerOwner(book).claim('a' * 32)['epoch']
    subs = service(book, probation_groups=0)
    held = lease(book, 90)
    subs.submit(held.lease_id, 'A', group(8))
    unchanged = [[] for _ in range(16)]
    original = subs._rewards_hold
    calls = []
    def grade(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(subs, '_rewards_hold', grade)
    accepted = subs.admit_training(held.lease_id, unchanged, owner_epoch=owner_epoch)
    retried = subs.admit_training(held.lease_id, unchanged, owner_epoch=owner_epoch)
    assert accepted['admission_id'] == retried['admission_id'] and len(calls) == 1
    with pytest.raises(ValueError, match='owner'):
        subs.report_training(held.lease_id, rollout_match=[1.0]*16, accepted=True,
                             admission_id=accepted['admission_id'], owner_epoch=owner_epoch + 1)
    monkeypatch.setattr(subs, '_rewards_hold', lambda *args: pytest.fail('post-update regrade'))
    subs.report_training(held.lease_id, rollout_match=[1.0]*16, accepted=True,
                         admission_id=accepted['admission_id'], owner_epoch=owner_epoch)
    assert subs.events()[-1]['detail']['admission_id'] == accepted['admission_id']


def test_changed_reward_cannot_be_reported_as_accepted(book):
    subs = service(book, probation_groups=0)
    held = lease(book, 91)
    subs.submit(held.lease_id, 'A', group(8))
    changed = [[] for _ in range(16)]
    changed[8] = [(7,text('4')[0])]
    final = subs.admit_training(held.lease_id, changed)
    assert final['accepted'] is False
    with pytest.raises(ValueError, match='final verdict'):
        subs.report_training(held.lease_id, rollout_match=[1.0]*16, accepted=True,
                             admission_id=final['admission_id'])
    subs.report_training(held.lease_id, rollout_match=[1.0]*16, accepted=False,
                         admission_id=final['admission_id'])
    assert subs.events()[-1]['detail']['reward_moved'] is True


@pytest.mark.parametrize('pair', [(True,4),(-1,4),(999,4),(1,-1),(1,True)])
def test_admission_validates_substitution_positions_and_tokens(book,pair):
    subs = service(book)
    held = lease(book, 92)
    subs.submit(held.lease_id, 'A', group(8))
    substitutions = [[pair]] + [[] for _ in range(15)]
    with pytest.raises(ValueError): subs.admit_training(held.lease_id,substitutions)
    assert not subs.events() and subs.buffer.get(held.lease_id) is not None


def _distributed_admission(fail):
    import pickle
    from reliquary.trainer.distributed import init_distributed
    ctx = init_distributed(device_type='cpu')
    called = []
    def authority(gate):
        called.append(ctx.rank)
        if fail: raise RuntimeError('authority offline')
        return [receipt('a'),receipt('b',False)]
    gate = DrawGate([[0],[1]], lease_ids=['a','b'], require_admission=True, admission_check=authority)
    if not ctx.is_main: gate = pickle.loads(pickle.dumps(gate))
    gate.tallies[ctx.rank] = [100,100,100,100]
    gate.forwarded[ctx.rank] = gate.admitted[ctx.rank] = 1
    try:
        gate.judge([],ctx)
        gate.check_commit(ctx)
        return gate.accepted, called
    except TrainingAdmissionError:
        return 'aborted',called


@pytest.mark.parametrize('fail',[False,True])
def test_all_ranks_receive_final_admission_or_abort(fail):
    from tests.unit.trainer_dist_helpers import run_ranks
    results = run_ranks(_distributed_admission,2,fail,timeout=60)
    expected = 'aborted' if fail else [True,False]
    assert results == {0:(expected,[0]),1:(expected,[])}


@pytest.mark.parametrize('expire_during_grade', [False, True])
def test_admission_rechecks_live_owner_when_committing_receipt(book, monkeypatch, expire_during_grade):
    from reliquary.validator.rl_trainer_owner import TrainerOwner, TrainerOwnerRefused
    subs = service(book)
    held = lease(book, 93)
    subs.submit(held.lease_id, 'A', group(8))
    owner = TrainerOwner(book)
    claim = owner.claim('a' * 32)
    initial_time = book.clock()
    unchanged = [[] for _ in range(16)]
    if not expire_during_grade:
        subs.admit_training(held.lease_id, unchanged, owner_epoch=claim['epoch'])
        book.clock = lambda: initial_time + 181
    else:
        def slow_grade(*args):
            book.clock = lambda: initial_time + 181
            return True
        monkeypatch.setattr(subs, '_rewards_hold', slow_grade)
    with pytest.raises(TrainerOwnerRefused):
        subs.admit_training(held.lease_id, unchanged, owner_epoch=claim['epoch'])
    assert subs.buffer.get(held.lease_id) is not None and not subs.events()


def test_legacy_accepted_detached_report_requires_precommit_receipt(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from reliquary.trainer.rl_link import TrainerLink, TrainerLinkError
    from reliquary.validator.rl_async_service import build_rl_trainer_router
    from reliquary.validator.rl_leases import LeaseRequest
    from tests.unit.test_rl_async_service import components, ENV, TOKEN
    rl = components(tmp_path)
    app = FastAPI()
    app.include_router(build_rl_trainer_router(rl, token=TOKEN))
    link = TrainerLink('http://testserver', TOKEN, client=TestClient(app), patience_s=0)
    rl.book.adopt_checkpoint(0)
    held = rl.book.grant([LeaseRequest('A', ENV, (3,), 0)])[0].leases[0]
    rl.submissions.submit(held.lease_id, 'A', group(8))
    link.take(1)
    try:
        with pytest.raises(TrainerLinkError) as error:
            link.report_training(held.lease_id, rollout_match=[1.0]*16, accepted=True)
        assert error.value.status == 422
        assert rl.buffer.get(held.lease_id) is not None and not rl.submissions.events()
    finally: rl.close()


def test_accepted_report_cannot_replace_precommitted_substitutions(book):
    subs = service(book)
    held = lease(book, 94)
    subs.submit(held.lease_id, 'A', group(8))
    unchanged = [[] for _ in range(16)]
    decision = subs.admit_training(held.lease_id, unchanged)
    other = [[(7, text('4')[0])]] + unchanged[1:]
    with pytest.raises(ValueError, match='substitutions changed'):
        subs.report_training(held.lease_id, rollout_match=[1.0]*16, accepted=True,
                             admission_id=decision['admission_id'], substitutions=other)
    assert subs.buffer.get(held.lease_id) is not None and not subs.events()


def test_lost_admission_reply_reuses_the_same_authority_decision(tmp_path, monkeypatch):
    import httpx
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from reliquary.trainer.rl_link import TrainerLink
    from reliquary.validator.rl_async_service import build_rl_trainer_router
    from reliquary.validator.rl_leases import LeaseRequest
    from tests.unit.test_rl_async_service import components, ENV, TOKEN
    rl = components(tmp_path)
    app = FastAPI()
    app.include_router(build_rl_trainer_router(rl, token=TOKEN))
    class Flaky:
        def __init__(self):
            self.client, self.lost, self.calls = TestClient(app), False, 0
        def request(self, method, url, **kwargs):
            response = self.client.request(method, url, **kwargs)
            if url.endswith('/admit'):
                self.calls += 1
                if not self.lost:
                    self.lost = True
                    raise httpx.ReadTimeout('lost admission reply')
            return response
    rl.book.adopt_checkpoint(0)
    held = rl.book.grant([LeaseRequest('A', ENV, (3,), 0)])[0].leases[0]
    rl.submissions.submit(held.lease_id, 'A', group(8))
    flaky = Flaky()
    link = TrainerLink('http://testserver', TOKEN, client=flaky, sleep=lambda s: None)
    calls = []
    monkeypatch.setattr(rl.submissions, '_rewards_hold', lambda *args: calls.append(1) or True)
    try:
        link.take(1)
        decision = link.admit_training(held.lease_id, [[] for _ in range(16)])
        assert decision['accepted'] is True and flaky.calls == 2 and calls == [1]
        assert rl.buffer.get(held.lease_id) is not None and not rl.submissions.events()
    finally: rl.close()


@pytest.mark.parametrize('takeover', [False, True])
def test_slow_reward_admission_allows_heartbeat_and_rejects_stale_owner(tmp_path, monkeypatch, takeover):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from reliquary.validator.rl_async_service import build_rl_trainer_router
    from reliquary.validator.rl_leases import LeaseRequest
    from reliquary.validator.rl_trainer_owner import OWNER_HEADER, EPOCH_HEADER
    from tests.unit.test_rl_async_service import components, ENV, TOKEN

    rl = components(tmp_path)
    now = [time.time()]
    rl.book.clock = lambda: now[0]
    rl.book.adopt_checkpoint(0)
    held = rl.book.grant([LeaseRequest('A', ENV, (3,), 0)])[0].leases[0]
    rl.submissions.submit(held.lease_id, 'A', group(8))
    rl.buffer.take(1)
    app = FastAPI()
    app.include_router(build_rl_trainer_router(rl, token=TOKEN))
    control, worker = TestClient(app), TestClient(app)
    auth = {'Authorization': f'Bearer {TOKEN}'}
    first = control.post('/rl/trainer/owner/claim', headers=auth, json={'owner': 'a' * 32})
    assert first.status_code == 200
    epoch = first.json()['epoch']
    headers = {**auth, OWNER_HEADER: 'a' * 32, EPOCH_HEADER: str(epoch)}
    substitutions = [[(7, text('4')[0])]] + [[] for _ in range(15)]
    entered, finish = threading.Event(), threading.Event()
    original = rl.submissions._rewards_hold

    def slow_grade(*args):
        entered.set()
        assert finish.wait(timeout=10), 'grading was not released'
        return original(*args)

    monkeypatch.setattr(rl.submissions, '_rewards_hold', slow_grade)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = pool.submit(worker.post, '/rl/trainer/admit', headers=headers,
                                  json={'lease_id': held.lease_id, 'substitutions': substitutions})
            try:
                assert entered.wait(timeout=5)
                now[0] += 181 if takeover else 150
                if takeover:
                    action = pool.submit(control.post, '/rl/trainer/owner/claim', headers=auth,
                                         json={'owner': 'b' * 32})
                else:
                    action = pool.submit(control.post, '/rl/trainer/owner/heartbeat', headers=headers, json={})
                renewed = action.result(timeout=3)
                assert renewed.status_code == 200
                if not takeover:
                    now[0] += 50  # past the first lease deadline, within the successful heartbeat extension
            finally:
                finish.set()
            response = pending.result(timeout=5)
        assert response.status_code == (409 if takeover else 200)
        assert rl.buffer.get(held.lease_id) is not None and not rl.submissions.events()
        rows = rl.book.db.execute('SELECT owner_epoch FROM rl_training_admissions').fetchall()
        assert rows == ([] if takeover else [(epoch,)])
        if takeover:
            replacement = {**auth, OWNER_HEADER: 'b' * 32, EPOCH_HEADER: str(renewed.json()['epoch'])}
            retried = control.post('/rl/trainer/admit', headers=replacement,
                                   json={'lease_id': held.lease_id, 'substitutions': substitutions})
            assert retried.status_code == 200 and retried.json()['accepted'] is True
    finally:
        finish.set()
        control.close()
        worker.close()
        rl.close()
