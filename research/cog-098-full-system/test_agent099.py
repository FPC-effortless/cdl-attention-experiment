from copy import deepcopy
from pathlib import Path

import pytest
import torch

from agent import FullCOGAgent
from tasks import Dialogue, Navigation, edits


class Echo:
    def __init__(self):
        self.called = []

    def step(self, action):
        self.called.append(action)
        return True, True


def decide(agent, env):
    agent.observe('fresh request', ('move', 'wait'), [1.,0.])
    action = agent.propose()
    return action, agent.act(env)


def test_feedback_changes_weights_and_checkpoint_replays():
    agent = FullCOGAgent()
    before = agent.policy.head[-1].weight.detach().clone()
    decide(agent, Echo())
    agent.learn(True,None,True)
    with pytest.raises(RuntimeError):
        agent.learn(False,None,True)
    agent.finish()
    assert not torch.equal(before, agent.policy.head[-1].weight)
    agent.freeze()
    restored = agent.restore(agent.snapshot())
    assert decide(restored,Echo()) == decide(agent,Echo())
    with pytest.raises(RuntimeError):
        restored.learn(True,None,True)
    assert restored.updates == 1


def test_stale_proposal_cannot_call_adapter():
    agent = FullCOGAgent()
    env = Echo()
    agent.observe('first', ('move',), [1.])
    agent.propose()
    with pytest.raises(RuntimeError):
        agent.observe('second', ('wait',), [1.])
    agent.version += 1
    with pytest.raises(RuntimeError):
        agent.act(env)
    assert not env.called


def test_reset_discards_staged_work_and_frozen_episodes_are_isolated():
    agent = FullCOGAgent()
    decide(agent, Echo())
    agent.learn(True,None,True)
    agent.finish()
    agent.freeze()
    graph = deepcopy(agent.graph)
    first = decide(agent,Echo())
    agent.reset()
    assert len(agent.history) == len(agent.workspace.nodes) == len(agent.visits) == 0
    assert decide(agent,Echo()) == first
    assert agent.graph == graph


def test_planning_uses_future_verified_reward_and_ablation_removes_it():
    agent = FullCOGAgent()
    agent.graph = {'s':{'go':[2,0.,{'next':2}]},'next':{'open':[1,1.,{}]}}
    assert agent.expected('s','go',1) == pytest.approx(.8)
    assert agent.expected('s','go',0) == 0.
    agent.memory = False
    assert agent.expected('s','go',1) == 0.


def test_latent_dialogue_goal_has_no_effect_on_public_menu_or_text():
    def episode(value):
        return Dialogue({'turns':[{'speaker':'USER','utterance':'Can you help?',
                    'frames':[{'service':'hotel','state':{'slot_values':{'hotel-area':[value]}}}]}]})
    env_a,env_b = episode('north'),episode('south')
    assert env_a.public() == env_b.public()
    action = 'hotel area north price unspecified'
    assert env_a.step(action) != env_b.step(action)


def test_repair_enumeration_needs_only_visible_buggy_source():
    code = 'def distance(x):\n return x + 1\n'
    candidates = edits(code)
    assert len(candidates) == len(set(candidates)) > 1
    assert all(compile(s,'edit','exec') for s in candidates.values())


def test_minigrid_observation_and_actions_follow_public_interface():
    env = Navigation('MiniGrid-DoorKey-6x6-v0',2000)
    try:
        text, menu, priors = env.public()
        assert env.obs['mission'] in text and len(menu) == len(priors) == 6
        _,done = env.step(menu[0])
        assert not done and env.steps == 1
    finally:
        env.env.close()


@pytest.mark.parametrize('policy', ['allowed','denied','stale'])
def test_frozen_agent_uses_native_planner_and_signed_ledger(tmp_path,policy):
    from issue_adapter import IssueWorkflow
    from boundary import OPERATIONS
    permitted = {(i, op) for i in range(4) for op in OPERATIONS} if policy!='denied' else set()
    env = IssueWorkflow(tmp_path/'issues.db', 'Please label ticket two', permitted, explicit=True)
    try:
        agent = FullCOGAgent()
        decide(agent, Echo())
        agent.learn(True,None,True)
        agent.finish()
        agent.freeze()
        agent = FullCOGAgent.restore(agent.snapshot())
        agent.observe(*env.public())
        agent.propose()
        if policy=='stale':
            env.ledger.db.execute('UPDATE policy SET epoch=epoch+1')
        ok, done = agent.act(env)
        assert done and ok == (policy=='allowed')
        assert env.status == {'allowed':'committed','denied':'denied','stale':'stale'}[policy]
        assert env.ledger.db.execute('SELECT count(*) FROM journal').fetchone()[0] == int(ok)
    finally:
        env.ledger.db.close()


def test_unassisted_adapters_hide_advice_but_keep_raw_observations():
    env = Navigation('MiniGrid-Empty-Random-6x6-v0',2000)
    try:
        obs, menu, prior = env.public()
        assert any(prior)
        assert env.public(assisted=False) == (obs,menu,[0.]*len(menu))
    finally:
        env.env.close()


def test_adapter_exception_cannot_be_learned_as_an_observed_outcome():
    class Fails:
        def step(self,_):
            raise ValueError('unavailable')
    agent = FullCOGAgent()
    agent.observe('try',('move',),[0.])
    agent.propose()
    with pytest.raises(ValueError):
        agent.act(Fails())
    with pytest.raises(RuntimeError):
        agent.learn(True,None,True)
    assert not agent.workspace.nodes and agent.pending is None
