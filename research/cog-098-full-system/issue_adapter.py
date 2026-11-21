"""Bridge the learned action selector to PR 74's native local issue transaction."""
import secrets
from pathlib import Path

from agent import FullCOGAgent  # also resolves pinned upstream package paths
from boundary import Authority, Ledger, OPERATIONS
from plans import Planner, checked, layout


class IssueWorkflow:
    """Session-scoped capabilities are issued externally, never inferred from language."""
    def __init__(self, path, text, permitted, explicit=False):
        key = secrets.token_bytes(32)
        self.ledger = Ledger(Path(path), key)
        self.authority = Authority(key)
        self.permitted = permitted
        self.explicit = explicit
        self.text = text
        self.planner = Planner('cache')
        self.proposals = {}
        self.status = None

    def public(self):
        self.proposals = {
            f'{op} ticket {ticket}': self.ledger.propose(secrets.token_hex(12), ticket, op)
            for op in OPERATIONS for ticket in range(4)
        }
        menu = tuple(self.proposals)
        return self.text, menu, [0.] * len(menu)

    def step(self, action):
        proposal = self.proposals.pop(action)
        self.proposals.clear()
        world = layout(proposal.ticket, OPERATIONS.index(proposal.op), proposal.epoch)
        path, _, _ = self.planner.solve(world)
        if not checked(world,path):
            raise RuntimeError('invalid workflow path')
        scope = (proposal.ticket, proposal.op) in self.permitted
        token = self.authority.issue(proposal,scope,self.explicit)
        self.status = self.ledger.commit(proposal,token)
        return self.status=='committed', True
