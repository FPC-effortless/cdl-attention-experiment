"""Explicit nanoCOG decoder adapters; these are engineering candidates, not trained models."""
from torch import nn
from .comm import PacketCommunication
from .state.records import StableRecordMemory


class PacketMixer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.packets = PacketCommunication(width)

    def forward(self, x, state=None):
        return self.packets(x, state, return_state=True)


class PacketAndTypedProbe(nn.Module):
    """Separate broker and stable-board path, for inspecting causality before training.

    The board receives the current normalized input directly, not a broker-selected
    payload. This combines outputs only; it does not learn to share or consume a
    common record identity across the packet and typed lanes.
    """
    def __init__(self, width: int):
        super().__init__()
        self.packets = PacketCommunication(width)
        self.records = StableRecordMemory(width)

    def forward(self, x, state=None):
        pst, rst = (None, None) if state is None else state
        packet_delta, pst = self.packets(x, pst, return_state=True)
        record_delta, rst = self.records(x, rst, return_state=True)
        return packet_delta + record_delta, (pst, rst)
