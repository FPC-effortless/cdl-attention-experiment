from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_REGISTERS = 4
VALUE_MODULUS = 16
NUM_OPERATORS = 8
OPERATOR_NAMES = (
    "copy",
    "add",
    "sub",
    "max",
    "min",
    "xor",
    "inc",
    "dec",
)
# Commands are deliberately opaque aliases. The model receives the alias while
# evaluation keeps the semantic operator id private for diagnostics/oracles.
OP_TO_ALIAS = (3, 6, 1, 7, 0, 4, 2, 5)
HELDOUT_BIGRAMS = frozenset((i, (i + 1) % NUM_OPERATORS) for i in range(NUM_OPERATORS))


@dataclass
class ProgramBatch:
    initial: torch.Tensor
    commands: torch.Tensor
    semantics: torch.Tensor
    arg_a: torch.Tensor
    arg_b: torch.Tensor
    dst: torch.Tensor
    target_states: torch.Tensor

    @property
    def depth(self) -> int:
        return int(self.commands.shape[1])

    def to(self, device: torch.device | str) -> "ProgramBatch":
        return ProgramBatch(**{k: v.to(device) for k, v in self.__dict__.items()})


def _apply_python(state: list[int], op: int, a: int, b: int, dst: int) -> list[int]:
    out = list(state)
    va, vb = state[a], state[b]
    if op == 0:
        value = va
    elif op == 1:
        value = (va + vb) % VALUE_MODULUS
    elif op == 2:
        value = (va - vb) % VALUE_MODULUS
    elif op == 3:
        value = max(va, vb)
    elif op == 4:
        value = min(va, vb)
    elif op == 5:
        value = va ^ vb
    elif op == 6:
        value = (va + 1) % VALUE_MODULUS
    elif op == 7:
        value = (va - 1) % VALUE_MODULUS
    else:
        raise ValueError(op)
    out[dst] = value
    return out


def oracle_transition(
    state: torch.Tensor,
    op: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor:
    va = state.gather(1, a[:, None]).squeeze(1)
    vb = state.gather(1, b[:, None]).squeeze(1)
    candidates = torch.stack(
        [
            va,
            torch.remainder(va + vb, VALUE_MODULUS),
            torch.remainder(va - vb, VALUE_MODULUS),
            torch.maximum(va, vb),
            torch.minimum(va, vb),
            torch.bitwise_xor(va, vb),
            torch.remainder(va + 1, VALUE_MODULUS),
            torch.remainder(va - 1, VALUE_MODULUS),
        ],
        dim=1,
    )
    value = candidates.gather(1, op[:, None]).squeeze(1)
    return state.scatter(1, dst[:, None], value[:, None])


def _sample_ops(rng: random.Random, depth: int, split: str) -> list[int]:
    if split not in {"train", "iid", "composition"}:
        raise ValueError(split)
    for _ in range(10000):
        ops = [rng.randrange(NUM_OPERATORS) for _ in range(depth)]
        has_heldout = any((x, y) in HELDOUT_BIGRAMS for x, y in zip(ops, ops[1:]))
        if split in {"train", "iid"} and not has_heldout:
            return ops
        if split == "composition" and depth >= 2 and has_heldout:
            return ops
    raise RuntimeError(f"could not sample {split} operator sequence at depth={depth}")


def make_program_batch(batch_size: int, depth: int, seed: int, split: str = "train") -> ProgramBatch:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if split == "composition" and depth < 2:
        raise ValueError("composition split requires depth >= 2")
    rng = random.Random(seed)
    initials = []
    commands = []
    semantics = []
    aa = []
    bb = []
    dd = []
    targets = []
    for _ in range(batch_size):
        state = [rng.randrange(VALUE_MODULUS) for _ in range(NUM_REGISTERS)]
        initials.append(list(state))
        ops = _sample_ops(rng, depth, split)
        commands.append([OP_TO_ALIAS[o] for o in ops])
        semantics.append(ops)
        row_a, row_b, row_d, row_targets = [], [], [], []
        for op in ops:
            a = rng.randrange(NUM_REGISTERS)
            b = rng.randrange(NUM_REGISTERS)
            dst = rng.randrange(NUM_REGISTERS)
            row_a.append(a)
            row_b.append(b)
            row_d.append(dst)
            state = _apply_python(state, op, a, b, dst)
            row_targets.append(list(state))
        aa.append(row_a)
        bb.append(row_b)
        dd.append(row_d)
        targets.append(row_targets)
    return ProgramBatch(
        initial=torch.tensor(initials, dtype=torch.long),
        commands=torch.tensor(commands, dtype=torch.long),
        semantics=torch.tensor(semantics, dtype=torch.long),
        arg_a=torch.tensor(aa, dtype=torch.long),
        arg_b=torch.tensor(bb, dtype=torch.long),
        dst=torch.tensor(dd, dtype=torch.long),
        target_states=torch.tensor(targets, dtype=torch.long),
    )


class StateCommandEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.value = nn.Embedding(VALUE_MODULUS, d_model)
        self.register = nn.Embedding(NUM_REGISTERS, d_model)
        self.command = nn.Embedding(NUM_OPERATORS, d_model)
        self.state_proj = nn.Sequential(
            nn.Linear(NUM_REGISTERS * d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.command_proj = nn.Sequential(
            nn.Linear(4 * d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        b = state.shape[0]
        regs = torch.arange(NUM_REGISTERS, device=state.device)[None, :].expand(b, -1)
        x = self.value(state) + self.register(regs)
        return self.state_proj(x.flatten(1))

    def encode_command(
        self,
        command: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                self.command(command),
                self.register(a),
                self.register(b),
                self.register(dst),
            ],
            dim=-1,
        )
        return self.command_proj(x)


class ExplicitOperatorMachine(nn.Module):
    """Typed state machine with modular learned operators and a transition verifier."""

    def __init__(self, d_model: int = 64, route_weight: float = 0.25, verifier_weight: float = 0.20):
        super().__init__()
        self.encoder = StateCommandEncoder(d_model)
        self.route_weight = float(route_weight)
        self.verifier_weight = float(verifier_weight)
        self.router = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.SiLU(), nn.Linear(d_model, NUM_OPERATORS)
        )
        op_input = 4 * d_model
        self.operators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(op_input, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, VALUE_MODULUS),
                )
                for _ in range(NUM_OPERATORS)
            ]
        )
        self.semantic = nn.Embedding(NUM_OPERATORS, d_model)
        self.verifier = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _gather_value(state: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return state.gather(1, index[:, None]).squeeze(1)

    def route_logits(self, state, command, a, b, dst):
        s = self.encoder.encode_state(state)
        c = self.encoder.encode_command(command, a, b, dst)
        return self.router(torch.cat([s, c], dim=-1))

    def operator_logits(self, state, a, b, dst):
        va = self.encoder.value(self._gather_value(state, a))
        vb = self.encoder.value(self._gather_value(state, b))
        vd = self.encoder.value(self._gather_value(state, dst))
        rd = self.encoder.register(dst)
        x = torch.cat([va, vb, vd, rd], dim=-1)
        return torch.stack([module(x) for module in self.operators], dim=1)

    def verifier_logits(self, state, command, a, b, dst, semantic, candidate_value):
        s = self.encoder.encode_state(state)
        c = self.encoder.encode_command(command, a, b, dst)
        o = self.semantic(semantic)
        v = self.encoder.value(candidate_value)
        return self.verifier(torch.cat([s, c, o, v], dim=-1)).squeeze(-1)

    def training_loss(self, batch: ProgramBatch) -> Dict[str, torch.Tensor]:
        route_losses = []
        operator_losses = []
        verifier_losses = []
        for t in range(batch.depth):
            state = batch.initial if t == 0 else batch.target_states[:, t - 1]
            command = batch.commands[:, t]
            semantic = batch.semantics[:, t]
            a, b, dst = batch.arg_a[:, t], batch.arg_b[:, t], batch.dst[:, t]
            target_state = batch.target_states[:, t]
            target_value = self._gather_value(target_state, dst)

            route = self.route_logits(state, command, a, b, dst)
            route_losses.append(F.cross_entropy(route, semantic))

            all_logits = self.operator_logits(state, a, b, dst)
            chosen = all_logits[torch.arange(state.shape[0], device=state.device), semantic]
            operator_losses.append(F.cross_entropy(chosen, target_value))

            pos = self.verifier_logits(state, command, a, b, dst, semantic, target_value)
            offset = torch.randint(1, VALUE_MODULUS, target_value.shape, device=state.device)
            corrupt = torch.remainder(target_value + offset, VALUE_MODULUS)
            neg = self.verifier_logits(state, command, a, b, dst, semantic, corrupt)
            verifier_losses.append(
                0.5
                * (
                    F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
                    + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
                )
            )
        operator_loss = torch.stack(operator_losses).mean()
        route_loss = torch.stack(route_losses).mean()
        verifier_loss = torch.stack(verifier_losses).mean()
        total = operator_loss + self.route_weight * route_loss + self.verifier_weight * verifier_loss
        return {
            "loss": total,
            "operator_loss": operator_loss,
            "route_loss": route_loss,
            "verifier_loss": verifier_loss,
        }

    @torch.no_grad()
    def rollout(
        self,
        batch: ProgramBatch,
        *,
        use_verifier: bool = True,
        oracle_routing: bool = False,
        oracle_execution: bool = False,
        verifier_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = batch.initial.clone()
        states = []
        routes = []
        for t in range(batch.depth):
            command = batch.commands[:, t]
            semantic = batch.semantics[:, t]
            a, b, dst = batch.arg_a[:, t], batch.arg_b[:, t], batch.dst[:, t]
            all_logits = self.operator_logits(state, a, b, dst)
            candidate_values = all_logits.argmax(dim=-1)
            if oracle_routing:
                selected = semantic
            else:
                scores = self.route_logits(state, command, a, b, dst)
                if use_verifier:
                    verifier_scores = []
                    for k in range(NUM_OPERATORS):
                        sem = torch.full_like(semantic, k)
                        verifier_scores.append(
                            self.verifier_logits(
                                state,
                                command,
                                a,
                                b,
                                dst,
                                sem,
                                candidate_values[:, k],
                            )
                        )
                    scores = scores + verifier_scale * torch.stack(verifier_scores, dim=1)
                selected = scores.argmax(dim=-1)
            routes.append(selected)
            if oracle_execution:
                state = oracle_transition(state, selected, a, b, dst)
            else:
                value = candidate_values.gather(1, selected[:, None]).squeeze(1)
                state = state.scatter(1, dst[:, None], value[:, None])
            states.append(state.clone())
        return torch.stack(states, dim=1), torch.stack(routes, dim=1)


class SharedTransitionModel(nn.Module):
    """Ablation: same typed recurrent state, one universal transition network."""

    def __init__(self, d_model: int = 96):
        super().__init__()
        self.encoder = StateCommandEncoder(d_model)
        self.transition = nn.Sequential(
            nn.Linear(6 * d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, VALUE_MODULUS),
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def step_logits(self, state, command, a, b, dst):
        s = self.encoder.encode_state(state)
        c = self.encoder.encode_command(command, a, b, dst)
        va = self.encoder.value(state.gather(1, a[:, None]).squeeze(1))
        vb = self.encoder.value(state.gather(1, b[:, None]).squeeze(1))
        vd = self.encoder.value(state.gather(1, dst[:, None]).squeeze(1))
        rd = self.encoder.register(dst)
        return self.transition(torch.cat([s, c, va, vb, vd, rd], dim=-1))

    def training_loss(self, batch: ProgramBatch) -> torch.Tensor:
        losses = []
        for t in range(batch.depth):
            state = batch.initial if t == 0 else batch.target_states[:, t - 1]
            dst = batch.dst[:, t]
            target = batch.target_states[:, t].gather(1, dst[:, None]).squeeze(1)
            logits = self.step_logits(
                state,
                batch.commands[:, t],
                batch.arg_a[:, t],
                batch.arg_b[:, t],
                dst,
            )
            losses.append(F.cross_entropy(logits, target))
        return torch.stack(losses).mean()

    @torch.no_grad()
    def rollout(self, batch: ProgramBatch) -> torch.Tensor:
        state = batch.initial.clone()
        states = []
        for t in range(batch.depth):
            dst = batch.dst[:, t]
            logits = self.step_logits(
                state,
                batch.commands[:, t],
                batch.arg_a[:, t],
                batch.arg_b[:, t],
                dst,
            )
            value = logits.argmax(dim=-1)
            state = state.scatter(1, dst[:, None], value[:, None])
            states.append(state.clone())
        return torch.stack(states, dim=1)


class GRUProgramBaseline(nn.Module):
    """Generic recurrent baseline with no explicit state-transition bottleneck."""

    def __init__(self, d_model: int = 96):
        super().__init__()
        self.encoder = StateCommandEncoder(d_model)
        self.initial = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, d_model)
        )
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.head = nn.Linear(d_model, NUM_REGISTERS * VALUE_MODULUS)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def logits(self, batch: ProgramBatch) -> torch.Tensor:
        h0 = self.initial(self.encoder.encode_state(batch.initial))[None, :, :]
        commands = []
        for t in range(batch.depth):
            commands.append(
                self.encoder.encode_command(
                    batch.commands[:, t],
                    batch.arg_a[:, t],
                    batch.arg_b[:, t],
                    batch.dst[:, t],
                )
            )
        x = torch.stack(commands, dim=1)
        h, _ = self.gru(x, h0)
        return self.head(h).view(
            batch.initial.shape[0], batch.depth, NUM_REGISTERS, VALUE_MODULUS
        )

    def training_loss(self, batch: ProgramBatch) -> torch.Tensor:
        logits = self.logits(batch)
        return F.cross_entropy(
            logits.reshape(-1, VALUE_MODULUS), batch.target_states.reshape(-1)
        )

    @torch.no_grad()
    def rollout(self, batch: ProgramBatch) -> torch.Tensor:
        return self.logits(batch).argmax(dim=-1)


def state_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    exact_steps = (pred == target).all(dim=-1)
    final_exact = exact_steps[:, -1].float().mean().item()
    step_exact = exact_steps.float().mean().item()
    register_accuracy = (pred == target).float().mean().item()
    return {
        "final_state_exact": final_exact,
        "step_state_exact": step_exact,
        "register_accuracy": register_accuracy,
    }


def aggregate_metrics(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
