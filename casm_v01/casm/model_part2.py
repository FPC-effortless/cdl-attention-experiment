        super().__init__()
        self.cfg = cfg
        self.proposals = nn.ModuleList([nn.Linear(cfg.d_model, cfg.memory_dim) for _ in range(cfg.state_slots)])
        self.gates = nn.ModuleList([nn.Linear(cfg.d_model + cfg.memory_dim, cfg.memory_dim) for _ in range(cfg.state_slots)])

    def update(self, state: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(self.cfg.state_slots):
            prev = state[:, i]
            proposal = torch.tanh(self.proposals[i](summary))
            gate = torch.sigmoid(self.gates[i](torch.cat([summary, prev], dim=-1)))
            outs.append(gate * prev + (1.0 - gate) * proposal)
        return torch.stack(outs, dim=1)


class CASM(nn.Module):
    def __init__(self, cfg: CASMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.memory_in = nn.Linear(cfg.d_model, cfg.memory_dim, bias=False)
        self.memory_context = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.router = CompressionRouter(cfg)
        self.state = PersistentState(cfg)
        self.memory_gate = nn.Linear(2 * cfg.d_model, cfg.d_model)
        self.memory_ffn_norm = RMSNorm(cfg.d_model)
        self.memory_ffn = SwiGLU(cfg)
        self.mtp_heads = nn.ModuleList([nn.Linear(cfg.d_model, cfg.vocab_size, bias=False) for _ in range(cfg.mtp_horizons - 1)])
        self.verify = nn.Sequential(nn.Linear(2 * cfg.d_model, cfg.d_model), nn.SiLU(), nn.Linear(cfg.d_model, 1))
        self.apply(self._init_weights)
        nn.init.zeros_(self.memory_gate.weight)
        nn.init.constant_(self.memory_gate.bias, -3.0)
        nn.init.normal_(self.router.score_mlp[-1].weight, mean=0.0, std=1e-4)
        nn.init.normal_(self.router.future_delta[-1].weight, mean=0.0, std=1e-4)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _init_memory(self, batch: int, device: torch.device, dtype: torch.dtype):
        ring = torch.zeros(batch, self.cfg.memory_slots, self.cfg.memory_dim, device=device, dtype=dtype)
        ring_valid = torch.zeros(batch, self.cfg.memory_slots, device=device, dtype=torch.bool)
        state = torch.zeros(batch, self.cfg.state_slots, self.cfg.memory_dim, device=device, dtype=dtype)
        return ring, ring_valid, state

    def _candidates(self, ring, valid, state):
        state_valid = torch.ones(state.shape[:2], device=state.device, dtype=torch.bool)
        return torch.cat([state, ring], dim=1), torch.cat([state_valid, valid], dim=1)

    def forward(self, tokens: torch.Tensor, return_aux: bool = True, external_teacher: Optional[torch.Tensor] = None, teacher_alpha: float = 0.0, target_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        b, t = tokens.shape
        if t < 2:
            raise ValueError("Need at least two tokens")
        x_in = tokens[:, :-1]
        targets = tokens[:, 1:]
        total_len = x_in.shape[1]
        ring, ring_valid, state = self._init_memory(b, tokens.device, self.embed.weight.dtype)
        logits_chunks=[]; target_chunks=[]; compression_losses=[]; compression_predictor_losses=[]; mtp_losses=[]; verifier_losses=[]; router_entropies=[]; gain_means=[]; gain_stds=[]; memory_gate_means=[]

        for chunk_idx, start in enumerate(range(0, total_len, self.cfg.chunk_size)):
            end = min(start + self.cfg.chunk_size, total_len)
            ids = x_in[:, start:end]
            y = targets[:, start:end]
            h = self.embed(ids)
            for block in self.blocks:
                h = block(h)
            h = self.norm(h)
            past_candidates, past_valid = self._candidates(ring, ring_valid, state)
            if self.cfg.use_memory:
                retrieved, _ = self.router.retrieve(h, past_candidates, past_valid)
                gate = torch.sigmoid(self.memory_gate(torch.cat([h, retrieved], dim=-1)))
                h = h + gate * retrieved
                memory_gate_means.append(gate.mean())
            h = h + self.memory_ffn(self.memory_ffn_norm(h))
            logits = self.lm_head(h)
            logits_chunks.append(logits); target_chunks.append(y)

            if return_aux and self.cfg.mtp_horizons > 1:
                for horizon, head in enumerate(self.mtp_heads, start=2):
                    valid_len = h.shape[1] - (horizon - 1)
                    if valid_len > 0:
                        pred = head(h[:, :valid_len])
                        tgt_start = start + horizon
                        tgt_end = min(start + h.shape[1] + 1, tokens.shape[1])
                        tgt = tokens[:, tgt_start:tgt_end]
                        if tgt.shape[1] == valid_len and (tgt != 256).any():
                            mtp_losses.append(F.cross_entropy(pred.transpose(1, 2), tgt, ignore_index=256))

            valid_tok = ids != 256
            last_idx = valid_tok.long().sum(dim=1).clamp_min(1) - 1
            summary = h[torch.arange(b, device=h.device), last_idx]
            state = self.state.update(state, summary)
            new_mem = torch.tanh(self.memory_in(summary))
            ring = torch.cat([ring[:, 1:], new_mem[:, None, :]], dim=1)
            ring_valid = torch.cat([ring_valid[:, 1:], torch.ones(b, 1, device=tokens.device, dtype=torch.bool)], dim=1)
            candidates, cand_valid = self._candidates(ring, ring_valid, state)

            if self.cfg.use_memory:
                scores = self.router.scores(summary, candidates)
                weights = F.softmax(scores.masked_fill(~cand_valid, -1e9), dim=-1)
                router_entropies.append((-(weights * weights.clamp_min(1e-8).log()).sum(dim=-1)).mean())
            else:
                scores = torch.zeros(b, candidates.shape[1], device=tokens.device)

            future_start = end
            future_end = min(future_start + self.cfg.compression_future_tokens, tokens.shape[1])
            need_aux = self.cfg.compression_loss_weight > 0 or self.cfg.compression_predictor_loss_weight > 0 or (external_teacher is not None and teacher_alpha > 0)
            if return_aux and self.cfg.use_memory and need_aux and future_end > future_start:
                future = tokens[:, future_start:future_end]
                target_dist, gain, predictor_loss = self.router.compression_target(summary, candidates, future, cand_valid)
                finite_gain = gain.masked_fill(~cand_valid, 0.0)
                valid_count = cand_valid.sum(dim=-1).clamp_min(1).to(gain.dtype)
                gain_mean_per = finite_gain.sum(dim=-1) / valid_count
                centered = (gain - gain_mean_per[:, None]).masked_fill(~cand_valid, 0.0)
                gain_std = torch.sqrt((centered.pow(2).sum(dim=-1) / valid_count).clamp_min(1e-12))
                gain_stds.append(gain_std.mean())
                if external_teacher is not None and teacher_alpha > 0 and chunk_idx < external_teacher.shape[1]:
                    ext = external_teacher[:, chunk_idx, :].to(target_dist.device, target_dist.dtype)
                    confidence = (gain_std.detach() / 0.08).clamp(0.0, 1.0)
                    ext_alpha = torch.maximum(torch.full_like(confidence, float(teacher_alpha)), 1.0 - confidence)
                    target_dist = ext_alpha[:, None] * ext + (1.0 - ext_alpha[:, None]) * target_dist
                    target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                pred_log = F.log_softmax(scores.masked_fill(~cand_valid, -1e9), dim=-1)
                compression_losses.append(-(target_dist * pred_log).sum(dim=-1).mean())
                compression_predictor_losses.append(predictor_loss)
                gain_means.append(finite_gain.sum() / cand_valid.sum().clamp_min(1))

            if return_aux and end < total_len:
                nxt_end = min(end + self.cfg.chunk_size, total_len)
                next_ids = x_in[:, end:nxt_end]
                if next_ids.shape[1] > 0:
                    next_emb = self.embed(next_ids).mean(dim=1)
                    pos = self.verify(torch.cat([summary, next_emb], dim=-1)).squeeze(-1)
                    if b > 1:
                        neg_emb = torch.roll(next_emb, shifts=1, dims=0)
                        neg = self.verify(torch.cat([summary, neg_emb], dim=-1)).squeeze(-1)
                        verifier_losses.append(0.5 * (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))))

        logits_all = torch.cat(logits_chunks, dim=1)
        targets_all = torch.cat(target_chunks, dim=1)
        token_nll = F.cross_entropy(logits_all.transpose(1, 2), targets_all, ignore_index=256, reduction="none")
        valid_targets = targets_all != 256
        weights = valid_targets.to(token_nll.dtype) if target_weights is None else target_weights.to(token_nll.dtype) * valid_targets.to(token_nll.dtype)
        if target_weights is not None and target_weights.shape != targets_all.shape:
            raise ValueError(f"target_weights {target_weights.shape} must match targets {targets_all.shape}")
        lm_loss = (token_nll * weights).sum() / weights.sum().clamp_min(1.0)
        zero = lm_loss.new_zeros(())
        compression_loss = torch.stack(compression_losses).mean() if compression_losses else zero
        compression_predictor_loss = torch.stack(compression_predictor_losses).mean() if compression_predictor_losses else zero
        mtp_loss = torch.stack(mtp_losses).mean() if mtp_losses else zero
        verifier_loss = torch.stack(verifier_losses).mean() if verifier_losses else zero
        total = lm_loss + self.cfg.compression_loss_weight * compression_loss + self.cfg.compression_predictor_loss_weight * compression_predictor_loss + self.cfg.mtp_loss_weight * mtp_loss + self.cfg.verifier_loss_weight * verifier_loss
        return {
            "loss": total,
            "lm_loss": lm_loss,
            "compression_loss": compression_loss,
            "compression_predictor_loss": compression_predictor_loss,
            "mtp_loss": mtp_loss,
            "verifier_loss": verifier_loss,
            "router_entropy": torch.stack(router_entropies).mean() if router_entropies else zero,
            "mean_compression_gain": torch.stack(gain_means).mean() if gain_means else zero,
            "compression_gain_std": torch.stack(gain_stds).mean() if gain_stds else zero,
            "memory_gate_mean": torch.stack(memory_gate_means).mean() if memory_gate_means else zero,
            "logits": logits_all,
        }
