"""
TinyGPT: decoder only, trained to predict the next token.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from tinychat.base import AttentionHead, Block, Embedding


class TinyGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = Embedding(config)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size)

    def forward(self, idx, targets=None, start_pos=0):
        x = self.embedding(idx, start_pos) # (B,T,C)
        x = self.blocks(x)      # (B,T,C)
        x = self.ln_f(x)        # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            # ignore_index=-100 lets later stages (SFT) skip the loss on prompt tokens
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T), ignore_index=-100)
        return logits, loss

    def set_kv_cache(self, enabled):
        # turn the KV cache of every attention head on (empty) or off
        for m in self.modules():
            if isinstance(m, AttentionHead):
                m.reset_cache(enabled)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, stop_token=None, use_cache=True):
        # idx is (B, T) array of indices in the current context
        # stop_token (e.g. <eos>): stop early once every row has produced it
        # use_cache: with a KV cache the prompt is read once, then every step only runs the newest token.
        #   Learned positions stop at block_size, so the cache is only used when everything fits;
        #   otherwise fall back to the sliding window that recomputes the last block_size tokens each step
        if use_cache and idx.shape[1] + max_new_tokens <= self.config.block_size:
            return self._generate_cached(idx, max_new_tokens, temperature, stop_token)
        done = torch.zeros(idx.shape[0], 1, dtype=torch.bool, device=idx.device) # (B, 1) rows that hit stop_token
        for _ in range(max_new_tokens):
            # only look at the last block_size tokens
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            idx_next = self._sample(logits[:, -1, :], temperature) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            if stop_token is not None:
                done = done | (idx_next == stop_token) # (B, 1)
                if done.all():
                    break
        return idx

    def _generate_cached(self, idx, max_new_tokens, temperature, stop_token):
        done = torch.zeros(idx.shape[0], 1, dtype=torch.bool, device=idx.device)
        self.set_kv_cache(True)
        try:
            logits, _ = self(idx) # prefill: read the whole prompt once, filling the cache
            for i in range(max_new_tokens):
                if i > 0:
                    # only the newest token goes through the model, at its position in the sequence
                    logits, _ = self(idx[:, -1:], start_pos=idx.shape[1] - 1)
                idx_next = self._sample(logits[:, -1, :], temperature) # (B, 1)
                idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
                if stop_token is not None:
                    done = done | (idx_next == stop_token)
                    if done.all():
                        break
        finally:
            self.set_kv_cache(False) # always clear it, so training forward passes are not affected
        return idx

    @staticmethod
    def _sample(logits, temperature):
        # scale by the temperature, softmax, and draw one token per row
        probs = F.softmax(logits / temperature, dim=-1) # (B, vocab_size)
        return torch.multinomial(probs, num_samples=1)  # (B, 1)
