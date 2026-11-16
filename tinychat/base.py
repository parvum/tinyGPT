"""
Shared building blocks for TinyGPT, TinyBERT and TinyT5.

The three models only differ in how attention is masked and how the blocks are wired:
- GPT  (decoder only):   Block         = causal self attention + MoE
- BERT (encoder only):   EncoderBlock  = full (unmasked) self attention + MoE
- T5   (encoder+decoder): EncoderBlock for the encoder,
                          DecoderBlock = causal self attention + cross attention + MoE
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 65    # size of the token table: characters + any special tokens
    block_size: int = 32    # maximum sequence length
    emb_dim: int = 64
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.0
    n_expert: int = 4       # experts in each MoE layer (n_expert=1, top_k=1 is a plain feed forward)
    top_k: int = 2          # each token is routed to its top_k experts


# -----------------------------------------------------------------------------
# Attention

class AttentionHead(nn.Module):
    """Individual causal attention head: each token only sees itself and the tokens before it"""
    causal = True

    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.emb_dim, head_size, bias=False)
        self.query = nn.Linear(config.emb_dim, head_size, bias=False)
        self.value = nn.Linear(config.emb_dim, head_size, bias=False)
        self.dropout_p = config.dropout
        self.reset_cache(enabled=False)

    def reset_cache(self, enabled):
        # KV cache for generation: keep the keys and values of the tokens seen so far, so each new
        # token only computes its own k, v instead of redoing the whole sequence
        self.cache_enabled = enabled
        self.k_cache = self.v_cache = None

    def forward(self, x, context=None):
        # context is where keys and values come from:
        # None -> x itself (self attention), encoder output -> cross attention
        if context is None:
            context = x
        k = self.key(context)   # (B,S,hs) S = context length, hs = head_size
        q = self.query(x)       # (B,T,hs)
        v = self.value(context) # (B,S,hs)
        causal = self.causal
        if self.cache_enabled:
            if self.k_cache is not None:
                # after the prompt, tokens come one at a time: the new token sees all cached tokens,
                # so no mask is needed (there is nothing after it to hide)
                assert x.shape[1] == 1, 'with a filled KV cache, feed one token at a time'
                k = torch.cat([self.k_cache, k], dim=1) # (B,S+1,hs)
                v = torch.cat([self.v_cache, v], dim=1)
                causal = False
            self.k_cache, self.v_cache = k, v
        # this one call does exactly the attention math below, but without storing the full (B,T,S)
        # weight matrix on the GPU (flash / memory efficient attention), which matters for long blocks:
        #   weight = q @ k.transpose(-2, -1) * hs**-0.5                      # (B, T, S) scaled scores
        #   weight = weight.masked_fill(tril[:T, :T] == 0, float('-inf'))   # only if causal
        #   weight = dropout(F.softmax(weight, dim=-1))                      # (B, T, S)
        #   out = weight @ v                                                 # (B, T, hs)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                             dropout_p=self.dropout_p if self.training else 0.0)
        return out # (B, T, hs)


class EncoderAttentionHead(AttentionHead):
    """No causal mask, every token sees the whole sequence (BERT, T5 encoder)"""
    causal = False


class CrossAttentionHead(AttentionHead):
    """Query from the decoder, key and value from the encoder output (pass it as context), no mask"""
    causal = False


class MultiAttentionHead(nn.Module):
    """A simple multi head attention layer, head_cls picks the kind of head"""
    def __init__(self, config, head_cls=AttentionHead):
        super().__init__()
        head_size = config.emb_dim // config.n_head
        self.heads = nn.ModuleList([head_cls(config, head_size) for _ in range(config.n_head)])
        self.projection = nn.Linear(config.emb_dim, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, context=None):
        x = torch.cat([h(x, context) for h in self.heads], dim=-1) # concatenate all heads, (B,T,hs) each -> (B,T,C)
        x = self.dropout(self.projection(x))
        return x


# -----------------------------------------------------------------------------
# Feed forward

class FeedForward(nn.Module):
    """A simple 2 layer MLP, expands to 4 * emb_dim in the middle"""
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.emb_dim, 4 * config.emb_dim),
            nn.ReLU(),
            nn.Linear(4 * config.emb_dim, config.emb_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class MoE(nn.Module):
    """A simple mixture of experts layer, each expert is just a FeedForward"""
    def __init__(self, config):
        super().__init__()
        self.top_k = config.top_k
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_expert)])
        # the router (gate) gives every token a score for every expert
        self.router = nn.Linear(config.emb_dim, config.n_expert)

    def forward(self, x):
        scores = self.router(x) # (B,T,n_expert)
        # keep only the top_k experts for each token, softmax over those k scores only
        topk_scores, topk_idx = scores.topk(self.top_k, dim=-1) # (B,T,k), (B,T,k)
        topk_weights = F.softmax(topk_scores, dim=-1) # (B,T,k)
        out = torch.zeros_like(x) # (B,T,C)
        for i, expert in enumerate(self.experts):
            hit = (topk_idx == i) # (B,T,k), True where expert i was picked
            token_mask = hit.any(dim=-1) # (B,T), tokens that go to expert i
            if not token_mask.any():
                continue
            w = (topk_weights * hit).sum(dim=-1) # (B,T), weight of expert i for each token (0 if not picked)
            # only run expert i on the tokens routed to it, (N,C) where N = number of those tokens
            out[token_mask] += w[token_mask].unsqueeze(-1) * expert(x[token_mask])
        return out


# -----------------------------------------------------------------------------
# Transformer blocks

class Block(nn.Module):
    """GPT transformer block: causal self attention + MoE, each with a layer norm and a residual"""
    head_cls = AttentionHead

    def __init__(self, config):
        super().__init__()
        self.mha = MultiAttentionHead(config, self.head_cls)
        self.ff = MoE(config)
        self.ln1 = nn.LayerNorm(config.emb_dim)
        self.ln2 = nn.LayerNorm(config.emb_dim)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class EncoderBlock(Block):
    """Same as Block but the self attention is not masked"""
    head_cls = EncoderAttentionHead


class DecoderBlock(Block):
    """Block with a cross attention layer between the self attention and the MoE"""
    def __init__(self, config):
        super().__init__(config) # gives self.mha (causal), self.ff (MoE), self.ln1, self.ln2
        self.cross = MultiAttentionHead(config, CrossAttentionHead)
        self.ln_cross = nn.LayerNorm(config.emb_dim)

    def forward(self, x, enc_out):
        x = x + self.mha(self.ln1(x))                 # look at the earlier decoder tokens
        x = x + self.cross(self.ln_cross(x), enc_out) # look at the encoder output
        x = x + self.ff(self.ln2(x))                  # MoE feed forward
        return x


# -----------------------------------------------------------------------------
# Embedding

class Embedding(nn.Module):
    """Token embedding + learned position embedding, (B,T) ids -> (B,T,C)"""
    def __init__(self, config):
        super().__init__()
        self.tok_embedding_table = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_embedding_table = nn.Embedding(config.block_size, config.emb_dim)

    def forward(self, idx, start_pos=0):
        # start_pos: position of the first token in idx (> 0 when generating with a KV cache)
        B,T = idx.shape
        tok_emb = self.tok_embedding_table(idx) # (B,T,C)
        pos_emb = self.pos_embedding_table(torch.arange(start_pos, start_pos + T, device=idx.device)) # (T,C), broadcast over B
        return tok_emb + pos_emb
