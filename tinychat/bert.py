"""
TinyBERT: encoder only, trained with masked language modeling (predict the hidden tokens).
"""
import torch.nn as nn
from torch.nn import functional as F

from tinychat.base import EncoderBlock, Embedding


class MLMHead(nn.Module):
    """BERT's masked language model head: a small transform before predicting the tokens"""
    def __init__(self, emb_dim, out_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(), # BERT uses GELU, a smooth version of ReLU
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, out_size), # one score per token
        )

    def forward(self, x):
        return self.net(x)


class TinyBERT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = Embedding(config)
        self.blocks = nn.Sequential(*[EncoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.emb_dim)
        self.mlm_head = MLMHead(config.emb_dim, config.vocab_size) # Linear -> GELU -> LayerNorm -> Linear

    def forward(self, idx, targets=None):
        x = self.embedding(idx) # (B,T,C)
        x = self.blocks(x)      # (B,T,C)
        x = self.ln_f(x)        # (B,T,C)
        logits = self.mlm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            # positions with target -100 (not masked) are skipped
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T), ignore_index=-100)
        return logits, loss
