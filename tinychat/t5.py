"""
TinyT5: encoder + decoder, trained with span corruption (fill in the spans cut out of the input).
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from tinychat.base import EncoderBlock, DecoderBlock, Embedding


class TinyT5(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # separate position tables for the encoder and decoder, but one shared token table, like T5
        self.enc_embedding = Embedding(config)
        self.dec_embedding = Embedding(config)
        self.dec_embedding.tok_embedding_table = self.enc_embedding.tok_embedding_table
        # encoder: unmasked blocks, then a layer norm
        self.encoder = nn.Sequential(*[EncoderBlock(config) for _ in range(config.n_layer)])
        self.ln_enc = nn.LayerNorm(config.emb_dim)
        # decoder: blocks take two inputs (x, enc_out) so nn.Sequential does not work, use a ModuleList
        self.decoder = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size)

    def run_encoder(self, src):
        x = self.enc_embedding(src) # (B,S,C)
        x = self.encoder(x)         # (B,S,C)
        return self.ln_enc(x)

    def run_decoder(self, tgt_in, enc_out):
        x = self.dec_embedding(tgt_in) # (B,T,C)
        for block in self.decoder:
            x = block(x, enc_out) # (B,T,C)
        x = self.ln_f(x)
        return self.lm_head(x) # (B,T,vocab_size)

    def forward(self, src, tgt_in, targets=None):
        enc_out = self.run_encoder(src)
        logits = self.run_decoder(tgt_in, enc_out)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, src, start_token, max_new_tokens):
        # run the encoder once, then decode one token at a time (greedy, always the most likely token)
        enc_out = self.run_encoder(src)
        idx = torch.full((src.shape[0], 1), start_token, dtype=torch.long, device=src.device) # (B,1)
        for _ in range(max_new_tokens):
            logits = self.run_decoder(idx[:, -self.config.block_size:], enc_out) # (B,T,vocab)
            idx_next = logits[:, -1, :].argmax(dim=-1, keepdim=True) # (B,1)
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
        return idx[:, 1:] # drop the start token
