"""
Small helpers shared by the training scripts.
"""
import math
import os
from dataclasses import asdict

import torch


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


@torch.no_grad()
def estimate_loss(model, get_batch, eval_iters):
    # average loss over eval_iters batches, get_batch(split) returns the model's inputs as a tuple
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            _, loss = model(*get_batch(split))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def bits_per_byte(loss, n_tokens, n_bytes):
    # the loss is nats per token, so it depends on the tokenizer: a BPE token holds ~4 characters and is
    # harder to guess than one character. Bits per byte of text is the same unit for every tokenizer:
    #   total nats = loss * n_tokens, / ln 2 -> bits, / n_bytes -> bits per byte
    return loss * n_tokens / (math.log(2) * n_bytes)


def save_checkpoint(path, model, config, tokenizer):
    # everything needed to rebuild the model and tokenizer later (e.g. for SFT)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'config': asdict(config),
        'tokenizer': tokenizer.state(), # chars for CharTokenizer, merges for BPETokenizer
    }, path)
    print(f'saved checkpoint to {path}')


def load_checkpoint(path, model_cls, device):
    # rebuild (model, config, tokenizer) from a file written by save_checkpoint
    from tinychat.base import ModelConfig
    from tinychat.tokenizer import tokenizer_from_state, CharTokenizer
    ckpt = torch.load(path, map_location=device)
    config = ModelConfig(**ckpt['config'])
    if 'tokenizer' in ckpt:
        tokenizer = tokenizer_from_state(ckpt['tokenizer'])
    else: # older checkpoints, from before BPE
        tokenizer = CharTokenizer(ckpt['chars'], ckpt['special_tokens'])
    model = model_cls(config)
    model.load_state_dict(ckpt['model'])
    return model.to(device), config, tokenizer
