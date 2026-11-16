"""
Pretrain TinyBERT on tiny shakespeare (masked language modeling).

Run from the project root:
    python -m scripts.train_bert
    python -m scripts.train_bert --max_iters 500   # quick test
"""
import argparse

import torch

from tinychat.base import ModelConfig
from tinychat.bert import TinyBERT
from tinychat.data import load_shakespeare, CharTokenizer, train_val_split, get_batch, mask_tokens
from tinychat.common import get_device, estimate_loss, save_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_iters', type=int, default=10000)
parser.add_argument('--eval_interval', type=int, default=100)
parser.add_argument('--eval_iters', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--mask_prob', type=float, default=0.15)
parser.add_argument('--out', type=str, default='out/bert.pt')
args = parser.parse_args()

torch.manual_seed(1337)
device = get_device()

# data, with one extra token for <MASK>
text = load_shakespeare()
tokenizer = CharTokenizer.from_text(text, special_tokens=['<MASK>'])
mask_token = tokenizer.token_id('<MASK>')
train_data, val_data = train_val_split(tokenizer.encode(text))

# model
config = ModelConfig(vocab_size=tokenizer.vocab_size)
model = TinyBERT(config).to(device)
print(config)
print(sum(p.numel() for p in model.parameters()), 'parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

def batch(split):
    data = train_data if split == 'train' else val_data
    x, _ = get_batch(data, args.batch_size, config.block_size, device) # BERT does not need the shifted y
    return mask_tokens(x, mask_token, tokenizer.n_chars, args.mask_prob) # (masked x, targets)

# training loop
for iter in range(args.max_iters):

    if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
        losses = estimate_loss(model, batch, args.eval_iters)
        print(f"step {iter}: train mlm loss {losses['train']:.4f}, val mlm loss {losses['val']:.4f}")

    xb, yb = batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

save_checkpoint(args.out, model, config, tokenizer)

# fill in the blanks, '_' marks the hidden characters (at most block_size characters)
@torch.no_grad()
def fill_mask(text, mask_char='_'):
    ids = [mask_token if ch == mask_char else tokenizer.stoi[ch] for ch in text]
    logits, _ = model(torch.tensor([ids], dtype=torch.long, device=device)) # (1,T,vocab)
    pred = logits[0].argmax(dim=-1).tolist()
    return ''.join(tokenizer.itos[p] if ch == mask_char else ch for ch, p in zip(text, pred))

model.eval()
for t in ["First Ci_izen:\nBefore we pr_ceed", "What _s your n_me?"]:
    print(repr(t), '->', repr(fill_mask(t)))
