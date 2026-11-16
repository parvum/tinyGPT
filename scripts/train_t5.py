"""
Pretrain TinyT5 on tiny shakespeare (span corruption).

Run from the project root:
    python -m scripts.train_t5
    python -m scripts.train_t5 --max_iters 500   # quick test
"""
import argparse

import torch

from tinychat.base import ModelConfig
from tinychat.t5 import TinyT5
from tinychat.data import load_shakespeare, CharTokenizer, train_val_split, get_batch, corrupt_spans
from tinychat.common import get_device, estimate_loss, save_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_iters', type=int, default=10000)
parser.add_argument('--eval_interval', type=int, default=100)
parser.add_argument('--eval_iters', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--n_spans', type=int, default=2)  # spans cut out of each sequence
parser.add_argument('--span_len', type=int, default=3) # tokens in each span
parser.add_argument('--out', type=str, default='out/t5.pt')
args = parser.parse_args()

torch.manual_seed(1337)
device = get_device()

# data, with extra tokens for the sentinels <X0>, <X1>, ... and the decoder start token <S>
text = load_shakespeare()
sentinels = [f'<X{i}>' for i in range(args.n_spans)]
tokenizer = CharTokenizer.from_text(text, special_tokens=sentinels + ['<S>'])
sentinel_tokens = [tokenizer.token_id(s) for s in sentinels]
start_token = tokenizer.token_id('<S>')
tgt_len = args.n_spans * (1 + args.span_len) # sentinel + span, for every span
train_data, val_data = train_val_split(tokenizer.encode(text))

# model
config = ModelConfig(vocab_size=tokenizer.vocab_size)
model = TinyT5(config).to(device)
print(config)
print(sum(p.numel() for p in model.parameters()), 'parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

def batch(split):
    data = train_data if split == 'train' else val_data
    x, _ = get_batch(data, args.batch_size, config.block_size, device)
    return corrupt_spans(x, sentinel_tokens, start_token, args.span_len) # (src, tgt_in, tgt)

# training loop
for iter in range(args.max_iters):

    if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
        losses = estimate_loss(model, batch, args.eval_iters)
        print(f"step {iter}: train t5 loss {losses['train']:.4f}, val t5 loss {losses['val']:.4f}")

    src, tgt_in, tgt = batch('train')
    logits, loss = model(src, tgt_in, tgt)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

save_checkpoint(args.out, model, config, tokenizer)

# fill in the missing spans of a few validation sequences
model.eval()
src, tgt_in, tgt = batch('val')
pred = model.generate(src[:3], start_token, tgt_len)
for i in range(3):
    print('input :', repr(tokenizer.decode(src[i].tolist())))
    print('target:', repr(tokenizer.decode(tgt[i].tolist())))
    print('output:', repr(tokenizer.decode(pred[i].tolist())))
    print()
