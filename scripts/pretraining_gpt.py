"""
Pretrain TinyGPT on TinyStories (next token prediction).

Run from the project root:
    python -m scripts.tok_train                                         # first: train the BPE tokenizer
    python -m scripts.pretraining_gpt                                   # default: first 100MB of TinyStories
    python -m scripts.pretraining_gpt --optimizer muon                  # Muon + AdamW, compare the val bpb
    python -m scripts.pretraining_gpt --tokenizer char --block_size 1024   # the old character level model
    python -m scripts.pretraining_gpt --max_mb 5 --max_iters 200 --emb_dim 64 --n_layer 2 --batch_size 4   # quick test
"""
import argparse
import os
import time

import torch

from tinychat.base import ModelConfig
from tinychat.gpt import TinyGPT
from tinychat.data import load_tinystories, get_batch, ASCII_CHARS, CHAT_SPECIAL_TOKENS
from tinychat.tokenizer import CharTokenizer, encode_documents, load_tokenizer
from tinychat.muon import setup_optimizers
from tinychat.common import get_device, estimate_loss, save_checkpoint, bits_per_byte

parser = argparse.ArgumentParser()
# data
parser.add_argument('--tokenizer', type=str, default='out/tokenizer.pt', help="BPE file from scripts.tok_train, or 'char'")
parser.add_argument('--max_mb', type=int, default=100, help='MB of the TinyStories train file to use (full file is ~2GB)')
# model
parser.add_argument('--block_size', type=int, default=512) # a story is ~750 chars = ~200 BPE tokens, SFT needs prompt + story
parser.add_argument('--emb_dim', type=int, default=256)
parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--n_layer', type=int, default=6)
parser.add_argument('--n_expert', type=int, default=4)
parser.add_argument('--top_k', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.0)
# training
parser.add_argument('--batch_size', type=int, default=16) # 16 x 512 = 8k tokens (~32k characters) per step
parser.add_argument('--max_iters', type=int, default=5000)
parser.add_argument('--eval_interval', type=int, default=250)
parser.add_argument('--eval_iters', type=int, default=50)
parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'muon'],
                    help='muon: Muon for the hidden matrices + AdamW for the rest')
parser.add_argument('--learning_rate', type=float, default=6e-4, help='AdamW learning rate')
parser.add_argument('--muon_lr', type=float, default=0.02, help='Muon learning rate (only with --optimizer muon)')
parser.add_argument('--out', type=str, default='out/gpt.pt')
args = parser.parse_args()

torch.manual_seed(1337)
device = get_device()

# data: stories are joined into one long stream with <eos> after each story
if args.tokenizer == 'char':
    tokenizer = CharTokenizer(ASCII_CHARS, special_tokens=CHAT_SPECIAL_TOKENS)
else:
    assert os.path.exists(args.tokenizer), f'{args.tokenizer} not found, run python -m scripts.tok_train first'
    tokenizer = load_tokenizer(args.tokenizer)
eos = tokenizer.token_id('<eos>')
t = time.time()
train_stories, val_stories = load_tinystories('train', max_mb=args.max_mb), load_tinystories('valid')
train_data = encode_documents(tokenizer, train_stories)
val_data = encode_documents(tokenizer, val_stories)
val_bytes = sum(len(s) for s in val_stories) # for bits per byte (the text is ASCII, 1 char = 1 byte)
print(f'train tokens: {len(train_data):,}, val tokens: {len(val_data):,}, vocab size: {tokenizer.vocab_size}, '
      f'{val_bytes / len(val_data):.2f} chars per token (encoded in {time.time() - t:.0f}s)')

# model
config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=args.block_size, emb_dim=args.emb_dim,
                     n_head=args.n_head, n_layer=args.n_layer, dropout=args.dropout,
                     n_expert=args.n_expert, top_k=args.top_k)
model = TinyGPT(config).to(device)
print(config)
print(f'{sum(p.numel() for p in model.parameters()):,} parameters')
optimizers = setup_optimizers(model, args.optimizer, lr=args.learning_rate, muon_lr=args.muon_lr)

def batch(split):
    data = train_data if split == 'train' else val_data
    return get_batch(data, args.batch_size, config.block_size, device) # (x, y)

# training loop
for iter in range(args.max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
        losses = estimate_loss(model, batch, args.eval_iters)
        bpb = bits_per_byte(losses['val'], len(val_data), val_bytes) # comparable between tokenizers
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val bpb {bpb:.4f}")

    xb, yb = batch('train')
    logits, loss = model(xb, yb)
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    loss.backward()
    for opt in optimizers:
        opt.step()

save_checkpoint(args.out, model, config, tokenizer)

# generate a story: start from <eos>, i.e. "a new story begins here"
model.eval()
context = torch.tensor([[eos]], dtype=torch.long, device=device)
print(tokenizer.decode(model.generate(context, max_new_tokens=300)[0].tolist()))
