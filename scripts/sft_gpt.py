"""
Supervised fine-tuning (SFT): teach the pretrained TinyGPT to follow instructions in the chat format
    <user>{prompt}<assistant>{response}<eos>
using TinyStoriesInstruct (write a story with these words/features, summarize this story).

Run from the project root, after scripts.pretraining_gpt:
    python -m scripts.sft_gpt
    python -m scripts.sft_gpt --max_mb 5 --max_iters 200   # quick test
"""
import argparse

import torch

from tinychat.gpt import TinyGPT
from tinychat.data import load_tinystories_instruct
from tinychat.chat import instruct_examples, render_examples, render_chat, sft_batch
from tinychat.common import get_device, estimate_loss, save_checkpoint, load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--init', type=str, default='out/gpt.pt', help='pretrained checkpoint to start from')
parser.add_argument('--out', type=str, default='out/sft.pt')
parser.add_argument('--max_mb', type=int, default=20, help='MB of the TinyStoriesInstruct train file to use (full file is ~2.6GB)')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_iters', type=int, default=2000)  # far fewer steps than pretraining
parser.add_argument('--eval_interval', type=int, default=200)
parser.add_argument('--eval_iters', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=1e-4) # lower than pretraining, we only nudge the model
args = parser.parse_args()

torch.manual_seed(1337)
device = get_device()

# start from the pretrained model, the chat tokens are already in its vocabulary
model, config, tokenizer = load_checkpoint(args.init, TinyGPT, device)
eos = tokenizer.token_id('<eos>')
print(config)

# data: (prompt, response) pairs rendered into the chat template
train_examples = instruct_examples(load_tinystories_instruct('train', max_mb=args.max_mb))
val_examples = instruct_examples(load_tinystories_instruct('valid'))
train_rendered = render_examples(tokenizer, train_examples, config.block_size)
val_rendered = render_examples(tokenizer, val_examples, config.block_size)
print(f'train examples: {len(train_rendered):,} (of {len(train_examples):,}), val examples: {len(val_rendered):,} (of {len(val_examples):,})')

optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

def batch(split):
    rendered = train_rendered if split == 'train' else val_rendered
    return sft_batch(rendered, args.batch_size, eos, device) # (x, y), y is -100 outside the responses

# training loop, the same as pretraining, only the data is different
for iter in range(args.max_iters):

    # step 0 shows how well the base model does on the chat data before any SFT
    if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
        losses = estimate_loss(model, batch, args.eval_iters)
        print(f"step {iter}: train sft loss {losses['train']:.4f}, val sft loss {losses['val']:.4f}")

    xb, yb = batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

save_checkpoint(args.out, model, config, tokenizer)

# try a few prompts
model.eval()
prompts = [
    'Write a story. Use the words: dog, ball, happy. It should include dialogue.',
    'Write a story. It should have a bad ending.',
    next(p for p, _ in val_examples if p.startswith('Summarize')), # a held out story to summarize
]
for prompt in prompts:
    ids, _ = render_chat(tokenizer, prompt)
    out = model.generate(torch.tensor([ids], dtype=torch.long, device=device), max_new_tokens=300,
                         temperature=0.8, stop_token=eos)
    print('=' * 80)
    print('USER:', prompt)
    print('ASSISTANT:', tokenizer.decode(out[0, len(ids):].tolist()))
