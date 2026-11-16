"""
Evaluate one or more checkpoints on held-out story prompts, side by side.

Metrics (averaged over all samples):
    reward     the GRPO reward (coverage + 0.2 stopped + 0.1 dialogue)
    coverage   fraction of the required words used
    all_words  fraction of stories that use every required word
    stopped    fraction of stories that ended with <eos> by themselves
    dialogue   fraction of stories with quotes, among prompts that asked for dialogue
    length     average story length in characters
    pretrain   loss on the TinyStories validation text (higher after SFT/RL = forgetting)
    bpb        the same loss in bits per byte, comparable between char and BPE checkpoints

Run from the project root:
    python -m scripts.eval_gpt                                          # out/sft.pt
    python -m scripts.eval_gpt --ckpt out/gpt.pt out/sft.pt out/grpo.pt # compare the stages
"""
import argparse
import os

import torch

from tinychat.gpt import TinyGPT
from tinychat.data import load_tinystories, load_tinystories_instruct, encode_documents, get_batch
from tinychat.rl import rl_prompts, evaluate
from tinychat.common import get_device, load_checkpoint, bits_per_byte

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, nargs='+', default=['out/sft.pt'])
parser.add_argument('--n_prompts', type=int, default=200)
parser.add_argument('--samples', type=int, default=1, help='answers sampled per prompt')
parser.add_argument('--temperature', type=float, default=0.8)
parser.add_argument('--max_new_tokens', type=int, default=300) # tokens: ~1200 chars with BPE, use 900 for --tokenizer char
parser.add_argument('--pretrain_iters', type=int, default=20, help='batches for the pretrain loss, 0 to skip')
args = parser.parse_args()

device = get_device()

# held-out prompts from the validation split (GRPO trains on prompts from the train split)
prompts = rl_prompts(load_tinystories_instruct('valid'))[:args.n_prompts]
val_stories = load_tinystories('valid') if args.pretrain_iters > 0 else None
val_data = {} # encoded once per tokenizer (checkpoints from the same pretraining share one)

@torch.no_grad()
def pretrain_loss(model, tokenizer, config):
    # next token loss on plain TinyStories text, the same as the pretraining val loss
    key = str(tokenizer.state())
    if key not in val_data:
        val_data[key] = encode_documents(tokenizer, val_stories)
    data = val_data[key]
    model.eval()
    losses = [model(*get_batch(data, 8, config.block_size, device))[1].item() for _ in range(args.pretrain_iters)]
    loss = sum(losses) / len(losses)
    return loss, bits_per_byte(loss, len(data), sum(len(s) for s in val_stories))

results = {}
for path in args.ckpt:
    torch.manual_seed(1337) # same random draws for every checkpoint, so the comparison is fair
    model, config, tokenizer = load_checkpoint(path, TinyGPT, device)
    metrics, examples = evaluate(model, tokenizer, prompts, args.samples, args.max_new_tokens, args.temperature)
    if args.pretrain_iters > 0:
        metrics['pretrain'], metrics['bpb'] = pretrain_loss(model, tokenizer, config)
    results[path] = metrics
    print(f'\n##### {path}: examples')
    for prompt, story, reward in examples[:2]:
        print(f'USER: {prompt}\nASSISTANT (reward {reward:.2f}): {story[:400]}{"..." if len(story) > 400 else ""}\n')

# side by side table
print(f'\n{len(prompts)} prompts x {args.samples} samples, temperature {args.temperature}')
print(f"{'metric':<12}" + ''.join(f'{os.path.basename(p):>20}' for p in results))
for key in next(iter(results.values())):
    print(f'{key:<12}' + ''.join(f'{m[key]:>20.4f}' for m in results.values()))
