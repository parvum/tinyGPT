"""
DPO (Direct Preference Optimization): learn from pairs of (chosen, rejected) answers, no RL loop.

1. Build preference pairs once (offline): for each prompt, sample a few stories from the SFT model,
   score them with story_reward (the same reward as GRPO), and pair the best one (chosen)
   with the worst one (rejected). The pairs are saved, so later runs reuse them.
   In real LLM training this is where human (or AI) labelers would pick the better answer.
2. Train on the fixed pairs. For each pair, with the frozen SFT model as the reference:
       chosen_gain   = log p(chosen)   - log p_ref(chosen)      how much more likely the model made it
       rejected_gain = log p(rejected) - log p_ref(rejected)
       loss = -log sigmoid(beta * (chosen_gain - rejected_gain))
   i.e. push the chosen answer up and the rejected one down, relative to where the SFT model started.
   beta plays the role of the KL penalty in GRPO: the reference keeps the model from drifting too far.

Compared with GRPO: no sampling while training (fast, simple, stable), but the model only ever
learns from answers the SFT model wrote, never from its own new answers (offline vs online).

Run from the project root, after scripts.sft_gpt:
    python -m scripts.dpo_gpt
    python -m scripts.dpo_gpt --n_prompts 8 --samples 2 --max_new_tokens 64 --steps 5 --eval_prompts 5   # quick test
"""
import argparse
import os
import time

import torch
from torch.nn import functional as F

from tinychat.gpt import TinyGPT
from tinychat.data import load_tinystories_instruct
from tinychat.rl import (rl_prompts, story_reward, sample_completions, completion_text, evaluate,
                         completion_batch, token_logprobs)
from tinychat.common import get_device, save_checkpoint, load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--init', type=str, default='out/sft.pt', help='SFT checkpoint to start from (also the reference)')
parser.add_argument('--out', type=str, default='out/dpo.pt')
# 1. preference pairs
parser.add_argument('--pairs', type=str, default='out/dpo_pairs.pt', help='where the pairs are saved / loaded from')
parser.add_argument('--rebuild_pairs', action='store_true', help='sample new pairs even if --pairs exists')
parser.add_argument('--max_mb', type=int, default=5, help='MB of the TinyStoriesInstruct train file to take prompts from')
parser.add_argument('--n_prompts', type=int, default=1000)
parser.add_argument('--samples', type=int, default=4, help='stories sampled per prompt, best vs worst becomes a pair')
parser.add_argument('--min_gap', type=float, default=0.2, help='skip pairs whose rewards are closer than this')
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--max_new_tokens', type=int, default=300) # tokens: ~1200 chars with BPE, use 900 for --tokenizer char
# 2. training
parser.add_argument('--steps', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=8, help='pairs per step')
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--beta', type=float, default=0.1)
parser.add_argument('--eval_interval', type=int, default=100)
parser.add_argument('--eval_prompts', type=int, default=50)
args = parser.parse_args()

torch.manual_seed(1337)
device = get_device()

model, config, tokenizer = load_checkpoint(args.init, TinyGPT, device)
eos = tokenizer.token_id('<eos>')


def sequence_logprobs(m, rows):
    # rows: list of (prompt ids, completion ids) -> (B,) total log p of each completion given its prompt
    x, y, mask = completion_batch(rows, eos, device)
    return (token_logprobs(m, x, y) * mask).sum(dim=1)


# -----------------------------------------------------------------------------
# 1. preference pairs

def build_pairs():
    prompts = rl_prompts(load_tinystories_instruct('train', max_mb=args.max_mb))
    order = torch.randperm(len(prompts)).tolist()[:args.n_prompts]
    model.eval()
    pairs, t = [], time.time()
    for n, i in enumerate(order):
        prompt = prompts[i]
        prompt_ids, completions = sample_completions(model, tokenizer, prompt, args.samples,
                                                     args.max_new_tokens, args.temperature)
        scored = sorted(((story_reward(prompt, completion_text(tokenizer, toks), stopped)['reward'], toks)
                         for toks, stopped in completions), key=lambda s: s[0])
        (worst_r, worst), (best_r, best) = scored[0], scored[-1]
        if best_r - worst_r >= args.min_gap: # too close to call -> not a useful preference
            pairs.append({'prompt': prompt_ids, 'chosen': best, 'rejected': worst,
                          'chosen_reward': best_r, 'rejected_reward': worst_r})
        if (n + 1) % 100 == 0:
            print(f'  sampled {n+1}/{len(order)} prompts, {len(pairs)} pairs ({time.time()-t:.0f}s)')
    # the reference log probs never change (the reference is the frozen SFT model, which is the model
    # right now), so compute them once here instead of keeping a second model in memory
    with torch.no_grad():
        for k in range(0, len(pairs), args.batch_size):
            chunk = pairs[k:k + args.batch_size]
            lp = sequence_logprobs(model, [(p['prompt'], p['chosen']) for p in chunk] +
                                          [(p['prompt'], p['rejected']) for p in chunk]).tolist()
            for j, p in enumerate(chunk):
                p['ref_chosen'], p['ref_rejected'] = lp[j], lp[len(chunk) + j]
    return pairs


if os.path.exists(args.pairs) and not args.rebuild_pairs:
    pairs = torch.load(args.pairs)['pairs']
    print(f'loaded {len(pairs)} pairs from {args.pairs} (use --rebuild_pairs after changing --init)')
else:
    print(f'building pairs from {args.n_prompts} prompts x {args.samples} samples ...')
    pairs = build_pairs()
    os.makedirs(os.path.dirname(args.pairs) or '.', exist_ok=True)
    torch.save({'pairs': pairs, 'init': args.init}, args.pairs)
    print(f'saved {len(pairs)} pairs to {args.pairs}')
assert len(pairs) >= 2, 'not enough pairs, try more --n_prompts or a lower --min_gap'

n_val = max(1, len(pairs) // 20) # hold out 5% of the pairs to see if the preference generalizes
train_pairs, val_pairs = pairs[n_val:], pairs[:n_val]
gap = sum(p['chosen_reward'] - p['rejected_reward'] for p in pairs) / len(pairs)
print(f'{len(train_pairs)} train pairs, {len(val_pairs)} val pairs, average reward gap {gap:.2f}')


# -----------------------------------------------------------------------------
# 2. DPO training

def dpo_loss(batch):
    B = len(batch)
    # chosen and rejected in one forward pass: first B rows chosen, last B rows rejected
    lp = sequence_logprobs(model, [(p['prompt'], p['chosen']) for p in batch] +
                                  [(p['prompt'], p['rejected']) for p in batch])
    ref_chosen = torch.tensor([p['ref_chosen'] for p in batch], device=device)
    ref_rejected = torch.tensor([p['ref_rejected'] for p in batch], device=device)
    chosen_gain = lp[:B] - ref_chosen     # > 0: the model made the chosen answer more likely than the SFT model did
    rejected_gain = lp[B:] - ref_rejected
    margin = chosen_gain - rejected_gain
    loss = -F.logsigmoid(args.beta * margin).mean()
    # accuracy: how often the model now prefers chosen over rejected (more than the reference did)
    return loss, (margin > 0).float().mean().item(), margin.mean().item()


@torch.no_grad()
def val_metrics():
    model.eval()
    losses, accs, margins = [], [], []
    for k in range(0, len(val_pairs), args.batch_size):
        loss, acc, margin = dpo_loss(val_pairs[k:k + args.batch_size])
        losses.append(loss.item())
        accs.append(acc)
        margins.append(margin)
    model.train()
    mean = lambda v: sum(v) / len(v)
    return mean(losses), mean(accs), mean(margins)


def run_eval(step):
    t = time.time()
    metrics, examples = evaluate(model, tokenizer, eval_prompts, 1, args.max_new_tokens, temperature=0.8)
    print(f'--- eval step {step}: ' + ', '.join(f'{k} {v:.3f}' for k, v in metrics.items()) + f' ({time.time()-t:.0f}s)')


eval_prompts = rl_prompts(load_tinystories_instruct('valid'))[:args.eval_prompts]
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)

run_eval(0) # the SFT model, the baseline
model.train()
for step in range(1, args.steps + 1):
    batch = [train_pairs[i] for i in torch.randint(len(train_pairs), (args.batch_size,)).tolist()]
    loss, acc, margin = dpo_loss(batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 10 == 0 or step == 1:
        # the loss starts at log(2) = 0.693 (model == reference, margin 0)
        print(f'step {step}: loss {loss.item():.4f}, accuracy {acc:.2f}, margin {margin:.2f}')
    if step % args.eval_interval == 0 or step == args.steps:
        v_loss, v_acc, v_margin = val_metrics()
        print(f'--- val pairs step {step}: loss {v_loss:.4f}, accuracy {v_acc:.2f}, margin {v_margin:.2f}')
        if step != args.steps:
            run_eval(step)

run_eval(args.steps)
save_checkpoint(args.out, model, config, tokenizer)
