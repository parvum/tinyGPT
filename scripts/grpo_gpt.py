"""
GRPO (Group Relative Policy Optimization): reinforcement learning with a reward checked by a script.

Starting from the SFT model, every step:
  1. take a few prompts like "Write a story. Use the words: dog, ball, happy."
  2. sample a group of answers for each prompt from the current model
  3. score each answer with story_reward (tinychat/rl.py)
  4. advantage = (reward - group mean) / group std: answers better than their siblings get pushed up,
     worse ones pushed down. The group average replaces the value network that PPO needs.
  5. loss = -advantage * log p(answer tokens) + beta * KL(model || frozen SFT model)
     the KL term keeps the model close to the SFT model, so it cannot drift into
     nonsense that just lists the words

Run from the project root, after scripts.sft_gpt:
    python -m scripts.grpo_gpt
    python -m scripts.grpo_gpt --steps 5 --prompts_per_step 2 --group_size 4 --max_new_tokens 64 --eval_prompts 5   # quick test
"""
import argparse
import time

import torch

from tinychat.gpt import TinyGPT
from tinychat.data import load_tinystories_instruct
from tinychat.rl import (rl_prompts, story_reward, sample_completions, completion_text, evaluate,
                         completion_batch, token_logprobs)
from tinychat.common import get_device, save_checkpoint, load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--init', type=str, default='out/sft.pt', help='SFT checkpoint to start from (also the KL reference)')
parser.add_argument('--out', type=str, default='out/grpo.pt')
parser.add_argument('--max_mb', type=int, default=5, help='MB of the TinyStoriesInstruct train file to take prompts from')
parser.add_argument('--steps', type=int, default=300)
parser.add_argument('--prompts_per_step', type=int, default=4)
parser.add_argument('--group_size', type=int, default=8, help='answers sampled per prompt')
parser.add_argument('--temperature', type=float, default=1.0, help='sampling temperature, 1.0 = explore')
parser.add_argument('--max_new_tokens', type=int, default=300) # tokens: ~1200 chars with BPE, use 900 for --tokenizer char
parser.add_argument('--learning_rate', type=float, default=1e-5) # much lower than SFT, RL updates are noisy
parser.add_argument('--beta', type=float, default=0.04, help='weight of the KL penalty')
parser.add_argument('--eval_interval', type=int, default=100)
parser.add_argument('--eval_prompts', type=int, default=50)
args = parser.parse_args()

torch.manual_seed(1337)
device = get_device()

# the policy (trained) and the reference (frozen copy of the SFT model, for the KL penalty)
model, config, tokenizer = load_checkpoint(args.init, TinyGPT, device)
ref_model, _, _ = load_checkpoint(args.init, TinyGPT, device)
ref_model.eval()
ref_model.requires_grad_(False)
eos = tokenizer.token_id('<eos>')

train_prompts = rl_prompts(load_tinystories_instruct('train', max_mb=args.max_mb))
eval_prompts = rl_prompts(load_tinystories_instruct('valid'))[:args.eval_prompts]
print(f'{len(train_prompts):,} train prompts, {len(eval_prompts)} eval prompts')

optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)


def group_loss(prompt_ids, completions, advantages):
    # one prompt, G completions: prompt + completion rows padded with <eos>, only answer tokens count
    x, y, mask = completion_batch([(prompt_ids, tokens) for tokens, _ in completions], eos, device)

    logp = token_logprobs(model, x, y) # (G,T)
    with torch.no_grad():
        ref_logp = token_logprobs(ref_model, x, y)
    # per token KL estimate used by GRPO: always >= 0, and 0 when the two models agree
    kl = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1
    # policy gradient: raise log p of answers with positive advantage, lower it for negative ones
    per_token = -advantages.unsqueeze(1) * logp + args.beta * kl
    loss = (per_token * mask).sum() / mask.sum() # average over all answer tokens in the group
    return loss, (kl * mask).sum().item() / mask.sum().item()


def run_eval(step):
    t = time.time()
    metrics, examples = evaluate(model, tokenizer, eval_prompts, 1, args.max_new_tokens, temperature=0.8)
    print(f'--- eval step {step}: ' + ', '.join(f'{k} {v:.3f}' for k, v in metrics.items()) + f' ({time.time()-t:.0f}s)')
    prompt, story, reward = examples[0]
    story = story[:300].replace('\n', '\n    ')
    print(f'    USER: {prompt}\n    ASSISTANT (reward {reward:.2f}): {story}')


run_eval(0) # the SFT model before any RL, the baseline
for step in range(1, args.steps + 1):
    t = time.time()
    picks = torch.randint(len(train_prompts), (args.prompts_per_step,)).tolist()

    # 1-3. sample a group of answers for each prompt and score them (no gradients here)
    model.eval()
    groups, all_rewards, stats = [], [], []
    for i in picks:
        prompt = train_prompts[i]
        prompt_ids, completions = sample_completions(model, tokenizer, prompt, args.group_size,
                                                     args.max_new_tokens, args.temperature)
        scores = [story_reward(prompt, completion_text(tokenizer, toks), stopped) for toks, stopped in completions]
        rewards = torch.tensor([s['reward'] for s in scores], device=device)
        # 4. group relative advantage. If every answer got the same reward, all advantages are 0
        #    and this prompt teaches nothing this step
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        groups.append((prompt_ids, completions, advantages))
        all_rewards.append(rewards)
        stats += scores

    # 5. one gradient step on all groups (accumulated one group at a time to save memory)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    kls = []
    for prompt_ids, completions, advantages in groups:
        loss, kl = group_loss(prompt_ids, completions, advantages)
        (loss / len(groups)).backward()
        kls.append(kl)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # RL gradients can spike, keep steps bounded
    optimizer.step()

    mean = lambda key: sum(s[key] for s in stats) / len(stats)
    print(f"step {step}: reward {torch.cat(all_rewards).mean():.3f}, coverage {mean('coverage'):.3f}, "
          f"stopped {mean('stopped'):.2f}, length {mean('length'):.0f}, kl {sum(kls)/len(kls):.4f} ({time.time()-t:.1f}s)")

    if step % args.eval_interval == 0 and step != args.steps:
        run_eval(step)

run_eval(args.steps)
save_checkpoint(args.out, model, config, tokenizer)
