"""
Pieces shared by GRPO training and evaluation: the prompts, the reward, sampling, and the metrics.

The task: "Write a story. Use the words: dog, ball, happy. It should include dialogue."
It can be checked by a script, so no reward model is needed:
    reward = fraction of the required words used       (0 .. 1, the main signal)
           + 0.2 if the model stopped by itself (<eos>)  (a story cut off at max_new_tokens gets nothing)
           + 0.1 if dialogue was asked for and the story has quotes
"""
import re

import torch
from torch.nn import functional as F

from tinychat.chat import story_prompt, render_chat


def rl_prompts(entries):
    # prompts with only the checkable parts (words, features) of TinyStoriesInstruct entries that have words
    return [story_prompt({k: f[k] for k in ('Words', 'Features') if k in f}) for f in entries if 'Words' in f]


def required_words(prompt):
    # "Use the words: dog, ball, happy." -> ['dog', 'ball', 'happy']
    m = re.search(r'Use the words: ([^.]*)\.', prompt)
    return [w.strip() for w in m.group(1).split(',') if w.strip()] if m else []


def story_reward(prompt, story, stopped):
    # returns the total reward and its parts (the parts are logged, only 'reward' is trained on)
    words = required_words(prompt)
    # \bdog\w* : a whole word starting with the required word, so 'dogs' counts but 'hotdog' does not
    found = sum(bool(re.search(rf'\b{re.escape(w)}\w*', story, re.IGNORECASE)) for w in words)
    coverage = found / len(words) if words else 0.0
    wants_dialogue = 'include dialogue' in prompt
    has_dialogue = '"' in story
    reward = coverage + 0.2 * stopped + 0.1 * (wants_dialogue and has_dialogue)
    return {
        'reward': reward,
        'coverage': coverage,
        'all_words': float(coverage == 1.0),
        'stopped': float(stopped),
        'dialogue': float(has_dialogue) if wants_dialogue else None, # only counted when asked for
        'length': len(story),
    }


@torch.no_grad()
def sample_completions(model, tokenizer, prompt, n, max_new_tokens, temperature):
    # sample n answers to one prompt in one batch (same prompt -> same length, so no padding is needed)
    # returns the prompt ids and a list of (completion ids, stopped), each completion cut after its first <eos>
    eos = tokenizer.token_id('<eos>')
    ids, _ = render_chat(tokenizer, prompt)
    max_new = min(max_new_tokens, model.config.block_size - len(ids)) # prompt + answer must fit in the context
    idx = torch.tensor([ids] * n, dtype=torch.long, device=next(model.parameters()).device)
    out = model.generate(idx, max_new, temperature=temperature, stop_token=eos)
    completions = []
    for row in out[:, len(ids):].tolist():
        stopped = eos in row
        if stopped:
            row = row[:row.index(eos) + 1] # keep the <eos>, the model should learn to produce it
        completions.append((row, stopped))
    return ids, completions


def completion_batch(pairs, pad_token, device):
    # pairs: list of (prompt ids, completion ids). Builds prompt + completion rows padded on the right
    # returns x, y (next token targets) and mask (True where y is a completion token)
    T = max(len(p) + len(c) for p, c in pairs)
    full = torch.full((len(pairs), T), pad_token, dtype=torch.long)
    is_answer = torch.zeros((len(pairs), T), dtype=torch.bool) # False on prompt and padding
    for b, (p, c) in enumerate(pairs):
        full[b, :len(p) + len(c)] = torch.tensor(p + c)
        is_answer[b, len(p):len(p) + len(c)] = True
    full, is_answer = full.to(device), is_answer.to(device)
    return full[:, :-1], full[:, 1:], is_answer[:, 1:]


def token_logprobs(model, x, y):
    # log p(y_t | x_<=t) for every position, (B,T)
    logits, _ = model(x)
    return F.log_softmax(logits, dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)


def completion_text(tokenizer, tokens):
    eos = tokenizer.token_id('<eos>')
    return tokenizer.decode([t for t in tokens if t != eos])


@torch.no_grad()
def evaluate(model, tokenizer, prompts, samples_per_prompt=1, max_new_tokens=300, temperature=0.8):
    # average reward parts over all samples, plus a few examples to read
    was_training = model.training
    model.eval()
    rows, examples = [], []
    for prompt in prompts:
        _, completions = sample_completions(model, tokenizer, prompt, samples_per_prompt, max_new_tokens, temperature)
        for tokens, stopped in completions:
            story = completion_text(tokenizer, tokens)
            rows.append(story_reward(prompt, story, stopped))
            if len(examples) < 3:
                examples.append((prompt, story, rows[-1]['reward']))
    metrics = {}
    for key in rows[0]:
        values = [r[key] for r in rows if r[key] is not None]
        metrics[key] = sum(values) / len(values) if values else float('nan')
    model.train(was_training)
    return metrics, examples
