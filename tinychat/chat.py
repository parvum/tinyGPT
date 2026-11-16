"""
Chat format and SFT data.

Every conversation is rendered with one fixed template:
    <user>{prompt}<assistant>{response}<eos>
The loss is only computed on the response and the final <eos>, so the model learns to answer
(and to stop), not to write questions.
"""
import torch

from tinychat.tokenizer import id_dtype

# how the TinyStoriesInstruct feature names are said in a prompt
FEATURE_TEXT = {
    'Dialogue': 'include dialogue',
    'BadEnding': 'have a bad ending',
    'MoralValue': 'teach a moral lesson',
    'Twist': 'have a plot twist',
    'Foreshadowing': 'use foreshadowing',
    'Conflict': 'include a conflict',
}


def story_prompt(fields):
    # turn the header fields of a TinyStoriesInstruct entry into an instruction
    parts = ['Write a story.']
    if 'Words' in fields:
        parts.append(f"Use the words: {fields['Words']}.")
    if 'Features' in fields:
        feats = [FEATURE_TEXT.get(f.strip(), f.strip().lower()) for f in fields['Features'].split(',')]
        parts.append('It should ' + ' and '.join(feats) + '.')
    if 'Random sentence' in fields:
        parts.append(f"Include the sentence: \"{fields['Random sentence']}\"")
    if 'Summary' in fields:
        parts.append(f"Summary: {fields['Summary']}")
    return ' '.join(parts)


def instruct_examples(entries):
    # two tasks from one dataset:
    #   1. write a story that follows the header (every entry)
    #   2. summarize the story (entries whose summary comes after the story)
    examples = []
    for f in entries:
        examples.append((story_prompt(f), f['Story']))
        if 'Summary after' in f:
            examples.append((f"Summarize this story:\n{f['Story']}", f['Summary after']))
    return examples


def render_chat(tokenizer, prompt, response=None):
    # returns (ids, mask): mask is 1 on the tokens the model should learn to produce (response + <eos>)
    # with response=None it renders just the prompt, ready for the model to answer
    ids = [tokenizer.token_id('<user>')] + tokenizer.encode(prompt) + [tokenizer.token_id('<assistant>')]
    mask = [0] * len(ids)
    if response is not None:
        r = tokenizer.encode(response) + [tokenizer.token_id('<eos>')]
        ids += r
        mask += [1] * len(r)
    return ids, mask


def render_examples(tokenizer, examples, block_size, min_response=32):
    # render every (prompt, response) pair to (ids, mask) tensors, cut to block_size + 1 tokens
    # (the +1 is because x and y are shifted by one). Examples whose prompt leaves fewer than
    # min_response tokens for the answer are dropped, a long story just gets its end cut off.
    rendered = []
    for prompt, response in examples:
        ids, mask = render_chat(tokenizer, prompt, response)
        if len(ids) - sum(mask) > block_size - min_response:
            continue
        ids, mask = ids[:block_size + 1], mask[:block_size + 1]
        rendered.append((torch.tensor(ids, dtype=id_dtype(tokenizer.vocab_size)), torch.tensor(mask, dtype=torch.bool)))
    return rendered


def sft_batch(rendered, batch_size, pad_token, device):
    # random examples, padded to the longest one in the batch
    #   x: tokens fed to the model, padded with pad_token
    #   y: next token targets, -100 on the prompt and on the padding so they are not in the loss
    picks = torch.randint(len(rendered), (batch_size,)).tolist()
    T = max(len(rendered[i][0]) for i in picks) - 1
    x = torch.full((batch_size, T), pad_token, dtype=torch.long)
    y = torch.full((batch_size, T), -100, dtype=torch.long)
    for b, i in enumerate(picks):
        ids, mask = rendered[i]
        ids = ids.long()
        n = len(ids) - 1
        x[b, :n] = ids[:-1]
        y[b, :n] = torch.where(mask[1:], ids[1:], torch.tensor(-100))
    return x.to(device), y.to(device)
