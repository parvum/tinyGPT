"""
Talk to the fine-tuned model in the terminal (single turn: every message is a fresh conversation).

Run from the project root, after scripts.sft_gpt:
    python -m scripts.chat_cli
    python -m scripts.chat_cli --ckpt out/gpt.pt   # compare with the base model (it will not stop by itself)
"""
import argparse

import torch

from tinychat.gpt import TinyGPT
from tinychat.chat import render_chat
from tinychat.common import get_device, load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='out/sft.pt')
parser.add_argument('--temperature', type=float, default=0.8)
parser.add_argument('--max_new_tokens', type=int, default=300)
args = parser.parse_args()

device = get_device()
model, config, tokenizer = load_checkpoint(args.ckpt, TinyGPT, device)
model.eval()
eos = tokenizer.token_id('<eos>')

print("Try: Write a story. Use the words: cat, tree, brave.   (empty line or Ctrl+C to quit)")
while True:
    try:
        prompt = input('\nYou: ').strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not prompt:
        break
    ids, _ = render_chat(tokenizer, prompt)
    ids = ids[-config.block_size:] # a very long prompt keeps only its end
    out = model.generate(torch.tensor([ids], dtype=torch.long, device=device), args.max_new_tokens,
                         temperature=args.temperature, stop_token=eos)
    reply = tokenizer.decode(out[0, len(ids):].tolist())
    print('Bot:', reply.removesuffix('<eos>'))
