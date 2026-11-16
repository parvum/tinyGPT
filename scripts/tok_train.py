"""
Train the BPE tokenizer on TinyStories (stage 0, before pretraining), then show how well it compresses.

Run from the project root:
    python -m scripts.tok_train                          # 4096 tokens from the first 10MB  -> out/tokenizer.pt
    python -m scripts.tok_train --max_mb 2 --vocab_size 1024   # quick test
"""
import argparse
import os
import time

from tinychat.data import load_tinystories, CHAT_SPECIAL_TOKENS, ASCII_CHARS
from tinychat.tokenizer import BPETokenizer, CharTokenizer, save_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--max_mb', type=int, default=10, help='MB of TinyStories to learn the merges from')
parser.add_argument('--vocab_size', type=int, default=4096, help='256 bytes + merges + special tokens')
parser.add_argument('--out', type=str, default='out/tokenizer.pt')
args = parser.parse_args()

# the stories joined by newlines, the special tokens are not learned, they are added on top
text = '\n'.join(load_tinystories('train', max_mb=args.max_mb))
print(f'training on {len(text):,} characters, vocab size {args.vocab_size}')
t = time.time()
tokenizer = BPETokenizer.train(text, args.vocab_size, special_tokens=CHAT_SPECIAL_TOKENS, verbose=True)
print(f'trained in {time.time() - t:.0f}s, {len(tokenizer.merges)} merges')
os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
save_tokenizer(args.out, tokenizer)
print(f'saved to {args.out}')

# ----------------------------------------------------------------------------- evaluation
# compression on held out text: characters (= bytes, the text is ASCII) per token
val = '\n'.join(load_tinystories('valid')[:2000])
char_tok = CharTokenizer(ASCII_CHARS, CHAT_SPECIAL_TOKENS)
for name, tok in [('char', char_tok), ('bpe', tokenizer)]:
    n = len(tok.encode(val))
    print(f'{name:>5}: vocab {tok.vocab_size:>5}, {n:>9,} tokens, {len(val) / n:.2f} chars per token')

print('\nlast merges learned (rarer chunks):', [tokenizer.vocab[i].decode() for i in range(tokenizer.vocab_size - len(CHAT_SPECIAL_TOKENS) - 10, tokenizer.vocab_size - len(CHAT_SPECIAL_TOKENS))])
sample = 'Once upon a time, a little dog named Max found a shiny ball. "Look!" he barked happily.'
ids = tokenizer.encode(sample)
print(f'\n{len(sample)} chars -> {len(ids)} tokens:')
print('|'.join(tokenizer.decode([i]) for i in ids))
assert tokenizer.decode(ids) == sample, 'decode(encode(x)) should give back x'
