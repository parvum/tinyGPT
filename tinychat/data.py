"""
Data loading and the batch makers (tokenizers are in tokenizer.py) for each training objective.
"""
import os
import urllib.request

import torch

from tinychat.tokenizer import CharTokenizer, BPETokenizer, encode_documents  # noqa: F401 (re-exported)

SHAKESPEARE_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
TINYSTORIES_URL = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/{}'
TINYSTORIES_FILES = {'train': 'TinyStoriesV2-GPT4-train.txt', 'valid': 'TinyStoriesV2-GPT4-valid.txt'}
TINYSTORIES_INSTRUCT_URL = 'https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/resolve/main/{}'
TINYSTORIES_INSTRUCT_FILES = {'train': 'TinyStories-Instruct-train.txt', 'valid': 'TinyStories-Instruct-valid.txt'}
END_OF_STORY = '<|endoftext|>' # separator between stories in the TinyStories files

# fixed character set: newline + all printable ASCII, so digits and symbols are there for later stages
ASCII_CHARS = ['\n'] + [chr(i) for i in range(32, 127)]
# special tokens for the chat stages, added now so SFT does not need to resize the model
CHAT_SPECIAL_TOKENS = ['<eos>', '<user>', '<assistant>']


def load_shakespeare(path='data/input.txt'):
    # download once, then read from disk
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def download(url, path, max_bytes=None):
    # download once, max_bytes keeps only the start of a big file
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f'downloading {url} -> {path}')
    tmp = path + '.part'
    with urllib.request.urlopen(url) as r, open(tmp, 'wb') as f:
        done = 0
        while max_bytes is None or done < max_bytes:
            chunk = r.read(1 << 20) # 1MB at a time
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
    os.replace(tmp, path) # only appears under the final name once complete


def to_ascii(text):
    # map the common non-ASCII characters in TinyStories to ASCII and drop the rest
    for a, b in {'’': "'", '‘': "'", '“': '"', '”': '"', '—': '-', '–': '-',
                 '…': '...', ' ': ' ', '\t': ' ', '\r': '', '\x00': ''}.items():
        text = text.replace(a, b)
    return text.encode('ascii', errors='ignore').decode('ascii')


def load_entries(url, name, max_mb=None, data_dir='data'):
    # download a TinyStories style file and split it into entries (ASCII strings) at <|endoftext|>
    # max_mb only downloads the first max_mb MB of the file
    path = os.path.join(data_dir, name if max_mb is None else name.replace('.txt', f'-{max_mb}MB.txt'))
    download(url.format(name), path, None if max_mb is None else max_mb * (1 << 20))
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    entries = text.split(END_OF_STORY)
    if max_mb is not None:
        entries = entries[:-1] # the last entry is probably cut in the middle by max_mb
    entries = [to_ascii(e).strip() for e in entries]
    return [e for e in entries if e]


def load_tinystories(split, max_mb=None, data_dir='data'):
    # returns a list of stories, the full train file is ~2GB, the valid file ~22MB
    return load_entries(TINYSTORIES_URL, TINYSTORIES_FILES[split], max_mb, data_dir)


def load_tinystories_instruct(split, max_mb=None, data_dir='data'):
    # returns a list of dicts like {'Words': 'meet, waffle, new', 'Features': 'Dialogue', 'Summary': ..., 'Story': ...}
    # the full train file is ~2.6GB, the valid file ~27MB
    entries = load_entries(TINYSTORIES_INSTRUCT_URL, TINYSTORIES_INSTRUCT_FILES[split], max_mb, data_dir)
    parsed = [parse_instruct_entry(e) for e in entries]
    return [p for p in parsed if p is not None]


def parse_instruct_entry(entry):
    # an entry looks like:
    #   Features: Dialogue, BadEnding      <- header lines in any order, each one optional
    #   Words: quit, oak, gloomy
    #   Summary: ...
    #   Random sentence: ...
    #   Story:
    #   Once upon a time ...
    #   Summary: ...                       <- sometimes the summary comes after the story instead
    head, sep, story = entry.partition('Story:')
    if not sep:
        return None
    fields = {}
    for line in head.strip().split('\n'):
        key, _, value = line.partition(':')
        if value.strip():
            fields[key.strip()] = value.strip()
    lines = story.strip().split('\n')
    if lines[-1].startswith('Summary:'):
        fields['Summary after'] = lines[-1][len('Summary:'):].strip()
        lines = lines[:-1]
    fields['Story'] = '\n'.join(lines).strip()
    return fields if fields['Story'] else None


def train_val_split(ids, val_frac=0.1):
    data = torch.tensor(ids, dtype=torch.long)
    n = int((1 - val_frac) * len(data))
    return data[:n], data[n:]


def get_batch(data, batch_size, block_size, device):
    # random chunks of block_size tokens, y is x shifted by one (next token targets)
    # data can be stored as uint8 to save memory, the model needs long ids
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).long()
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).long()
    return x.to(device), y.to(device)


# -----------------------------------------------------------------------------
# BERT: masked language modeling

def mask_tokens(x, mask_token, n_chars, mask_prob=0.15):
    # returns (masked x, targets), targets are -100 where nothing was hidden
    targets = x.clone()
    picked = torch.rand(x.shape, device=x.device) < mask_prob # (B,T) ~15% of the positions
    targets[~picked] = -100 # -100 is ignored by F.cross_entropy
    # BERT's 80/10/10 rule: of the picked tokens, 80% -> <MASK>, 10% -> a random char, 10% -> left unchanged
    x = x.clone()
    r = torch.rand(x.shape, device=x.device)
    x[picked & (r < 0.8)] = mask_token
    random_pos = picked & (r >= 0.8) & (r < 0.9)
    x[random_pos] = torch.randint(n_chars, (int(random_pos.sum()),), device=x.device)
    return x, targets


# -----------------------------------------------------------------------------
# T5: span corruption

def corrupt_spans(x, sentinel_tokens, start_token, span_len):
    # x is (B, T). Cut len(sentinel_tokens) spans of span_len tokens out of every row.
    # src:    the row with each span replaced by its sentinel  "ab<X0>ef<X1>i"
    # tgt:    the sentinels followed by the missing spans      "<X0>cd<X1>gh"
    # tgt_in: tgt shifted right with the start token in front  (teacher forcing)
    # every row gets the same number and size of spans, so all rows have the same length (no padding)
    n_spans = len(sentinel_tokens)
    srcs, tgts = [], []
    for row in x.tolist():
        seg = len(row) // n_spans # one span per segment, so spans never overlap
        src, tgt = [], []
        for i in range(n_spans):
            segment = row[i*seg:(i+1)*seg]
            s = torch.randint(seg - span_len + 1, (1,)).item() # random start of the span inside this segment
            src += segment[:s] + [sentinel_tokens[i]] + segment[s+span_len:]
            tgt += [sentinel_tokens[i]] + segment[s:s+span_len]
        src += row[n_spans*seg:] # leftover tokens if T is not divisible by n_spans
        srcs.append(src)
        tgts.append(tgt)
    src = torch.tensor(srcs, dtype=torch.long, device=x.device) # (B, S)
    tgt = torch.tensor(tgts, dtype=torch.long, device=x.device) # (B, n_spans * (1 + span_len))
    start = torch.full((tgt.shape[0], 1), start_token, dtype=torch.long, device=x.device)
    tgt_in = torch.cat([start, tgt[:, :-1]], dim=1)
    return src, tgt_in, tgt
