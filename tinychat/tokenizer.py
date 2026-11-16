"""
Tokenizers: the character level one and a byte level BPE (Byte Pair Encoding) one, like GPT-2 / nanochat.

BPE in one sentence: start from the 256 bytes, then repeatedly glue together the most frequent
adjacent pair of tokens into a new token, until the vocabulary has vocab_size tokens.
    "the dog" -> t h e _ d o g -> th e _ d o g -> the _ d o g -> the _do g -> the _dog
Encoding new text replays the learned merges in the same order. Decoding just concatenates bytes.

Both tokenizers have the same interface: encode(text) -> ids, decode(ids) -> text, token_id('<eos>'),
vocab_size, special_tokens, state() (for saving in a checkpoint).
"""
import collections
import re
from array import array

import torch

# split the text into chunks first, merges never cross a chunk boundary, so " dog" and " dog." share
# the token " dog" and never a token "g.". This is the GPT-2 pattern, written for ASCII text:
#   's 't 'll ...  |  optional space + letters  |  optional space + 1-3 digits  |
#   optional space + punctuation  |  whitespace (not the last space before a word)  |  whitespace
SPLIT_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]{1,3}| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")


def merge(ids, pair, new_id):
    # replace every occurrence of pair in ids with new_id:  merge([1,2,3,1,2], (1,2), 9) -> [9,3,9]
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte level BPE: ids 0-255 are bytes, 256.. are merges, then the special tokens"""

    def __init__(self, merges, special_tokens=()):
        self.merges = [tuple(m) for m in merges] # merge i glues (a, b) into token 256 + i
        self.special_tokens = list(special_tokens)
        self.ranks = {pair: 256 + i for i, pair in enumerate(self.merges)} # pair -> id, lower = learned earlier
        self.vocab = [bytes([i]) for i in range(256)] # id -> the bytes it stands for
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.special_ids = {tok: len(self.vocab) + j for j, tok in enumerate(self.special_tokens)}
        self.special_names = {i: tok for tok, i in self.special_ids.items()}
        self.cache = {} # chunk -> ids, stories reuse the same few thousand words over and over

    @classmethod
    def train(cls, text, vocab_size, special_tokens=(), verbose=False):
        n_merges = vocab_size - 256 - len(special_tokens)
        assert n_merges >= 0
        # 1. count each distinct chunk once ("the" appears a million times, it is one entry with count 1M)
        counts = collections.Counter(SPLIT_PATTERN.findall(text))
        words = [list(w.encode('utf-8')) for w in counts]
        freqs = list(counts.values())
        # 2. how often every adjacent pair appears, and in which words (so a merge only touches those words)
        pair_counts = collections.Counter()
        where = collections.defaultdict(set)
        for wi, w in enumerate(words):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += freqs[wi]
                where[pair].add(wi)
        # 3. repeatedly merge the most frequent pair
        merges = []
        for i in range(n_merges):
            if not pair_counts:
                break # every chunk is a single token already
            pair = max(pair_counts, key=pair_counts.get)
            new_id = 256 + i
            merges.append(pair)
            for wi in list(where[pair]):
                w, f = words[wi], freqs[wi]
                for p in zip(w, w[1:]): # take the word's old pairs out of the counts ...
                    pair_counts[p] -= f
                    if pair_counts[p] == 0:
                        del pair_counts[p]
                    where[p].discard(wi)
                w = words[wi] = merge(w, pair, new_id)
                for p in zip(w, w[1:]): # ... and put its new pairs in
                    pair_counts[p] += f
                    where[p].add(wi)
            if verbose and (i < 10 or (i + 1) % 500 == 0):
                tok = cls(merges).vocab[new_id] if i < 10 else None
                print(f'  merge {i+1}/{n_merges}' + (f': {tok!r}' if tok is not None else ''))
        return cls(merges, special_tokens)

    @property
    def vocab_size(self):
        return len(self.vocab) + len(self.special_tokens)

    def token_id(self, special_token):
        return self.special_ids[special_token]

    def _encode_chunk(self, chunk):
        ids = self.cache.get(chunk)
        if ids is None:
            ids = list(chunk.encode('utf-8'))
            while len(ids) >= 2:
                # apply the earliest learned merge that is possible here, like during training
                pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float('inf')))
                if pair not in self.ranks:
                    break # nothing left to merge
                ids = merge(ids, pair, self.ranks[pair])
            self.cache[chunk] = ids
        return ids

    def encode(self, s):
        # plain text only, special tokens are added by id (so a user typing "<eos>" cannot inject one)
        return [i for chunk in SPLIT_PATTERN.findall(s) for i in self._encode_chunk(chunk)]

    def decode(self, ids):
        # special tokens are shown by name. A token can be half of a multi-byte character, hence 'replace'
        out = bytearray()
        for i in ids:
            out += self.special_names[i].encode() if i in self.special_names else self.vocab[i]
        return out.decode('utf-8', errors='replace')

    def state(self):
        return {'type': 'bpe', 'merges': self.merges, 'special_tokens': self.special_tokens}


class CharTokenizer:
    """Character level tokenizer, special tokens (like <MASK>) get the ids after the characters"""

    def __init__(self, chars, special_tokens=()):
        self.chars = list(chars)
        self.special_tokens = list(special_tokens)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        for j, tok in enumerate(self.special_tokens):
            self.stoi[tok] = len(self.chars) + j
            self.itos[len(self.chars) + j] = tok

    @classmethod
    def from_text(cls, text, special_tokens=()):
        return cls(sorted(set(text)), special_tokens)

    @property
    def n_chars(self):
        return len(self.chars)

    @property
    def vocab_size(self):
        return len(self.chars) + len(self.special_tokens)

    def token_id(self, special_token):
        return self.stoi[special_token]

    def encode(self, s):
        # plain text only, special tokens are added by id. characters the tokenizer does not know are skipped
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        # special tokens are shown by name, e.g. '<MASK>'
        return ''.join(self.itos[i] for i in ids)

    def state(self):
        return {'type': 'char', 'chars': self.chars, 'special_tokens': self.special_tokens}


def tokenizer_from_state(state):
    if state['type'] == 'bpe':
        return BPETokenizer(state['merges'], state['special_tokens'])
    return CharTokenizer(state['chars'], state['special_tokens'])


def save_tokenizer(path, tokenizer):
    torch.save(tokenizer.state(), path)


def load_tokenizer(path):
    return tokenizer_from_state(torch.load(path))


def id_dtype(vocab_size):
    # smallest dtype that holds every id, to keep 100M+ tokens in memory cheaply
    return torch.uint8 if vocab_size <= 256 else torch.int16


def encode_documents(tokenizer, docs):
    # encode a list of documents into one long tensor of ids, with <eos> after each document
    eos = tokenizer.token_id('<eos>')
    if isinstance(tokenizer, CharTokenizer):
        # fast path: a lookup table on the raw bytes, much faster than a python loop over 100M characters
        lut = torch.full((128,), -1, dtype=torch.long) # ascii byte -> token id, -1 = not in the tokenizer
        for ch in tokenizer.chars:
            lut[ord(ch)] = tokenizer.stoi[ch]
        lut[0] = eos # \x00 marks the end of each document (to_ascii removed real \x00)
        text = '\x00'.join(docs) + '\x00'
        ids = lut[torch.frombuffer(bytearray(text.encode('ascii')), dtype=torch.uint8).long()]
        return ids[ids >= 0].to(id_dtype(tokenizer.vocab_size)) # drop characters not in the tokenizer
    ids = array('h') # int16, compact while the stream grows
    for doc in docs:
        ids.extend(tokenizer.encode(doc))
        ids.append(eos)
    return torch.frombuffer(ids, dtype=torch.int16).clone()
