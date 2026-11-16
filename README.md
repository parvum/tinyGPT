# tinychat

Tiny GPT / BERT / T5 for learning, laid out like nanochat: model code in `tinychat/`, runnable scripts in `scripts/`.

```
tinychat/
  base.py     shared building blocks: ModelConfig, attention heads, MoE, Block / EncoderBlock / DecoderBlock, Embedding
  gpt.py      TinyGPT   (decoder only, next token prediction)
  bert.py     TinyBERT  (encoder only, masked language modeling)
  t5.py       TinyT5    (encoder + decoder, span corruption)
  tokenizer.py  BPETokenizer (byte level BPE, trained by us), CharTokenizer, encode_documents
  data.py     TinyStories / TinyStoriesInstruct / shakespeare download, batch makers
  chat.py     chat template <user>..<assistant>..<eos>, instruct -> (prompt, response), SFT batches
  rl.py       RL prompts, the checkable story reward, sampling, evaluation metrics
  muon.py     Muon optimizer (orthogonalized momentum) for the hidden matrices, AdamW for the rest
  common.py   get_device, estimate_loss, save_checkpoint, load_checkpoint
scripts/
  tok_train.py         0. train the BPE tokenizer (4096 tokens)          -> out/tokenizer.pt
  pretraining_gpt.py   1. pretrain TinyGPT on TinyStories               -> out/gpt.pt
  sft_gpt.py           2. SFT on TinyStoriesInstruct                    -> out/sft.pt
  grpo_gpt.py          3a. GRPO (online RL) with a script-checked reward -> out/grpo.pt
  dpo_gpt.py           3b. DPO (offline preference pairs, same reward)   -> out/dpo.pt
  eval_gpt.py          compare checkpoints: reward, word coverage, stopping, forgetting
  chat_cli.py          talk to the model in the terminal
  train_bert.py, train_t5.py   (still on tiny shakespeare)
```

Run everything (tokenizer -> pretrain -> SFT -> GRPO -> DPO -> eval) and get `out/report.md`:

```
bash speedrun.sh            # full run (GPU)
QUICK=1 bash speedrun.sh    # tiny settings, a few minutes, checks the pipeline works
```

Or run the stages one by one, from this folder (the `-m` form lets the scripts import `tinychat`):

```
python -m scripts.tok_train                       # BPE, ~4 characters per token
python -m scripts.pretraining_gpt                 # first 100MB of TinyStories, BPE tokens, block_size 512
python -m scripts.pretraining_gpt --optimizer muon   # Muon + AdamW instead of AdamW only
python -m scripts.pretraining_gpt --tokenizer char --block_size 1024   # the character level version
python -m scripts.sft_gpt                         # starts from out/gpt.pt
python -m scripts.grpo_gpt                        # starts from out/sft.pt
python -m scripts.dpo_gpt                         # starts from out/sft.pt, pairs cached in out/dpo_pairs.pt
python -m scripts.eval_gpt --ckpt out/gpt.pt out/sft.pt out/grpo.pt out/dpo.pt   # before / after table
python -m scripts.chat_cli                        # chat with out/sft.pt (--ckpt out/grpo.pt for the RL model)
python -m scripts.pretraining_gpt --max_mb 5 --max_iters 200 --emb_dim 64 --n_layer 2 --batch_size 4   # quick test
python -m scripts.train_bert --max_iters 500
python -m scripts.train_t5 --n_spans 3 --span_len 2
```

The loss per token depends on the tokenizer (a BPE token is harder to guess than one character), so
pretraining and eval also print **bpb**, bits per byte of text, which is comparable between tokenizers.

Checkpoints go to `out/`, the dataset to `data/`. The original notebook `gge-gpt.ipynb` is unchanged.
