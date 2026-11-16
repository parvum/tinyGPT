#!/usr/bin/env bash
# Run the whole tinychat pipeline and write a report:
#   tokenizer -> pretrain -> SFT -> GRPO -> DPO -> eval (all checkpoints side by side) -> $OUT/report.md
#
# Usage, from the project root:
#   bash speedrun.sh                 # full run (GPU)
#   QUICK=1 bash speedrun.sh         # tiny settings, checks the pipeline works in a few minutes on CPU
#   OUT=runs/try2 bash speedrun.sh   # keep runs apart
#   GRPO_ARGS="--steps 600 --beta 0.02" bash speedrun.sh   # change one stage's settings
#
# On Colab: put this folder on Google Drive (so checkpoints survive a disconnect), then
#   from google.colab import drive; drive.mount('/content/drive')
#   %cd /content/drive/MyDrive/<this folder>
#   !bash speedrun.sh
set -euo pipefail

OUT=${OUT:-out}
PY=${PY:-python}
export PYTHONUNBUFFERED=1 # print training lines as they happen, not in chunks
mkdir -p "$OUT/logs"
: > "$OUT/logs/times.log"

if [ "${QUICK:-0}" = 1 ]; then
  TOK_ARGS=${TOK_ARGS:-"--max_mb 2 --vocab_size 1024"}
  PRETRAIN_ARGS=${PRETRAIN_ARGS:-"--max_mb 5 --max_iters 300 --eval_interval 150 --eval_iters 5 --emb_dim 64 --n_head 4 --n_layer 2 --batch_size 4 --learning_rate 2e-3"}
  SFT_ARGS=${SFT_ARGS:-"--max_mb 2 --max_iters 100 --eval_interval 50 --eval_iters 5 --batch_size 4 --learning_rate 1e-3"}
  GRPO_ARGS=${GRPO_ARGS:-"--max_mb 2 --steps 4 --prompts_per_step 2 --group_size 4 --max_new_tokens 100 --eval_interval 2 --eval_prompts 4"}
  DPO_ARGS=${DPO_ARGS:-"--max_mb 2 --n_prompts 20 --min_gap 0.05 --max_new_tokens 100 --steps 10 --batch_size 4 --eval_interval 5 --eval_prompts 4"}
  EVAL_ARGS=${EVAL_ARGS:-"--n_prompts 10 --max_new_tokens 100 --pretrain_iters 3"}
else
  # empty = the defaults written in each script
  TOK_ARGS=${TOK_ARGS:-""}
  PRETRAIN_ARGS=${PRETRAIN_ARGS:-""}
  SFT_ARGS=${SFT_ARGS:-""}
  GRPO_ARGS=${GRPO_ARGS:-""}
  DPO_ARGS=${DPO_ARGS:-""}
  EVAL_ARGS=${EVAL_ARGS:-""}
fi

# run <name> <module> <args...>: prints and saves the output to $OUT/logs/<name>.log, records the time
run() {
  local name=$1; shift
  echo; echo "=================== $name: $*"
  local start=$(date +%s)
  $PY -m "$@" 2>&1 | tee "$OUT/logs/$name.log"
  echo "$name: $(( $(date +%s) - start ))s" | tee -a "$OUT/logs/times.log"
}

run tok      scripts.tok_train --out "$OUT/tokenizer.pt" $TOK_ARGS
run pretrain scripts.pretraining_gpt --tokenizer "$OUT/tokenizer.pt" --out "$OUT/gpt.pt" $PRETRAIN_ARGS
run sft      scripts.sft_gpt  --init "$OUT/gpt.pt" --out "$OUT/sft.pt" $SFT_ARGS
run grpo     scripts.grpo_gpt --init "$OUT/sft.pt" --out "$OUT/grpo.pt" $GRPO_ARGS
# --rebuild_pairs: the SFT model was just retrained, old pairs would be from a different model
run dpo      scripts.dpo_gpt  --init "$OUT/sft.pt" --out "$OUT/dpo.pt" --pairs "$OUT/dpo_pairs.pt" --rebuild_pairs $DPO_ARGS
run eval     scripts.eval_gpt --ckpt "$OUT/gpt.pt" "$OUT/sft.pt" "$OUT/grpo.pt" "$OUT/dpo.pt" $EVAL_ARGS

# ----------------------------------------------------------------------------- report
L="$OUT/logs"
{
  echo "# tinychat speedrun report"
  echo
  echo "- date: $(date)"
  echo "- gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
  echo "- quick mode: ${QUICK:-0}"
  echo
  echo "## Time per stage"
  echo '```'; cat "$L/times.log"; echo '```'
  echo
  echo "## 0. Tokenizer"
  echo '```'; grep -E 'chars per token|trained in' "$L/tok.log"; echo '```'
  echo
  echo "## 1. Pretraining (first and last loss)"
  echo '```'; grep -E 'parameters|^ModelConfig|train tokens' "$L/pretrain.log"; grep '^step' "$L/pretrain.log" | sed -n '1p;$p'; echo '```'
  echo
  echo "## 2. SFT (step 0 = base model on chat data)"
  echo '```'; grep '^step' "$L/sft.log" | sed -n '1p;$p'; echo '```'
  echo
  echo "## 3a. GRPO (held-out eval during training)"
  echo '```'; grep -- '--- eval' "$L/grpo.log"; echo '```'
  echo
  echo "## 3b. DPO (pairs, held-out pairs, eval)"
  echo '```'; grep -E 'train pairs|--- val pairs|--- eval' "$L/dpo.log"; echo '```'
  echo
  echo "## Final comparison"
  echo '```'; sed -n '/prompts x/,$p' "$L/eval.log"; echo '```'
} > "$OUT/report.md"

echo; echo "=================== done, report: $OUT/report.md"; echo
cat "$OUT/report.md"
