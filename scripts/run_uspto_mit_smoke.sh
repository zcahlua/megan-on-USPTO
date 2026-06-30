#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/logs/uspto_mit_smoke"
DATA_ROOT="${SMOKE_DATA_DIR:-$ROOT_DIR/data_smoke}"
MODEL_DIR="$ROOT_DIR/models/uspto_mit_smoke"
SAMPLE_SIZE="${SMOKE_SAMPLE_SIZE:-12}"

mkdir -p "$LOG_DIR" "$DATA_ROOT" "$ROOT_DIR/models"

export PROJECT_ROOT="$ROOT_DIR"
export DATA_DIR="$DATA_ROOT"
export CONFIGS_DIR="$ROOT_DIR/configs"
export CONFIG_DIR="$ROOT_DIR/configs"
export LOGS_DIR="$LOG_DIR"
export MODELS_DIR="$ROOT_DIR/models"
export RANDOM_SEED="${RANDOM_SEED:-132435}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export N_JOBS="${N_JOBS:-1}"

if ! python - <<'PY' > "$LOG_DIR/environment.txt" 2>&1
import sys
print('python', sys.version.replace('\n', ' '))
missing = []
for name in ['argh', 'gin', 'numpy', 'pandas', 'rdkit', 'torch']:
    try:
        module = __import__(name)
        print(name, getattr(module, '__version__', 'imported'))
    except Exception as exc:
        print(name, 'IMPORT_FAILED', exc)
        missing.append(name)
try:
    import torch
    print('cuda_available', torch.cuda.is_available())
except Exception as exc:
    print('cuda_available', 'UNKNOWN', exc)
if missing:
    raise SystemExit('Missing required Python packages: ' + ', '.join(missing))
PY
then
  cat "$LOG_DIR/environment.txt"
  echo "Environment check failed. Create it with: conda env create -f env.yml && conda activate megan" >&2
  exit 1
fi

python bin/acquire.py uspto_mit 2>&1 | tee "$LOG_DIR/acquire.log"
test -s "$DATA_ROOT/uspto_mit/feat/data/train.txt" || { echo "Missing raw train split" >&2; exit 1; }
test -s "$DATA_ROOT/uspto_mit/x.tsv" || { echo "Missing processed x.tsv" >&2; exit 1; }
test -s "$DATA_ROOT/uspto_mit/metadata.tsv" || { echo "Missing processed metadata.tsv" >&2; exit 1; }
test -s "$DATA_ROOT/uspto_mit/default_split.csv" || { echo "Missing default_split.csv" >&2; exit 1; }

python - "$DATA_ROOT/uspto_mit" "$SAMPLE_SIZE" <<'PY' 2>&1 | tee "$LOG_DIR/sample.log"
import os, sys
import pandas as pd
base, n = sys.argv[1], int(sys.argv[2])
x = pd.read_csv(os.path.join(base, 'x.tsv'), sep='\t')
meta = pd.read_csv(os.path.join(base, 'metadata.tsv'), sep='\t')
split = pd.read_csv(os.path.join(base, 'default_split.csv'))
keep = []
for col in ['train', 'valid', 'test']:
    inds = split.index[split[col] == 1].tolist()[:n]
    keep.extend(inds)
keep = sorted(set(keep))
x.iloc[keep].reset_index(drop=True).to_csv(os.path.join(base, 'x.tsv'), sep='\t')
meta.iloc[keep].reset_index(drop=True).to_csv(os.path.join(base, 'metadata.tsv'), sep='\t')
split.iloc[keep].reset_index(drop=True).to_csv(os.path.join(base, 'default_split.csv'))
print('Smoke dataset rows:', len(keep))
PY

N_JOBS=1 python bin/featurize.py uspto_mit megan_for_8_dfs_cano 2>&1 | tee "$LOG_DIR/featurize.log"
FEAT_DIR="$DATA_ROOT/uspto_mit/feat/megan_for_default_8_dfs"
test -s "$FEAT_DIR/metadata.csv" || { echo "Missing featurized metadata.csv" >&2; exit 1; }
test -s "$FEAT_DIR/sample_data.npz" || { echo "Missing featurized sample_data.npz" >&2; exit 1; }

rm -rf "$MODEL_DIR"
python bin/train.py uspto_mit "$MODEL_DIR" --max_n_epochs 1 --train_samples_per_epoch 4 --valid_samples_per_epoch 4 --batch_size 1 --megan_warmup_epochs 0 2>&1 | tee "$LOG_DIR/train.log"
test -s "$MODEL_DIR/model.pt" || { echo "Missing smoke model checkpoint" >&2; exit 1; }
if [ ! -s "$MODEL_DIR/model_best.pt" ]; then cp "$MODEL_DIR/model.pt" "$MODEL_DIR/model_best.pt"; fi

python bin/eval.py "$MODEL_DIR" --beam-size 1 --beam-batch-size 1 --show-every 1 --split-key valid --max-gen-steps 2 2>&1 | tee "$LOG_DIR/eval.log"
compgen -G "$MODEL_DIR/eval_valid_1_2*.txt" >/dev/null || { echo "Missing eval output" >&2; exit 1; }
compgen -G "$MODEL_DIR/pred_valid_1_2*.txt" >/dev/null || { echo "Missing prediction output" >&2; exit 1; }

echo "USPTO-MIT smoke test completed. Logs: $LOG_DIR"
