#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
LOG_DIR="$ROOT_DIR/logs/uspto_mit_full"
mkdir -p "$LOG_DIR" "$ROOT_DIR/models" "$ROOT_DIR/data"

export PROJECT_ROOT="$ROOT_DIR"
export DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
export CONFIGS_DIR="$ROOT_DIR/configs"
export CONFIG_DIR="$ROOT_DIR/configs"
export LOGS_DIR="$LOG_DIR"
export MODELS_DIR="$ROOT_DIR/models"
export RANDOM_SEED="${RANDOM_SEED:-132435}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export N_JOBS="${N_JOBS:-4}"

python bin/acquire.py uspto_mit 2>&1 | tee "$LOG_DIR/acquire.log"
test -s "$DATA_DIR/uspto_mit/feat/data/train.txt" || { echo "Missing raw train split" >&2; exit 1; }
test -s "$DATA_DIR/uspto_mit/x.tsv" || { echo "Missing processed x.tsv" >&2; exit 1; }
test -s "$DATA_DIR/uspto_mit/metadata.tsv" || { echo "Missing processed metadata.tsv" >&2; exit 1; }

N_JOBS=${N_JOBS:-4} python bin/featurize.py uspto_mit megan_for_8_dfs_cano 2>&1 | tee "$LOG_DIR/featurize.log"
FEAT_DIR="$DATA_DIR/uspto_mit/feat/megan_for_default_8_dfs"
test -s "$FEAT_DIR/metadata.csv" || { echo "Missing featurized metadata.csv" >&2; exit 1; }

python bin/train.py uspto_mit models/uspto_mit_mix 2>&1 | tee "$LOG_DIR/train_mix.log"
test -s models/uspto_mit_mix/model_best.pt || { echo "Missing mixed model_best.pt" >&2; exit 1; }
python bin/train.py uspto_mit_sep models/uspto_mit_sep 2>&1 | tee "$LOG_DIR/train_sep.log"
test -s models/uspto_mit_sep/model_best.pt || { echo "Missing separated model_best.pt" >&2; exit 1; }

python bin/eval.py models/uspto_mit_mix --beam-size 10 --show-every 1000 2>&1 | tee "$LOG_DIR/eval_mix.log"
compgen -G "models/uspto_mit_mix/eval_*.txt" >/dev/null || { echo "Missing mixed eval output" >&2; exit 1; }
compgen -G "models/uspto_mit_mix/pred_*.txt" >/dev/null || { echo "Missing mixed predictions" >&2; exit 1; }
python bin/eval.py models/uspto_mit_sep --beam-size 10 --show-every 1000 2>&1 | tee "$LOG_DIR/eval_sep.log"
compgen -G "models/uspto_mit_sep/eval_*.txt" >/dev/null || { echo "Missing separated eval output" >&2; exit 1; }
compgen -G "models/uspto_mit_sep/pred_*.txt" >/dev/null || { echo "Missing separated predictions" >&2; exit 1; }

echo "USPTO-MIT full pipeline completed. Logs: $LOG_DIR"
