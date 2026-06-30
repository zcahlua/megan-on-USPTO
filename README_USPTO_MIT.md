# MEGAN on USPTO-MIT / USPTO-480K

This repository (`https://github.com/zcahlua/megan-on-USPTO`) includes native support for the dataset key `uspto_mit`.
The upstream USPTO-MIT / USPTO-480K source describes `USPTO/data.zip` as approximately 480K fully atom-mapped reactions.

**Important direction note:** the repository's USPTO-MIT configs are forward-synthesis configs, not retrosynthesis configs:

- `configs/uspto_mit.gin`: USPTO-MIT forward synthesis, mixed
- `configs/uspto_mit_sep.gin`: USPTO-MIT forward synthesis, separated/reactant-marked atom features

Do not report a USPTO-MIT retrosynthesis run from these configs unless you explicitly reverse the reaction direction and add a separate retrosynthesis config.

## Dataset source

- Dataset name: **USPTO-MIT / USPTO-480K**
- Download URL: `https://github.com/wengong-jin/nips17-rexgen/raw/master/USPTO/data.zip`
- Original dataset repository: `https://github.com/wengong-jin/nips17-rexgen/tree/master/USPTO`

`src/datasets/uspto_mit.py` downloads this zip file into `data/uspto_mit/feat/data.zip`, extracts it under `data/uspto_mit/feat/data/`, and preprocesses the split text files into MEGAN's tabular files.
The downloader retries transient HTTP failures, reuses existing extracted split files, and reports a manual fallback path if GitHub is unavailable.

## Environment setup

The historical upstream dependency stack is old and is most reproducible through conda:

```bash
conda env create -f env.yml
conda activate megan
source env.sh
```

The repository now includes both `env.yml` and `env.yaml`; `env.yml` is the documented setup file. A Dockerfile is also provided for environments where old conda/PyPI packages are difficult to resolve:

```bash
docker build -t megan-uspto-mit .
docker run --gpus all --rm -it -v "$PWD":/workspace/megan-on-USPTO megan-uspto-mit
```

CUDA is recommended for training. TensorFlow is used only for TensorBoard summaries; PyTorch is used for featurization/training/evaluation model execution.

## One-command smoke test

Run:

```bash
scripts/run_uspto_mit_smoke.sh
```

The smoke script uses an isolated smoke data directory (`data_smoke/`) so it does not corrupt full-data artifacts. It:

1. prints Python, RDKit, PyTorch, and CUDA status to `logs/uspto_mit_smoke/environment.txt`;
2. acquires USPTO-MIT;
3. samples a small number of train/valid/test rows;
4. runs featurization with `megan_for_8_dfs_cano`;
5. trains for one tiny epoch with gin overrides;
6. evaluates the tiny model on the validation split with beam size 1;
7. validates expected raw, processed, feature, checkpoint, evaluation, and prediction files.

The smoke test is a functional check only. It is not a meaningful model-quality run.

Useful overrides:

```bash
SMOKE_SAMPLE_SIZE=20 N_JOBS=1 scripts/run_uspto_mit_smoke.sh
```

## Full USPTO-MIT pipeline

Run:

```bash
scripts/run_uspto_mit_full.sh
```

This executes the repository-native full pipeline and writes logs to `logs/uspto_mit_full/`.
Full training may take many hours or days depending on hardware.

Equivalent explicit commands:

```bash
python bin/acquire.py uspto_mit
N_JOBS=${N_JOBS:-4} python bin/featurize.py uspto_mit megan_for_8_dfs_cano
python bin/train.py uspto_mit models/uspto_mit_mix
python bin/train.py uspto_mit_sep models/uspto_mit_sep
python bin/eval.py models/uspto_mit_mix --dataset-key uspto_mit --beam-size 10 --show-every 1000
python bin/eval.py models/uspto_mit_sep --dataset-key uspto_mit --beam-size 10 --show-every 1000
```

## Expected output files

After acquisition:

- `data/uspto_mit/feat/data/train.txt`
- `data/uspto_mit/feat/data/valid.txt`
- `data/uspto_mit/feat/data/test.txt`
- `data/uspto_mit/x.tsv`
- `data/uspto_mit/metadata.tsv`
- `data/uspto_mit/default_split.csv`

After featurization:

- `data/uspto_mit/feat/megan_for_default_8_dfs/metadata.csv`
- `data/uspto_mit/feat/megan_for_default_8_dfs/sample_data.npz`
- `data/uspto_mit/feat/megan_for_default_8_dfs/nodes.npz`
- `data/uspto_mit/feat/megan_for_default_8_dfs/adj.npz`
- action and property vocabulary JSON files in the same feature directory

After training:

- `models/uspto_mit_mix/model.pt`
- `models/uspto_mit_mix/model_best.pt`
- `models/uspto_mit_sep/model.pt`
- `models/uspto_mit_sep/model_best.pt`

After evaluation:

- `models/uspto_mit_mix/eval_*.txt`
- `models/uspto_mit_mix/pred_*.txt`
- `models/uspto_mit_sep/eval_*.txt`
- `models/uspto_mit_sep/pred_*.txt`

## Hardware expectations

- Acquisition requires enough disk for the zip, extracted USPTO-MIT split files, and processed TSVs.
- Full featurization is CPU and memory intensive; set `N_JOBS` to fit your machine.
- Full training is designed for CUDA-capable PyTorch and may take many hours. The historical README reports approximately 10 hours for USPTO-50k and approximately 60 hours for USPTO-FULL on a GTX 1070; USPTO-MIT sits between these workloads but remains a long run.
- Evaluation can also take a long time for large beam sizes.

## Failure behavior and manual fallback

The scripts use `set -euo pipefail` and validate outputs after each stage. If a stage fails, inspect the matching log under `logs/uspto_mit_smoke/` or `logs/uspto_mit_full/`.

If GitHub downloads fail, manually download:

```text
https://github.com/wengong-jin/nips17-rexgen/raw/master/USPTO/data.zip
```

and place it at the path shown in the error message, typically `data/uspto_mit/feat/data.zip`, then rerun acquisition. Acquisition preserves that archive after extraction, and reruns skip download/unzip if `train.txt`, `valid.txt`, and `test.txt` already exist under `data/uspto_mit/feat/data/`.
