# ubrain

Experiments on how cross-depth residual connections affect GPT training efficiency, model size, memory use, and validation performance.

The central question is whether dense or sparse U-Net-style skip connections let a shallower transformer match a deeper sequential model at lower parameter or compute cost.

## Model variants

Two architecture baselines are available:

- `--arch=gpt2`: pre-LN GPT-2 with learned absolute position embeddings and GELU MLPs
- `--arch=modern`: RMSNorm, RoPE, SwiGLU, QK-Norm, and bias-free linear layers

Normalization placement:

- `--norm_placement=pre`: `x + sublayer(norm(x))`
- `--norm_placement=peri`: `x + norm(sublayer(norm(x)))`

Cross-depth residual topology:

- `--residual=baseline`: sequential transformer stack
- `--residual=dense`: learned gates from every earlier block state
- `--residual=unet`: learned mirrored skips from the first half to the second half
- `--residual=dense_ungated`: fixed normalized dense sums

Learned skip gates start at zero, making `dense` and `unet` bit-identical to the corresponding baseline at initialization. This isolates learned connectivity from initialization-scale changes.

## Setup

Python 3.10+ and PyTorch 2.x are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Run the invariant checks:

```bash
python test_residual.py
```

## Fast local debugging

Shakespeare-char is useful for checking code paths, but it overfits quickly and is not used for reported architectural conclusions.

```bash
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py \
  --device=auto \
  --compile=False \
  --arch=modern \
  --norm_placement=pre \
  --residual=baseline \
  --dropout=0.0
```

`--device=auto` selects CUDA, then MPS, then CPU.

## FineWeb-Edu experiments

The experiment config uses a pinned FineWeb-Edu revision, GPT-2 BPE tokens, 12 transformer layers, no dropout, unique output directories, and W&B logging.

Prepare 500M training tokens and 5M validation tokens:

```bash
python data/finewebedu/prepare.py
```

Override the dataset size for a smoke test:

```bash
TARGET_TOKENS=1000000 VAL_TOKENS=100000 \
  python data/finewebedu/prepare.py
```

Train a baseline:

```bash
python train.py config/train_finewebedu.py \
  --device=cuda \
  --arch=modern \
  --norm_placement=pre \
  --residual=baseline \
  --seed=1337
```

Change only the residual topology for a controlled comparison:

```bash
python train.py config/train_finewebedu.py --device=cuda --residual=dense --seed=1337
python train.py config/train_finewebedu.py --device=cuda --residual=unet --seed=1337
python train.py config/train_finewebedu.py --device=cuda --residual=dense_ungated --seed=1337
```

Run at least three seeds for results intended for comparison or publication.

## Shallower-model comparison

To test whether skip connectivity compensates for depth, compare a shallower variant against both a depth-matched sequential model and the full-depth baseline:

```bash
# Full-depth reference
python train.py config/train_finewebedu.py --n_layer=12 --residual=baseline --seed=1337

# Depth-matched controls
python train.py config/train_finewebedu.py --n_layer=6 --residual=baseline --seed=1337
python train.py config/train_finewebedu.py --n_layer=6 --residual=dense --seed=1337
python train.py config/train_finewebedu.py --n_layer=6 --residual=unet --seed=1337
```

Compare validation loss against tokens, estimated training FLOPs, parameters, peak memory, and wall-clock time. These metrics can favor different models.

## Logging and checkpoints

Each run records:

- training and validation loss
- best validation loss
- tokens processed
- estimated training FLOPs
- parameter counts
- device and dtype
- memory use and wall-clock time

With `--unique_out_dir=True`, checkpoints are separated by architecture, normalization, residual topology, depth, and seed. `ckpt.pt` is the latest resumable state and `best.pt` is the best evaluated state.

Resume a single-device run with the same configuration and output directory:

```bash
python train.py config/train_finewebedu.py --init_from=resume
```

Exact DDP resume is intentionally disabled until per-rank data and RNG states are checkpointed correctly.

## Repository layout

- `model.py`: GPT architectures and residual topologies
- `train.py`: training, evaluation, checkpointing, metrics, and device handling
- `config/train_finewebedu.py`: primary experiment configuration
- `data/finewebedu/prepare.py`: reproducible streaming dataset preparation
- `test_residual.py`: topology and initialization invariants

## Attribution

The original training stack was derived from [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) at commit `3adf61e`. This repository retains nanoGPT's license while substantially changing the model, experiment controls, data pipeline, checkpointing, metrics, and device support.
