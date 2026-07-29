#!/usr/bin/env bash
# One-shot setup for a fresh multi-GPU host, then prints the launch commands.
#   bash setup_gpu.sh            # full setup incl. tokenizing FineWeb-Edu (~30 min)
#   bash setup_gpu.sh --no-data  # skip tokenizing (data already present)
set -euo pipefail
cd "$(dirname "$0")"

echo "=== host ==="
python3 -c "import sys; print('python', sys.version.split()[0])"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv || echo "no nvidia-smi"

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
. .venv/bin/activate
pip install -q --upgrade pip
# torch first, matched to the host CUDA; the default index is usually right on a GPU box
python -c "import torch" 2>/dev/null || pip install torch
pip install -q -r requirements.txt

echo
echo "=== torch / GPU ==="
python - <<'PY'
import torch
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpus', torch.cuda.device_count())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print(f'gpu0 {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}')
    # train.py picks bf16 only on sm_80+; on older cards it uses fp16 + GradScaler,
    # because pre-Ampere bf16 is emulated and measured ~12x slower on a T4.
    print('dtype will be:', 'bfloat16' if cap >= (8, 0) else 'float16')
    print('arch list:', torch.cuda.get_arch_list())
PY

echo
echo "=== correctness checks ==="
python test_residual.py
python test_ddp.py 2>&1 | tail -12

if [ "${1:-}" != "--no-data" ]; then
    echo
    echo "=== data (260M train + 5M val tokens, ~30 min) ==="
    if [ -f data/finewebedu/train.bin ]; then
        echo "already present: $(ls -l data/finewebedu/train.bin | awk '{print $5}') bytes"
    else
        TARGET_TOKENS=260000000 python data/finewebedu/prepare.py
    fi
fi

NG=$(python -c "import torch; print(torch.cuda.device_count())")
cat <<EOF

=== launch ===
wandb login   # or export WANDB_API_KEY=...

tokens/iter must stay 122,880 to stay comparable with every run so far. That is
gradient_accumulation_steps * batch_size * block_size, independent of GPU count -- but
DDP requires accum to be divisible by the GPU count, so keep accum*batch_size = 120:

  GPUs   accum  batch    note
  1      15     8        the single-GPU setting used for all Kaggle runs
  1-8    24     5        divisible by 1,2,3,4,6,8 -- use this for DDP
  8      8      15       fewer, larger micro-batches; needs ~40GB/GPU

Detected $NG GPU(s). Single run:

  torchrun --standalone --nproc_per_node=$NG train.py config/train_finewebedu.py \\
    --batch_size=5 --gradient_accumulation_steps=24 \\
    --residual=dense --lr_schedule=constant --max_iters=2000 --warmup_iters=100 \\
    --seed=1337 --wandb_group=gpu-flat-s1337

Notes:
  - DDP resume is not supported (train.py asserts on it); finish runs in one go.
  - compile=True is the default and wants sm_70+; train.py disables it below that.
  - Seed spread on an identical config measured 0.022 at 2,000 iters, which is the size
    of every architectural effect seen so far. Budget >=3 seeds per reported config.
EOF
