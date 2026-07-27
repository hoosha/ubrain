"""Kaggle script kernel: screening sweep over residual topologies on one GPU.

Attaches the prep kernel's output rather than re-tokenizing, reads the W&B key
from Kaggle Secrets (so no key is committed or printed), and runs the three
topologies back to back under one W&B group.

Each run is ~2h at 2,000 iters on a T4, so all three fit inside the 12h session
cap. Runs are independent processes: if one fails the others still execute, and
their W&B history plus checkpoint artifacts survive the session ending.
"""
import os
import subprocess
import sys

REPO = 'https://github.com/hoosha/ubrain.git'
CLONE = '/tmp/ubrain'
DATA_IN = '/kaggle/input/ubrain-prep'
VARIANTS = ('baseline', 'dense', 'unet')
MAX_ITERS = os.environ.get('MAX_ITERS', '2000')
SEED = os.environ.get('SEED', '1337')
GROUP = os.environ.get('WANDB_GROUP', f'finewebedu-screen-s{SEED}')

subprocess.run(['git', 'clone', '--depth', '1', '-b', 'main', REPO, CLONE], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tiktoken', 'wandb'], check=True)

from kaggle_secrets import UserSecretsClient  # noqa: E402  (available only on Kaggle)
try:
    os.environ['WANDB_API_KEY'] = UserSecretsClient().get_secret('WANDB_API_KEY')
except Exception as e:
    # Secrets are stored per account but must be attached to each kernel that reads
    # them, so this is the usual first-run failure. Fail loudly before burning GPU time.
    raise SystemExit(
        f'could not read the WANDB_API_KEY secret ({e}). In this kernel: '
        'Edit -> Add-ons -> Secrets -> toggle WANDB_API_KEY on, then re-run.'
    )

data_dir = os.path.join(CLONE, 'data/finewebedu')
os.makedirs(data_dir, exist_ok=True)
for name in ('train.bin', 'val.bin'):
    src, dst = os.path.join(DATA_IN, name), os.path.join(data_dir, name)
    if not os.path.exists(dst):
        os.symlink(src, dst)
    print(name, os.path.getsize(src), 'bytes')

failures = []
for residual in VARIANTS:
    print(f'=== residual={residual} ===', flush=True)
    result = subprocess.run(
        [sys.executable, 'train.py', 'config/train_finewebedu.py',
         '--device=cuda', f'--residual={residual}', f'--seed={SEED}',
         f'--max_iters={MAX_ITERS}', f'--lr_decay_iters={MAX_ITERS}', '--warmup_iters=100',
         '--out_dir=/kaggle/working/out', f'--wandb_group={GROUP}'],
        cwd=CLONE,
    )
    if result.returncode != 0:
        failures.append(residual)
        print(f'!! residual={residual} exited {result.returncode}', flush=True)

print('done. failures:', failures or 'none')
sys.exit(1 if failures else 0)
