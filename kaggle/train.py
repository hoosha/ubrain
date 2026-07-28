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
# Sweeps, selected by ACTIVE. push.py rewrites the ACTIVE line for a variant kernel,
# so one file serves several sweeps without duplicating the harness below.
SWEEPS = {
    # 2x2 minus the control we already have (plain baseline, val 3.9849 in
    # finewebedu-screen-s1337): {no cross-depth skips, ungated unet skips} x {no block
    # scale, learned}. Run 2 vs run 4 decides whether rewiring adds anything beyond
    # progressive layer wake-up, which is a known-good effect on its own.
    'alive': (
        dict(residual='baseline', block_scale='learned'),      # does wake-up alone help?
        dict(residual='unet_ungated', block_scale='none'),     # do plain-sum skips alone?
        dict(residual='unet_ungated', block_scale='learned'),  # the combination
    ),
    # ReZero's selling point is that zero-init block scales remove the need for LR
    # warmup, so holding warmup_iters=100 may have handicapped both alpha runs -- the
    # 'alive' pair only reached alpha/sum_abs 2.25 of 24. Same two configs with warmup
    # removed; everything else identical, so this is a clean two-cell comparison.
    'nowarmup': (
        dict(residual='baseline', block_scale='learned', warmup_iters=0, run_suffix='nowarm'),
        dict(residual='unet_ungated', block_scale='learned', warmup_iters=0, run_suffix='nowarm'),
    ),
    # Every earlier sweep annealed 6e-4 -> 6e-5 over the run, so a late-schedule drop
    # (~0.09 in the last 400 iters) sat on top of every variant difference. Hold LR flat
    # at 6e-4 instead: loss at any step then reflects optimisation progress rather than
    # schedule position, and no variant can win by interacting with the anneal. Split
    # across two kernels so six ~2h runs finish in ~6h wall-clock, not 12h.
    'flat_a': (
        dict(residual='baseline', run_suffix='flat'),
        dict(residual='dense', run_suffix='flat'),
        dict(residual='unet', run_suffix='flat'),
    ),
    'flat_b': (
        dict(residual='unet_ungated', run_suffix='flat'),
        # zero-init block scales are the case ReZero says needs no warmup
        dict(residual='baseline', block_scale='learned', warmup_iters=0,
             run_suffix='flat-nowarm'),
        dict(residual='unet_ungated', block_scale='learned', warmup_iters=0,
             run_suffix='flat-nowarm'),
    ),
    # The central hypothesis: can a *shallower* model with rewired skips match a deeper
    # sequential baseline? Half the depth against the 12-layer flat baseline, with its own
    # 6-layer control so the comparison is depth-vs-skips rather than depth alone.
    'shallow': (
        dict(residual='baseline', n_layer=6, run_suffix='flat'),
        dict(residual='dense', n_layer=6, run_suffix='flat'),
    ),
}
# per-sweep overrides layered between BASE and each run's own spec
SWEEP_BASE = {k: dict(lr_schedule='constant') for k in ('flat_a', 'flat_b', 'shallow')}
# sweeps that share a W&B group, so runs split across kernels still compare in one place
SWEEP_GROUP = {'flat_a': 'flat', 'flat_b': 'flat', 'shallow': 'flat'}
ACTIVE = 'alive'  # push.py rewrites this line
MAX_ITERS = os.environ.get('MAX_ITERS', '2000')
SEED = os.environ.get('SEED', '1337')
GROUP = os.environ.get('WANDB_GROUP',
                       f'finewebedu-{SWEEP_GROUP.get(ACTIVE, ACTIVE)}-s{SEED}')

subprocess.run(['git', 'clone', '--depth', '1', '-b', 'main', REPO, CLONE], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tiktoken', 'wandb'], check=True)

def wandb_key():
    """Find the W&B key. Pushing a new kernel version via the API drops the UI secret
    attachment, so try several sources before giving up."""
    if os.environ.get('WANDB_API_KEY'):
        return os.environ['WANDB_API_KEY'], 'env'
    # A private dataset in dataset_sources survives `kernels push`, unlike a UI secret.
    # Walk rather than hardcode: the mount is /kaggle/input/datasets/<user>/<slug>/ in
    # some sessions and /kaggle/input/<slug>/ in others, and a wrong guess here costs
    # the whole run's dashboards.
    for root, _dirs, files in os.walk('/kaggle/input'):
        for name in files:
            if name in ('wandb_key.txt', 'wandb_key'):
                with open(os.path.join(root, name)) as f:
                    return f.read().strip(), f'dataset:{os.path.join(root, name)}'
    try:
        from kaggle_secrets import UserSecretsClient  # available only on Kaggle
        return UserSecretsClient().get_secret('WANDB_API_KEY'), 'secret'
    except Exception as e:
        print(f'no W&B key available ({e})', flush=True)
        return None, None


key, source = wandb_key()
if key:
    os.environ['WANDB_API_KEY'] = key
    print(f'W&B key loaded from {source}', flush=True)
else:
    # Train anyway and keep the metrics locally; losing hours of GPU time to a missing
    # credential is worse than losing live dashboards. Sync afterwards with `wandb sync`.
    os.environ['WANDB_MODE'] = 'offline'
    print('WARNING: running with WANDB_MODE=offline; metrics will need syncing', flush=True)

def find_tokens():
    """Locate train.bin/val.bin under /kaggle/input.

    The mount name depends on whether the tokens arrive as a dataset or as another
    kernel's output, and UI-side attachments are not preserved across API pushes, so
    search rather than hard-coding one path.
    """
    for root, _dirs, files in os.walk('/kaggle/input'):
        if 'train.bin' in files and 'val.bin' in files:
            return root
    tree = [os.path.join(r, f) for r, _d, fs in os.walk('/kaggle/input') for f in fs][:40]
    raise SystemExit('could not find train.bin/val.bin under /kaggle/input. Contents:\n  '
                     + ('\n  '.join(tree) or '(empty - no data source attached)'))


DATA_IN = find_tokens()
print('tokens from', DATA_IN, flush=True)
data_dir = os.path.join(CLONE, 'data/finewebedu')
os.makedirs(data_dir, exist_ok=True)
for name in ('train.bin', 'val.bin'):
    src, dst = os.path.join(DATA_IN, name), os.path.join(data_dir, name)
    if not os.path.exists(dst):
        os.symlink(src, dst)
    print(name, os.path.getsize(src), 'bytes', flush=True)

BASE = dict(device='cuda', seed=SEED, max_iters=MAX_ITERS, lr_decay_iters=MAX_ITERS,
            warmup_iters=100, out_dir='/kaggle/working/out', wandb_group=GROUP)

failures = []
for spec in SWEEPS[ACTIVE]:
    args = {**BASE, **SWEEP_BASE.get(ACTIVE, {}), **spec}  # narrowest wins
    label = '/'.join(f'{k}={v}' for k, v in spec.items())
    print(f'=== {label} ===', flush=True)
    result = subprocess.run(
        [sys.executable, 'train.py', 'config/train_finewebedu.py']
        + [f'--{k}={v}' for k, v in args.items()],
        cwd=CLONE,
    )
    if result.returncode != 0:
        failures.append(label)
        print(f'!! {label} exited {result.returncode}', flush=True)

print('done. failures:', failures or 'none')
sys.exit(1 if failures else 0)
