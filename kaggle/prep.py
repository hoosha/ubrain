"""Kaggle script kernel: build the FineWeb-Edu token files.

Runs without a GPU so it does not consume the weekly accelerator quota. The
resulting train.bin/val.bin land in /kaggle/working and become this kernel's
output, which the training kernel attaches instead of re-tokenizing (~35 min)
every time a runtime is recycled.

Requires notebook internet access to be enabled.
"""
import os
import shutil
import subprocess
import sys

REPO = 'https://github.com/hoosha/ubrain.git'
CLONE = '/tmp/ubrain'  # keep the repo out of /kaggle/working so it isn't part of the output
TARGET_TOKENS = os.environ.get('TARGET_TOKENS', '260000000')
VAL_TOKENS = os.environ.get('VAL_TOKENS', '5000000')

subprocess.run(['git', 'clone', '--depth', '1', '-b', 'main', REPO, CLONE], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tiktoken'], check=True)

subprocess.run(
    [sys.executable, 'data/finewebedu/prepare.py'],
    cwd=CLONE, check=True,
    env={**os.environ, 'TARGET_TOKENS': TARGET_TOKENS, 'VAL_TOKENS': VAL_TOKENS},
)

for name in ('train.bin', 'val.bin', 'dataset.json'):
    # shutil.move, not os.replace: /tmp and /kaggle/working are different devices,
    # so a rename raises EXDEV after the (slow) tokenisation has already succeeded.
    shutil.move(os.path.join(CLONE, 'data/finewebedu', name), os.path.join('/kaggle/working', name))

print('prepared:', {f: os.path.getsize(f'/kaggle/working/{f}') for f in ('train.bin', 'val.bin')})
