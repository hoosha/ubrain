"""Push the Kaggle kernels, filling in the authenticated username.

  python kaggle/push.py prep     # CPU kernel that builds the token files
  python kaggle/push.py train    # GPU kernel that runs the screening sweep

Metadata is generated here rather than committed so the kernel ids follow
whichever Kaggle account is authenticated.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

KERNELS = {
    'prep': dict(code='prep.py', gpu=False, sources=[]),
    'train': dict(code='train.py', gpu=True, sources=['{user}/ubrain-prep']),
}


def username():
    for var in ('KAGGLE_USERNAME',):
        if os.environ.get(var):
            return os.environ[var]
    path = os.path.expanduser('~/.kaggle/kaggle.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)['username']
    out = subprocess.run(['kaggle', 'config', 'view'], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if 'username' in line:
            return line.split(':')[-1].strip()
    raise SystemExit('could not determine Kaggle username; set KAGGLE_USERNAME')


def main(which):
    spec = KERNELS[which]
    user = username()
    with tempfile.TemporaryDirectory() as tmp:
        # kaggle kernels push requires the code file and metadata in one directory
        code = spec['code']
        with open(os.path.join(HERE, code)) as src, open(os.path.join(tmp, code), 'w') as dst:
            dst.write(src.read())
        meta = {
            'id': f'{user}/ubrain-{which}',
            'title': f'ubrain-{which}',
            'code_file': code,
            'language': 'python',
            'kernel_type': 'script',
            'is_private': True,
            'enable_gpu': spec['gpu'],
            'enable_internet': True,
            'dataset_sources': [],
            'kernel_sources': [s.format(user=user) for s in spec['sources']],
            'competition_sources': [],
        }
        with open(os.path.join(tmp, 'kernel-metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(json.dumps(meta, indent=2))
        subprocess.run(['kaggle', 'kernels', 'push', '-p', tmp], check=True)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in KERNELS:
        raise SystemExit(f'usage: push.py {{{"|".join(KERNELS)}}}')
    main(sys.argv[1])
