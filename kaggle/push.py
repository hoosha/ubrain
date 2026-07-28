"""Push the Kaggle kernels, filling in the authenticated username.

  python kaggle/push.py prep     # CPU kernel that builds the token files
  python kaggle/push.py train    # GPU kernel that runs the screening sweep

Metadata is generated here rather than committed so the kernel ids follow
whichever Kaggle account is authenticated.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BRANCH = 'main'  # the branch kaggle/train.py clones; keep the two in step
# Prefer the CLI installed alongside this interpreter (e.g. inside a venv) over PATH.
KAGGLE = (shutil.which('kaggle', path=os.path.dirname(sys.executable))
          or shutil.which('kaggle') or 'kaggle')

KERNELS = {
    'prep': dict(code='prep.py', gpu=False, sources=[], datasets=[], accelerator=None),
    # Attach data and credentials as datasets, not UI state: secret attachments and
    # kernel-output sources are both dropped when a new kernel version is pushed,
    # whereas dataset_sources travel with this metadata.
    # accelerator: Kaggle otherwise hands out either a T4 or a P100, and its PyTorch
    # build has no sm_60 kernels, so a P100 fails immediately ("no kernel image is
    # available"). Triton also needs sm_70+ for torch.compile.
    'train': dict(code='train.py', gpu=True, sources=[], accelerator='NvidiaTeslaT4',
                  datasets=['{user}/ubrain-finewebedu', '{user}/wandb-key']),
    # a second kernel over the same harness, so it can run while 'train' is still busy
    # instead of re-pushing that one and cancelling its in-flight run
    'train-nowarmup': dict(code='train.py', gpu=True, sources=[], sweep='nowarmup',
                           accelerator='NvidiaTeslaT4',
                           datasets=['{user}/ubrain-finewebedu', '{user}/wandb-key']),
    'train-flat-a': dict(code='train.py', gpu=True, sources=[], sweep='flat_a',
                         accelerator='NvidiaTeslaT4',
                         datasets=['{user}/ubrain-finewebedu', '{user}/wandb-key']),
    'train-flat-b': dict(code='train.py', gpu=True, sources=[], sweep='flat_b',
                         accelerator='NvidiaTeslaT4',
                         datasets=['{user}/ubrain-finewebedu', '{user}/wandb-key']),
}


def username():
    """Resolve the Kaggle username across the OAuth, token-file and env auth paths."""
    if os.environ.get('KAGGLE_USERNAME'):
        return os.environ['KAGGLE_USERNAME']
    legacy = os.path.expanduser('~/.kaggle/kaggle.json')  # retired, still honoured if present
    if os.path.exists(legacy):
        with open(legacy) as f:
            return json.load(f)['username']
    out = subprocess.run([KAGGLE, 'config', 'view'], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if 'username' in line.lower():
            return line.split(':')[-1].strip()
    raise SystemExit(
        'could not determine Kaggle username. Run `kaggle auth login`, or set KAGGLE_USERNAME.')


def assert_remote_has_head():
    """The kernel clones `main` from GitHub, so pushing a kernel while local commits have
    not reached origin/main trains *stale code* -- and the failure surfaces hours later as
    an unknown-config-key crash, or worse, silently as a run of the previous config. This
    has bitten twice; check it instead of remembering it."""
    def git(*a):
        return subprocess.run(['git', *a], capture_output=True, text=True).stdout.strip()
    subprocess.run(['git', 'fetch', '-q', 'origin', BRANCH], check=False)
    head, remote = git('rev-parse', 'HEAD'), git('rev-parse', f'origin/{BRANCH}')
    if not remote:
        raise SystemExit(f'cannot resolve origin/{BRANCH}')
    if head != remote and subprocess.run(
            ['git', 'merge-base', '--is-ancestor', head, remote]).returncode != 0:
        raise SystemExit(
            f'origin/{BRANCH} is at {remote[:7]} and does not contain HEAD ({head[:7]}).\n'
            f'The kernel clones {BRANCH}, so it would train stale code.\n'
            f'Fix:  git branch -f {BRANCH} HEAD && git push origin {BRANCH}')
    dirty = git('status', '--porcelain')
    if dirty:
        print(f'WARNING: uncommitted changes will NOT reach the kernel:\n{dirty}')


def dataset_exists(ref):
    r = subprocess.run([KAGGLE, 'datasets', 'files', ref], capture_output=True, text=True)
    return r.returncode == 0


def main(which):
    spec = KERNELS[which]
    if spec['gpu']:
        assert_remote_has_head()
    user = username()
    with tempfile.TemporaryDirectory() as tmp:
        # kaggle kernels push requires the code file and metadata in one directory
        code = spec['code']
        with open(os.path.join(HERE, code)) as src:
            body = src.read()
        if spec.get('sweep'):   # point this kernel's copy at a different sweep
            body, n = re.subn(r"^ACTIVE = .*$", f"ACTIVE = {spec['sweep']!r}", body, flags=re.M)
            if n != 1:
                raise SystemExit(f'expected one ACTIVE line in {code}, found {n}')
        with open(os.path.join(tmp, code), 'w') as dst:
            dst.write(body)
        meta = {
            'id': f'{user}/ubrain-{which}',
            'title': f'ubrain-{which}',
            'code_file': code,
            'language': 'python',
            'kernel_type': 'script',
            'is_private': True,
            'enable_gpu': spec['gpu'],
            'enable_internet': True,
            # skip datasets that do not exist yet, otherwise the push itself is rejected
            'dataset_sources': [d for d in (x.format(user=user) for x in spec['datasets'])
                                if dataset_exists(d)],
            'kernel_sources': [s.format(user=user) for s in spec['sources']],
            'competition_sources': [],
        }
        with open(os.path.join(tmp, 'kernel-metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(json.dumps(meta, indent=2))
        cmd = [KAGGLE, 'kernels', 'push', '-p', tmp]
        if spec.get('accelerator'):
            cmd += ['--accelerator', spec['accelerator']]
        print('$', ' '.join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in KERNELS:
        raise SystemExit(f'usage: push.py {{{"|".join(KERNELS)}}}')
    main(sys.argv[1])
