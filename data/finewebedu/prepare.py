"""
FineWeb-Edu -> nanoGPT's train.bin / val.bin (raw np.uint16, GPT-2 BPE).

Streams the dataset instead of downloading the whole sample-10BT (~27GB):

  python data/finewebedu/prepare.py                 # 500M train + 5M val
  TARGET_TOKENS=100000000 python data/finewebedu/prepare.py

Validation is the first VAL_TOKENS of the deterministic stream and training starts
at the immediately following token. Files are exact token counts, so the split can
cut a document; no tokens are discarded or shared between splits.
"""
import json
import os
import sys

import numpy as np
import tiktoken
from datasets import load_dataset

TARGET_TOKENS = int(os.environ.get('TARGET_TOKENS', 500_000_000))
VAL_TOKENS = int(os.environ.get('VAL_TOKENS', 5_000_000))
NAME = os.environ.get('FINEWEB_SUBSET', 'sample-10BT')
# Pin the dataset so repeated experiments consume identical bytes.
REVISION = os.environ.get('FINEWEB_REVISION', '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9')
HERE = os.path.dirname(__file__)

enc = tiktoken.get_encoding('gpt2')
EOT = enc._special_tokens['<|endoftext|>']
assert TARGET_TOKENS > 0 and VAL_TOKENS > 0


def token_stream():
    ds = load_dataset('HuggingFaceFW/fineweb-edu', name=NAME, split='train',
                      revision=REVISION, streaming=True)
    for doc in ds:
        yield from enc.encode_ordinary(doc['text'])
        yield EOT


def write_split(stream, path, n_tokens):
    """Write exactly n_tokens from one continuous stream to a temporary file."""
    tmp = path + '.tmp'
    written = 0
    with open(tmp, 'wb') as f:
        while written < n_tokens:
            chunk = np.fromiter(stream, dtype=np.uint16, count=min(1_000_000, n_tokens - written))
            if not len(chunk):
                raise RuntimeError(f'stream exhausted at {written:,}/{n_tokens:,} tokens')
            chunk.tofile(f)
            written += len(chunk)
            print(f'  {os.path.basename(path)}: {written:,}/{n_tokens:,} tokens', end='\r')
    print()
    return tmp


if __name__ == '__main__':
    stream = token_stream()
    print(f'writing {VAL_TOKENS:,} val + {TARGET_TOKENS:,} train tokens from fineweb-edu/{NAME}@{REVISION}')
    val = os.path.join(HERE, 'val.bin')
    train = os.path.join(HERE, 'train.bin')
    val_tmp = write_split(stream, val, VAL_TOKENS)
    train_tmp = write_split(stream, train, TARGET_TOKENS)
    assert os.path.getsize(val_tmp) == 2 * VAL_TOKENS
    assert os.path.getsize(train_tmp) == 2 * TARGET_TOKENS
    os.replace(val_tmp, val)
    os.replace(train_tmp, train)
    with open(os.path.join(HERE, 'dataset.json'), 'w') as f:
        json.dump(dict(dataset='HuggingFaceFW/fineweb-edu', subset=NAME, revision=REVISION,
                       tokenizer='gpt2', val_tokens=VAL_TOKENS, train_tokens=TARGET_TOKENS), f, indent=2)
    # no meta.pkl: train.py falls back to vocab_size 50304, GPT-2 BPE padded for efficiency
    print('done. train.py will use vocab_size=50304')
    # The HF streaming reader keeps a background thread alive; interpreter finalization
    # can abort in it (SIGABRT) long after the outputs above are written and verified,
    # which would otherwise report a false failure. Exit now that the work is done.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
