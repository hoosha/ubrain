"""DDP sanity check for every residual variant.

  python test_ddp.py                                   # world_size 1, gloo, no networking
  torchrun --nproc_per_node=8 test_ddp.py              # the real check on a multi-GPU host

What this is actually for: DDP's reducer raises if a parameter has requires_grad=True but
receives no gradient, and it errors on zero-element parameters. This repo has both shapes
lying around -- skip_gate is a ParameterList with empty entries for blocks that have no
skips, and ungated variants hold their coefficients frozen -- so the variants can break
DDP in ways single-GPU training never reveals. Cheap to check, expensive to discover
halfway through a multi-GPU run.

Forced to loopback because gloo rendezvous resolves the hostname otherwise, which fails on
machines whose hostname has no resolvable address (any corporate laptop).
"""
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig

VARIANTS = [
    ('baseline', 'none'), ('baseline', 'learned'),
    ('unet', 'none'), ('unet', 'learned'),
    ('unet_ungated', 'none'), ('unet_ungated', 'learned'),
    ('dense', 'none'), ('dense_ungated', 'none'),
]


def main():
    launched = 'RANK' in os.environ
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29577')
    cuda = torch.cuda.is_available()
    backend = 'nccl' if cuda and launched else 'gloo'
    dist.init_process_group(backend=backend, rank=rank, world_size=world)
    if cuda:
        torch.cuda.set_device(rank % max(torch.cuda.device_count(), 1))
    dev = f'cuda:{rank % max(torch.cuda.device_count(), 1)}' if cuda else 'cpu'

    if rank == 0:
        print(f'backend={backend} world_size={world} device={dev}')
    failures = []
    for residual, block_scale in VARIANTS:
        torch.manual_seed(1337)
        cfg = GPTConfig(n_layer=6, n_head=2, n_embd=64, block_size=32, vocab_size=64,
                        dropout=0.0, arch='modern', residual=residual,
                        block_scale=block_scale)
        model = GPT(cfg).to(dev)
        ddp_model = DDP(model, device_ids=[torch.cuda.current_device()] if cuda else None)
        opt = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95), 'cuda' if cuda else 'cpu')
        x = torch.randint(0, 64, (2, 32), device=dev)
        try:
            # two steps: the reducer only rebuilds buckets and checks unused params after
            # the first backward, so a single step can pass while the second fails
            for _ in range(2):
                _, loss = ddp_model(x, x)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            status = f'ok    loss={loss.item():.4f}'
        except Exception as e:
            status = f'FAIL  {type(e).__name__}: {str(e).splitlines()[0][:150]}'
            failures.append(f'{residual}/{block_scale}')
        if rank == 0:
            print(f'  {residual:14s} block_scale={block_scale:8s} {status}')
        del ddp_model, model, opt

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print('all variants ok' if not failures else f'FAILURES: {failures}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
