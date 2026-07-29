"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import json
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # retained for upstream config compatibility
wandb_group = '' # optional comparison group
wandb_artifact_every_evals = 0 # upload ckpt.pt as a W&B artifact every N evals (0 = only at the end)
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# architecture / experiment knobs
arch = 'gpt2' # 'gpt2' reproduces the published baseline; 'modern' = RMSNorm+RoPE+SwiGLU+QK-Norm
norm_placement = 'pre' # 'pre' or 'peri'
residual = 'baseline' # baseline|dense|unet|dense_ungated|unet_ungated
# 'learned': zero-init scale on each sublayer output, so layers come alive during
# training instead of being fully on from step 0 (ReZero / LayerScale).
block_scale = 'none' # 'none' | 'learned'
seed = 1337 # vary this for multi-seed runs
# appended to run_name/out_dir; use it when a run differs by something the variant label
# does not encode (e.g. warmup_iters), which would otherwise collide with an earlier run
run_suffix = ''
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
# 'cosine': warmup then cosine to min_lr. 'constant': warmup then hold learning_rate,
# so a variant comparison is not confounded by where each run sits in an anneal -- the
# late-schedule drop is large enough to mask or invent differences between variants.
lr_schedule = 'cosine' # 'cosine' | 'constant'
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'auto' # auto, cpu, mps, cuda, cuda:0, ...
dtype = 'auto' # auto, float32, bfloat16, or float16 (the latter uses a GradScaler)
compile = True # use PyTorch 2.0 to compile the model to be faster
unique_out_dir = False # derive a per-config/seed directory; enable for experiment sweeps
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Human-readable variant labels for run names, W&B and out_dir. "gated" vs "sum"/"mean"
# is the distinction that matters once both learned and fixed coefficients exist, and
# "-alpha" is legible where "-ls" was not.
VARIANT_LABEL = {
    'baseline': 'baseline',
    'dense': 'dense-gated',
    'unet': 'unet-gated',
    'dense_ungated': 'dense-mean',
    'unet_ungated': 'unet-sum',
}

def variant_label(residual, block_scale):
    label = VARIANT_LABEL.get(residual, residual)
    return label + ('-alpha' if block_scale == 'learned' else '')


# various inits, derived attributes, I/O setup
if device == 'auto':
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
if unique_out_dir:
    tag = f"{arch}-{norm_placement}-{variant_label(residual, block_scale)}"
    out_dir = os.path.join(out_dir, f"{tag}-L{n_layer}-s{seed}{'-' + run_suffix if run_suffix else ''}")
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    # CUDA only when there is CUDA: this lets the whole DDP path be exercised on CPU with
    # the gloo backend, which is the only way to test it without a multi-GPU host.
    if torch.cuda.is_available():
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0, (
        f'gradient_accumulation_steps={gradient_accumulation_steps} is not divisible by '
        f'world_size={ddp_world_size}. tokens/iter is accum*batch*block regardless of '
        f'world size, so pick an accum that divides your GPU count and keep '
        f'accum*batch_size fixed to hold tokens/iter comparable (e.g. 24x5 = 120 works '
        f'for 1,2,3,4,6,8 GPUs; the single-GPU 15x8 only works for 1,3,5,15).')
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = torch.device(device).type
# Resolve dtype only after the final device is known. This keeps --device=cpu
# deterministic even on a CUDA host, and avoids unsupported MPS autocast modes.
if dtype == 'auto':
    # Require *native* bf16 (Ampere+, SM 8.0). torch.cuda.is_bf16_supported() also
    # returns True for pre-Ampere emulation: on a T4 (SM 7.5) a 4096^2 matmul measured
    # 66ms in bf16 vs 5.2ms in fp16, so trusting it would cost ~12x throughput.
    dtype = ('bfloat16' if torch.cuda.get_device_capability() >= (8, 0) else 'float16') \
        if device_type == 'cuda' else 'float32'
if device_type != 'cuda' and dtype != 'float32':
    print(f"note: {dtype} autocast is not used on {device_type}; using float32")
    dtype = 'float32'
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if dtype == 'float32' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
print(f"resolved device={device} device_type={device_type} dtype={dtype}")
if device_type == 'cuda':
    cap = torch.cuda.get_device_capability()
    print(f"gpu={torch.cuda.get_device_name(0)} capability=sm_{cap[0]}{cap[1]}")
    supported = torch.cuda.get_arch_list()
    if f'sm_{cap[0]}{cap[1]}' not in supported:
        # e.g. a Kaggle P100 (sm_60) against a torch build shipping sm_70+ only. Without
        # this the first compile dies in Inductor with "no kernel image is available".
        raise SystemExit(f"this torch build has no kernels for sm_{cap[0]}{cap[1]} "
                         f"(supports {supported}); request a newer GPU")
    if compile and cap < (7, 0):
        print('disabling torch.compile: Triton requires sm_70+')
        compile = False

# poor man's data loader. Independent generators keep training and both evaluation
# streams identical across architectures, regardless of model-init RNG consumption.
data_dir = os.path.join('data', dataset)
train_rng = torch.Generator().manual_seed(seed + seed_offset)
train_eval_rng = torch.Generator().manual_seed(seed + 10_000 + seed_offset)
val_eval_rng = torch.Generator().manual_seed(seed + 20_000 + seed_offset)
def get_batch(split, rng=None):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    rng = rng or train_rng
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=rng)
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  arch=arch, norm_placement=norm_placement, residual=residual,
                  block_scale=block_scale) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    # A correct DDP resume needs one data/RNG state per rank; loading rank 0's state
    # everywhere silently duplicates batches. Keep single-device resume exact instead.
    assert not ddp, "DDP resume is not yet supported; restart or resume on one device"
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # Backward-compatible with upstream checkpoints created before these knobs existed.
    for k, default in [('arch', 'gpt2'), ('norm_placement', 'pre'), ('residual', 'baseline'),
                       ('block_scale', 'none')]:
        model_args[k] = checkpoint_model_args.get(k, default)
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'arch',
              'norm_placement', 'residual', 'block_scale']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(device_type == 'cuda' and dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    if 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
    if 'rng_state' in checkpoint:
        # torch.load(map_location=device) also moves RNG byte tensors; CPU generators
        # only accept CPU ByteTensor state even when the model lives on MPS/CUDA.
        torch.set_rng_state(checkpoint['rng_state'].cpu())
        train_rng.set_state(checkpoint['train_rng_state'].cpu())
        old_eval_state = checkpoint.get('eval_rng_state')
        train_eval_rng.set_state(checkpoint.get('train_eval_rng_state', old_eval_state).cpu())
        val_eval_rng.set_state(checkpoint.get('val_eval_rng_state', old_eval_state).cpu())
        if device_type == 'cuda' and 'device_rng_state' in checkpoint:
            torch.cuda.set_rng_state(checkpoint['device_rng_state'].cpu())
# keep checkpoint until the prefetched next batch is restored below

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        rng = train_eval_rng if split == 'train' else val_eval_rng
        for k in range(eval_iters):
            X, Y = get_batch(split, rng)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) constant schedule: hold the peak LR after warmup
    if lr_schedule == 'constant':
        return learning_rate
    # 3) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# training loop
if init_from == 'resume' and checkpoint is not None and 'next_batch' in checkpoint:
    X, Y = (t.to(device) for t in checkpoint['next_batch'])
else:
    X, Y = get_batch('train') # fetch the very first batch
checkpoint = None # free up resume state
t0 = time.time()
t_start = t0
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

# --- run metrics -------------------------------------------------------------
# Log every budget axis so "did the smaller model match the baseline?" stays
# answerable on whichever axis matters (params / FLOPs / memory / wall-clock)
# without re-running anything. These can disagree: cross-depth skips add ~no
# parameters but do keep early block outputs alive, which costs memory.
run_cfg = raw_model.config # authoritative on resume; CLI globals may describe another model
variant = variant_label(run_cfg.residual, run_cfg.block_scale)
run_name = (f"{run_cfg.arch}-{run_cfg.norm_placement}-{variant}-L{run_cfg.n_layer}-s{seed}"
            f"{'-' + run_suffix if run_suffix else ''}")
n_params = raw_model.get_num_params()
total_params = raw_model.get_num_params(non_embedding=False)
flops_per_token = raw_model.flops_per_token()

def peak_memory_bytes():
    if device_type == 'cuda':
        return torch.cuda.max_memory_allocated()
    if device_type == 'mps':
        return torch.mps.driver_allocated_memory()
    return 0

def log_metrics(**kw):
    if not master_process:
        return
    record = dict(run=run_name, variant=variant, arch=run_cfg.arch,
                  norm_placement=run_cfg.norm_placement,
                  residual=run_cfg.residual, block_scale=run_cfg.block_scale,
                  n_layer=run_cfg.n_layer, n_embd=run_cfg.n_embd,
                  seed=seed, dataset=dataset,
                  device_type=device_type, dtype=dtype, params=n_params,
                  total_params=total_params, flops_per_token=flops_per_token,
                  peak_mem_bytes=peak_memory_bytes(),
                  wallclock_s=time.time() - t_start, **kw)
    with open('runs.jsonl', 'a') as f:
        f.write(json.dumps(record, default=float) + '\n')

if master_process:
    print(f"run {run_name}: params={n_params:,} flops/token={flops_per_token:,} tokens/iter={tokens_per_iter:,}")
if wandb_log and master_process:
    import wandb
    wandb_id_path = os.path.join(out_dir, 'wandb_id.txt')
    wandb_id = open(wandb_id_path).read().strip() if os.path.exists(wandb_id_path) else wandb.util.generate_id()
    with open(wandb_id_path, 'w') as f:
        f.write(wandb_id)
    wandb.init(project=wandb_project, name=run_name, id=wandb_id, resume='allow',
               config={**config, 'device': device, 'dtype': dtype, 'out_dir': out_dir,
                       'params': n_params, 'total_params': total_params,
                       'flops_per_token': flops_per_token, 'variant': variant},
               group=wandb_group or None,
               # tags so the UI can filter on either experimental axis directly
               tags=['controlled-comparison', dataset, run_cfg.arch, f'variant:{variant}',
                     f'skip:{run_cfg.residual}', f'alpha:{run_cfg.block_scale}'])

eval_count = 0

def gate_metrics():
    """Per-edge skip-gate values and the gradient actually reaching them.

    Zero-init gates mean a variant that never opens them is identical to the baseline,
    so a null result is ambiguous without this: "no gradient arrived" (a bug) has to be
    distinguishable from "gradient arrived and the network declined the edge".

    The gradient proxy is Adam's second moment rather than p.grad, which is None here
    because zero_grad(set_to_none=True) ran after the last step; sqrt(exp_avg_sq) is
    also a smoothed estimate rather than one micro-batch's noise.
    """
    if raw_model.skip_gate is None:
        return {}, None
    out, vals, grads = {}, [], []
    for i, g in enumerate(raw_model.skip_gate):
        if g.numel() == 0:
            continue
        v = g.detach().float().cpu()
        out[f'gate/rms/L{i}'] = v.pow(2).mean().sqrt().item()
        out[f'gate/absmax/L{i}'] = v.abs().max().item()
        # every edge individually: which ones open is the result, not an aggregate of it
        for k, s in enumerate(raw_model.skip_src[i]):
            out[f'gate/e/L{i}_t{s}'] = v[k].item()
        vals.append(v)
        st = optimizer.state.get(g, {})
        if 'exp_avg_sq' in st:
            gr = st['exp_avg_sq'].detach().float().cpu().sqrt()
            out[f'gate/grad_rms/L{i}'] = gr.mean().item()
            grads.append(gr)
    if not vals:
        return {}, None
    # ungated variants hold their coefficients fixed, so there is no optimizer state and
    # no gradient to report; say so rather than printing an absence that reads as a fault
    out['gate/trainable'] = int(any(g.requires_grad for g in raw_model.skip_gate))
    allv = torch.cat(vals)
    out['gate/rms_all'] = allv.pow(2).mean().sqrt().item()
    out['gate/absmax_all'] = allv.abs().max().item()
    # a run whose gates never leave zero is a baseline in disguise; make that one number
    out['gate/frac_open'] = (allv.abs() > 0.01).float().mean().item()
    if grads:
        out['gate/grad_rms_all'] = torch.cat(grads).mean().item()
        out['gate/grad_dead_edges'] = int((torch.cat(grads) == 0).sum())
    return out, allv

def block_scale_metrics():
    """Per-sublayer alpha and the gradient reaching it.

    This is the primary observable for the "layers come alive" question: alpha=0 means
    the sublayer is switched off entirely, so the trajectory says *which* layers wake up
    and *when*. Same Adam-second-moment gradient proxy and the same reason as
    gate_metrics(): p.grad is None at eval time.
    """
    blocks = raw_model.transformer.h
    if not getattr(blocks[0], 'scaled', False):
        return {}, None
    out, vals, grads, prof = {}, [], [], []
    for i, b in enumerate(blocks):
        for tag, p in (('attn', b.ls_1), ('mlp', b.ls_2)):
            v = p.detach().float().cpu()
            out[f'alpha/v/L{i}_{tag}'] = v.mean().item()
            vals.append(v.reshape(-1))
            prof.append((i, v.abs().mean().item()))
            st = optimizer.state.get(p, {})
            if 'exp_avg_sq' in st:
                g = st['exp_avg_sq'].detach().float().cpu().sqrt().reshape(-1)
                out[f'alpha/grad_rms/L{i}_{tag}'] = g.mean().item()
                grads.append(g)
    allv = torch.cat(vals)
    mass = sum(m for _, m in prof)
    out['alpha/mean_abs'] = allv.abs().mean().item()
    out['alpha/max_abs'] = allv.abs().max().item()
    # total computation switched on: at 0 the net is a pure residual chain, at 24 every
    # sublayer contributes at unit scale
    out['alpha/sum_abs'] = mass
    out['alpha/frac_alive'] = (allv.abs() > 0.01).float().mean().item()
    # where the live mass sits in depth: below (n_layer-1)/2 means shallow layers woke
    # first, above it means deep ones did
    out['alpha/depth_centroid'] = (sum(i * m for i, m in prof) / mass) if mass else 0.0
    if grads:
        g = torch.cat(grads)
        out['alpha/grad_rms_all'] = g.mean().item()
        out['alpha/grad_dead'] = int((g == 0).sum())
    return out, allv


def upload_checkpoint(reason):
    """Persist ckpt.pt to W&B so a run survives an ephemeral runtime being reclaimed."""
    if not (wandb_log and master_process):
        return
    path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(path):
        return
    artifact = wandb.Artifact(f'ckpt-{run_name}', type='model',
                              metadata={'iter': iter_num, 'reason': reason,
                                        'best_val_loss': float(best_val_loss)})
    artifact.add_file(path)
    wandb.log_artifact(artifact)

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        gates, gate_values = gate_metrics()
        alphas, alpha_values = block_scale_metrics()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
              # 'grad n/a' only at iter 0, before the optimizer has any state to read;
              # printing 0 there would look identical to the dead-gradient failure case
              + (f", gate rms {gates['gate/rms_all']:.4f}"
                 + (f", grad {gates['gate/grad_rms_all']:.2e}, dead {gates['gate/grad_dead_edges']}"
                    if 'gate/grad_rms_all' in gates
                    else ('' if not gates['gate/trainable'] else ', grad n/a')) if gates else '')
              + (f", alive {alphas['alpha/sum_abs']:.2f}/{2 * run_cfg.n_layer}"
                 f" depth {alphas['alpha/depth_centroid']:.2f}"
                 + (f" grad {alphas['alpha/grad_rms_all']:.2e} dead {alphas['alpha/grad_dead']}"
                    if 'alpha/grad_rms_all' in alphas else ' grad n/a') if alphas else ''))
        log_metrics(iter=iter_num, train_loss=losses['train'], val_loss=losses['val'],
                    best_val_loss=min(best_val_loss, losses['val'].item()), lr=lr,
                    tokens=iter_num * tokens_per_iter,
                    total_flops=flops_per_token * iter_num * tokens_per_iter,
                    **gates, **alphas)
        if wandb_log:
            wandb.log({
                **gates, **alphas,
                **({'gate/values': wandb.Histogram(gate_values.tolist())} if gates else {}),
                **({'alpha/values': wandb.Histogram(alpha_values.tolist())} if alphas else {}),
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "val/best_loss": min(best_val_loss, losses['val'].item()),
                "lr": lr,
                "tokens": iter_num * tokens_per_iter,
                "estimated_train_flops": flops_per_token * iter_num * tokens_per_iter,
                "system/wallclock_s": time.time() - t_start,
                "system/memory_bytes": peak_memory_bytes(),
                "model/params": n_params,
                "model/total_params": total_params,
                "model/flops_per_token": flops_per_token,
                "mfu": running_mfu*100, # A100-relative; use wall-clock across devices
            }, step=iter_num)
        improved = losses['val'] < best_val_loss
        if improved:
            best_val_loss = losses['val']
        if iter_num > 0:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': {**config, 'device': device, 'dtype': dtype, 'out_dir': out_dir},
                'rng_state': torch.get_rng_state(),
                'train_rng_state': train_rng.get_state(),
                'train_eval_rng_state': train_eval_rng.get_state(),
                'val_eval_rng_state': val_eval_rng.get_state(),
                # X/Y is already prefetched for the next update; preserving it avoids
                # skipping/replacing one batch after resume.
                'next_batch': (X.cpu(), Y.cpu()),
            }
            if device_type == 'cuda':
                checkpoint['device_rng_state'] = torch.cuda.get_rng_state()
            # ckpt.pt is always the latest resumable state; best.pt is the best model.
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            if improved or always_save_checkpoint:
                torch.save(checkpoint, os.path.join(out_dir, 'best.pt'))
            print(f"saved latest checkpoint to {out_dir}")
            eval_count += 1
            if wandb_artifact_every_evals and eval_count % wandb_artifact_every_evals == 0:
                upload_checkpoint('periodic')
    if (iter_num == 0 and eval_only) or iter_num >= max_iters:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        if wandb_log:
            # Per-iteration training loss, so progress is visible between evals rather
            # than only at eval_interval. Same step key as the eval row, so W&B merges.
            wandb.log({'train/batch_loss': lossf, 'system/iter_time_ms': dt * 1000,
                       'tokens': iter_num * tokens_per_iter, 'lr': lr}, step=iter_num)
    iter_num += 1
    local_iter_num += 1

if master_process:
    log_metrics(iter=iter_num, final=True, best_val_loss=best_val_loss,
                tokens=iter_num * tokens_per_iter,
                total_flops=flops_per_token * iter_num * tokens_per_iter,
                # final gate/alpha state is a headline result, not a side metric
                **gate_metrics()[0], **block_scale_metrics()[0])
    print(f"done {run_name}: best val loss {float(best_val_loss):.4f} "
          f"in {time.time() - t_start:.0f}s, peak mem {peak_memory_bytes()/1e9:.2f}GB")

if wandb_log and master_process:
    upload_checkpoint('final')
    wandb.summary.update({'val/best_loss': float(best_val_loss), 'completed_iters': iter_num,
                          'tokens': iter_num * tokens_per_iter,
                          'system/peak_memory_bytes': peak_memory_bytes()})
    wandb.finish()
if ddp:
    destroy_process_group()
