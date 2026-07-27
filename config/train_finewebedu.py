# The config the REPORTED results come from: modern baseline, 12 layers, data-rich.
#
# Sized so ~490M tokens is roughly Chinchilla-optimal for the non-embedding param
# count (~24 tokens/param) rather than badly undertrained -- val loss needs to track
# train loss, otherwise architecture deltas just measure regularization.
# 12 layers also gives the U-Net mirror pattern 6 real skips to work with; the
# shakespeare_char default (6 layers) leaves almost no depth to span.
#
# Vary the experiment with CLI overrides, e.g.:
#   python train.py config/train_finewebedu.py --residual=dense
#   python train.py config/train_finewebedu.py --residual=unet --norm_placement=peri
#   python train.py config/train_finewebedu.py --n_layer=6 --residual=dense   # shallow-vs-deep
# and 3 seeds per reported config: --seed=1337 / 1338 / 1339
#
# batch_size/gradient_accumulation_steps are the knobs to retune per GPU; keep their
# product (tokens per iter) fixed so runs stay comparable across machines.

out_dir = 'out-finewebedu'
eval_interval = 200
eval_iters = 100
log_interval = 20
always_save_checkpoint = False

wandb_log = True
wandb_project = 'residual-rewiring'
wandb_run_name = 'finewebedu'

dataset = 'finewebedu'
unique_out_dir = True # prevent seed/variant checkpoints from overwriting each other
# Sized for a 16GB GPU (measured 8.35GB peak on a Colab T4). Throughput is flat
# across micro-batch sizes here, so keep the product batch_size*accum = 120 fixed
# when retuning per GPU; that preserves 122,880 tokens per optimizer step.
gradient_accumulation_steps = 15
batch_size = 8
block_size = 1024
# -> 122,880 tokens per iter

# modern baseline (RMSNorm + RoPE + SwiGLU + QK-Norm, bias=False forced by arch)
arch = 'modern'
norm_placement = 'pre'
residual = 'baseline'

n_layer = 12
n_head = 6
n_embd = 384
dropout = 0.0 # data-rich regime: dropout would confound the architecture comparison

learning_rate = 6e-4
max_iters = 4000 # ~491M tokens
lr_decay_iters = 4000
min_lr = 6e-5
warmup_iters = 200
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
