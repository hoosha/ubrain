"""
Checks that make the skip-connection experiment interpretable. Run: python test_residual.py

The load-bearing one is init equivalence: a gated variant must produce a loss
IDENTICAL to the baseline at step 0. Without that, a measured difference later
cannot be separated from the variant having started somewhere else.
"""
import torch
from model import GPT, GPTConfig, skip_sources

CFG = dict(block_size=32, vocab_size=97, n_layer=6, n_head=4, n_embd=64, dropout=0.0)


def build(seed, **kw):
    torch.manual_seed(seed)
    return GPT(GPTConfig(**CFG, **kw))


def output_of(model, seed=1234):
    torch.manual_seed(seed)
    idx = torch.randint(0, CFG['vocab_size'], (2, CFG['block_size']))
    model.eval()
    with torch.no_grad():
        return model(idx, idx)


def loss_of(model, seed=1234):
    return output_of(model, seed)[1].item()


def test_topology():
    assert skip_sources('baseline', 4) == [[], [], [], []]
    # dense: block i reads every earlier stream state (outs[0] is the embedding)
    assert skip_sources('dense', 4) == [[], [0], [0, 1], [0, 1, 2]]
    # unet: second half mirrors the first. Source must always be strictly earlier
    # than the current block, else the skip degenerates into doubling x.
    unet = skip_sources('unet', 12)
    assert unet[:6] == [[]] * 6
    assert unet[6:] == [[5], [4], [3], [2], [1], [0]]
    for depth in (5, 12):
        for i, src in enumerate(skip_sources('unet', depth)):
            assert all(s < i for s in src), f"block {i} reads a non-earlier state {src}"
    for name in ('baseline', 'dense', 'unet', 'dense_ungated'):
        assert len(skip_sources(name, 8)) == 8
    try:
        skip_sources('nonsense', 4)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown variant should raise")


def test_init_equivalence():
    """Zero-init gates => variant is numerically identical to baseline at step 0."""
    for arch in ('gpt2', 'modern'):
        for placement in ('pre', 'peri'):
            base_logits, base_loss = output_of(build(0, arch=arch, norm_placement=placement, residual='baseline'))
            for variant in ('dense', 'unet'):
                logits, loss = output_of(build(0, arch=arch, norm_placement=placement, residual=variant))
                assert torch.equal(logits, base_logits) and torch.equal(loss, base_loss), (
                    f"{arch}/{placement}/{variant} must be bit-identical to baseline at init"
                )
    # the ungated variant deliberately breaks equivalence -- that is its purpose
    base = loss_of(build(0, residual='baseline'))
    ungated = loss_of(build(0, residual='dense_ungated'))
    assert ungated != base, "dense_ungated should NOT match baseline at init"


def test_gates_train_and_are_undecayed():
    """Gates must receive gradient, and must escape weight decay (which would
    pull them back to zero and fight the experiment)."""
    model = build(0, residual='dense')
    loss = model(torch.randint(0, CFG['vocab_size'], (2, CFG['block_size'])).clone(),
                 torch.randint(0, CFG['vocab_size'], (2, CFG['block_size'])))[1]
    loss.backward()
    grads = [g.grad for g in model.skip_gate if g.numel()]
    assert grads and all(g is not None and torch.any(g != 0) for g in grads), \
        "zero-init gates must still receive nonzero gradient"

    # every gate is 1-D, so nanoGPT's existing configure_optimizers routes it to nodecay
    assert all(g.dim() < 2 for g in model.skip_gate)
    assert all(g.numel() > 0 for g in model.skip_gate if g.requires_grad)

    ungated = build(0, residual='dense_ungated')
    assert all(not g.requires_grad for g in ungated.skip_gate)
    trainable = {id(p) for p in ungated.parameters() if p.requires_grad}
    assert not any(id(g) in trainable for g in ungated.skip_gate)


def test_no_wpe_under_rope():
    """get_num_params() used to unconditionally subtract wpe, which RoPE removes."""
    modern = build(0, arch='modern')
    assert 'wpe' not in modern.transformer
    assert modern.get_num_params() > 0
    assert 'wpe' in build(0, arch='gpt2').transformer
    modern.crop_block_size(16) # must not blow up without wpe
    assert modern.config.block_size == 16


def test_shallow_variant_vs_deep_baseline_params():
    """The headline comparison: skips must not smuggle in parameters."""
    deep = GPT(GPTConfig(**{**CFG, 'n_layer': 12}, arch='modern', residual='baseline'))
    shallow_plain = GPT(GPTConfig(**{**CFG, 'n_layer': 6}, arch='modern', residual='baseline'))
    shallow_dense = GPT(GPTConfig(**{**CFG, 'n_layer': 6}, arch='modern', residual='dense'))
    added = shallow_dense.get_num_params() - shallow_plain.get_num_params()
    assert added == sum(range(6)), f"dense should add exactly one scalar per skip, got {added}"
    assert shallow_dense.get_num_params() < deep.get_num_params()


def test_gate_gradients_are_nonzero():
    """A zero-init variant that never receives gate gradient is a baseline in disguise.
    Catch that here rather than after a multi-hour run reports a false null result."""
    for variant in ('dense', 'unet'):
        torch.manual_seed(1337)
        m = GPT(GPTConfig(n_layer=6, n_head=2, n_embd=64, block_size=32,
                          arch='modern', residual=variant, dropout=0.0, vocab_size=64))
        x = torch.randint(0, 64, (2, 32))
        m(x, x)[1].backward()
        edges = 0
        for i, g in enumerate(m.skip_gate):
            if g.numel() == 0:
                continue
            assert g.grad is not None, f"{variant} L{i}: no gradient reached the gate"
            assert (g.grad != 0).all(), f"{variant} L{i}: zero gate gradient {g.grad}"
            edges += g.numel()
        assert edges == sum(map(len, skip_sources(variant, 6))), variant
        print(f"  {variant}: all {edges} gate gradients nonzero at init")


if __name__ == '__main__':
    import contextlib, io
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            with contextlib.redirect_stdout(io.StringIO()): # models print their param count
                fn()
            print(f"ok  {name}")
    print("all checks passed")
