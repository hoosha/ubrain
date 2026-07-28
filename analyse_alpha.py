"""Visualise how block scales wake up over training, from W&B history.

  python analyse_alpha.py <run-name-or-id> [more runs...] -o alpha.html

One heatmap per run: rows are sublayers deepest-last, columns are eval steps, colour is
alpha. The question these runs exist to answer is *which* layers come alive and *when* --
a single scalar per eval cannot show that, and the per-sublayer series are 24 lines that
overplot into noise, so a depth-by-time grid is the readable form.

Alpha is one-sided in practice (it starts at 0 and grows), so this uses the sequential
ramp rather than the diverging one; sign is still shown, since a negative alpha means the
sublayer subtracts its own output and is worth seeing.
"""
import argparse
import sys

import wandb

from analyse_gates import MID, POS, NEG, ink, _rgb

PROJECT = 'hooshaya-ucl/residual-rewiring'


def fetch(api, ref):
    """Accept a run id or a run name."""
    try:
        run = api.run(f'{PROJECT}/{ref}')
    except Exception:
        matches = [r for r in api.runs(PROJECT) if r.name == ref]
        if not matches:
            raise SystemExit(f'no run named or id {ref!r} in {PROJECT}')
        run = matches[0]
    n_layer = run.config['n_layer']
    keys = ['iter'] + [f'alpha/v/L{i}_{t}' for i in range(n_layer) for t in ('attn', 'mlp')]
    rows = [r for r in run.history(keys=keys, pandas=False) if r.get('iter') is not None]
    if not rows:
        raise SystemExit(f'{run.name} logged no alpha history (block_scale off?)')
    return run, n_layer, sorted(rows, key=lambda r: r['iter'])


def colour(v, vmax):
    t = min(abs(v) / vmax, 1.0) if vmax else 0.0
    pole = POS if v >= 0 else NEG
    return '#%02x%02x%02x' % tuple(round(m + (p - m) * t) for m, p in zip(MID, pole))


def panel(run, n_layer, rows, vmax):
    iters = [r['iter'] for r in rows]
    # thin the columns so labels stay legible on a 21-eval run
    show = {it for k, it in enumerate(iters) if k % max(1, len(iters) // 10) == 0 or k == len(iters) - 1}
    head = ''.join(f'<th>{it if it in show else ""}</th>' for it in iters)
    body = []
    for i in range(n_layer):
        for tag in ('attn', 'mlp'):
            tds = []
            for r in rows:
                v = r.get(f'alpha/v/L{i}_{tag}')
                if v is None:
                    tds.append('<td class="na"></td>')
                    continue
                bg = colour(v, vmax)
                tds.append(f'<td style="background:{bg};color:{ink(_rgb(bg))}" '
                           f'title="L{i} {tag} @ {r["iter"]}: {v:+.4f}"></td>')
            body.append(f'<tr><th>L{i} {tag}</th>{"".join(tds)}</tr>')
    final = [r for r in rows][-1]
    tot = sum(abs(final.get(f'alpha/v/L{i}_{t}') or 0)
              for i in range(n_layer) for t in ('attn', 'mlp'))
    return (f'<figure><figcaption>{run.name}<br><span class=sub>final &Sigma;|&alpha;| = '
            f'{tot:.2f} of {2 * n_layer} &middot; val {run.summary.get("val/loss"):.4f}'
            f'</span></figcaption>'
            f'<table><tr><th></th>{head}</tr>{"".join(body)}</table></figure>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('-o', '--out', default='alpha.html')
    a = ap.parse_args()
    api = wandb.Api()
    fetched = [fetch(api, r) for r in a.runs]
    vmax = max(abs(v) for _run, n, rows in fetched for r in rows
               for k, v in r.items() if k.startswith('alpha/v/') and v is not None) or 1.0
    panels = ''.join(panel(*f, vmax) for f in fetched)
    with open(a.out, 'w') as f:
        f.write(f"""<!doctype html><meta charset=utf-8><title>block scale wake-up</title>
<style>
 :root {{ --surface:#fcfcfb; --ink:#0b0b0b; --muted:#898781; }}
 @media (prefers-color-scheme:dark) {{
   :root {{ --surface:#1a1a19; --ink:#fff; --muted:#898781; }} }}
 body {{ background:var(--surface); color:var(--ink); margin:32px;
        font:13px/1.5 ui-sans-serif,system-ui,sans-serif; }}
 h1 {{ font-size:16px; font-weight:600; margin:0 0 4px; }}
 p.lead {{ color:var(--muted); margin:0 0 24px; max-width:70ch; }}
 .grid {{ display:flex; flex-direction:column; gap:32px; }}
 figure {{ margin:0; }} figcaption {{ margin-bottom:8px; }}
 .sub {{ color:var(--muted); }}
 table {{ border-collapse:separate; border-spacing:1px; }}
 th {{ color:var(--muted); font-weight:400; font-size:10px; text-align:right;
       padding-right:4px; white-space:nowrap; }}
 td {{ width:20px; height:13px; border-radius:2px; }}
 td.na {{ background:transparent; }}
</style>
<h1>Block-scale wake-up &mdash; rows are sublayers (deepest last), columns are eval steps</h1>
<p class=lead>Colour is &alpha; on a shared scale to &plusmn;{vmax:.3f}. &alpha;=0 means the
sublayer is switched off and contributes nothing; the network starts as a pure chain of
the ordinary intra-block residuals and layers switch on from there. Blue is positive, red
negative (the sublayer subtracting its own output). Hover for exact values.</p>
<div class=grid>{panels}</div>
""")
    print(f'wrote {a.out} ({len(fetched)} run(s), vmax={vmax:.4f})')
    for run, n_layer, rows in fetched:
        f = rows[-1]
        prof = [(i, sum(abs(f.get(f'alpha/v/L{i}_{t}') or 0) for t in ('attn', 'mlp')))
                for i in range(n_layer)]
        print(f'\n{run.name}  (iter {f["iter"]})')
        for i, m in prof:
            print(f'  L{i:<3d} {"#" * int(round(m / max(x for _, x in prof) * 40)):40s} {m:.4f}')


if __name__ == '__main__':
    main()
