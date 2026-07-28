"""Visualise learned skip-gate matrices from checkpoints.

  python analyse_gates.py out/ckpt.pt                       # one panel
  python analyse_gates.py /tmp/gatechk/v*/ckpt.pt -o g.html # small multiples over training

Writes a standalone HTML heatmap (row = destination block, column = source tap) and
prints a glyph version for terminals. Signed data, so the scale is diverging with a
neutral midpoint and a symmetric range; cell values are printed in-cell because the
question "which edges opened" is answered by the numbers, with colour only guiding
the eye to the pattern.
"""
import argparse
import torch
from model import skip_sources

# Diverging pair: blue (positive) <-> red (negative), neutral gray midpoint.
# Poles and midpoint from the design-system diverging spec; interpolated in sRGB,
# which is close enough for a heatmap whose cells are also labelled.
POS = (0x2a, 0x78, 0xd6)
NEG = (0xe3, 0x49, 0x48)
MID = (0xf0, 0xef, 0xec)
# Magnitude view: sequential, one hue light->dark (never a rainbow), same blue ramp.
SEQ = (0x0d, 0x36, 0x6b)


def ink(rgb):
    """Readable text on an arbitrary cell colour: the deep end of either ramp is far
    too dark for the near-black ink the light cells need."""
    lin = [(c / 255) / 12.92 if c / 255 <= 0.04045 else (((c / 255) + 0.055) / 1.055) ** 2.4
           for c in rgb]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return '#ffffff' if lum < 0.45 else '#0b0b0b'


def load(path):
    """Return a full (n_layer x n_layer) matrix of gate values, None where no edge exists.

    The stored gate vector is indexed by position within skip_src[i], NOT by source
    depth -- those coincide for dense (skip_src[i] == range(i)) but not for U-Net,
    whose single entry belongs in column n_layer-1-i. Map through skip_sources()
    rather than assuming they line up.
    """
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ck['model'].items()}
    n_layer = ck['model_args']['n_layer']
    src = skip_sources(ck['model_args']['residual'], n_layer)
    mat = [[None] * n_layer for _ in range(n_layer)]
    for i in range(n_layer):
        key = f'skip_gate.{i}'
        if key not in sd or not sd[key].numel():
            continue
        for k, s in enumerate(src[i]):
            mat[i][s] = sd[key][k].item()
    return ck['model_args'], ck.get('iter_num', ck.get('iter')), mat


def colour(v, vmax, signed=True):
    """Interpolate midpoint -> pole. Alpha-free so it prints correctly.

    signed=True gives the diverging scale (sign carries meaning); signed=False gives
    the sequential magnitude scale, one hue light->dark.
    """
    t = min(abs(v) / vmax, 1.0) if vmax else 0.0
    pole = (POS if v >= 0 else NEG) if signed else SEQ
    return '#%02x%02x%02x' % tuple(round(m + (p - m) * t) for m, p in zip(MID, pole))


def glyph_panels(mat, n_layer):
    """Two magnitude grids, one per sign. Splitting by sign keeps a monospace
    rendering unambiguous, where a single diverging glyph ramp could not be."""
    # 5 magnitude bands; the top band is separated because two embedding taps are
    # ~3x anything else and merging them into '█' would hide the main result.
    BANDS = ((0.03, '░'), (0.07, '▒'), (0.13, '▓'), (0.25, '█'))
    def cell(v, want_pos):
        if v is None:
            return ' '          # no such edge in this topology
        if abs(v) < 0.01:
            return '·'          # unused edge: sign-agnostic, so it shows in both panels
        if (v >= 0) != want_pos:
            return ' '
        for thr, ch in BANDS:
            if abs(v) < thr:
                return ch
        return '▉'              # >=0.25, off the top of the ramp
    out = []
    for want_pos, title in ((True, 'positive  — block ADDS the earlier state'),
                            (False, 'negative  — block SUBTRACTS it')):
        lines = [title, '       ' + ''.join(f't{j:<3d}' for j in range(n_layer - 1))]
        for i in range(1, n_layer):
            lines.append(f'  B{i:<4d}' + ''.join(cell(mat[i][j], want_pos) * 2 + ' '
                                                 for j in range(n_layer - 1)))
        out.append('\n'.join(lines))
    out.append('  · <0.01 (unused)   ░ <0.03   ▒ <0.07   ▓ <0.13   █ <0.25   ▉ >=0.25')
    return out


def grid(panels, vmax, n_layer, signed):
    cells = []
    for _args, it, mat in panels:
        rows = []
        for i in range(1, n_layer):
            tds = []
            for j in range(n_layer - 1):
                v = mat[i][j]
                if v is None:
                    tds.append('<td class="na"></td>')
                    continue
                bg = colour(v, vmax, signed)
                txt = f'{v:+.2f}' if signed else f'{abs(v):.2f}'
                tds.append(f'<td style="background:{bg};color:{ink(_rgb(bg))}" '
                           f'title="B{i} &larr; t{j}: {v:+.4f}">{txt}</td>')
            rows.append(f'<tr><th>B{i}</th>{"".join(tds)}</tr>')
        head = ''.join(f'<th>t{j}</th>' for j in range(n_layer - 1))
        cells.append(f'<figure><figcaption>iter {it}</figcaption>'
                     f'<table><tr><th></th>{head}</tr>{"".join(rows)}</table></figure>')
    return f'<div class=grid>{"".join(cells)}</div>'


def _rgb(hexstr):
    return tuple(int(hexstr[k:k + 2], 16) for k in (1, 3, 5))


def html(panels, vmax, n_layer):
    steps = [-vmax + 2 * vmax * k / 16 for k in range(17)]
    leg_signed = ''.join(f'<span style="background:{colour(v, vmax)}"></span>' for v in steps)
    leg_abs = ''.join(f'<span style="background:{colour(vmax * k / 16, vmax, False)}"></span>'
                      for k in range(17))
    return f"""<!doctype html><meta charset=utf-8>
<title>skip-gate values</title>
<style>
 :root {{ --surface:#fcfcfb; --ink:#0b0b0b; --muted:#898781; --line:#e1e0d9; }}
 @media (prefers-color-scheme:dark) {{
   :root {{ --surface:#1a1a19; --ink:#fff; --muted:#898781; --line:#2c2c2a; }} }}
 body {{ background:var(--surface); color:var(--ink); margin:32px;
        font:13px/1.5 ui-sans-serif,system-ui,sans-serif; }}
 h1 {{ font-size:16px; font-weight:600; margin:0 0 4px; }}
 p.sub {{ color:var(--muted); margin:0 0 24px; max-width:64ch; }}
 .grid {{ display:flex; flex-wrap:wrap; gap:28px; }}
 figure {{ margin:0; }}
 figcaption {{ color:var(--muted); margin-bottom:6px; }}
 table {{ border-collapse:separate; border-spacing:2px; }}
 th {{ color:var(--muted); font-weight:400; font-size:11px; }}
 td {{ width:44px; height:24px; text-align:center; font-variant-numeric:tabular-nums;
       font-size:10px; border-radius:3px; }}
 td.na {{ background:transparent; }}
 h2 {{ font-size:13px; font-weight:600; margin:36px 0 2px; }}
 h2 + p {{ color:var(--muted); margin:0 0 16px; max-width:64ch; }}
 .legend {{ display:flex; align-items:center; gap:8px; margin-top:16px;
            color:var(--muted); }}
 .legend span {{ width:16px; height:12px; display:inline-block; }}
 .legend .bar {{ display:flex; }}
</style>
<h1>Learned skip-gate values &mdash; row = destination block, column = source tap</h1>
<p class=sub>t0 is the embedding output; t<sub>j</sub> is the output of block
j&minus;1. Blank cells are edges the topology does not contain. Hover any cell for the
exact value &mdash; both views below show the same numbers, so colour never hides
anything.</p>

<h2>Signed &mdash; direction of the edge</h2>
<p>Diverging scale, symmetric at &plusmn;{vmax:.2f}. Blue adds the earlier state,
red subtracts it, gray is an unused edge.</p>
{grid(panels, vmax, n_layer, True)}
<div class=legend>{-vmax:+.2f}<span class=bar>{leg_signed}</span>{vmax:+.2f}
&nbsp;&nbsp;red = subtract &middot; gray = unused &middot; blue = add</div>

<h2>Absolute value &mdash; strength of the edge</h2>
<p>Sequential scale, one hue light&rarr;dark, 0 to {vmax:.2f}. Ignores direction so
the sparsity pattern reads directly: pale cells are edges the network declined.</p>
{grid(panels, vmax, n_layer, False)}
<div class=legend>0.00<span class=bar>{leg_abs}</span>{vmax:.2f}
&nbsp;&nbsp;pale = unused &middot; dark = strong (either sign)</div>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt', nargs='+')
    ap.add_argument('-o', '--out', default='gates.html')
    a = ap.parse_args()
    panels = sorted((load(p) for p in a.ckpt), key=lambda x: x[1] or 0)
    n_layer = panels[0][0]['n_layer']
    vmax = max(abs(v) for _, _, m in panels for row in m for v in row if v is not None)
    with open(a.out, 'w') as f:
        f.write(html(panels, vmax, n_layer))
    print(f'wrote {a.out}  (vmax={vmax:.4f}, {len(panels)} panel(s))\n')
    for p in glyph_panels(panels[-1][2], n_layer):
        print(p, '\n')


if __name__ == '__main__':
    main()
