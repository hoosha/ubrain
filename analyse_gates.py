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


def arch_svg(mat, it, n_layer, vmax):
    """The network itself, with each skip drawn as an arc coloured by its gate value.

    Both endpoints sit on the residual stream and the arc bulges left in proportion
    to the span, so U-Net's mirror pattern shows up as nested arches. Colour is the
    same diverging scale as the heatmap; stroke width redundantly encodes |g| so the
    strong edges are legible without relying on colour alone.
    """
    PITCH, TOP, BOX_W, BOX_H = 34, 46, 46, 20
    gutter = 24 + 14 * (n_layer - 1)  # must clear the widest arc: bulge at max span
    x = gutter + BOX_W / 2 + 4        # residual stream, with the arc gutter to its left
    W = x + 62                        # room for the centred bottom label
    H = TOP + PITCH * n_layer + 54

    def y_block(i):
        return TOP + PITCH * i

    def y_tap(i):                     # the wire entering block i
        return y_block(i) - 12

    parts = [f'<svg viewBox="0 0 {W:.0f} {H}" width="{W:.0f}" height="{H}" '
             f'role="img" aria-label="skip topology, arcs coloured by gate value">']
    parts.append(f'<line x1="{x}" y1="{TOP - 26}" x2="{x}" y2="{y_block(n_layer) + 14}" '
                 f'class="stream"/>')
    parts.append(f'<text x="{x}" y="{TOP - 30}" class="lbl" text-anchor="middle">embed</text>')
    parts.append(f'<text x="{x}" y="{y_block(n_layer) + 28}" class="lbl" '
                 f'text-anchor="middle">norm &rarr; logits</text>')

    # weakest first so the strong edges land on top rather than behind the clutter
    edges = [(i, j, v) for i in range(n_layer) for j in range(n_layer)
             if (v := mat[i][j]) is not None]
    for i, j, v in sorted(edges, key=lambda e: abs(e[2])):
        y0, y1 = y_tap(j), y_tap(i)
        bulge = 18 + 14 * abs(i - j)
        parts.append(
            f'<path d="M {x} {y0} Q {x - bulge} {(y0 + y1) / 2:.0f} {x} {y1}" '
            f'stroke="{colour(v, vmax)}" stroke-width="{1 + 4 * abs(v) / vmax:.2f}" '
            f'fill="none"><title>B{i} &larr; t{j}: {v:+.4f}</title></path>')

    for i in range(n_layer):
        parts.append(f'<rect x="{x - BOX_W / 2}" y="{y_block(i)}" width="{BOX_W}" '
                     f'height="{BOX_H}" rx="3" class="blk"/>')
        parts.append(f'<text x="{x}" y="{y_block(i) + 14}" class="blbl" '
                     f'text-anchor="middle">B{i}</text>')
    parts.append('</svg>')
    return (f'<figure><figcaption>topology &mdash; iter {it}</figcaption>'
            f'{"".join(parts)}</figure>')


def arch_svg_u(mat, it, n_layer, vmax):
    """The U-Net folded at its midpoint: encoder descending, decoder ascending beside
    it, so mirrored blocks sit on the same row and their skip is a short crossbar.

    This is the shape the topology is named for. The flat layout draws the same edges
    as nested arches, and the matrix as an anti-diagonal; neither looks like a U.

    Flow runs *down* the left arm and *up* the right one, so a left block's input wire
    is above it while a right block's is below. The crossbars therefore slant slightly
    rather than being truly horizontal -- that is the honest geometry: each one runs
    from the wire entering B_r to the wire entering its mirror.
    """
    h = (n_layer + 1) // 2
    PITCH, TOP, BOX_W, BOX_H = 48, 44, 46, 22
    XL, XR = 62, 214
    W, H = XR + 68, TOP + PITCH * h + 66

    def y_row(r):
        return TOP + PITCH * r

    lo_l = y_row(h - 1) + BOX_H                  # bottom of the encoder arm
    lo_r = y_row(n_layer - 1 - h) + BOX_H        # bottom of the decoder arm (may differ if odd)
    y_bot = max(lo_l, lo_r) + 26

    p = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-label="U-Net folded topology, crossbars coloured by gate value">']
    # the sequential path, as one stroke: down the left arm, round the bottom, up the right
    p.append(f'<path class="stream" fill="none" d="M {XL} {TOP - 22} L {XL} {lo_l} '
             f'L {XL} {y_bot} L {XR} {y_bot} L {XR} {lo_r} L {XR} {TOP - 22}"/>')
    p.append(f'<text x="{XL}" y="{TOP - 28}" class="lbl" text-anchor="middle">embed</text>')
    p.append(f'<text x="{XR}" y="{TOP - 28}" class="lbl" text-anchor="middle">logits</text>')

    for r in range(h):
        i = n_layer - 1 - r                      # the decoder block mirroring encoder block r
        if i < h:
            continue
        v = mat[i][r]
        if v is None:
            continue
        y0, y1 = y_row(r) - 12, y_row(r) + BOX_H + 12
        p.append(f'<line x1="{XL + BOX_W / 2}" y1="{y0}" x2="{XR - BOX_W / 2}" y2="{y1}" '
                 f'stroke="{colour(v, vmax)}" stroke-width="{1 + 4 * abs(v) / vmax:.2f}">'
                 f'<title>B{i} &larr; t{r}: {v:+.4f}</title></line>')
        p.append(f'<text x="{(XL + XR) / 2}" y="{(y0 + y1) / 2 - 4}" class="lbl" '
                 f'text-anchor="middle">{v:+.2f}</text>')

    for i in range(n_layer):
        x = XL if i < h else XR
        r = i if i < h else n_layer - 1 - i
        p.append(f'<rect x="{x - BOX_W / 2}" y="{y_row(r)}" width="{BOX_W}" '
                 f'height="{BOX_H}" rx="3" class="blk"/>')
        p.append(f'<text x="{x}" y="{y_row(r) + 15}" class="blbl" '
                 f'text-anchor="middle">B{i}</text>')
    p.append('</svg>')
    return (f'<figure><figcaption>topology, folded &mdash; iter {it}</figcaption>'
            f'{"".join(p)}</figure>')


def arch_svg_inv_u(mat, it, n_layer, vmax, thresh=0.0):
    """The sequence folded into an inverted U: B0 bottom-left, rising to B{h-1} at the
    apex, then descending the right arm to the last block.

    Works for any topology, not just the mirror one -- the fold is a layout choice, so
    dense's arbitrary (i, j) edges are drawn as chords through the interior while
    same-arm edges bulge outward. Flow runs *up* the left arm and *down* the right, so
    every block's input wire is below it on the left and above it on the right.

    thresh drops |g| <= thresh. The caption reports how many edges were hidden, since a
    filtered figure that does not say so reads as if it showed everything.
    """
    h = (n_layer + 1) // 2
    PITCH, TOP, BOX_W, BOX_H = 52, 58, 46, 22
    gut = 20 + 12 * (h - 1)                    # room for the outside bulges
    XL = gut + BOX_W / 2 + 4
    XR = XL + 152
    W, H = XR + gut + BOX_W / 2 + 4, TOP + PITCH * h + 74

    def row(i):
        return h - 1 - i if i < h else h - 1 - (n_layer - 1 - i)

    def xb(i):
        return XL if i < h else XR

    def yb(i):
        return TOP + PITCH * row(i)

    def tap(j):                                # the wire entering block j
        return (XL, yb(j) + BOX_H + 12) if j < h else (XR, yb(j) - 12)

    shown = [(i, j, v) for i in range(n_layer) for j in range(n_layer)
             if (v := mat[i][j]) is not None and abs(v) > thresh]
    total = sum(1 for i in range(n_layer) for j in range(n_layer) if mat[i][j] is not None)

    y_apex, y_in, y_out = TOP - 24, yb(0) + BOX_H + 26, yb(n_layer - 1) + BOX_H + 26
    p = [f'<svg viewBox="0 0 {W:.0f} {H}" width="{W:.0f}" height="{H}" role="img" '
         f'aria-label="topology folded into an inverted U, edges coloured by gate value">']
    p.append(f'<path class="stream" fill="none" d="M {XL} {y_in} L {XL} {y_apex} '
             f'L {XR} {y_apex} L {XR} {y_out}"/>')
    p.append(f'<text x="{XL}" y="{y_in + 16}" class="lbl" text-anchor="middle">embed</text>')
    p.append(f'<text x="{XR}" y="{y_out + 16}" class="lbl" text-anchor="middle">logits</text>')

    for i, j, v in sorted(shown, key=lambda e: abs(e[2])):   # weakest first, strong on top
        (x0, y0), (x1, y1) = tap(j), tap(i)
        if x0 == x1:                           # same arm: bulge away from the interior
            out = -1 if x0 == XL else 1
            cx, cy = x0 + out * (20 + 12 * abs(row(i) - row(j))), (y0 + y1) / 2
        else:                                  # cross-arm: a chord through the interior
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + 26
        p.append(f'<path d="M {x0:.0f} {y0} Q {cx:.0f} {cy:.0f} {x1:.0f} {y1}" '
                 f'stroke="{colour(v, vmax)}" stroke-width="{1 + 4 * abs(v) / vmax:.2f}" '
                 f'fill="none"><title>B{i} &larr; t{j}: {v:+.4f}</title></path>')

    for i in range(n_layer):
        p.append(f'<rect x="{xb(i) - BOX_W / 2}" y="{yb(i)}" width="{BOX_W}" '
                 f'height="{BOX_H}" rx="3" class="blk"/>')
        p.append(f'<text x="{xb(i)}" y="{yb(i) + 15}" class="blbl" '
                 f'text-anchor="middle">B{i}</text>')
    p.append('</svg>')
    hid = total - len(shown)
    cap = f'topology &mdash; iter {it}'
    if thresh:
        cap += (f' &middot; {len(shown)} of {total} edges (|g| &gt; {thresh:g}; '
                f'{hid} weaker edge{"s" if hid != 1 else ""} hidden)')
    return f'<figure><figcaption>{cap}</figcaption>{"".join(p)}</figure>'


def arch_svg_towers(mat, it, n_layer, vmax, thresh=0.0, mirror_only=False, name=''):
    """Two towers with B{h-1} bridging at the top: the first arm rises, the bridge block
    crosses, the second descends.

    The point of this layout is that a mirror edge (i + j == n_layer - 1) lands with its
    source tap and its destination on the *same row*, so it draws as a horizontal
    crossbar. That makes two different topologies directly comparable edge-for-edge on
    one scale -- which the arch and matrix views do not.

    Non-mirror edges have no horizontal home here and are drawn as curves, or dropped
    entirely under mirror_only. Reads best at even depth; at odd depth the bridge
    consumes a tap that has no partner on the far tower.
    """
    h = (n_layer + 1) // 2
    nrow = n_layer - h                         # tap rows == blocks on the second tower
    PITCH, TOP, BOX_W, BOX_H = 54, 104, 46, 22
    XL, XR = 62, 258
    W, H = XR + 78, TOP + PITCH * nrow + 64

    def y_row(r):
        return TOP + PITCH * r

    def tap(j):                                # the wire entering block j
        return ((XL, y_row(n_layer - 1 - j - h)) if j <= n_layer - 1 - h
                else (XR, y_row(j - h)))

    inject = tap                               # a skip lands on its destination's input wire
    y_top, y_bot = y_row(0), y_row(nrow - 1)
    y_bridge = TOP - 62
    XM = (XL + XR) / 2

    p = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-label="two towers bridged at the top, crossbars coloured by gate value">']
    # left tower rises, bridge crosses the top, right tower descends
    p.append(f'<path class="stream" fill="none" d="M {XL} {y_bot + 40} L {XL} {y_bridge} '
             f'L {XM - 34} {y_bridge}"/>')
    p.append(f'<path class="stream" fill="none" d="M {XM + 34} {y_bridge} L {XR} {y_bridge} '
             f'L {XR} {y_bot + 40}"/>')
    p.append(f'<rect x="{XM - 34}" y="{y_bridge - BOX_H / 2}" width="68" height="{BOX_H}" '
             f'rx="3" class="blk"/>')
    p.append(f'<text x="{XM}" y="{y_bridge + 4}" class="blbl" text-anchor="middle">'
             f'B{h - 1}</text>')
    p.append(f'<text x="{XL}" y="{y_bot + 56}" class="lbl" text-anchor="middle">embed</text>')
    p.append(f'<text x="{XR}" y="{y_bot + 56}" class="lbl" text-anchor="middle">logits</text>')

    mirror = [(i, n_layer - 1 - i) for i in range(h, n_layer)]
    drawn = 0
    total = sum(1 for i in range(n_layer) for j in range(n_layer) if mat[i][j] is not None)
    for i, j in mirror:                        # horizontal crossbars, the comparable set
        v = mat[i][j]
        if v is None or abs(v) <= thresh:
            continue
        y = y_row(i - h)
        p.append(f'<line x1="{XL}" y1="{y}" x2="{XR - 8}" y2="{y}" '
                 f'stroke="{colour(v, vmax)}" stroke-width="{1 + 4 * abs(v) / vmax:.2f}">'
                 f'<title>B{i} &larr; t{j}: {v:+.4f}</title></line>')
        p.append(f'<text x="{XM}" y="{y - 6}" class="lbl" text-anchor="middle">{v:+.2f}</text>')
        drawn += 1
    if not mirror_only:
        for i in range(n_layer):
            for j in range(n_layer):
                v = mat[i][j]
                if v is None or abs(v) <= thresh or (i, j) in mirror:
                    continue
                (x0, y0), (x1, y1) = tap(j), inject(i)
                cx = (x0 + x1) / 2 + (0 if x0 != x1 else (-40 if x0 == XL else 40))
                p.append(f'<path d="M {x0:.0f} {y0} Q {cx:.0f} {(y0 + y1) / 2 + 18:.0f} '
                         f'{x1:.0f} {y1}" stroke="{colour(v, vmax)}" fill="none" '
                         f'stroke-width="{1 + 4 * abs(v) / vmax:.2f}" stroke-dasharray="3 3">'
                         f'<title>B{i} &larr; t{j}: {v:+.4f}</title></path>')
                drawn += 1

    for i in list(range(h - 1)) + list(range(h, n_layer)):
        if i < h:                              # left tower: sits between its two taps
            x, y = XL, tap(i)[1] - PITCH / 2 - BOX_H / 2
        else:
            x, y = XR, y_row(i - h) + 14
        p.append(f'<rect x="{x - BOX_W / 2}" y="{y:.0f}" width="{BOX_W}" height="{BOX_H}" '
                 f'rx="3" class="blk"/>')
        p.append(f'<text x="{x}" y="{y + 15:.0f}" class="blbl" text-anchor="middle">B{i}</text>')
    p.append('</svg>')

    cap = f'{name} &mdash; iter {it}' if name else f'topology &mdash; iter {it}'
    note = 'mirror edges only' if mirror_only else 'dashed = non-mirror'
    # state the scale here: this view rescales to its own edges, so the section legend
    # below (which covers the heatmaps) does not describe these colours
    cap += f' &middot; {drawn} of {total} edges ({note}) &middot; scale &plusmn;{vmax:.2f}'
    return f'<figure><figcaption>{cap}</figcaption>{"".join(p)}</figure>'


def grid(panels, vmax, n_layer, signed, arch='none', thresh=0.0, mirror_only=False):
    draw = {'linear': arch_svg, 'u': arch_svg_u}.get(arch)
    if arch == 'invu':
        cells = [arch_svg_inv_u(m, it, n_layer, vmax, thresh) for _a, it, m in panels]
    elif arch == 'towers':
        # Scale to the edges actually drawn, shared across panels. Keeping the document
        # vmax would set the range from an edge this view hides, washing every crossbar
        # out; the section text states the scale so it is not mistaken for the heatmap's.
        mir = [(i, n_layer - 1 - i) for i in range((n_layer + 1) // 2, n_layer)]
        cand = [abs(m[i][j]) for _a, _it, m in panels for i, j in mir
                if m[i][j] is not None and abs(m[i][j]) > thresh]
        if not mirror_only:
            cand += [abs(v) for _a, _it, m in panels for r in m for v in r
                     if v is not None and abs(v) > thresh]
        tvmax = max(cand) if cand else vmax
        cells = [arch_svg_towers(m, it, n_layer, tvmax, thresh, mirror_only,
                                 a.get('residual', '')) for a, it, m in panels]
    else:
        cells = [draw(m, it, n_layer, vmax) for _a, it, m in panels] if draw else []
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
        who = _args.get('residual', '')
        cells.append(f'<figure><figcaption>{who + " &mdash; " if who else ""}iter {it}'
                     f'</figcaption>'
                     f'<table><tr><th></th>{head}</tr>{"".join(rows)}</table></figure>')
    return f'<div class=grid>{"".join(cells)}</div>'


def _rgb(hexstr):
    return tuple(int(hexstr[k:k + 2], 16) for k in (1, 3, 5))


def html(panels, vmax, n_layer, arch, thresh, mirror_only):
    steps = [-vmax + 2 * vmax * k / 16 for k in range(17)]
    leg_signed = ''.join(f'<span style="background:{colour(v, vmax)}"></span>' for v in steps)
    leg_abs = ''.join(f'<span style="background:{colour(vmax * k / 16, vmax, False)}"></span>'
                      for k in range(17))
    return f"""<!doctype html><meta charset=utf-8>
<title>skip-gate values</title>
<style>
 :root {{ --surface:#fcfcfb; --ink:#0b0b0b; --muted:#898781; --line:#e1e0d9; --baseline:#c3c2b7; }}
 @media (prefers-color-scheme:dark) {{
   :root {{ --surface:#1a1a19; --ink:#fff; --muted:#898781; --line:#2c2c2a; --baseline:#383835; }} }}
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
 .stream {{ stroke:var(--line); stroke-width:2; }}
 .blk {{ fill:var(--surface); stroke:var(--baseline); stroke-width:1; }}
 .lbl {{ fill:var(--muted); font-size:10px; }}
 .blbl {{ fill:var(--ink); font-size:10px; }}
 svg {{ overflow:visible; }}
</style>
<h1>Learned skip-gate values &mdash; row = destination block, column = source tap</h1>
<p class=sub>t0 is the embedding output; t<sub>j</sub> is the output of block
j&minus;1. Blank cells are edges the topology does not contain. Hover any cell for the
exact value &mdash; both views below show the same numbers, so colour never hides
anything.</p>

<h2>Signed &mdash; direction of the edge</h2>
<p>Diverging scale, symmetric at &plusmn;{vmax:.2f}. Blue adds the earlier state,
red subtracts it, gray is an unused edge.</p>
{grid(panels, vmax, n_layer, True, arch, thresh, mirror_only)}
<div class=legend>{-vmax:+.2f}<span class=bar>{leg_signed}</span>{vmax:+.2f}
&nbsp;&nbsp;red = subtract &middot; gray = unused &middot; blue = add &middot;
thicker = stronger</div>

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
    # off by default: at dense's 66 edges a topology drawing is unreadable clutter,
    # so it is only worth rendering for a sparse variant you asked for it on.
    ap.add_argument('--arch', choices=('none', 'linear', 'u', 'invu', 'towers'), default='none',
                    help="draw the topology beside the heatmaps: 'u' folds it at "
                         "the midpoint (U), 'invu' apex-up, 'towers' two towers bridged at "
                         "the top, 'linear' one column")
    ap.add_argument('--min-gate', type=float, default=0.0,
                    help='drop edges with |g| <= this from the topology drawing')
    ap.add_argument('--mirror-only', action='store_true',
                    help='towers layout: draw only i+j==n_layer-1 edges, the set two '
                         'topologies share, so different variants compare edge-for-edge')
    a = ap.parse_args()
    panels = sorted((load(p) for p in a.ckpt),
                key=lambda x: (x[1] or 0, x[0].get('residual', '')))
    n_layer = panels[0][0]['n_layer']
    vmax = max(abs(v) for _, _, m in panels for row in m for v in row if v is not None)
    with open(a.out, 'w') as f:
        f.write(html(panels, vmax, n_layer, a.arch, a.min_gate, a.mirror_only))
    print(f'wrote {a.out}  (vmax={vmax:.4f}, {len(panels)} panel(s))\n')
    for p in glyph_panels(panels[-1][2], n_layer):
        print(p, '\n')


if __name__ == '__main__':
    main()
