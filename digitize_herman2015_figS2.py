#!/usr/bin/env python3
"""Digitize the erosion-rate curves of Fig. S2A of Herman et al. (2015) from the vector paths of the supplement PDF.

Usage: python digitize_herman2015_figS2.py <SupplementaryMaterial.pdf> [out.csv]
"""
import sys
import numpy as np
import pymupdf

X_TICKS = [2000.0, 4000.0, 6000.0, 8000.0, 10000.0]   # panel A x tick values
Y_TICKS = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0]  # panel A y tick values


def _calibrate(pg):
    """Find panel A (the erosion panel) and fit pt->data linear maps from ticks."""
    dr = pg.get_drawings()
    thin = [q for q in dr if q.get('color') and max(q['color']) < 0.05
            and (q.get('width') or 0) < 0.9 and len(q['items']) == 1]
    frames = [q for q in dr if q.get('color') and max(q['color']) < 0.05
              and len(q['items']) == 3]
    if not frames:
        raise RuntimeError('no axis frame found')
    # panel A is the upper panel => smallest y0
    frame = min(frames, key=lambda q: q['rect'].y0)
    fr = frame['rect']
    band = (fr.y0 - 15, fr.y1 + 15)

    # y ticks: short horizontal stubs at the left edge, inside the band
    yt = sorted({round(q['rect'].y0, 1) for q in thin
                 if abs(q['rect'].y1 - q['rect'].y0) < 1.5
                 and q['rect'].x0 <= fr.x0 + 2 and band[0] <= q['rect'].y0 <= band[1]},
                reverse=True)
    # x ticks: short vertical stubs on the bottom axis
    xt = sorted({round(q['rect'].x0, 1) for q in thin
                 if abs(q['rect'].x1 - q['rect'].x0) < 1.5
                 and fr.y1 - 2 <= q['rect'].y1 <= fr.y1 + 6})
    if len(yt) < len(Y_TICKS) or len(xt) < len(X_TICKS):
        raise RuntimeError(f'tick detection failed (found {len(yt)} y, {len(xt)} x)')
    yt, xt = yt[:len(Y_TICKS)], xt[-len(X_TICKS):]
    ay, by = np.polyfit(yt, Y_TICKS, 1)
    ax, bx = np.polyfit(xt, X_TICKS, 1)
    return (ax, bx), (ay, by), band


def to_data(x_pt, y_pt, cal):
    (ax, bx), (ay, by) = cal
    return ax * x_pt + bx, ay * y_pt + by


_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy<2 compat


def polylines_from_items(items):
    """Split a path's line items into connected polylines (detect pen-up)."""
    runs, cur = [], []
    for it in items:
        if it[0] != 'l':
            continue
        p1, p2 = it[1], it[2]
        if not cur:
            cur = [(p1.x, p1.y), (p2.x, p2.y)]
            continue
        lx, ly = cur[-1]
        if abs(p1.x - lx) < 0.15 and abs(p1.y - ly) < 0.15:
            cur.append((p2.x, p2.y))
        else:
            runs.append(cur); cur = [(p1.x, p1.y), (p2.x, p2.y)]
    if cur:
        runs.append(cur)
    return runs


def _find_page(doc):
    """Page carrying Fig S2 (standalone file: page 0; supplement: search text)."""
    if len(doc) == 1:
        return doc[0]
    for pg in doc:
        t = pg.get_text() or ''
        if 'Distance to sampling station' in t and 'Erosion rate' in t:
            return pg
    raise RuntimeError('Fig S2 page not found')


def main(pdf, out_csv):
    pg = _find_page(pymupdf.open(pdf))
    calx, caly, PANEL_A_YRANGE = _calibrate(pg)
    cal = (calx, caly)
    print(f"# auto-calibration: x = {calx[0]:.6g}*pt + {calx[1]:.6g}; "
          f"y = {caly[0]:.6g}*pt + {caly[1]:.6g}")
    print(f"# panel A y-band = {PANEL_A_YRANGE[0]:.0f}..{PANEL_A_YRANGE[1]:.0f} pt")
    curves = []
    for p in pg.get_drawings():
        col = p.get('color')
        if not col or max(col) > 0.05:      # black only
            continue
        if (p.get('width') or 0) < 0.9:     # skip thin axis/tick strokes
            continue
        r = p['rect']
        if not (PANEL_A_YRANGE[0] <= r.y0 and r.y1 <= PANEL_A_YRANGE[1]):
            continue
        for run in polylines_from_items(p['items']):
            if len(run) < 5:                # drop legend key stubs
                continue
            curves.append(np.array([to_data(x, y, cal) for x, y in run]))

    curves.sort(key=lambda c: -_trapz(c[:, 1], c[:, 0]))
    def to_bins(c):
        out = []
        for (x1, y1), (x2, y2) in zip(c[:-1], c[1:]):
            if abs(x2 - x1) > 1.0 and abs(y2 - y1) < 1e-6:   # horizontal run
                out.append((x1, x2, y1))
        return out
    binned = [to_bins(c) for c in curves]
    names = ['upper_1sigma', 'geometric_mean', 'lower_1sigma'][:len(curves)]
    print(f"# recovered {len(curves)} black polyline(s) in panel A")
    for nm, c in zip(names, curves):
        print(f"#   {nm:15s} n={len(c):3d}  x={c[:,0].min():.0f}..{c[:,0].max():.0f} m"
              f"  y={c[:,1].min():.2f}..{c[:,1].max():.2f} mm/yr")

    # report recovered bin geometry
    w = [b[1] - b[0] for b in binned[0]]
    print(f"# staircase bins recovered: n={len(binned[0])}, "
          f"width {min(w):.1f}-{max(w):.1f} m (mean {np.mean(w):.1f} m)")

    if out_csv and binned[0]:
        # align all three curves on the mean curve's bin edges
        ref = binned[names.index('geometric_mean')] if 'geometric_mean' in names else binned[0]
        with open(out_csv, 'w') as f:
            f.write("# Observed erosion rate, digitized from the VECTOR paths of Fig. S2A,\n")
            f.write("# Supplementary Material of Herman et al. (2015), Erosion by an Alpine glacier.\n")
            f.write("# Black curves = geometric mean +/- 1-sigma (lognormal), per the figure caption.\n")
            f.write("# The figure is a STAIRCASE plot, so these are BIN values; bin_lo/bin_hi are the\n")
            f.write("# horizontal run limits, which recover Herman's own binning interval (~698 m).\n")
            f.write("# distance_m = bin centre = distance to the sampling station (gauge), metres.\n")
            f.write("bin_lo_m,bin_hi_m,distance_m,geometric_mean,lower_1sigma,upper_1sigma\n")
            for lo, hi, _ in ref:
                vals = {}
                for nm, bl in zip(names, binned):
                    hit = [v for (a, b, v) in bl if a < hi - 1 and b > lo + 1]
                    vals[nm] = hit[0] if hit else float('nan')
                f.write(f"{lo:.1f},{hi:.1f},{0.5*(lo+hi):.1f},"
                        f"{vals.get('geometric_mean', float('nan')):.4f},"
                        f"{vals.get('lower_1sigma', float('nan')):.4f},"
                        f"{vals.get('upper_1sigma', float('nan')):.4f}\n")
        print(f"# wrote {out_csv}")
    return binned


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
