#!/usr/bin/env python3
"""Bayesian inference of K_g (and optionally D, S_c) from the nine OSL erosion rates.

The prediction is linear in K_g and D, so the 2e6-draw sampling reduces to two multiplies per draw.

Usage: python bayesian_erosion_inference.py <output.nc> [--l 1 2] [--ndraw 2000000] [--fix-D VALUE] [--fix-Sc DEG] [--out file.json]
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

OSL_CSV = 'data_osl/osl_erosion_rates.csv'
SWATH_M = 780.0
WINDOW_CELLS = 3         # OSL prediction window, N x N cells centred on the sample
THIN_ICE_M = 1.0         # "thin-ice threshold" below which hillslope applies.
SC_GRID = np.deg2rad(np.linspace(25.0, 65.0, 161))   # tan of these = slope
ACC_TRUNK_FRAC = 0.02    # flow-accumulation threshold for "channel", as a
                         # fraction of the domain's max accumulation
TANGENT_SPAN = 4         # cells up/down channel used for the local tangent


# ----------------------------------------------------------------- terrain ---
def d8_flowdirs(z):
    """D8 steepest-descent neighbour index for every cell (-1 = pit/outlet)."""
    ny, nx = z.shape
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    best = np.full((ny, nx), -1, dtype=np.int64)
    bestdrop = np.zeros((ny, nx))
    for k, (dy, dx) in enumerate(offs):
        zs = np.full_like(z, np.nan)
        ys = slice(max(0, -dy), ny - max(0, dy))
        xs = slice(max(0, -dx), nx - max(0, dx))
        yd = slice(max(0, dy), ny - max(0, -dy))
        xd = slice(max(0, dx), nx - max(0, -dx))
        zs[ys, xs] = z[yd, xd]
        dist = np.hypot(dy, dx)
        drop = (z - zs) / dist
        upd = np.isfinite(drop) & (drop > bestdrop)
        bestdrop[upd] = drop[upd]
        iy, ix = np.nonzero(upd)
        best[iy, ix] = np.ravel_multi_index(
            (np.clip(iy + dy, 0, ny - 1), np.clip(ix + dx, 0, nx - 1)), (ny, nx))
    return best


def flow_accum(z, fd):
    """Number of upstream cells draining through each cell (topological order)."""
    ny, nx = z.shape
    acc = np.ones(ny * nx)
    order = np.argsort(z.ravel())[::-1]          # high to low
    f = fd.ravel()
    for i in order:
        j = f[i]
        if j >= 0:
            acc[j] += acc[i]
    return acc.reshape(ny, nx)


def bed_slope(z, dx, dy):
    gy, gx = np.gradient(z, dy, dx)
    return np.hypot(gx, gy)


# ----------------------------------------------------------------- transects ---
def upstream_map(fd, n):
    """For each cell, the list of cells draining directly into it (CSR-style)."""
    f = fd.ravel()
    src = np.nonzero(f >= 0)[0]
    order = np.argsort(f[src], kind='stable')
    src_sorted = src[order]
    tgt_sorted = f[src_sorted]
    bounds = np.searchsorted(tgt_sorted, np.arange(n + 1))
    return src_sorted, bounds


def local_tangent(i0, fd, trunk_flat, up_src, up_bounds, shape):
    """Unit tangent of the channel at trunk cell i0, from a local D8 chord."""
    ny, nx = shape
    f = fd.ravel()
    down = i0
    for _ in range(TANGENT_SPAN):
        j = f[down]
        if j < 0 or not trunk_flat[j]:
            break
        down = int(j)
    up = i0
    for _ in range(TANGENT_SPAN):
        cand = up_src[up_bounds[up]:up_bounds[up + 1]]
        cand = cand[trunk_flat[cand]]
        if cand.size == 0:
            break
        up = int(cand[0])
    y0, x0 = np.unravel_index(up, (ny, nx))
    y1, x1 = np.unravel_index(down, (ny, nx))
    v = np.array([x1 - x0, y1 - y0], float)
    nv = np.linalg.norm(v)
    return v / nv if nv > 0 else np.array([1.0, 0.0])


def transect_cells(xs, ys, x, y, tangent, dx, dy):
    """Cell indices along a +/-SWATH_M line through (xs,ys) perpendicular to tangent."""
    perp = np.array([-tangent[1], tangent[0]])
    step = 0.5 * (abs(dx) + abs(dy))
    s = np.arange(-SWATH_M, SWATH_M + step, step)
    px = xs + perp[0] * s
    py = ys + perp[1] * s
    ix = np.clip(np.searchsorted(x, px), 0, len(x) - 1)
    iy = np.clip(np.searchsorted(y, py), 0, len(y) - 1)
    keep = (px >= x.min()) & (px <= x.max()) & (py >= y.min()) & (py <= y.max())
    return iy[keep], ix[keep]


# ----------------------------------------------------------------- main ---
def build_operators(nc, exps, thin_ice=THIN_ICE_M, verbose=True):
    ds = xr.open_dataset(nc)
    x, y = ds.x.values, ds.y.values
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    topg = ds.topg.values
    if topg.ndim == 3:
        if np.nanmax(np.abs(topg - topg[0])) > 1e-9:
            raise RuntimeError('topg evolves in time; this script assumes a fixed bed '
                               '(erosion diagnostic, not coupled)')
        topg = topg[0]
    thk = ds.thk.values
    ub = ds.velbase_mag.values
    t = ds.time.values
    ds.close()

    w = np.gradient(t)
    w = w / w.sum()                     # time-integration weights
    S = bed_slope(topg, abs(dx), abs(dy))
    fd = d8_flowdirs(topg)
    acc = flow_accum(topg, fd)
    trunk = acc >= ACC_TRUNK_FRAC * acc.max()
    if verbose:
        print(f"# grid {topg.shape}, dx={dx:.3f} dy={dy:.3f}; {len(t)} saved steps "
              f"({t[0]:.1f} .. {t[-1]:.1f})")
        print(f"# bed slope: median {np.median(S):.3f}, p99 {np.percentile(S,99):.3f}, "
              f"max {S.max():.3f}")
        print(f"# trunk cells (acc >= {ACC_TRUNK_FRAC:.0%} of max): {trunk.sum()}")

    trunk_flat = trunk.ravel()
    up_src, up_bounds = upstream_map(fd, topg.size)
    tk = np.nonzero(trunk_flat)[0]
    tyx = np.column_stack(np.unravel_index(tk, topg.shape))

    from osl_sites import load_sites, describe
    osl = load_sites()
    print('  ' + describe(osl))
    tr = Transformer.from_crs('EPSG:4326', 'EPSG:32759', always_xy=True)
    ux, uy = tr.transform(osl.longitude.values, osl.latitude.values)

    A = {l: np.zeros(len(osl)) for l in exps}
    B = np.zeros((len(osl), len(SC_GRID)))
    meta = []
    for i, (sx, sy) in enumerate(zip(ux, uy)):
        r = WINDOW_CELLS // 2
        cy = int(np.argmin(np.abs(y - sy))); cx = int(np.argmin(np.abs(x - sx)))
        wy = np.arange(max(cy - r, 0), min(cy + r + 1, len(y)))
        wx = np.arange(max(cx - r, 0), min(cx + r + 1, len(x)))
        iy, ix = np.meshgrid(wy, wx, indexing='ij')
        iy, ix = iy.ravel(), ix.ravel()
        d2 = (x[tyx[:, 1]] - sx) ** 2 + (y[tyx[:, 0]] - sy) ** 2
        k = int(np.argmin(d2))
        icy = thk[:, iy, ix] > thin_ice                     # (time, npts)
        ubt = ub[:, iy, ix]
        St = S[iy, ix][None, :]
        for l in exps:
            gl = np.where(icy, np.abs(ubt) ** l, 0.0).mean(axis=1)
            A[l][i] = float((w * gl).sum())
        for j, scr in enumerate(SC_GRID):
            sc = np.tan(scr)
            r = np.clip(St / sc, 0.0, 0.999)                # Roering law diverges at S_c
            h = np.where(~icy, St / (1.0 - r ** 2), 0.0).mean(axis=1)
            B[i, j] = float((w * h).sum())
        meta.append(dict(sample=osl['sample'][i], npts=int(len(iy)),
                         trunk_dist_m=float(np.sqrt(d2[k])),
                         frac_time_ice=float(icy.mean())))
    return osl, A, B, meta


def infer(osl, A, B, l, ndraw, seed=0, verbose=True, fix_D=None, fix_Sc=None):
    e = osl.erosion_rate.values
    sig = 0.5 * (np.log10(e + osl.sigma_plus.values)
                 - np.log10(np.maximum(e - osl.sigma_minus.values, 1e-6)))
    rng = np.random.default_rng(seed)
    lo_kg, hi_kg = (1e-6, 1e-3) if l == 1 else (1e-9, 1e-5)
    kg = 10 ** rng.uniform(np.log10(lo_kg), np.log10(hi_kg), ndraw)
    D = (np.full(ndraw, float(fix_D)) if fix_D is not None else
         10 ** rng.uniform(np.log10(1e-8), np.log10(1e1), ndraw))
    scd = (np.full(ndraw, float(fix_Sc)) if fix_Sc is not None else
           rng.uniform(25.0, 65.0, ndraw))
    Bi = np.empty((len(e), ndraw))
    for i in range(len(e)):
        Bi[i] = np.interp(np.deg2rad(scd), SC_GRID, B[i])
    # e_hat in m/yr -> mm/yr to match the observations
    pred = (kg[None, :] * A[l][:, None] + D[None, :] * Bi) * 1e3
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum(((np.log10(pred) - np.log10(e)[:, None]) / sig[:, None]) ** 2,
                         axis=0)
    chi2[~np.isfinite(chi2)] = np.inf
    w = np.exp(-0.5 * (chi2 - chi2.min()))
    w = w / w.sum()
    ess = 1.0 / np.sum(w ** 2)
    idx = rng.choice(ndraw, size=100000, replace=True, p=w)
    out = {}
    for nm, v, logsp in [('Kg', kg, True), ('D', D, True), ('Sc', scd, False)]:
        smp = v[idx]
        if logsp:
            q = 10 ** np.percentile(np.log10(smp), [16, 50, 84])
        else:
            q = np.percentile(smp, [16, 50, 84])
        out[nm] = dict(median=float(q[1]), lo68=float(q[0]), hi68=float(q[2]))
    pri = dict(Kg=(lo_kg, hi_kg, True), D=(1e-8, 1e1, True), Sc=(25.0, 65.0, False))
    out['prior_limited'] = {}
    for nm in ('Kg', 'D', 'Sc'):
        plo, phi, logsp = pri[nm]
        m = out[nm]['median']
        if logsp:
            f = (np.log10(m) - np.log10(plo)) / (np.log10(phi) - np.log10(plo))
        else:
            f = (m - plo) / (phi - plo)
        out['prior_limited'][nm] = bool(f < 0.05 or f > 0.95)
        out[nm]['prior_frac'] = float(f)
    out['chi2_min'] = float(chi2.min())
    out['chi2_min_per_site'] = float(chi2.min() / len(e))
    out['ESS'] = float(ess)
    out['l'] = l
    if verbose:
        print(f"\n=== l = {l} ===  (chi2_min {chi2.min():.2f}, "
              f"{chi2.min()/len(e):.2f}/site, ESS {ess:.0f})")
        for nm, u in [('Kg', 'm/yr per (m/yr)^l'), ('D', 'm2/yr'), ('Sc', 'deg')]:
            r = out[nm]
            flag = (f"  <-- PRIOR-LIMITED (at {r['prior_frac']:.2f} of prior range)"
                    if out['prior_limited'][nm] else
                    f"  [{r['prior_frac']:.2f} of prior range]")
            print(f"  {nm:>3} = {r['median']:.3g}   68% CI [{r['lo68']:.3g}, "
                  f"{r['hi68']:.3g}]   {u}{flag}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('nc')
    ap.add_argument('--l', type=float, nargs='+', default=[1.0, 2.0])
    ap.add_argument('--ndraw', type=int, default=2_000_000)
    ap.add_argument('--thin-ice', type=float, default=THIN_ICE_M)
    ap.add_argument('--out')
    ap.add_argument('--fix-D', type=float, default=None,
                    help='hold D at this value (m/yr) instead of sampling it')
    ap.add_argument('--fix-Sc', type=float, default=None,
                    help='hold S_c at this value (degrees) instead of sampling it')
    a = ap.parse_args()
    exps = [int(v) if float(v).is_integer() else v for v in a.l]
    osl, A, B, meta = build_operators(a.nc, exps, thin_ice=a.thin_ice)
    print("\n# per-site transect diagnostics")
    print(f"{'sample':>8} {'npts':>5} {'trunk_d_m':>10} {'frac_time_ice':>14} "
          + ' '.join(f'A(l={l})'.rjust(12) for l in exps))
    for m, i in zip(meta, range(len(meta))):
        print(f"{m['sample']:>8} {m['npts']:>5} {m['trunk_dist_m']:>10.0f} "
              f"{m['frac_time_ice']:>14.3f} "
              + ' '.join(f'{A[l][i]:12.4g}' for l in exps))
    res = {str(l): infer(osl, A, B, l, a.ndraw,
                         fix_D=getattr(a, 'fix_D'), fix_Sc=getattr(a, 'fix_Sc'))
           for l in exps}
    if a.out:
        json.dump(dict(source=a.nc, thin_ice_m=a.thin_ice, results=res,
                       meta=meta), open(a.out, 'w'), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == '__main__':
    main()
