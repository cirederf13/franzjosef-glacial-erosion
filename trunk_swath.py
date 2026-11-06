#!/usr/bin/env python3
"""Franz Josef catchment and the +/-780 m swath around the D8-routed main trunk, with distance measured from the sampling station.

Usage: python trunk_swath.py <output.nc> [--png out.png]
"""
import heapq

import numpy as np
from scipy.ndimage import distance_transform_edt

X0, Y0 = 414909.10342628224, 5172956.631058902
XGAUGE, YGAUGE = X0 + 18.38e3, Y0 + 17.48e3         # Herman et al. gauge
SWATH_M = 780.0


SNAP_M = 500.0     # radius for snapping the gauge onto the channel


def snap_to_channel(acc, x, y, xg=XGAUGE, yg=YGAUGE, radius_m=SNAP_M):
    """Move the pour point onto the stream (standard 'snap pour point' step)."""
    dx, dy = abs(float(x[1] - x[0])), abs(float(y[1] - y[0]))
    iy = int(np.argmin(np.abs(y - yg))); ix = int(np.argmin(np.abs(x - xg)))
    ry = max(1, int(round(radius_m / dy))); rx = max(1, int(round(radius_m / dx)))
    sl = (slice(max(iy - ry, 0), iy + ry + 1), slice(max(ix - rx, 0), ix + rx + 1))
    sub = acc[sl]
    j = np.unravel_index(int(np.argmax(sub)), sub.shape)
    return (sl[0].start + j[0], sl[1].start + j[1]), (iy, ix)


def catchment(fd, seed_rc, shape):
    """Every cell whose D8 path reaches seed_rc (the catchment of the pour point)."""
    f = fd.ravel()
    src = np.nonzero(f >= 0)[0]
    order = np.argsort(f[src], kind='stable')
    ss, ts = src[order], f[src[order]]
    bounds = np.searchsorted(ts, np.arange(shape[0] * shape[1] + 1))
    seed = int(np.ravel_multi_index(seed_rc, shape))
    mask = np.zeros(shape[0] * shape[1], bool)
    mask[seed] = True
    stack = [seed]
    while stack:
        i = stack.pop()
        for k in ss[bounds[i]:bounds[i + 1]]:
            if not mask[k]:
                mask[k] = True; stack.append(int(k))
    return mask.reshape(shape)


CONFLUENCE_JUMP = 1.30   # a >30% one-step area jump means a tributary joined


def pour_point(acc, fd, x, y, shape):
    """The Franz Josef catchment outlet: walk downstream from the snapped gauge"""
    dx, dy = abs(float(x[1] - x[0])), abs(float(y[1] - y[0]))
    cell = dx * dy / 1e6
    seed, raw = snap_to_channel(acc, x, y)
    f = fd.ravel()
    cur = int(np.ravel_multi_index(seed, shape))
    best = cur
    area = catchment(fd, np.unravel_index(cur, shape), shape).sum() * cell
    for _ in range(400):
        nxt = f[cur]
        if nxt < 0:
            break
        a_new = catchment(fd, np.unravel_index(int(nxt), shape), shape).sum() * cell
        if a_new > area * CONFLUENCE_JUMP:
            break
        cur, area, best = int(nxt), a_new, int(nxt)
    return np.unravel_index(best, shape), raw, area

_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def fill_depressions(z, eps=1e-4):
    """Priority-flood depression filling WITH an epsilon gradient (Barnes et al. 2014)."""
    ny, nx = z.shape
    out = np.full_like(z, np.inf, dtype=float)
    closed = np.zeros((ny, nx), bool)
    h = []
    for i in range(ny):
        for j in (0, nx - 1):
            heapq.heappush(h, (z[i, j], i, j)); closed[i, j] = True; out[i, j] = z[i, j]
    for j in range(nx):
        for i in (0, ny - 1):
            if not closed[i, j]:
                heapq.heappush(h, (z[i, j], i, j)); closed[i, j] = True; out[i, j] = z[i, j]
    while h:
        zc, i, j = heapq.heappop(h)
        for di, dj in _OFFS:
            a, b = i + di, j + dj
            if 0 <= a < ny and 0 <= b < nx and not closed[a, b]:
                out[a, b] = max(z[a, b], zc + eps)
                closed[a, b] = True
                heapq.heappush(h, (out[a, b], a, b))
    return out


def d8(z):
    """D8 steepest-descent target index per cell (-1 if none)."""
    ny, nx = z.shape
    best = np.full((ny, nx), -1, np.int64)
    drop = np.zeros((ny, nx))
    for di, dj in _OFFS:
        zs = np.full_like(z, np.nan)
        ys = slice(max(0, -di), ny - max(0, di)); xs = slice(max(0, -dj), nx - max(0, dj))
        yd = slice(max(0, di), ny - max(0, -di)); xd = slice(max(0, dj), nx - max(0, -dj))
        zs[ys, xs] = z[yd, xd]
        d = (z - zs) / np.hypot(di, dj)
        u = np.isfinite(d) & (d > drop)
        drop[u] = d[u]
        iy, ix = np.nonzero(u)
        best[iy, ix] = np.ravel_multi_index(
            (np.clip(iy + di, 0, ny - 1), np.clip(ix + dj, 0, nx - 1)), (ny, nx))
    return best


def accumulate(z, fd):
    acc = np.ones(z.size)
    f = fd.ravel()
    for i in np.argsort(z.ravel())[::-1]:
        if f[i] >= 0:
            acc[f[i]] += acc[i]
    return acc.reshape(z.shape)


def trace_trunk(acc, fd, seed_rc, shape, max_steps=100000):
    """Walk upstream from the gauge, always taking the highest-accumulation feeder."""
    ny, nx = shape
    f = fd.ravel()
    src = np.nonzero(f >= 0)[0]
    order = np.argsort(f[src], kind='stable')
    ss, ts = src[order], f[src[order]]
    bounds = np.searchsorted(ts, np.arange(z_size := ny * nx + 1))
    cur = int(np.ravel_multi_index(seed_rc, shape))
    path = [cur]
    seen = {cur}
    for _ in range(max_steps):
        feeders = ss[bounds[cur]:bounds[cur + 1]]
        feeders = [int(k) for k in feeders if k not in seen]
        if not feeders:
            break
        cur = max(feeders, key=lambda k: acc.ravel()[k])
        seen.add(cur); path.append(cur)
    return np.array(path)


def site_distance_km(dkm, x, y, lon, lat):
    """Along-trunk distance from the gauge, in km, for arbitrary sample coordinates."""
    from pyproj import Transformer
    tr = Transformer.from_crs('EPSG:4326', 'EPSG:32759', always_xy=True)
    ux, uy = tr.transform(np.asarray(lon), np.asarray(lat))
    dist, offset = [], []
    for sx, sy in zip(np.atleast_1d(ux), np.atleast_1d(uy)):
        cy = int(np.argmin(np.abs(y - sy))); cx = int(np.argmin(np.abs(x - sx)))
        dist.append(dkm[cy, cx])
        offset.append(np.hypot(x[cx] - sx, y[cy] - sy))
    return np.array(dist), np.array(offset)


def swath_mask(topg, x, y, swath_m=SWATH_M):
    """(mask, trunk_mask, along_trunk_distance_km_from_gauge)"""
    dx, dy = abs(float(x[1] - x[0])), abs(float(y[1] - y[0]))
    zf = fill_depressions(np.asarray(topg, float))
    fd = d8(zf)
    acc = accumulate(zf, fd)
    seed, raw, _area = pour_point(acc, fd, x, y, topg.shape)
    path = trace_trunk(acc, fd, seed, topg.shape)
    trunk = np.zeros(topg.shape, bool)
    py, px = np.unravel_index(path, topg.shape)
    trunk[py, px] = True
    # swath = within swath_m of the trunk line
    d = distance_transform_edt(~trunk, sampling=(dy, dx))
    mask = d <= swath_m
    step = np.hypot(np.diff(px) * dx, np.diff(py) * dy)
    along = np.concatenate([[0.0], np.cumsum(step)])
    # zero the axis at the trunk point nearest the gauge
    gi = int(np.argmin((x[px] - XGAUGE) ** 2 + (y[py] - YGAUGE) ** 2))
    along = along - along[gi]
    _, (ny_i, nx_i) = distance_transform_edt(~trunk, sampling=(dy, dx),
                                             return_indices=True)
    lut = np.full(topg.shape, np.nan)
    lut[py, px] = along
    dist_km = lut[ny_i, nx_i] / 1000.0
    catch = catchment(fd, seed, topg.shape)
    return mask, trunk, dist_km, catch


if __name__ == '__main__':
    import argparse, xarray as xr
    ap = argparse.ArgumentParser(); ap.add_argument('nc'); ap.add_argument('--png')
    a = ap.parse_args()
    ds = xr.open_dataset(a.nc)
    x, y = ds.x.values, ds.y.values
    topg = ds.topg.values; topg = topg[0] if topg.ndim == 3 else topg
    thk = ds.thk.values[-1]; t = float(ds.time.values[-1]); ds.close()
    m, tr, dk = swath_mask(topg, x, y)
    cell = abs(x[1] - x[0]) * abs(y[1] - y[0]) / 1e6
    print(f"# {a.nc}  grid {topg.shape}  cell {cell*1e6:.0f} m2")
    print(f"  trunk traced: {tr.sum()} cells, length {np.nanmax(dk):.2f} km")
    print(f"  swath (+/-780 m): {m.sum()} cells = {m.sum()*cell:.1f} km2")
    print(f"  at t={t:.1f}:  ice in swath = {((thk>1)&m).sum()*cell:.1f} km2"
          f"   volume in swath = {np.nansum(thk[m])*cell*1e6/1e9:.2f} km3")
    print(f"  swath touches domain border: "
          f"{bool(m[0,:].any() or m[-1,:].any() or m[:,0].any() or m[:,-1].any())}")
    if a.png:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(topg, cmap='gray', origin='lower')
        ov = np.where(m, 1.0, np.nan); ax.imshow(ov, cmap='autumn', alpha=.35, origin='lower')
        ax.imshow(np.where(tr, 1.0, np.nan), cmap='cool', origin='lower')
        ax.imshow(np.where((thk > 1), thk, np.nan), cmap='Blues', alpha=.6, origin='lower')
        ax.plot(np.argmin(np.abs(x - XGAUGE)), np.argmin(np.abs(y - YGAUGE)),
                'r*', ms=16, label='gauge')
        ax.legend(); ax.set_title(f'trunk + $\\pm$780 m swath, ice at t={t:.0f}')
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(); fig.savefig(a.png, dpi=140)
        print(f"  wrote {a.png}")
