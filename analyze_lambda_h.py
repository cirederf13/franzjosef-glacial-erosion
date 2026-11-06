#!/usr/bin/env python3
"""Lambda_h L-curve of the uniform-tau inversion, with replicates.

Usage: python analyze_lambda_h.py [sweep_dir]
"""
import glob
import os
import sys
import numpy as np
import xarray as xr
from scipy.ndimage import binary_erosion, laplace


def metrics(nc):
    ds = xr.open_dataset(nc)
    mask = ds.icemask.isel(iterations=-1).values > 0.5
    thk = ds.thk.isel(iterations=-1).values
    vs = ds.velsurf_mag.isel(iterations=-1).values
    vo = ds.velsurfobs_mag.isel(iterations=-1).values
    ub = ds.velbase_mag.isel(iterations=-1).values
    tot = ds.da_cost_total_hist.values
    dat = ds.da_cost_data_hist.values
    ds.close()
    if len(tot) == 0 or len(dat) == 0:
        return None   # optimize.nc still being written (run in flight)
    it = binary_erosion(mask, iterations=3)
    res = (vs - vo)[it]
    nb = len(tot)
    w = max(1, nb // 6)
    return dict(
        nbit=nb,
        cost_data=float(dat[-1]),
        rmse=float(np.sqrt(np.mean(res**2))),
        bias=float(res.mean()),
        thk_rough=float(np.abs(laplace(thk))[it].mean()),
        ub_rough=float(np.abs(laplace(ub))[it].mean()),
        thk_mean=float(thk[it].mean()),
        thk_max=float(thk[it].max()),
        ub_mean=float(ub[it].mean()),
        tail_pct=float((tot[-1] - tot[-w]) / abs(tot[-1]) * 100),
    )


def main(base='outputs/lambda_h_sweep'):
    runs = {}
    for d in sorted(glob.glob(f'{base}/lam*_rep*')):
        nc = os.path.join(d, 'optimize.nc')
        if not os.path.isfile(nc):
            print(f'# MISSING optimize.nc: {d}'); continue
        name = os.path.basename(d)
        lam = float(name.split('_')[0][3:])
        try:
            m = metrics(nc)
        except Exception as exc:
            print(f'# SKIP {name}: {exc}'); continue
        if m is None:
            print(f'# SKIP {name}: empty cost history (still running)'); continue
        runs.setdefault(lam, []).append((name, m))

    print(f"{'lam':>8} {'rep':>4} {'nbit':>6} {'RMSE':>8} {'bias':>8} {'thk_rgh':>8} "
          f"{'ub_rgh':>8} {'thk_mn':>7} {'thk_mx':>7} {'ub_mn':>7} {'tail%':>7}")
    print('-'*95)
    for lam in sorted(runs):
        for nm, m in runs[lam]:
            print(f"{lam:>8.0e} {nm[-1]:>4} {m['nbit']:>6} {m['rmse']:>8.2f} {m['bias']:>8.2f} "
                  f"{m['thk_rough']:>8.3f} {m['ub_rough']:>8.3f} {m['thk_mean']:>7.1f} "
                  f"{m['thk_max']:>7.1f} {m['ub_mean']:>7.1f} {m['tail_pct']:>7.2f}")
        print()

    print("=== replicate spread vs across-lambda trend ===")
    print(f"{'lam':>8} {'RMSE mean+/-sd':>22} {'thk_rough mean+/-sd':>26}")
    stats = {}
    for lam in sorted(runs):
        r = np.array([m['rmse'] for _, m in runs[lam]])
        g = np.array([m['thk_rough'] for _, m in runs[lam]])
        stats[lam] = (r.mean(), r.std(ddof=1) if len(r) > 1 else 0.0,
                      g.mean(), g.std(ddof=1) if len(g) > 1 else 0.0)
        print(f"{lam:>8.0e} {r.mean():>12.2f} +/- {r.std(ddof=1) if len(r)>1 else 0:>6.2f} "
              f"{g.mean():>16.3f} +/- {g.std(ddof=1) if len(g)>1 else 0:>6.3f}")

    lams = sorted(stats)
    rm = np.array([stats[l][0] for l in lams]); rs = np.array([stats[l][1] for l in lams])
    gm = np.array([stats[l][2] for l in lams]); gs = np.array([stats[l][3] for l in lams])
    print()
    print(f"across-lambda RMSE range      : {rm.min():.2f}-{rm.max():.2f} "
          f"(spread {rm.max()-rm.min():.2f}); typical replicate sd {np.mean(rs):.2f}")
    print(f"across-lambda thk_rough range : {gm.min():.3f}-{gm.max():.3f} "
          f"(spread {gm.max()-gm.min():.3f}); typical replicate sd {np.mean(gs):.3f}")
    snr_r = (rm.max()-rm.min())/max(np.mean(rs), 1e-12)
    snr_g = (gm.max()-gm.min())/max(np.mean(gs), 1e-12)
    print(f"\nsignal-to-noise: RMSE {snr_r:.1f}x, thickness roughness {snr_g:.1f}x")
    print("  (<~2x means the lambda effect is not resolvable above initialization noise)")
    mono_g = all(gm[i] >= gm[i+1] for i in range(len(gm)-1))
    mono_r = all(rm[i] <= rm[i+1] for i in range(len(rm)-1))
    print(f"monotonic? roughness decreasing with lambda: {mono_g}; "
          f"RMSE increasing with lambda: {mono_r}")
    print("  (a usable L-curve needs both, or at least a clear knee)")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'outputs/lambda_h_sweep')
