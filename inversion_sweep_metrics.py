#!/usr/bin/env python3
"""Collect and cache the metrics of the inversion calibration sweeps.

Usage: python inversion_sweep_metrics.py [--refresh]
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_erosion, laplace

DIAG = 'outputs/inversion_diagnostics'
EROSION_ITER = 3          # px eroded off the ice mask, to avoid mask-edge artifacts
from erosion_params import osl_kg
OSL_KG = osl_kg()


def metrics(nc):
    """Interior metrics of one finished inversion. None if it never wrote a cost history."""
    ds = xr.open_dataset(nc)
    last = dict(iterations=-1)
    mask = ds.icemask.isel(**last).values > 0.5
    thk = ds.thk.isel(**last).values
    vs = ds.velsurf_mag.isel(**last).values
    vo = ds.velsurfobs_mag.isel(**last).values
    ub = ds.velbase_mag.isel(**last).values
    tau = ds.tau_ref.isel(**last).values
    tot = ds.da_cost_total_hist.values
    dat = ds.da_cost_data_hist.values
    ds.close()
    if len(tot) == 0 or len(dat) == 0:
        return None
    it = binary_erosion(mask, iterations=EROSION_ITER)
    res = (vs - vo)[it]
    w = max(1, len(tot) // 6)
    return dict(
        nbit=len(tot),
        cost_data=float(dat[-1]),
        rmse=float(np.sqrt(np.mean(res**2))),
        bias=float(res.mean()),
        us_ratio=float(vs[it].sum() / vo[it].sum()),
        thk_rough=float(np.abs(laplace(thk))[it].mean()),
        ub_rough=float(np.abs(laplace(ub))[it].mean()),
        vs_rough=float(np.abs(laplace(vs))[it].mean()),
        thk_mean=float(thk[it].mean()),
        thk_max=float(thk[it].max()),
        ub_mean=float(ub[it].mean()),
        tau_sd=float(tau.std()),
        # % change in total cost over the last sixth of the run: ~0 means converged
        tail_pct=float((tot[-1] - tot[-w]) / abs(tot[-1]) * 100),
    )


def _collect(pattern, parse, csv, refresh):
    path = os.path.join(DIAG, csv)
    if os.path.isfile(path) and not refresh:
        return pd.read_csv(path, comment='#')
    rows = []
    for d in sorted(glob.glob(pattern)):
        nc = os.path.join(d, 'optimize.nc')
        if not os.path.isfile(nc):
            print(f'  skip (no optimize.nc): {d}'); continue
        name = os.path.basename(d)
        try:
            m = metrics(nc)
        except Exception as exc:
            print(f'  skip {name}: {exc}'); continue
        if m is None:
            print(f'  skip {name}: empty cost history (still running?)'); continue
        val, rep = parse(name)
        rows.append(dict(run=name, value=val, rep=rep, path=nc, **m))
    df = pd.DataFrame(rows).sort_values(['value', 'rep'])
    os.makedirs(DIAG, exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(f'# {csv} -- written by inversion_sweep_metrics.py\n')
        fh.write(f'# interior = ice mask eroded by {EROSION_ITER} px; roughness = mean |Laplacian|\n')
        df.to_csv(fh, index=False)
    print(f'  wrote {path}  ({len(df)} runs)')
    return df


def lambda_h(refresh=False):
    return _collect('outputs/lambda_h_sweep/lam*_rep*',
                    lambda n: (float(n.split('_')[0][3:]), int(n[-1])),
                    'sweep_lambda_h.csv', refresh)


def tau_ref(refresh=False):
    return _collect('outputs/tau_ref_sweep/tau*_rep*',
                    lambda n: (float(n.split('_')[0][3:]), int(n[-1])),
                    'sweep_tau_ref.csv', refresh)


def kg_by_tau(refresh=False):
    """Modern Kg per tau_ref value, on the BEST (lowest-RMSE) replicate of each."""
    path = os.path.join(DIAG, 'kg_by_tau_ref.csv')
    if os.path.isfile(path) and not refresh:
        return pd.read_csv(path, comment='#')
    import subprocess
    import sys
    obs = pd.read_csv('data_osl/herman2015_erosion_profile_figS2A.csv', comment='#')
    df = tau_ref()
    best = df.loc[df.groupby('value').rmse.idxmin()]
    rows = []
    for _, r in best.iterrows():
        out = subprocess.run([sys.executable, 'geodesic_profile.py', r.path, '--herman-bins'],
                             capture_output=True, text=True)
        lines = [l for l in out.stdout.splitlines() if l and not l.startswith('#')]
        mod = pd.DataFrame([l.split(',') for l in lines[1:]],
                           columns=lines[0].split(',')).astype(float)
        n = min(len(obs), len(mod))
        e = obs.geometric_mean.values[:n]
        sig = 0.5*(np.log10(obs.upper_1sigma.values[:n]) - np.log10(obs.lower_1sigma.values[:n]))
        ub = mod.ub_mean.values[:n]
        logu, loge = np.log10(ub), np.log10(e)
        rec = dict(tau_ref=r.value, run=r.run, rmse=r.rmse, us_ratio=r.us_ratio)
        for l in (1.0, 2.0):
            # Kg in mm/yr per (m/yr)^l from the fit; /1e3 -> m/yr, the eq.-1 units
            kg = 10**(np.sum((loge - l*logu)/sig**2)/np.sum(1/sig**2)) * 1e-3
            rec[f'Kg_l{int(l)}'] = kg
            rec[f'ratio_l{int(l)}'] = max(kg/OSL_KG[l], OSL_KG[l]/kg)
        A = np.vstack([np.ones_like(logu), logu]).T
        W = np.diag(1/sig**2)
        b = np.linalg.solve(A.T@W@A, A.T@W@loge)
        rec['free_l'] = b[1]
        rows.append(rec)
        print(f"  tau={r.value:.4f}  Kg(l=1)={rec['Kg_l1']:.3e} ({rec['ratio_l1']:.2f}x)  "
              f"Kg(l=2)={rec['Kg_l2']:.3e} ({rec['ratio_l2']:.2f}x)  free l={b[1]:.2f}")
    outdf = pd.DataFrame(rows).sort_values('tau_ref')
    with open(path, 'w') as fh:
        fh.write('# kg_by_tau_ref.csv -- written by inversion_sweep_metrics.py\n')
        fh.write('# best (lowest-RMSE) replicate per tau_ref; Kg in m/yr per (m/yr)^l\n')
        fh.write(f'# ratio = agreement factor vs the OSL posteriors {OSL_KG}\n')
        outdf.to_csv(fh, index=False)
    print(f'  wrote {path}')
    return outdf


def summarise(df, label):
    g = df.groupby('value')
    s = pd.DataFrame(dict(
        rmse=g.rmse.mean(), rmse_sd=g.rmse.std(ddof=1),
        thk_rough=g.thk_rough.mean(), thk_rough_sd=g.thk_rough.std(ddof=1),
        ub_rough=g.ub_rough.mean(), thk_mean=g.thk_mean.mean(),
        us_ratio=g.us_ratio.mean(), ub_mean=g.ub_mean.mean(), n=g.rmse.size()))
    span_r = s.rmse.max() - s.rmse.min()
    span_g = s.thk_rough.max() - s.thk_rough.min()
    print(f'\n=== {label} ===')
    print(s.to_string(float_format=lambda v: f'{v:.3f}'))
    print(f'signal-to-noise:  RMSE {span_r/max(s.rmse_sd.mean(),1e-12):.1f}x   '
          f'thk roughness {span_g/max(s.thk_rough_sd.mean(),1e-12):.1f}x')
    print('  (<~2x means the effect is not resolvable above CNN-initialization noise)')
    return s


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true')
    a = ap.parse_args()
    summarise(lambda_h(a.refresh), 'lambda_h sweep (tau_ref fixed)')
    summarise(tau_ref(a.refresh), 'tau_ref sweep (lambda_h = 1e6)')
    print('\n=== modern Kg vs tau_ref (best replicate each) ===')
    print(kg_by_tau(a.refresh).to_string(index=False))
