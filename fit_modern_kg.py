#!/usr/bin/env python3
"""Present-day K_g and exponent l: weighted log-space regression of the observed erosion profile on inverted u_b.

Usage: python fit_modern_kg.py <optimize.nc> [label]
"""
import subprocess
import sys
import numpy as np
import pandas as pd

OBS = 'data_osl/herman2015_erosion_profile_figS2A.csv'


def model_profile(nc):
    out = subprocess.run([sys.executable, 'geodesic_profile.py', nc, '--herman-bins'],
                         capture_output=True, text=True)
    rows = [l for l in out.stdout.splitlines() if l and not l.startswith('#')]
    if not rows:
        raise RuntimeError('geodesic_profile.py produced no rows:\n' + out.stdout + out.stderr)
    hdr = rows[0].split(',')
    data = [r.split(',') for r in rows[1:]]
    df = pd.DataFrame(data, columns=hdr).astype(float)
    return df


def fit(logu, loge, w, l_fixed=None):
    """Weighted least squares of log10(e) = log10(Kg) + l*log10(u)."""
    if l_fixed is None:
        A = np.vstack([np.ones_like(logu), logu]).T
        W = np.diag(1.0 / w**2)
        beta = np.linalg.solve(A.T @ W @ A, A.T @ W @ loge)
        logKg, l = beta
    else:
        l = float(l_fixed)
        logKg = np.sum((loge - l * logu) / w**2) / np.sum(1.0 / w**2)
    pred = logKg + l * logu
    chi2 = float(np.sum(((loge - pred) / w)**2))
    return 10**logKg, l, chi2, pred


def main(nc, label):
    obs = pd.read_csv(OBS, comment='#')
    mod = model_profile(nc)
    n = min(len(obs), len(mod))
    obs, mod = obs.iloc[:n], mod.iloc[:n]
    d = np.abs(obs.distance_m.values - mod.distance_m.values)
    if d.max() > 60:
        raise RuntimeError(f'bin centres disagree by up to {d.max():.0f} m -- grids not aligned')

    e = obs.geometric_mean.values
    sig = 0.5 * (np.log10(obs.upper_1sigma.values) - np.log10(obs.lower_1sigma.values))
    ub = mod.ub_mean.values
    ok = (ub > 0) & (e > 0)
    logu, loge, w = np.log10(ub[ok]), np.log10(e[ok]), sig[ok]

    print(f"=== {label} ===")
    print(f"bins used: {ok.sum()}/{n}   (bin-centre max mismatch {d.max():.1f} m)")
    print(f"u_b range {ub[ok].min():.1f}-{ub[ok].max():.1f} m/yr; "
          f"obs erosion {e[ok].min():.2f}-{e[ok].max():.2f} mm/yr")
    print()
    print(f"{'fit':>12} {'Kg':>14} {'l':>7} {'chi2':>9} {'chi2/dof':>9}")
    for nm, lf, dof in [('free l', None, ok.sum() - 2),
                        ('l = 1 fixed', 1.0, ok.sum() - 1),
                        ('l = 2 fixed', 2.0, ok.sum() - 1)]:
        Kg, l, chi2, _ = fit(logu, loge, w, lf)
        # Kg here is in mm/yr per (m/yr)^l because e is in mm/yr; convert to m/yr
        print(f"{nm:>12} {Kg*1e-3:>14.4g} {l:>7.3f} {chi2:>9.2f} {chi2/max(dof,1):>9.2f}")
    print()
    print("Kg reported in m/yr per (m/yr)^l, i.e. the units of the manuscript's eq. 1.")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])
