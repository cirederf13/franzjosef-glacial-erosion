#!/usr/bin/env python3
"""Glacial-cycle run: forcing, ice history, erosion maps and swath profile.

Usage: python plot_cycle_results.py <output.nc> <outdir> <tag> [--kg1 V --kg2 V --D V --sc V]
"""
import argparse, os
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import figstyle
figstyle.apply()
from matplotlib.colors import LogNorm
from erosion_params import (ref, hillslope_mm_yr, hillslope_range, spinup_index,
                            GLACIAL_RANGE,
                            PLOT_D_MM_YR, PLOT_SC_DEG)
from mapdeco import graticule, scalebar
from trunk_swath import swath_mask, site_distance_km, XGAUGE, YGAUGE

OSL = 'data_osl/osl_erosion_rates.csv'
HERMAN = 'data_osl/herman2015_erosion_profile_figS2A.csv'


def bed_slope(z, dx, dy):
    gy, gx = np.gradient(z, dy, dx)
    return np.hypot(gx, gy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('nc'); ap.add_argument('outdir'); ap.add_argument('tag')
    ap.add_argument('--kg1', type=float, default=3.47e-5)
    ap.add_argument('--kg2', type=float, default=1.85e-7)
    ap.add_argument('--D', type=float, default=PLOT_D_MM_YR,
                    help='hillslope rate, mm/yr')
    ap.add_argument('--sc', type=float, default=PLOT_SC_DEG,
                    help='critical slope, degrees')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    ds = xr.open_dataset(a.nc)
    x, y = ds.x.values, ds.y.values
    dx, dy = abs(float(x[1]-x[0])), abs(float(y[1]-y[0])); cell = dx*dy/1e6
    topg = ds.topg.values; topg = topg[0] if topg.ndim == 3 else topg
    thk = ds.thk.values; ub = ds.velbase_mag.values; t = ds.time.values
    ds.close()

    sw, trunk, dkm, catch = swath_mask(topg, x, y)
    S = bed_slope(topg, dx, dy)
    p1 = ref(1, kg=a.kg1, D=a.D, Sc=a.sc)
    p2 = ref(2, kg=a.kg2, D=a.D, Sc=a.sc)
    hs_norm_range = hillslope_range(hillslope_mm_yr(S, p1['D'], p1['Sc']))
    hill1 = hillslope_mm_yr(S, p1['D'], p1['Sc'], floor=hs_norm_range[0])     # mm/yr
    hill2 = hillslope_mm_yr(S, p2['D'], p2['Sc'], floor=hs_norm_range[0])     # mm/yr

    ice = thk > 1
    w = np.gradient(t); w = w/w.sum()
    # per-time series inside the catchment
    vol = np.array([np.nansum(thk[i][catch])*cell*1e6/1e9 for i in range(len(t))])
    area = np.array([(ice[i] & catch).sum()*cell for i in range(len(t))])
    ero1 = np.array([np.where(ice[i], p1['Kg']*ub[i]*1e3, hill1)[catch].mean()
                     for i in range(len(t))])
    ero2 = np.array([np.where(ice[i], p2['Kg']*ub[i]**2*1e3, hill2)[catch].mean()
                     for i in range(len(t))])
    f = np.loadtxt('glacial_cycle_smb_rescaled_lgm_v5.txt', skiprows=1)

    k0 = spinup_index(ub, ice, catch)
    if k0:
        print(f'  excluding {k0} spin-up sample(s) from the history panels '
              f'(t = {t[0]/1000:.2f} to {t[k0-1]/1000:.2f} ka)')
    fig, ax = plt.subplots(3, 1, figsize=(11, 9.2), sharex=True)
    ax[0].plot(f[:,0]/1000, f[:,3], 'k-')
    ax[0].set_ylabel('ELA (m)'); ax[0].invert_yaxis()
    a2 = ax[0].twinx(); a2.plot(f[:,0]/1000, f[:,4], color='tab:blue')
    a2.set_ylabel('accmax (m/yr)', color='tab:blue')
    a2.tick_params(axis='y', labelcolor='tab:blue')
    figstyle.panel(ax[0], 0)
    ax[1].plot(t[k0:]/1000, vol[k0:], 'k-')
    ax[1].set_ylabel('ice volume\nin catchment (km$^3$)')
    figstyle.panel(ax[1], 1)
    ax[2].plot(t[k0:]/1000, ero1[k0:], label=f'$l$=1, $K_g$={p1["Kg"]:.2e}')
    ax[2].plot(t[k0:]/1000, ero2[k0:], label=f'$l$=2, $K_g$={p2["Kg"]:.2e}')
    ax[2].set_ylabel('erosion rate\n(mm/yr, catchment mean)')
    ax[2].set_xlabel('time (ka)'); ax[2].legend()
    figstyle.panel(ax[2], 2)
    for b in ax: b.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f'{a.outdir}/cycle_history_{a.tag}.png', dpi=140)
    plt.close(fig); print(f'wrote {a.outdir}/cycle_history_{a.tag}.png')

    picks = [np.argmin(np.abs(t-v)) for v in (-99000,-88000,-77000,-66000,-55000,
                                              -44000,-33000,-22000,-11000,-100)]
    fig, axes = plt.subplots(2, 5, figsize=(21, 9.0))
    gl_norm = LogNorm(*GLACIAL_RANGE); hs_norm = LogNorm(*hs_norm_range)
    for n, (axx, i) in enumerate(zip(axes.ravel(), picks)):
        e = np.where(ice[i], p1['Kg']*ub[i]*1e3, np.nan)
        h = np.where(~ice[i], hill1, np.nan)
        imh = axx.imshow(h, cmap='YlOrRd', origin='lower', norm=hs_norm)
        img = axx.imshow(e, cmap='Blues', origin='lower', norm=gl_norm)
        axx.contour(catch.astype(float), [.5], colors='k', linewidths=.9, linestyles='--')
        axx.text(.03, .965, f't = {t[i]/1000:.1f} ka\n{(ice[i]&catch).sum()*cell:.0f} km$^2$ ice',
                 transform=axx.transAxes, va='top', ha='left',
                 fontsize=mpl.rcParams['xtick.labelsize'],
                 bbox=dict(fc='white', ec='none', alpha=.78, pad=2))
        figstyle.panel(axx, n)
        axx.set_xticks([]); axx.set_yticks([])
        axx.contour(ice[i].astype(float), [.5], colors='#08306b', linewidths=.8)
        graticule(axx, x, y, nlines=2, fontsize=7)
        if n == 0:
            scalebar(axx, dx, fontsize=7.5)
    cb1 = fig.colorbar(img, ax=axes[0,:], fraction=.018, pad=.01)
    cb1.set_label('glacial erosion (mm/yr, log)', fontsize=9)
    cb2 = fig.colorbar(imh, ax=axes[1,:], fraction=.018, pad=.01)
    cb2.set_label('hillslope erosion (mm/yr, log)', fontsize=9)
    fig.savefig(f'{a.outdir}/erosion_maps_{a.tag}.png', dpi=130, bbox_inches='tight')
    plt.close(fig); print(f'wrote {a.outdir}/erosion_maps_{a.tag}.png')

    # ---------- FIG 3: swath profile vs observations ----------
    tot1 = np.zeros_like(topg); tot2 = np.zeros_like(topg)
    for i in range(len(t)):
        tot1 += w[i]*np.where(ice[i], p1['Kg']*ub[i]*1e3, hill1)/1e3
        tot2 += w[i]*np.where(ice[i], p2['Kg']*ub[i]**2*1e3, hill2)/1e3   # back to m/yr
    edges = np.arange(0, np.nanmax(dkm[catch])+0.7, 0.7)
    prof = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = catch & (dkm >= lo) & (dkm < hi)
        if m.sum() == 0: continue
        prof.append((0.5*(lo+hi), tot1[m].mean()*1e3, tot2[m].mean()*1e3, m.sum()))
    prof = np.array(prof)
    from osl_sites import load_sites, describe
    osl = load_sites(); her = pd.read_csv(HERMAN, comment='#')
    print('  ' + describe(osl))
    from pyproj import Transformer
    tr = Transformer.from_crs('EPSG:4326', 'EPSG:32759', always_xy=True)
    oux, ouy = tr.transform(osl.longitude.values, osl.latitude.values)
    od, _off = site_distance_km(dkm, x, y, osl.longitude.values, osl.latitude.values)
    od = list(od)
    om1, om2 = [], []
    for sx, sy in zip(oux, ouy):
        cy = int(np.argmin(np.abs(y - sy))); cx = int(np.argmin(np.abs(x - sx)))
        wy = slice(max(cy-1, 0), cy+2); wx = slice(max(cx-1, 0), cx+2)
        om1.append(tot1[wy, wx].mean()*1e3)
        om2.append(tot2[wy, wx].mean()*1e3)
    od, om1, om2 = np.array(od), np.array(om1), np.array(om2)

    fig, axx = plt.subplots(figsize=(10.5, 6.2))
    axx.plot(prof[:,0], prof[:,1], '-', color='tab:blue', lw=1.2, alpha=.45,
             label='model $l$=1, swath mean (corridor average)')
    axx.plot(prof[:,0], prof[:,2], '-', color='tab:red', lw=1.2, alpha=.45,
             label='model $l$=2, swath mean (corridor average)')
    # pair each observation with the model AT THE SAME POINT
    for i, d in enumerate(od):
        axx.plot([d, d], [osl.erosion_rate.values[i], om1[i]], '-',
                 color='0.6', lw=.9, zorder=1)
    axx.errorbar(od, osl.erosion_rate.values,
                 yerr=[osl.sigma_minus.values, osl.sigma_plus.values],
                 fmt='D', color='tab:green', ms=8, capsize=3, mec='k', mew=.7,
                 zorder=4, label='OSL observed (this study, ~10-100 ka)')
    axx.plot(od, om1, 'o', color='tab:blue', ms=7, mec='k', mew=.7, zorder=5,
             label='MODEL at OSL sites, $l$=1 (3$\\times$3 window)')
    axx.plot(od, om2, 's', color='tab:red', ms=6, mec='k', mew=.7, zorder=5,
             label='MODEL at OSL sites, $l$=2 (3$\\times$3 window)')
    axx.errorbar(her.distance_m/1000, her.geometric_mean,
                 yerr=[her.geometric_mean-her.lower_1sigma,
                       her.upper_1sigma-her.geometric_mean],
                 fmt='^', color='0.35', ms=5, capsize=2, alpha=.75, zorder=2,
                 label='Herman+2015 modern (5 months)')
    r1 = np.median(om1/osl.erosion_rate.values); r2 = np.median(om2/osl.erosion_rate.values)
    axx.set(yscale='log', xlabel='along-trunk distance from gauge (km)',
            ylabel='erosion rate (mm yr$^{-1}$)',)
    axx.legend(loc='upper right'); axx.grid(alpha=.3, which='both')
    fig.tight_layout(); fig.savefig(f'{a.outdir}/swath_profile_{a.tag}.png', dpi=140)
    plt.close(fig); print(f'wrote {a.outdir}/swath_profile_{a.tag}.png')
    print(f'  model/obs at OSL sites: l=1 median {r1:.2f} (range '
          f'{np.min(om1/osl.erosion_rate.values):.2f}-{np.max(om1/osl.erosion_rate.values):.2f}), '
          f'l=2 median {r2:.2f}')

    print(f"\ncatchment {catch.sum()*cell:.1f} km2 "
          f"(swath no longer used for integration: {sw.sum()*cell:.1f} km2)")
    print(f"present-day: vol {vol[-1]:.2f} km3, area {area[-1]:.1f} km2")
    print(f"cycle-mean catchment erosion: l=1 {(w*ero1).sum():.3f} mm/yr, "
          f"l=2 {(w*ero2).sum():.3f} mm/yr")


if __name__ == '__main__':
    main()
