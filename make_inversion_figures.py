#!/usr/bin/env python3
"""Figures from the 2013 velocity inversion: observed vs inverted velocity, and the erosion/velocity profiles with K_g and l fits.

Usage: python make_inversion_figures.py <optimize.nc> <outdir> [tag]
"""
import os
import re
import subprocess
import sys
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import figstyle
figstyle.apply()
from matplotlib.colors import LightSource, Normalize
from scipy.ndimage import binary_erosion, laplace
from mapdeco import graticule, scalebar

OBS_PROFILE = 'data_osl/herman2015_erosion_profile_figS2A.csv'
from erosion_params import osl_kg
OSL_KG = osl_kg()
HERMAN_FREE = (2.5e-7, 2.02)             # Herman et al. (2015) own NLS fit


def load(nc):
    ds = xr.open_dataset(nc)
    g = dict(x=ds.x.values, y=ds.y.values,
             mask=ds.icemask.isel(iterations=-1).values > 0.5,
             thk=ds.thk.isel(iterations=-1).values,
             usurf=ds.usurf.isel(iterations=-1).values,
             vs=ds.velsurf_mag.isel(iterations=-1).values,
             vo=ds.velsurfobs_mag.isel(iterations=-1).values,
             ub=ds.velbase_mag.isel(iterations=-1).values,
             _nc=nc)
    ds.close()
    return g


def profile(nc):
    r = subprocess.run([sys.executable, 'geodesic_profile.py', nc, '--herman-bins'],
                       capture_output=True, text=True)
    rows = [l for l in r.stdout.splitlines() if l and not l.startswith('#')]
    return pd.DataFrame([x.split(',') for x in rows[1:]], columns=rows[0].split(',')).astype(float)


def crop(g, pad=6):
    ys, xs = np.where(g['mask'])
    return (slice(max(ys.min()-pad, 0), ys.max()+pad+1),
            slice(max(xs.min()-pad, 0), xs.max()+pad+1))


SPATIAL_REF = 'outputs/inversion_reg_sweep_v2/lam100/optimize.nc'
LAMBDA_H_CHOSEN = 1e6
TAU_REF_CHOSEN = 0.13


def _bed_hillshade(sl):
    """Shaded relief for the map panels, on the same row order as the netCDF fields."""
    with rasterio.open('data_inversion/topg.tif') as src:
        bed = src.read(1).astype(float)
    bed = bed[::-1]
    return LightSource(azdeg=315, altdeg=45).hillshade(bed[sl], vert_exag=1.5,
                                                       dx=77.9, dy=-78.0)


def _rough(fld, mask):
    """Mean |Laplacian| over the ice interior (a roughness metric)."""
    return float(np.abs(laplace(fld))[binary_erosion(mask, iterations=3)].mean())


def _map(ax, hs, fld, mask, sl, cmap, vmin, vmax, title, cblab, diverging=False,
         xy=None, bar=False):
    ax.imshow(hs, cmap='gray', origin='lower')
    im = ax.imshow(np.where(mask, fld, np.nan)[sl], cmap=cmap, origin='lower',
                   vmin=vmin, vmax=vmax, alpha=0.88)
    ax.contour(mask[sl].astype(float), levels=[0.5], colors='w' if not diverging else 'k',
               linewidths=0.9)
    m = re.match(r'\((\w)\)\s*(.*)', title, flags=re.S)
    if m:
        figstyle.panel(ax, m.group(1))
        note = m.group(2)
    else:
        note = title
    if note:
        ax.text(.03, .965, note, transform=ax.transAxes, va='top', ha='left',
                fontsize=mpl.rcParams['xtick.labelsize']*.82,
                bbox=dict(fc='white', ec='none', alpha=.80, pad=2.5))
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label(cblab, fontsize=8)
    if xy is not None:
        graticule(ax, xy[0], xy[1], nlines=2, fontsize=6.5)
    if bar:
        scalebar(ax, 77.9, fontsize=7.5)
    return im


def fig9(g, out, tag):
    """The inversion, end to end: the fit, every 2D field, and the two calibrations."""
    import inversion_sweep_metrics as ism

    sl = crop(g)
    XY = (g['x'][sl[1]], g['y'][sl[0]])      # cropped axes -- see mapdeco.graticule
    hs = _bed_hillshade(sl)
    mask = g['mask']

    sp = load(SPATIAL_REF) if os.path.isfile(SPATIAL_REF) else None

    fig = plt.figure(figsize=(16.5, 15.2))
    gs = fig.add_gridspec(3, 3, hspace=.30, wspace=.30,
                          left=.035, right=.955, top=.925, bottom=.045)

    # ---- row 1: the velocity fit, which is the only real data constraint ----
    vmax = float(np.nanpercentile(g['vo'][mask], 99))
    _map(fig.add_subplot(gs[0, 0]), hs, g['vo'], mask, sl, 'magma', 0, vmax,
         '(a) Observed surface velocity, 2013', 'm yr$^{-1}$', xy=XY, bar=True)
    _map(fig.add_subplot(gs[0, 1]), hs, g['vs'], mask, sl, 'magma', 0, vmax,
         '(b) Inverted surface velocity', 'm yr$^{-1}$', xy=XY)
    res = g['vs'] - g['vo']
    rl = float(np.nanpercentile(np.abs(res[mask]), 98))
    it = binary_erosion(mask, iterations=3)
    _map(fig.add_subplot(gs[0, 2]), hs, res, mask, sl, 'RdBu_r', -rl, rl,
         f"(c) Residual, inverted $-$ observed\nRMSE {np.sqrt((res[it]**2).mean()):.0f}, "
         f"bias {res[it].mean():+.0f} m yr$^{{-1}}$, "
         f"$\\Sigma u_s/\\Sigma u_s^{{obs}}$ = {g['vs'][it].sum()/g['vo'][it].sum():.3f}",
         'm yr$^{-1}$', diverging=True, xy=XY)

    # ---- row 2: the inverted field, and the noise question, side by side ----
    _map(fig.add_subplot(gs[1, 0]), hs, g['thk'], mask, sl, 'YlGnBu',
         0, float(np.nanpercentile(g['thk'][mask], 99.5)),
         f"(d) Inverted ice thickness  (the ONLY inverted field)\n"
         f"mean {g['thk'][it].mean():.0f} m, max {g['thk'][it].max():.0f} m; "
         f"roughness {_rough(g['thk'], mask):.1f}", 'm', xy=XY)

    ubmax = float(np.nanpercentile(g['ub'][mask], 99))
    if sp is not None:
        ubmax = max(ubmax, float(np.nanpercentile(sp['ub'][sp['mask']], 99)))
    _map(fig.add_subplot(gs[1, 1]), hs, g['ub'], mask, sl, 'viridis', 0, ubmax,
         f"(e) Basal sliding $u_b$ -- ADOPTED\nuniform prescribed "
         f"$\\tau_{{ref}}$; roughness {_rough(g['ub'], mask):.1f}", 'm yr$^{-1}$', xy=XY)

    axf = fig.add_subplot(gs[1, 2])
    if sp is not None:
        _map(axf, hs, sp['ub'], sp['mask'], sl, 'viridis', 0, ubmax,
             f"(f) Basal sliding $u_b$ -- REJECTED\nspatially-variable "
             f"$\\tau_{{ref}}$ ($\\lambda_\\tau$=100); "
             f"roughness {_rough(sp['ub'], sp['mask']):.1f}", 'm yr$^{-1}$', xy=XY)
    else:
        axf.set_axis_off(); axf.text(.5, .5, 'spatial reference run not on disk',
                                     ha='center', va='center', transform=axf.transAxes)

    lam = ism.lambda_h()
    axg = fig.add_subplot(gs[2, 0])
    for v, d in lam.groupby('value'):
        axg.plot(d.thk_rough, d.rmse, 'o', ms=3.5, color='0.72', zorder=1)
    m = lam.groupby('value').agg(rr=('thk_rough', 'mean'), rm=('rmse', 'mean'),
                                 rs=('rmse', 'std'), gs_=('thk_rough', 'std')).reset_index()
    axg.errorbar(m.rr, m.rm, yerr=m.rs, xerr=m.gs_, fmt='-o', color='tab:blue',
                 ms=6, lw=1.5, capsize=3, zorder=2, label='mean of 3 replicates')
    for _, r in m.iterrows():
        pick = abs(r.value - LAMBDA_H_CHOSEN) < 1
        axg.annotate(f'$10^{{{np.log10(r.value):.0f}}}$', (r.rr, r.rm),
                     textcoords='offset points', xytext=(8, 7),
                     fontsize=9.5 if pick else 8.5,
                     fontweight='bold' if pick else 'normal',
                     color='crimson' if pick else '0.25')
    k = m[abs(m.value - LAMBDA_H_CHOSEN) < 1]
    if len(k):
        axg.plot(k.rr, k.rm, 'o', ms=13, mfc='none', mec='crimson', mew=2, zorder=3,
                 label='adopted $\\lambda_h = 10^6$ (knee)')
    axg.set(xscale='log', xlabel='thickness roughness, mean $|\\nabla^2 h|$',
            ylabel='velocity RMSE (m yr$^{-1}$)',
            )
    figstyle.panel(axg, 'g')
    axg.set_xlim(m.rr.min()*0.55, m.rr.max()*2.1)
    axg.grid(alpha=.3, which='both'); axg.legend(loc='upper center')

    tau = ism.tau_ref(); kg = ism.kg_by_tau()
    axh = fig.add_subplot(gs[2, 1])
    t = tau.groupby('value').agg(rm=('rmse', 'mean'), rs=('rmse', 'std')).reset_index()
    axh.errorbar(t.value, t.rm, yerr=t.rs, fmt='-o', color='tab:blue', ms=6, lw=1.6,
                 capsize=3, label='velocity RMSE')
    axh.set_xlabel('prescribed $\\tau_{ref}$ (MPa)')
    axh.set_ylabel('velocity RMSE (m yr$^{-1}$)', color='tab:blue')
    axh.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = axh.twinx()
    ax2.plot(kg.tau_ref, kg.ratio_l1, 's--', color='tab:red', ms=5, lw=1.3,
             label='$K_g$ vs OSL, $l$=1')
    ax2.plot(kg.tau_ref, kg.ratio_l2, '^--', color='tab:orange', ms=5, lw=1.3,
             label='$K_g$ vs OSL, $l$=2')
    ax2.axhline(1.0, ls=':', color='0.4', lw=1)
    ax2.set_ylabel('$K_g$ disagreement factor vs OSL', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    axh.axvline(TAU_REF_CHOSEN, color='crimson', lw=1.4, alpha=.8)
    axh.annotate(f'adopted\n$\\tau_{{ref}}$ = {TAU_REF_CHOSEN}', (TAU_REF_CHOSEN, .58),
                 xycoords=('data', 'axes fraction'), textcoords='offset points',
                 xytext=(10, 0), fontsize=8.5, color='crimson', fontweight='bold')
    axh.axvline(0.2153697218796329, color='0.55', lw=1.2, ls='--')
    axh.annotate("IGM default\n(uncalibrated)", (0.2153697218796329, .12),
                 xycoords=('data', 'axes fraction'), textcoords='offset points',
                 xytext=(-63, 0), fontsize=7.5, color='0.4')
    figstyle.panel(axh, 'h')
    h1, l1 = axh.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axh.legend(h1+h2, l1+l2, loc='upper left'); axh.grid(alpha=.3)

    axi = fig.add_subplot(gs[2, 2])
    with xr.open_dataset(g['_nc']) as ds:
        tot = ds.da_cost_total_hist.values
        dat = ds.da_cost_data_hist.values
        reg = ds.da_cost_reg_hist.values if 'da_cost_reg_hist' in ds else None
    axi.plot(tot, color='k', lw=1.5, label='total cost')
    axi.plot(dat, color='tab:blue', lw=1.2, label='data misfit')
    if reg is not None and len(reg) == len(tot):
        axi.plot(reg, color='tab:green', lw=1.2, label='thickness regularization')
    if sp is not None:
        with xr.open_dataset(SPATIAL_REF) as ds2:
            axi.plot(ds2.da_cost_total_hist.values, color='0.6', lw=1.0, ls='--',
                     label='rejected spatial run (total)')
    axi.axvline(len(tot)-1, color='crimson', lw=1.2, ls=':')
    axi.annotate(f"stopped at iteration {len(tot)} by the\noptimizer's own patience, not by\n"
                 f"the 6000-iteration budget", (len(tot)-1, .58),
                 xycoords=('data', 'axes fraction'), textcoords='offset points',
                 xytext=(16, 0), fontsize=8, color='crimson',
                 arrowprops=dict(arrowstyle='-', color='crimson', lw=.8))
    axi.set(xlabel='optimizer iteration', ylabel='cost',
            )
    figstyle.panel(axi, 'i')
    axi.legend(); axi.grid(alpha=.3)

    p = f'{out}/fig9_velocity_maps.png'
    fig.savefig(p, dpi=140); plt.close(fig)
    print('wrote', p)


def wls(logu, loge, w, l_fixed=None):
    if l_fixed is None:
        A = np.vstack([np.ones_like(logu), logu]).T
        W = np.diag(1/w**2)
        b = np.linalg.solve(A.T@W@A, A.T@W@loge)
        return 10**b[0], b[1]
    l = float(l_fixed)
    return 10**(np.sum((loge-l*logu)/w**2)/np.sum(1/w**2)), l


def fig10(nc, out, tag):
    obs = pd.read_csv(OBS_PROFILE, comment='#')
    mod = profile(nc)
    n = min(len(obs), len(mod)); obs, mod = obs.iloc[:n], mod.iloc[:n]
    d = obs.distance_m.values/1000.0
    e, lo, hi = obs.geometric_mean.values, obs.lower_1sigma.values, obs.upper_1sigma.values
    sig = 0.5*(np.log10(hi)-np.log10(lo))
    ub = mod.ub_mean.values

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    # (a) observed erosion profile
    ax[0].fill_between(d, lo, hi, color='0.75', alpha=.7, label='observed $\\pm1\\sigma$')
    ax[0].plot(d, e, 'ko-', ms=4, lw=1.6, label='observed geometric mean')
    ax[0].set(xlabel='distance to sampling station (km)', ylabel='erosion rate (mm yr$^{-1}$)')
    figstyle.panel(ax[0], 0)
    ax[0].legend(); ax[0].grid(alpha=.3)

    # (b) velocity profiles
    ax[1].plot(d, mod.usurfobs_mean, 'k^-', ms=4, lw=1.4, label='observed $u_s$')
    ax[1].plot(d, mod.usurf_mean, 'o-', color='tab:blue', ms=4, lw=1.4, label='inverted $u_s$')
    ax[1].plot(d, mod.ub_mean, 's-', color='tab:red', ms=4, lw=1.4, label='inverted $u_b$')
    ax[1].set(xlabel='distance to sampling station (km)', ylabel='velocity (m yr$^{-1}$)')
    figstyle.panel(ax[1], 1)
    ax[1].legend(); ax[1].grid(alpha=.3)

    # (c) erosion vs basal velocity
    ax[2].errorbar(ub, e, yerr=[e-lo, hi-e], fmt='ko', ms=5, lw=1, capsize=2,
                   label='observations')
    uu = np.linspace(ub.min()*0.85, ub.max()*1.15, 100)
    logu, loge = np.log10(ub), np.log10(e)
    for lf, c, ls in [(None, 'tab:green', '-'), (1.0, 'tab:blue', '--'), (2.0, 'tab:red', '-.')]:
        Kg, l = wls(logu, loge, sig, lf)
        lab = (f'free $l$: $l$={l:.2f}, $K_g$={Kg*1e-3:.2e}' if lf is None
               else f'$l$={l:.0f} fixed: $K_g$={Kg*1e-3:.2e}')
        ax[2].plot(uu, Kg*uu**l, ls, color=c, lw=1.6, label=lab)
    for l, kg in OSL_KG.items():
        ax[2].plot(uu, kg*1e3*uu**l, ':', color='0.45', lw=1.3,
                   label=f'OSL posterior, $l$={l:.0f}' if l == 1.0 else None)
    ax[2].plot(uu, HERMAN_FREE[0]*1e3*uu**HERMAN_FREE[1], ':', color='tab:orange', lw=1.4,
               label=f'Herman+2015 NLS ($l$={HERMAN_FREE[1]})')
    ax[2].set(xscale='log', yscale='log', xlabel='inverted basal velocity $u_b$ (m yr$^{-1}$)',
              ylabel='erosion rate (mm yr$^{-1}$)')
    figstyle.panel(ax[2], 2)
    ax[2].legend(); ax[2].grid(alpha=.3, which='both')
    fig.tight_layout()
    p = f'{out}/fig10_kg_l_inversion.png'
    fig.savefig(p, dpi=150); plt.close(fig)
    print('wrote', p)


if __name__ == '__main__':
    nc, out = sys.argv[1], sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else 'uniform tau_ref'
    g = load(nc)
    fig9(g, out, tag)
    fig10(nc, out, tag)
