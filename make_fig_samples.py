#!/usr/bin/env python3
"""Sample map: bedrock topography, ice extent, catchment and the nine OSL samples.

The dashed LGM limit is the modelled extent at t = -26.03 ka. Cells reconstructed in the DEM repair are shown as terrain with --plain.

Usage: python make_fig_samples.py [out.png] [--plain] [--sea] [--tonly | --zoom [--zoom2]] [--suggate lonlat.txt]
"""
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import figstyle
figstyle.apply()
import matplotlib.patheffects as pe
from matplotlib.colors import LightSource, ListedColormap, LinearSegmentedColormap
from scipy.ndimage import binary_fill_holes
from trunk_swath import swath_mask, XGAUGE, YGAUGE
from mapdeco import graticule, scalebar

OSL = 'data_osl/osl_erosion_rates.csv'
DEM_TIF = 'data/topg.tif'
DEM_TIF_PRE = 'data/topg_v1_buggy_edges.tif'   # pre-repair, to recover the nodata mask
DEM_NC = 'outputs/glacial_cycle_v6_native_final/output.nc'
OBS_NC = 'outputs/tau_ref_sweep/tau0.13_rep3/optimize.nc'   # observed 2013 ice mask
LGM_T = -26030.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', nargs='?', default='outputs/figures_v6final/fig1_samples_map.png')
    ap.add_argument('--nc', default=DEM_NC, help='the cycle run to take the LGM extent from')
    ap.add_argument('--inset', action='store_true',
                    help='zoom inset on the p84286-p84290 elevation transect '
                         '(off by default; see the comment at the inset block)')
    ap.add_argument('--zoom', action='store_true',
                    help='bottom-right inset zoomed on all 9 OSL samples + gauge')
    ap.add_argument('--plain', action='store_true',
                    help='do not highlight the reconstructed DEM (drops the blue sea strips, '
                         'the grey overlay and its legend entry)')
    ap.add_argument('--tonly', action='store_true',
                    help='only the transect (286-290) zoom inset, bottom-right, linked to the map')
    ap.add_argument('--zoom2', action='store_true',
                    help='with --zoom: stack a second-level zoom on the 286-290 transect')
    ap.add_argument('--sea', action='store_true',
                    help='with --plain: draw a clean blue sea mask (no streaks)')
    ap.add_argument('--suggate', default=None,
                    help='optional 2-column (lon lat) published LGM limit')
    a = ap.parse_args()

    ds = xr.open_dataset(a.nc)
    x, y = ds.x.values, ds.y.values
    dx, dy = abs(float(x[1]-x[0])), abs(float(y[1]-y[0])); cell = dx*dy/1e6
    topg = ds.topg.values; topg = topg[0] if topg.ndim == 3 else topg
    t = ds.time.values; thk = ds.thk.values
    ds.close()
    lgm = thk[int(np.argmin(np.abs(t - LGM_T)))] > 1

    o = xr.open_dataset(OBS_NC)
    obs = binary_fill_holes(o.icemask.isel(iterations=-1).values > 0.5)
    o.close()

    sw, trunk, dkm, catch = swath_mask(topg, x, y)

    import rasterio
    with rasterio.open(DEM_TIF) as src:
        _fixed = src.read(1).astype(float)
    with rasterio.open(DEM_TIF_PRE) as src:
        _pre = src.read(1).astype(float)
    repaired = (_fixed != _pre)[::-1]   # rasterio is north-up, the netCDF south-up

    sea = (topg <= 0.0) & ~repaired
    if a.plain:
        if a.sea:
            from scipy.ndimage import label, binary_closing
            P = 12
            low = np.pad(topg <= 8.0, P, mode='edge')   # pad: closing must not erode the border
            cand = binary_closing(low, structure=np.ones((5, 5)), iterations=2)[P:-P, P:-P]
            lab, n = label(cand)
            ny_, nx_ = topg.shape
            corner = lab[ny_-20:, :20]                  # NW block (row ny-1 = north)
            seed = np.bincount(corner[corner > 0]).argmax()
            sea = binary_fill_holes(lab == seed)
        else:
            sea = np.zeros_like(sea)
        repaired = np.zeros_like(repaired)
    # dy negated: y ascends and imshow uses origin="lower"
    shade = LightSource(azdeg=315, altdeg=45).hillshade(topg, vert_exag=1.5, dx=dx, dy=-dy)

    from osl_sites import load_sites, describe
    df = load_sites()
    print('  ' + describe(df))
    from pyproj import Transformer
    tr = Transformer.from_crs('EPSG:4326', 'EPSG:32759', always_xy=True)
    sx, sy = tr.transform(df.longitude.values, df.latitude.values)
    si = np.array([np.argmin(np.abs(y-v)) for v in sy])
    sj = np.array([np.argmin(np.abs(x-v)) for v in sx])
    gi = int(np.argmin(np.abs(y-YGAUGE))); gj = int(np.argmin(np.abs(x-XGAUGE)))
    ndist = len({(i, j) for i, j in zip(si, sj)})

    terr = LinearSegmentedColormap.from_list(
        'terrain_land', plt.get_cmap('terrain')(np.linspace(0.25, 1.0, 256)))

    fig, ax = plt.subplots(figsize=(9.2, 8.6))
    im = ax.imshow(np.where(sea | repaired, np.nan, topg), cmap=terr, origin='lower',
                   vmin=0, vmax=3000)
    ax.imshow(shade, cmap='gray', origin='lower', alpha=.30)
    ax.imshow(np.where(sea, 1.0, np.nan), cmap=ListedColormap(['#9fc3de']), origin='lower')
    ax.imshow(np.where(repaired, 1.0, np.nan), cmap=ListedColormap(['0.80']),
              origin='lower', alpha=.85)
    if not a.plain:
        ax.plot([], [], 's', color='0.80', mec='0.5', ms=8,
                label='reconstructed DEM (no data in source)')

    if a.suggate:
        lonlat = np.loadtxt(a.suggate)
        px, py = tr.transform(lonlat[:, 0], lonlat[:, 1])
        ax.plot(np.interp(px, x, np.arange(len(x))),
                np.interp(py, y, np.arange(len(y))),
                '--', color='k', lw=2.0, label='LGM ice extent (Suggate, 2005)')
    else:
        ax.contour(lgm.astype(float), [.5], colors='k', linewidths=2.0, linestyles='--')
        ax.plot([], [], '--', color='k', lw=2.0,
                label='LGM ice extent, modelled ($t=-26.03$ ka)')

    ax.contour(obs.astype(float), [.5], colors='#08306b', linewidths=1.6)
    ax.imshow(np.where(obs, 1.0, np.nan), cmap=ListedColormap(['#f7fbff']),
              origin='lower', alpha=.75)
    ax.plot([], [], '-', color='#08306b', lw=1.6, label='present-day ice (2013)')

    ax.contour(catch.astype(float), [.5], colors='0.15', linewidths=1.1, linestyles='dashed')
    ax.plot([], [], color='0.15', lw=1.1, ls=(0, (5, 2)),
            label=f'Franz Josef catchment ({catch.sum()*cell:.0f} km$^2$)')

    prof = [i for i, nm in enumerate(df['sample'])
            if nm[-3:] in ('286', '287', '288', '289', '290')]
    order = sorted(prof, key=lambda i: -df.elev_m[i]) if prof else []
    if len(prof) > 1:
        ax.plot(sj[order], si[order], '-', color='red', lw=1.8, alpha=.85, zorder=5)
        lo, hi = df.elev_m[prof].min(), df.elev_m[prof].max()
        ax.plot([], [], '-o', color='red', mfc='red', mec='k', mew=.9, ms=7, lw=1.8,
                label=f'elevation transect, {lo:.0f}-{hi:.0f} m '
                      f'({len(prof)} samples)')
    ax.plot(sj, si, 'o', mfc='red', mec='k', mew=.9, ms=9, zorder=6,
            label=f'OSL samples ($n$={len(df)})')
    for i, nm in enumerate(df['sample']):
        if i in prof:
            continue
        ax.annotate(nm[-3:], (sj[i], si[i]), xytext=(8, 4),
                    textcoords='offset points', fontsize=8.5, zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.4, foreground='white')])

    if a.inset and len(prof) > 1:
        hp = 6                # cells; ~470 m half-window, so the 366 m transect
                              # fills the inset instead of sitting in its middle
        ci, cj = float(si[prof].mean()), float(sj[prof].mean())
        axz = ax.inset_axes([0.605, 0.045, 0.375, 0.315])
        axz.imshow(np.where(sea | repaired, np.nan, topg), cmap=terr,
                   origin='lower', vmin=0, vmax=3000)
        axz.imshow(shade, cmap='gray', origin='lower', alpha=.30)
        axz.imshow(np.where(obs, 1.0, np.nan), cmap=ListedColormap(['#f7fbff']),
                   origin='lower', alpha=.75)
        axz.contour(obs.astype(float), [.5], colors='#08306b', linewidths=1.6)
        axz.plot(sj[order], si[order], '-o', color='red', mfc='red', mec='k',
                 mew=.9, ms=8, lw=1.8, zorder=6)
        for k, i in enumerate(order):
            right = k % 2 == 0
            axz.annotate(f"{df['sample'][i][-3:]}  {df.elev_m[i]:.0f} m",
                         (sj[i], si[i]),
                         xytext=(11, 4) if right else (-11, -4),
                         textcoords='offset points',
                         ha='left' if right else 'right', va='center',
                         fontsize=8.2, zorder=8,
                         path_effects=[pe.withStroke(linewidth=2.4,
                                                     foreground='white')])
        axz.set_xlim(cj-hp, cj+hp); axz.set_ylim(ci-hp, ci+hp)
        axz.set_xticks([]); axz.set_yticks([])
        for sp in axz.spines.values():
            sp.set_edgecolor('red'); sp.set_linewidth(1.6)
        ax.indicate_inset_zoom(axz, edgecolor='red', lw=1.4, alpha=.95)
    if a.zoom and not a.tonly:
        pad = 12                                    # cells, ~0.9 km
        ii = np.append(si, gi); jj = np.append(sj, gj)
        bx = [0.715, 0.375, 0.265, 0.30] if a.zoom2 else [0.715, 0.035, 0.265, 0.40]
        axz = ax.inset_axes(bx)
        asp = (bx[3]*8.6) / (bx[2]*9.2)             # inset height/width in inches
        cj_, ci_ = (jj.min()+jj.max())/2, (ii.min()+ii.max())/2
        hw = max((jj.max()-jj.min())/2 + pad, ((ii.max()-ii.min())/2 + pad)/asp)
        hh = hw*asp
        axz.imshow(np.where(sea | repaired, np.nan, topg), cmap=terr, origin='lower',
                   vmin=0, vmax=3000)
        axz.imshow(shade, cmap='gray', origin='lower', alpha=.30)
        axz.imshow(np.where(obs, 1.0, np.nan), cmap=ListedColormap(['#f7fbff']),
                   origin='lower', alpha=.75)
        axz.contour(obs.astype(float), [.5], colors='#08306b', linewidths=1.4)
        axz.contour(catch.astype(float), [.5], colors='0.15', linewidths=1.0,
                    linestyles='dashed')
        if len(prof) > 1:
            axz.plot(sj[order], si[order], '-', color='red', lw=1.4, alpha=.85, zorder=5)
        axz.plot(sj, si, 'o', mfc='red', mec='k', mew=.8, ms=7, zorder=6)
        axz.plot(gj, gi, '*', color='gold', mec='k', mew=.8, ms=15, zorder=7)
        halo = [pe.withStroke(linewidth=2.4, foreground='white')]
        for i, nm in enumerate(df['sample']):
            if i in prof:
                continue
            axz.annotate(nm[-3:], (sj[i], si[i]), xytext=(7, 3),
                         textcoords='offset points', fontsize=8, zorder=8,
                         path_effects=halo)
        if len(prof) > 1:       # five labels 53-143 m apart cannot be separated: one tag
            axz.annotate('286-290', (sj[prof].mean(), si[prof].mean()),
                         xytext=(-9, 0), textcoords='offset points', ha='right',
                         va='center', fontsize=8, zorder=8, path_effects=halo)
        axz.set_xlim(cj_-hw, cj_+hw); axz.set_ylim(ci_-hh, ci_+hh)
        axz.set_xticks([]); axz.set_yticks([])
        for sp in axz.spines.values():
            sp.set_edgecolor('k'); sp.set_linewidth(1.4)
        # 1 km scale bar inside the inset
        L = 1000/dx
        x0, y0 = cj_-hw+hw*.08, ci_-hh+hh*.07
        axz.plot([x0, x0+L], [y0, y0], '-', color='k', lw=3, zorder=9,
                 solid_capstyle='butt', path_effects=[pe.withStroke(linewidth=5, foreground='white')])
        axz.text(x0+L/2, y0+hh*.035, '1 km', ha='center', va='bottom', fontsize=8,
                 zorder=9, path_effects=halo)
        ax.indicate_inset_zoom(axz, edgecolor='k', lw=1.2, alpha=.95)
        if a.zoom2 and len(prof) > 1:
            hp = 7
            ci2, cj2 = float(si[prof].mean()), float(sj[prof].mean()) + 1.8   # shift so the labels fit
            axb = ax.inset_axes([0.715, 0.035, 0.265, 0.30])
            axb.imshow(np.where(sea | repaired, np.nan, topg), cmap=terr, origin='lower',
                       vmin=0, vmax=3000)
            axb.imshow(shade, cmap='gray', origin='lower', alpha=.30)
            axb.imshow(np.where(obs, 1.0, np.nan), cmap=ListedColormap(['#f7fbff']),
                       origin='lower', alpha=.75)
            axb.contour(obs.astype(float), [.5], colors='#08306b', linewidths=1.4)
            axb.plot(sj[order], si[order], '-o', color='red', mfc='red', mec='k',
                     mew=.9, ms=8, lw=1.6, zorder=6)
            for k, i in enumerate(order):
                right = k % 2 == 0
                axb.annotate(f"{df['sample'][i][-3:]} \u00b7 {df.elev_m[i]:.0f} m",
                             (sj[i], si[i]),
                             xytext=(10, 3) if right else (-10, -3),
                             textcoords='offset points',
                             ha='left' if right else 'right', va='center',
                             fontsize=8, zorder=8, path_effects=halo)
            axb.set_xlim(cj2-hp, cj2+hp); axb.set_ylim(ci2-hp*asp, ci2+hp*asp)
            axb.set_xticks([]); axb.set_yticks([])
            for sp in axb.spines.values():
                sp.set_edgecolor('red'); sp.set_linewidth(1.5)
            L2 = 200/dx
            x0b, y0b = cj2-hp+hp*.12, ci2-hp*asp+hp*asp*.10
            axb.plot([x0b, x0b+L2], [y0b, y0b], '-', color='k', lw=3, zorder=9,
                     solid_capstyle='butt',
                     path_effects=[pe.withStroke(linewidth=5, foreground='white')])
            axb.text(x0b+L2/2, y0b+hp*asp*.05, '200 m', ha='center', va='bottom',
                     fontsize=8, zorder=9, path_effects=halo)
            axz.indicate_inset_zoom(axb, edgecolor='red', lw=1.2, alpha=.95)

    if a.tonly and len(prof) > 1:
        hp = 7
        asp = 0.34*8.6/(0.265*9.2)
        halo = [pe.withStroke(linewidth=2.4, foreground='white')]
        ci2, cj2 = float(si[prof].mean()), float(sj[prof].mean()) + 1.8
        axb = ax.inset_axes([0.715, 0.035, 0.265, 0.34])
        axb.imshow(np.where(sea | repaired, np.nan, topg), cmap=terr, origin='lower',
                   vmin=0, vmax=3000)
        axb.imshow(shade, cmap='gray', origin='lower', alpha=.30)
        axb.imshow(np.where(obs, 1.0, np.nan), cmap=ListedColormap(['#f7fbff']),
                   origin='lower', alpha=.75)
        axb.contour(obs.astype(float), [.5], colors='#08306b', linewidths=1.4)
        axb.plot(sj[order], si[order], '-o', color='red', mfc='red', mec='k',
                 mew=.9, ms=8, lw=1.6, zorder=6)
        for k, i in enumerate(order):
            right = k % 2 == 0
            axb.annotate(f"{df['sample'][i][-3:]} \u00b7 {df.elev_m[i]:.0f} m",
                         (sj[i], si[i]), xytext=(10, 3) if right else (-10, -3),
                         textcoords='offset points', ha='left' if right else 'right',
                         va='center', fontsize=8, zorder=8, path_effects=halo)
        axb.set_xlim(cj2-hp, cj2+hp); axb.set_ylim(ci2-hp*asp, ci2+hp*asp)
        axb.set_xticks([]); axb.set_yticks([])
        for sp in axb.spines.values():
            sp.set_edgecolor('red'); sp.set_linewidth(1.5)
        L2 = 200/dx
        x0b, y0b = cj2-hp+hp*.12, ci2-hp*asp+hp*asp*.10
        axb.plot([x0b, x0b+L2], [y0b, y0b], '-', color='k', lw=3, zorder=9,
                 solid_capstyle='butt',
                 path_effects=[pe.withStroke(linewidth=5, foreground='white')])
        axb.text(x0b+L2/2, y0b+hp*asp*.05, '200 m', ha='center', va='bottom',
                 fontsize=8, zorder=9, path_effects=halo)
        ax.indicate_inset_zoom(axb, edgecolor='red', lw=1.2, alpha=.95)

    ax.plot(gj, gi, '*', color='gold', mec='k', mew=.9, ms=19, zorder=7,
            label='sampling station (Herman et al., 2015)')

    ax.text((sj.min()-30) if a.zoom2 else sj.mean()+26,
            si.mean()+(62 if a.zoom2 else (72 if a.zoom else 30)),
            'Franz Josef Glacier /\nKā Roimata o Hine Hukatere',
            ha='right' if a.zoom2 else 'left', va='center', fontsize=mpl.rcParams['xtick.labelsize'],
            path_effects=[pe.withStroke(linewidth=2.8, foreground='white')])
    if not a.plain or a.sea:
      ax.annotate('Tasman Sea', xy=(0.055, 0.945), xycoords='axes fraction',
                fontsize=11, style='italic', color='#26547c',
                path_effects=[pe.withStroke(linewidth=2.6, foreground='white')])

    graticule(ax, x, y, nlines=4, fontsize=9)
    scalebar(ax, dx, fontsize=9)
    cb = plt.colorbar(im, ax=ax, fraction=.040, pad=.02)
    cb.set_label('bedrock elevation (m a.s.l.)', fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)
    ax.legend(loc='upper right', framealpha=.92)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print(f'wrote {a.out}')
    print(f'  {len(df)} OSL samples -> {ndist} distinct grid cells '
          f'(exact coordinates; see osl_sites.py)')
    if not a.suggate:
        print('  NOTE: dashed limit is the MODELLED LGM extent; the published '
              'Suggate (2005) outline and the Waiho Loop are not drawn '
              '(not available digitally)')


if __name__ == '__main__':
    main()
