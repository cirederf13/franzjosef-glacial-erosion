#!/usr/bin/env python3
"""Ice-thickness panels through the glacial cycle.

Usage: python ice_extent_panel.py <output.nc> <out.png> [tag]
"""
import sys
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import figstyle
figstyle.apply()
from matplotlib.colors import LightSource, ListedColormap
from trunk_swath import swath_mask, XGAUGE, YGAUGE
from mapdeco import graticule, scalebar


def main(nc, png, tag=''):
    ds = xr.open_dataset(nc)
    x, y = ds.x.values, ds.y.values
    dx, dy = abs(float(x[1]-x[0])), abs(float(y[1]-y[0])); cell = dx*dy/1e6
    topg = ds.topg.values; topg = topg[0] if topg.ndim == 3 else topg
    thk = ds.thk.values; t = ds.time.values
    ds.close()
    sw, tr, dkm, cat = swath_mask(topg, x, y)
    shade = LightSource(azdeg=315, altdeg=50).hillshade(topg, vert_exag=1.4, dx=dx, dy=-dy)
    sea = topg <= 0.0
    gi = int(np.argmin(np.abs(y-YGAUGE))); gj = int(np.argmin(np.abs(x-XGAUGE)))
    picks = [int(np.argmin(np.abs(t-v))) for v in
             (-99000, -88000, -77000, -66000, -55000, -44000, -33000, -22000, -11000, -100)]
    fig, axes = plt.subplots(2, 5, figsize=(21, 9.6))
    for ax, i in zip(axes.ravel(), picks):
        ax.imshow(topg, cmap='gist_earth', origin='lower', vmin=-400, vmax=3000)
        ax.imshow(shade, cmap='gray', origin='lower', alpha=.35)
        ax.imshow(np.where(sea, 1.0, np.nan), cmap=ListedColormap(['#2b6ca3']), origin='lower')
        ice = thk[i] > 1
        ax.imshow(np.where(ice, 1.0, np.nan), cmap=ListedColormap(['#f7fbff']),
                  origin='lower', alpha=.95)
        ax.contour(ice.astype(float), [.5], colors='#08306b', linewidths=1.1)
        ax.contour(cat.astype(float), [.5], colors='k', linewidths=1.1, linestyles='--')
        ax.plot(gj, gi, 'r*', ms=11)
        ax.text(.03, .965, f"t = {t[i]/1000:.1f} ka\n{ice.sum()*cell:.0f} km$^2$ total\n"
                f"{(ice&cat).sum()*cell:.0f} km$^2$ in catchment\nterminus bed {topg[ice].min():.0f} m",
                transform=ax.transAxes, va='top', ha='left',
                fontsize=mpl.rcParams['xtick.labelsize']*.92,
                bbox=dict(fc='white', ec='none', alpha=.78, pad=2))
        figstyle.panel(ax, list(axes.ravel()).index(ax))
        ax.set_xticks([]); ax.set_yticks([])
        graticule(ax, x, y, nlines=2, fontsize=7)
        if ax is axes.ravel()[0]:
            scalebar(ax, dx, fontsize=7.5)
    fig.tight_layout(); fig.savefig(png, dpi=130)
    print(f'wrote {png}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else '')
