#!/usr/bin/env python3
"""Lat/lon graticule and scale bar for maps drawn with imshow in pixel-index space."""
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

CRS = 'EPSG:32759'
_cache = {}


def lonlat(x, y):
    """2D lon/lat for the projected grid axes x, y. Cached -- the transform is slow."""
    key = (float(x[0]), float(x[-1]), len(x), float(y[0]), float(y[-1]), len(y))
    if key not in _cache:
        from pyproj import Transformer
        tr = Transformer.from_crs(CRS, 'EPSG:4326', always_xy=True)
        X, Y = np.meshgrid(x, y)
        lo, la = tr.transform(X, Y)
        _cache[key] = (lo, la)
    return _cache[key]


def _nice(span, target=4):
    """A round graticule step giving roughly `target` lines across `span` degrees."""
    raw = span/max(target, 1)
    for st in (0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0):
        if st >= raw:
            return st
    return 5.0


def _dm(v, pos_suffix, neg_suffix):
    """Format a decimal degree as degrees and minutes, e.g. -43.4 -> 43°24'S."""
    suf = pos_suffix if v >= 0 else neg_suffix
    a = abs(v)
    d = int(np.floor(a + 1e-9))
    m = (a - d)*60.0
    if abs(m) < 0.05:
        return f"{d}\u00b0{suf}"
    return f"{d}\u00b0{m:02.0f}'{suf}"


def graticule(ax, x, y, color='0.25', lw=.45, alpha=.45, fontsize=8, nlines=4,
              ticks=True, lines=True, labelsize=None):
    """Lat/lon graticule for a map drawn in PIXEL-INDEX space."""
    lo, la = lonlat(x, y)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if lines:
        for fld in (lo, la):
            step = _nice(float(fld.max()-fld.min()), nlines)
            lv = np.arange(np.ceil(fld.min()/step)*step, fld.max()+step/2, step)
            lv = lv[(lv > fld.min()) & (lv < fld.max())]
            if len(lv):
                ax.contour(fld, levels=lv, colors=color, linewidths=lw, alpha=alpha)
        ax.set_xlim(xlim); ax.set_ylim(ylim)   # contours must not re-scale the panel

    if not ticks:
        return
    lon_edge, lat_edge = lo[0, :], la[:, 0]
    for fld_edge, axis in ((lon_edge, 'x'), (lat_edge, 'y')):
        step = _nice(float(fld_edge.max()-fld_edge.min()), nlines)
        lv = np.arange(np.ceil(fld_edge.min()/step)*step, fld_edge.max()+step/2, step)
        lv = lv[(lv > fld_edge.min()) & (lv < fld_edge.max())]
        if not len(lv):
            continue
        asc = fld_edge[-1] > fld_edge[0]
        src = fld_edge if asc else fld_edge[::-1]
        idx = np.arange(len(fld_edge)) if asc else np.arange(len(fld_edge))[::-1]
        pos = np.interp(lv, src, idx)
        lab = [_dm(v, 'E', 'W') if axis == 'x' else _dm(v, 'N', 'S') for v in lv]
        if axis == 'x':
            ax.set_xticks(pos); ax.set_xticklabels(lab)
        else:
            ax.set_yticks(pos); ax.set_yticklabels(lab)
    ax.tick_params(labelsize=labelsize or fontsize, length=3, width=.7,
                   direction='out', colors='0.15')
    for sp in ax.spines.values():
        sp.set_linewidth(.8)


def scalebar(ax, dx_m, loc=(.055, .045), frac=.26, color='k', fontsize=8.5,
             facecolor='white', alpha=.78):
    """A distance scale bar. dx_m is the grid spacing in metres (1 px = dx_m)."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    wpx = abs(x1-x0)
    # round the bar to a human number of km, close to `frac` of the map width
    km_raw = wpx*dx_m/1000.0*frac
    km = min((1, 2, 5, 10, 20, 25, 50, 100), key=lambda v: abs(np.log(v/max(km_raw, 1e-9))))
    bar = km*1000.0/dx_m
    bx = x0 + loc[0]*wpx
    by = y0 + loc[1]*abs(y1-y0)
    h = 0.011*abs(y1-y0)
    ax.add_patch(Rectangle((bx-0.14*bar, by-1.5*h), bar+0.28*bar, 5.2*h,
                           facecolor=facecolor, edgecolor='none', alpha=alpha,
                           zorder=8, transform=ax.transData))
    # two-tone bar: reads as a scale bar rather than a stray line
    for i in range(2):
        ax.add_patch(Rectangle((bx+i*bar/2, by), bar/2, h, zorder=9,
                               facecolor=color if i == 0 else 'white',
                               edgecolor=color, lw=.6))
    ax.text(bx+bar/2, by+1.5*h, f'{km:g} km', ha='center', va='bottom',
            fontsize=fontsize, color=color, zorder=9)
