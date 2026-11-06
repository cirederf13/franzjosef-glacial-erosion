#!/usr/bin/env python3
"""Typography and line weights shared by the figure scripts."""
import matplotlib as mpl

SCALE = 1.38        # typography: +38%, in the 30-40% band asked for
LW_SCALE = 2.0      # lines: twice as thick
BASE = 10.0


def apply(scale=SCALE, lw_scale=LW_SCALE):
    """Install the paper's typography and line weights as matplotlib defaults."""
    f = BASE*scale
    mpl.rcParams.update({
        'font.size':          f,
        'axes.labelsize':     f,
        'axes.titlesize':     f,          # panels should not carry titles, but if
                                          # one is kept it must not look like a header
        'xtick.labelsize':    f*0.88,
        'ytick.labelsize':    f*0.88,
        'legend.fontsize':    f*0.80,
        'figure.titlesize':   f,
        'lines.linewidth':    1.5*lw_scale,
        'lines.markersize':   6.0*1.25,
        'axes.linewidth':     0.8*lw_scale,
        'grid.linewidth':     0.8*lw_scale*0.6,
        'xtick.major.width':  0.8*lw_scale,
        'ytick.major.width':  0.8*lw_scale,
        'xtick.major.size':   3.5*1.3,
        'ytick.major.size':   3.5*1.3,
        'patch.linewidth':    1.0*lw_scale,
        'errorbar.capsize':   3.0*1.4,
        'savefig.bbox':       'tight',
        'figure.constrained_layout.use': False,
    })


def lw(base=1.0, lw_scale=LW_SCALE):
    """A line width for calls that set linewidths explicitly (contours, outlines)."""
    return base*lw_scale


_SEQ = 'abcdefghijklmnopqrstuvwxyz'


def panel(ax, i, dx=0.0, dy=1.02, fontsize=None, weight='bold', color='k'):
    """Letter a panel (a), (b), ... in its upper-left, outside the axes frame."""
    lab = _SEQ[i] if isinstance(i, int) else str(i)
    return ax.text(dx, dy, f'({lab})', transform=ax.transAxes,
                   ha='left', va='bottom', fontweight=weight, color=color,
                   fontsize=fontsize or mpl.rcParams['axes.labelsize']*1.05,
                   clip_on=False)
