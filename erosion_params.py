#!/usr/bin/env python3
"""Erosion-law parameters and colour scaling shared by the figure scripts."""
import numpy as np

EARLIER_FIGURES = {
    1: dict(Kg=3.50e-05, D=0.80, Sc=32.0),
    2: dict(Kg=2.18e-07, D=1.50, Sc=29.0),
}

GLACIAL_RANGE = (0.05, 20.0)


def ref(l, kg=None, D=None, Sc=None, posterior=None):
    """Parameters for exponent l."""
    r = dict(EARLIER_FIGURES[int(l)])
    if posterior:
        for k, key in (('Kg', 'Kg'), ('D', 'D'), ('Sc', 'Sc')):
            if key in posterior and 'median' in posterior[key]:
                v = float(posterior[key]['median'])
                if k == 'D':
                    v *= 1e3
                r[k] = v
    for k, v in (('Kg', kg), ('D', D), ('Sc', Sc)):
        if v is not None:
            r[k] = float(v)
    return r


def hillslope_mm_yr(slope, D_mm_yr, Sc_deg, floor=None):
    """Roering et al. (1999) nonlinear hillslope diffusion, in mm/yr."""
    sc = np.tan(np.deg2rad(Sc_deg))
    h = D_mm_yr * slope / (1.0 - np.clip(slope/sc, 0.0, 0.999)**2)
    return h if floor is None else np.maximum(h, floor)


def hillslope_range(hill_mm_yr, lo_pct=35, hi_pct=99.9, decades_max=4):
    """A log colour range matched to the hillslope field actually being plotted."""
    v = np.asarray(hill_mm_yr)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return GLACIAL_RANGE
    hi = float(np.percentile(v, hi_pct))
    lo = float(np.percentile(v, lo_pct))
    lo = max(lo, hi/10**decades_max)
    # round outward to whole decades so the bar ticks land on powers of ten
    return (10**np.floor(np.log10(lo)), 10**np.ceil(np.log10(hi)))


def spinup_index(ub, ice, region, tol=0.90, look=5):
    """First sample index at which the ice dynamics have settled."""
    n = len(ub)
    def mean_ub(k):
        m = ice[k] & region
        return float(ub[k][m].mean()) if m.any() else 0.0
    v = [mean_ub(k) for k in range(min(n, 40))]
    for k in range(len(v)):
        ref_v = float(np.median(v[k+1:k+1+look])) if k+1 < len(v) else v[k]
        if ref_v <= 0:
            continue
        r = v[k]/ref_v
        if tol <= r <= 1.0/tol:
            return k
    return 0


POSTERIOR_JSON = 'outputs/pipeline_v6native/bayes_v6native_exact_fixedD.json'

PLOT_D_MM_YR = 0.026
PLOT_SC_DEG = 36.9


def osl_posterior(path=POSTERIOR_JSON):
    """{1.0: {...}, 2.0: {...}} medians from the canonical sec. 4.3 inference."""
    import json
    r = json.load(open(path))['results']
    return {float(l): {k: r[l][k]['median'] for k in ('Kg', 'D', 'Sc')}
            for l in r}


def osl_kg(path=POSTERIOR_JSON):
    """{1.0: Kg, 2.0: Kg} -- the long-term Kg the modern values are compared to."""
    return {l: v['Kg'] for l, v in osl_posterior(path).items()}
