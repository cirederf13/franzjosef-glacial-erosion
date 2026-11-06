#!/usr/bin/env python3
"""Loader for the nine OSL samples (exact coordinates where available, rounded otherwise)."""
import numpy as np
import pandas as pd

OSL_CSV   = 'data_osl/osl_erosion_rates.csv'
EXACT_TXT = 'data_osl/osl_sample_coordinates_exact.txt'


def load_sites(csv=OSL_CSV, exact=EXACT_TXT, prefer_exact=True):
    """All nine samples with the best position available for each."""
    osl = pd.read_csv(csv, comment='#')
    osl['sample'] = osl['sample'].str.lower()
    osl['exact'] = False
    osl['elev_m'] = np.nan
    if not prefer_exact:
        return osl
    try:
        ex = pd.read_csv(exact, sep=r'\s+', header=None,
                         names=['sample', 'longitude', 'latitude', 'elev_m'])
    except FileNotFoundError:
        return osl
    ex['sample'] = ex['sample'].str.lower()
    ex = ex.drop_duplicates('sample').set_index('sample')
    for i, s in enumerate(osl['sample']):
        if s in ex.index:
            osl.loc[i, 'longitude'] = float(ex.longitude[s])
            osl.loc[i, 'latitude'] = float(ex.latitude[s])
            osl.loc[i, 'elev_m'] = float(ex.elev_m[s])
            osl.loc[i, 'exact'] = True
    return osl


def describe(osl):
    n = int(osl['exact'].sum())
    miss = list(osl.loc[~osl['exact'], 'sample'])
    return (f'{n}/{len(osl)} OSL samples at exact coordinates'
            + (f'; rounded (~1 km) for {", ".join(miss)}' if miss else ''))
