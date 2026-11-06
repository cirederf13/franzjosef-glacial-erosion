#!/usr/bin/env python3
"""Along-flow u_b(d), u_s(d) profiles from the inversion, binned by geodesic (in-ice) distance from the sampling station.

Usage: python geodesic_profile.py <optimize.nc> [out_csv] [--bin 500] [--herman-bins]
"""
import sys
import numpy as np
import xarray as xr
from skimage.graph import MCP_Geometric

X0, Y0 = 414909.10342628224, 5172956.631058902
XGAUGE_REL, YGAUGE_REL = 18.38e3, 17.48e3
XGAUGE, YGAUGE = X0 + XGAUGE_REL, Y0 + YGAUGE_REL
BIN_M = 500.0
# Herman et al. (2015) Fig. S2A staircase bin grid (see digitize_herman2015_figS2.py)
HERMAN_LO, HERMAN_W, HERMAN_N = 1400.5, 699.9, 12
HERMAN_BINS = False


def geodesic_distance(mask, seed_rc, dx, dy):
    """Shortest in-ice path length (m) from seed to every ice cell."""
    aniso = abs(dx - dy) / max(dx, dy)
    if aniso > 0.01:
        raise ValueError(f"grid too anisotropic dx={dx} dy={dy} ({aniso:.1%}) for MCP_Geometric")
    if aniso > 1e-9:
        print(f"# note: grid anisotropy {aniso:.2%} (dx={dx}, dy={dy}); using mean spacing")
    dx = 0.5 * (dx + dy)
    costs = np.where(mask, 1.0, np.inf)
    mcp = MCP_Geometric(costs, fully_connected=True)
    cum, _ = mcp.find_costs([seed_rc])
    return cum * dx  # cumulative cost is in cell units -> metres


def main(nc, out_csv, bin_m):
    ds = xr.open_dataset(nc)
    x, y = ds.x.values, ds.y.values
    dx, dy = float(abs(x[1] - x[0])), float(abs(y[1] - y[0]))
    mask = ds.icemask.isel(iterations=-1).values > 0.5
    ub = ds.velbase_mag.isel(iterations=-1).values
    us = ds.velsurf_mag.isel(iterations=-1).values
    uo = ds.velsurfobs_mag.isel(iterations=-1).values
    thk = ds.thk.isel(iterations=-1).values
    ds.close()

    # gauge is off-ice: seed from the nearest ice cell, keep the offset
    iy = np.abs(y - YGAUGE).argmin()
    ix = np.abs(x - XGAUGE).argmin()
    ice_rc = np.argwhere(mask)
    d2 = ((x[ice_rc[:, 1]] - XGAUGE) ** 2 + (y[ice_rc[:, 0]] - YGAUGE) ** 2)
    k = int(d2.argmin())
    seed = tuple(ice_rc[k])
    offset = float(np.sqrt(d2[k]))
    print(f"# gauge UTM  = ({XGAUGE:.1f}, {YGAUGE:.1f})")
    print(f"# gauge cell = (iy={iy}, ix={ix}); on_ice={bool(mask[iy, ix])}")
    print(f"# geodesic seed cell = {seed}; gauge->seed offset = {offset:.1f} m")

    gd = geodesic_distance(mask, seed, dx, dy) + offset
    gd[~mask] = np.nan
    reach = np.isfinite(gd) & mask
    print(f"# ice cells = {mask.sum()}; reachable from seed = {reach.sum()}"
          f" ({100*reach.sum()/mask.sum():.1f}%)")
    print(f"# geodesic distance range = {np.nanmin(gd[reach]):.0f} .. {np.nanmax(gd[reach]):.0f} m")

    if HERMAN_BINS:
        # Herman et al. (2015) Fig. S2A bin grid, recovered from its vector paths
        edges = HERMAN_LO + HERMAN_W * np.arange(0, HERMAN_N + 1)
        print(f"# binning on Herman et al. (2015) grid: {HERMAN_N} bins of "
              f"{HERMAN_W} m from {HERMAN_LO} m")
    else:
        edges = np.arange(0.0, np.nanmax(gd[reach]) + bin_m, bin_m)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = reach & (gd >= lo) & (gd < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        rows.append((0.5 * (lo + hi), n,
                     float(ub[sel].mean()), float(us[sel].mean()),
                     float(uo[sel].mean()), float(thk[sel].mean())))

    hdr = "distance_m,n_cells,ub_mean,usurf_mean,usurfobs_mean,thk_mean"
    print(hdr)
    for r in rows:
        print(f"{r[0]:.0f},{r[1]},{r[2]:.3f},{r[3]:.3f},{r[4]:.3f},{r[5]:.2f}")
    if out_csv:
        with open(out_csv, "w") as f:
            f.write("# geodesic-binned profile per manuscript sec 3.3.3\n")
            f.write(f"# source={nc}\n# bin={bin_m} m (OUR choice; see docstring caveat)\n")
            f.write(f"# gauge=({XGAUGE:.1f},{YGAUGE:.1f}) seed={seed} offset={offset:.1f} m\n")
            f.write(hdr + "\n")
            for r in rows:
                f.write(f"{r[0]:.0f},{r[1]},{r[2]:.3f},{r[3]:.3f},{r[4]:.3f},{r[5]:.2f}\n")
        print(f"# wrote {out_csv}")


if __name__ == "__main__":
    nc = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    bm = BIN_M
    if "--bin" in sys.argv:
        bm = float(sys.argv[sys.argv.index("--bin") + 1])
    if "--herman-bins" in sys.argv:
        globals()["HERMAN_BINS"] = True
    main(nc, out, bm)
