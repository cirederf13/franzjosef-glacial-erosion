#!/usr/bin/env python3
"""Scalar tau_ref calibration against observed 2013 surface velocity (see run_tau_ref_sweep.sh)."""
import glob, os, sys
import numpy as np, xarray as xr
from scipy.ndimage import binary_erosion

def metrics(nc):
    ds=xr.open_dataset(nc)
    mask=ds.icemask.isel(iterations=-1).values>0.5
    thk=ds.thk.isel(iterations=-1).values
    vs=ds.velsurf_mag.isel(iterations=-1).values
    vo=ds.velsurfobs_mag.isel(iterations=-1).values
    ub=ds.velbase_mag.isel(iterations=-1).values
    tot=ds.da_cost_total_hist.values; ds.close()
    if len(tot)==0: return None
    it=binary_erosion(mask,iterations=3)
    res=(vs-vo)[it]
    return dict(nbit=len(tot), rmse=float(np.sqrt(np.mean(res**2))),
                bias=float(res.mean()),
                ratio=float(vs[it].mean()/vo[it].mean()),
                thk_mean=float(thk[it].mean()), thk_max=float(thk[it].max()),
                ub_mean=float(ub[it].mean()), ub_max=float(ub[it].max()))

def main(base='outputs/tau_ref_sweep'):
    runs={}
    for d in sorted(glob.glob(f'{base}/tau*_rep*')):
        nc=os.path.join(d,'optimize.nc')
        if not os.path.isfile(nc): continue
        tau=float(os.path.basename(d).split('_')[0][3:])
        m=metrics(nc)
        if m: runs.setdefault(tau,[]).append(m)
    print(f"{'tau_ref':>10} {'n':>3} {'RMSE':>14} {'bias':>14} {'us/uso':>14} "
          f"{'thk_mn':>8} {'ub_mn':>8} {'ub_mx':>8}")
    print('-'*95)
    best=None
    for tau in sorted(runs):
        M=runs[tau]; g=lambda k: np.array([m[k] for m in M])
        r,b,ra=g('rmse'),g('bias'),g('ratio')
        sd=lambda a: a.std(ddof=1) if len(a)>1 else 0.0
        print(f"{tau:>10.4f} {len(M):>3} {r.mean():>7.2f}+/-{sd(r):>5.2f} "
              f"{b.mean():>7.2f}+/-{sd(b):>5.2f} {ra.mean():>7.3f}+/-{sd(ra):>5.3f} "
              f"{g('thk_mean').mean():>8.1f} {g('ub_mean').mean():>8.1f} {g('ub_max').mean():>8.1f}")
        if best is None or r.mean()<best[1]: best=(tau,r.mean(),ra.mean())
    print()
    print(f"best misfit at tau_ref = {best[0]:.4f}  (RMSE {best[1]:.2f} m/yr, "
          f"usurf/usurfobs {best[2]:.3f})")
    # ratio closest to 1 = least biased velocity, which is what K_g needs
    cand=min(runs, key=lambda t: abs(np.mean([m['ratio'] for m in runs[t]])-1.0))
    rr=np.mean([m['ratio'] for m in runs[cand]])
    print(f"least-biased velocity at tau_ref = {cand:.4f}  (ratio {rr:.3f})")
    print()
    print("reference: IGM default (inherited, uncalibrated) tau_ref = 0.2153697218796329")
    print("           spatially-variable lam_tau=100 inversion: RMSE 70.6, ratio ~0.91")

if __name__=='__main__':
    main(sys.argv[1] if len(sys.argv)>1 else 'outputs/tau_ref_sweep')
