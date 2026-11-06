# Franz Josef Glacier: glacial erosion over a glacial cycle — code

Code, configuration and small input files behind the paper *"Glacial erosion during a
glacial cycle: insights from the Franz Josef Glacier / Kā Roimata o Hine Hukatere"*
[authors, journal, DOI: **to be filled in**].

The ice-flow model is [IGM](https://github.com/instructed-glacier-model/igm)
(Instructed Glacier Model), **not** included here: the results were produced with
release `v3.2.0`, commit `c5b3434045f625145f41bc49de36e08a4706f42a`, TensorFlow 2.17,
Python 3.11, on one NVIDIA RTX 4090.

> **Status:** this repository contains the analysis and run code exactly as used, but
> several inputs and one component cannot be redistributed or are not yet added. See
> [What is not included](#what-is-not-included) before trying to reproduce anything.

## What the code does

1. **Terrain** — `prepare_topg.py` builds the 78 m DEM (`data/topg.tif`) from the raw
   clipped DEM, filling nodata cells (sentinel values and a border-connected zero region)
   with the nearest valid pixel.
2. **Modern basal friction (2013 inversion).** IGM's `field_inversion` infers ice thickness
   from the observed 2013 surface velocities with a uniform, prescribed Weertman `tau_ref`
   (`experiment/inversion_2013_uniform_tau.yaml`, inputs in `data_inversion/`).
   - `run_lambda_h_sweep.sh`, `analyze_lambda_h.py`: thickness-regularisation L-curve.
   - `run_tau_ref_sweep.sh`, `analyze_tau_ref.py`: `tau_ref` calibration (adopted: 0.13 MPa).
   - `geodesic_profile.py`, `fit_modern_kg.py`, `digitize_herman2015_figS2.py`: along-flow
     `u_b(d)` profile on the bins of Herman et al. (2015) and the present-day `K_g` and exponent.
3. **Forcing.** `glacial_cycle_smb.txt` is the proxy-derived ELA(t) curve;
   `make_smb_rescaled_lgm_v4.py` applies the anchor-preserving rescale; the maximum
   accumulation was then rescaled to give `glacial_cycle_smb_rescaled_lgm_v5.txt`, the
   forcing used in the paper (that last step is not scripted).
4. **Glacial cycle.** `./run_glacial_cycle.sh` runs
   `experiment/glacial_cycle_rescaled_lgm_v6_native.yaml` (about 100 ka at native 78 m,
   `tau_ref = 0.13`, about 30 h on one GPU). The run used in the paper was resumed at
   t = -98030.8 yr from a snapshot of an identical earlier run, with `make_restart.py`, so
   its saved output starts there.
5. **OSL-thermochronometry (MATLAB, `osl/`).** Heat-advection plus trapping-kinetics model
   converting each sample's measured *n/N* into an erosion rate; see `osl/README.md`.
6. **Erosion and figures** (run from the repository root):

| Manuscript figure | Command |
|---|---|
| Sample map | `python make_fig_samples.py out.png --tonly --plain --sea` |
| ELA history / erosion maps / swath profile | `python plot_cycle_results.py <output.nc> <outdir> <tag> --kg1 V --kg2 V` |
| Ice extent through the cycle | `python ice_extent_panel.py <output.nc> out.png <tag>` |
| Bayesian inference | `python make_bayes_figures.py <output.nc> <bayes.json> <outdir> <tag>` |
| Velocity maps, `K_g` and `l` from the inversion | `python make_inversion_figures.py <optimize.nc> <outdir> <tag>` |

`plot_cycle_results.py` writes `cycle_history_<tag>.png`, `erosion_maps_<tag>.png` and
`swath_profile_<tag>.png`. The erosion posterior comes from
`bayesian_erosion_inference.py`. Hillslope parameters were held fixed
(`D = 0.026 mm/yr`, `S_c = 36.9 deg`) with `K_g` the only free amplitude, and
`erosion_params.py` reads the result from
`outputs/pipeline_v6native/bayes_v6native_exact_fixedD.json`:

```bash
python bayesian_erosion_inference.py outputs/glacial_cycle_v6_native_final/output.nc \
    --fix-D 2.6e-5 --fix-Sc 36.9 --out outputs/pipeline_v6native/bayes_v6native_exact_fixedD.json
```
(`2.6e-5` is the value of 0.026 mm/yr that the script takes; its own output labels the unit `m2/yr`.)

Shared modules: `trunk_swath.py` (catchment and swath), `osl_sites.py` (sample loader),
`erosion_params.py`, `figstyle.py`, `mapdeco.py`.

Some scripts read fixed paths: the final cycle output at
`outputs/glacial_cycle_v6_native_final/output.nc` (`make_fig_samples.py`) and the observed
2013 ice mask from `outputs/tau_ref_sweep/tau0.13_rep3/optimize.nc`.

## Install

```bash
conda env create -f environment.yml     # then check IGM installs (see below)
conda activate igm-erosion
```

`environment.yml` was written from the working environment but **has not been rebuilt
from scratch**; `environment-lock.txt` is the exact package list of that environment.
IGM is pinned to the commit above. NumPy must stay on 1.x: `make_restart.py` relies on
NumPy 1 value-based casting.

All scripts expect to be run **from the repository root** with paths relative to it
(`data/`, `data_inversion/`, `data_osl/`, `outputs/`, `experiment/`); IGM resolves the
forcing files and restart inputs relative to the directory `igm_run` is launched from.

## What is not included

| Item | Why | What to do |
|---|---|---|
| IGM | separate project (GPL-3.0) | install the pinned commit |
| `data/topg.tif` (DEM), raw `FranzJosefClipUTM.tif` | redistribution not cleared | place your copy in `data/`; run `prepare_topg.py`. **Note:** `data_inversion/topg.tif` is a copy of the same DEM |
| `data_osl/osl_erosion_rates.csv` (paper Table 2) | not added | see `data_osl/README.md` |
| `data_osl/osl_sample_coordinates_exact.txt` | exact sample locations, kept private | see `data_osl/README.md`; `osl_sites.py` needs it |
| `data_osl/herman2015_erosion_profile_figS2A.csv` | digitised from Herman et al. (2015) Fig. S2A | regenerate with `digitize_herman2015_figS2.py` |
| Simulation outputs (`output.nc`, about 1 GB each) | size | deposit on Zenodo and link here |
| Script building `glacial_cycle_smb.txt` from the isotope record; the accumulation rescale giving the v5 curve | not in this repository | the curves are provided as data |
| Step reducing `osl/output/*.mat` to the Table 2 rates | not scripted | see `osl/README.md` |

## Caveats

- **Not bit-for-bit reproducible.** IGM sets no random seeds; GPU floating-point
  non-associativity makes runs differ, and a run can blow up (NaN) early and succeed on
  a retry. `run_glacial_cycle.sh` retries up to three times and rejects NaN output.
- **The Bayesian inference script is a reconstruction.** The original (section 4.3) was lost;
  this version was rebuilt from the manuscript's description and checked against the
  previously published posteriors (agreement 3% for `l=1`, 15% for `l=2`, on an earlier
  run). The 2e6 draws are cheap because the prediction is linear in `K_g` and `D`.

## Licence

MIT — see `LICENSE`. Copyright holders to be confirmed before release.
