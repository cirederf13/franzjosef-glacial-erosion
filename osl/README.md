# OSL-thermochronometry: erosion rates from trapped-charge measurements (MATLAB)

Forward model that turns a measured trapped-charge fraction *n/N* into an erosion
(exhumation) rate, as described in the paper's section "Inversion of OSL data into
erosion rates". Requires MATLAB (`xlsread`, `interp1`); not tested with Octave.

| File | Role |
|---|---|
| `nN2erosion.m` | 1-D steady heat-advection model for the crustal column (κ = 30 km²/Myr, 25 km thick, 10 °C at the top and 700 °C at the base, 6 Myr), extracts the temperature–time path of a rock advected to the surface, predicts *n/N* with `trapping_BTM_FAD.m`, and compares it with the measurement. 500 random trial exhumation rates per run (`Vya1`, drawn in 0.1–11.9 mm/yr, unseeded), misfit = squared difference scaled by a 10 % uncertainty. Saves `[rate, misfit]` to `output/<name>.mat` |
| `trapping_BTM_FAD.m` | Trapping/detrapping kinetics with band-tail states and anomalous fading (King et al. 2016; Kars et al. 2008) — the rate equation in the paper |
| `P8xxxx.xlsx`, `P8xxxxminus.xlsx`, `P8xxxxplus.xlsx` | Per-sample inputs: columns `D_dot` (Gy/ka), `D0` (Gy), `log10_s`, `log10_rho_prime`, `Et` (eV), `Eu` (eV), `nN` (measured *n/N*), plus `nNss` and `Age`. The `minus`/`plus` files differ from the nominal file in `nN` only (the lower and upper bound of the measurement), so each sample gives three runs and hence a rate with asymmetric bounds |

Nine samples: P84252, P84253, P84268, P84269, P84286–P84290.

## Running

`nN2erosion.m` is a **single-sample template**: the input and output names are written
into it (lines 5–6, currently `P84290minus.xlsx` → `output/P84290minus.mat`). To process
every sample, edit those two lines for each of the 27 spreadsheets (or wrap the script
in a loop); run from this directory so `xlsread` finds the spreadsheets and `output/`
exists. The files are included exactly as supplied.

## Not included

How the 27 `output/*.mat` results were reduced to the erosion rates and uncertainties
in the paper's Table 2 (selecting the best-fit rate) is not scripted here.
