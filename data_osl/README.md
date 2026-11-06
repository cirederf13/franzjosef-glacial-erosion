Inputs read by `osl_sites.py`, `bayesian_erosion_inference.py`, the figure scripts and
`fit_modern_kg.py`. Not distributed here.

- `osl_erosion_rates.csv` — the nine OSL erosion rates with asymmetric uncertainties and
  rounded coordinates (paper Tables 1-2); `#`-prefixed header lines, then one row per sample.
- `osl_sample_coordinates_exact.txt` — whitespace-separated, no header:
  `sample  longitude  latitude  elevation_m` (surveyed positions; the paper's Table 1 is
  rounded to ~1 km, which is coarser than the sample spacing).
- `herman2015_erosion_profile_figS2A.csv` — Herman et al. (2015) Fig. S2A, produced by
  `../digitize_herman2015_figS2.py` from the supplement PDF.
