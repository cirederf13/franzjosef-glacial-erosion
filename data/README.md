Place the 78 m DEM here as `topg.tif` (EPSG:32759, 483 x 516 cells), built by
`../prepare_topg.py` from the raw clipped DEM `FranzJosefClipUTM.tif` (place that file in
the repository root). `topg_v1_buggy_edges.tif` (the pre-repair DEM) is also read by
`make_fig_samples.py` and `lgm_and_present_extent.py` to identify reconstructed cells.
The DEM is not distributed with this repository.
