
import numpy as np
import rasterio
from scipy import ndimage

RAW = "FranzJosefClipUTM.tif"
OUT = "data/topg.tif"

with rasterio.open(RAW) as src:
    z = src.read(1).astype("float64")
    profile = src.profile

sentinel_mask = z == -32768

zero_mask = z == 0
labeled, _ = ndimage.label(zero_mask)
border_labels = set(labeled[0, :]) | set(labeled[-1, :]) | set(labeled[:, 0]) | set(labeled[:, -1])
border_labels.discard(0)
border_zero_mask = np.isin(labeled, list(border_labels))

bad_mask = sentinel_mask | border_zero_mask
print(f"bad cells: {bad_mask.sum()} / {z.size} ({100 * bad_mask.sum() / z.size:.1f}%)")
print(f"  sentinel (-32768): {sentinel_mask.sum()}")
print(f"  border-connected zero blob: {border_zero_mask.sum()}")
print(f"  interior zero cells preserved as real data: {(zero_mask & ~border_zero_mask).sum()}")

_, indices = ndimage.distance_transform_edt(bad_mask, return_distances=True, return_indices=True)
filled = z.copy()
filled[bad_mask] = z[tuple(idx[bad_mask] for idx in indices)]

print(f"post-fill min/max: {filled.min():.1f} / {filled.max():.1f}")
assert not np.any(filled == -32768)

out_profile = profile.copy()
out_profile.update(dtype="float32", nodata=None)
with rasterio.open(OUT.replace("topg.tif", "topg_fixed.tif"), "w", **out_profile) as dst:
    dst.write(filled.astype("float32"), 1)

print("wrote data/topg_fixed.tif (candidate -- not yet swapped in)")
