import osgeo
import numpy as np
import pandas as pd
import rasterio

# -----------------------------
# Inputs
# -----------------------------
raster_1990 = r"D:\Khalil\PHD_doc\software\GeoSOS-FLUS\Python\Newcode\Class\c1990.tif"
raster_2010 = r"D:\Khalil\PHD_doc\software\GeoSOS-FLUS\Python\Newcode\Class\c2010.tif"
out_excel   = r"D:\Khalil\PHD_doc\software\GeoSOS-FLUS\Python\Newcode\Class\transition_matrix_1990_2010.xlsx"

# Class names (0..10)
CLASS_NAMES = {
    0:  "Building area",
    1:  "Grassland",
    2:  "Deciduous forest",
    3:  "Coniferous forest",
    4:  "Mixed forest",
    5:  "Transitional forest",
    6:  "Clearings, areas with low",
    7:  "Dwarf pine",
    8:  "Peatland",
    9:  "Rocks, stone seas",
    10: "Water bodies and streams",
}

N = 11  # number of classes

# -----------------------------
# Read rasters and validate
# -----------------------------
with rasterio.open(raster_1990) as src90, rasterio.open(raster_2010) as src10:
    if (src90.width != src10.width) or (src90.height != src10.height):
        raise ValueError("Raster shapes differ. You must align/resample them first.")
    if src90.transform != src10.transform:
        raise ValueError("Raster transforms differ. You must align/resample them first.")
    if src90.crs != src10.crs:
        raise ValueError("Raster CRS differ. You must align/resample them first.")

    a90 = src90.read(1)
    a10 = src10.read(1)

    nod90 = src90.nodata
    nod10 = src10.nodata

    # Pixel area (in map units^2) if you want area outputs
    pixel_area = abs(src90.transform.a * src90.transform.e)  # a=px width, e=-px height

# -----------------------------
# Mask valid pixels
# -----------------------------
valid = np.ones(a90.shape, dtype=bool)

if nod90 is not None:
    valid &= (a90 != nod90)
if nod10 is not None:
    valid &= (a10 != nod10)

# Keep only classes 0..10
valid &= (a90 >= 0) & (a90 < N) & (a10 >= 0) & (a10 < N)

from_cls = a90[valid].astype(np.int64)
to_cls   = a10[valid].astype(np.int64)

# -----------------------------
# Transition matrix (counts)
# -----------------------------
# Encode pair (from,to) as single index: from*N + to
pair_index = from_cls * N + to_cls
counts = np.bincount(pair_index, minlength=N * N).reshape(N, N)

# Labels in correct order 0..10
labels = [CLASS_NAMES[i] for i in range(N)]

df_counts = pd.DataFrame(counts, index=labels, columns=labels)

# Optional: row-normalized percentages (1990 class -> distribution in 2010)
row_sums = df_counts.sum(axis=1).replace(0, np.nan)
df_ratio = (df_counts.div(row_sums, axis=0)).round(2)

# Optional: area matrix (same shape) if CRS unit is meters (then m²)
df_area = (df_counts * pixel_area)

# -----------------------------
# Save to Excel (multiple sheets)
# -----------------------------
with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
    df_counts.to_excel(writer, sheet_name="Counts")
    df_ratio.to_excel(writer, sheet_name="Rio_1990to2010")
    df_area.to_excel(writer, sheet_name="Area_units2")

print("Saved:", out_excel)
