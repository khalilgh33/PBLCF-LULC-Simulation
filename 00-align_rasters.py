import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# === INPUTS ===
input_folder = r"E:\Khalil\PHD_doc\software\Python\Final\Class\c2000.tif"   # folder with your .tif files
output_folder = r"E:\Khalil\PHD_doc\software\Python\Final\Variable" # folder for aligned rasters

# Pick one raster as reference (rows, cols, resolution, CRS, extent)
reference_file = r"D:\Khalil\Říčany\Urban_Atlas\MOLUSCE\ŘíčanyUA2012.tif"


os.makedirs(output_folder, exist_ok=True)

# ---------- helpers ----------
def dtype_range(dtype_str):
    info = np.iinfo(dtype_str) if np.dtype(dtype_str).kind in ("i", "u") else np.finfo(dtype_str)
    return info.min, info.max

def default_nodata_for_dtype(dtype_str):
    kind = np.dtype(dtype_str).kind
    if kind == "u":  # unsigned int
        # choose the max value to avoid clashing with typical class ranges
        return dtype_range(dtype_str)[1]   # e.g., 255 for uint8, 65535 for uint16
    if kind == "i":  # signed int
        return -9999 if dtype_range(dtype_str)[0] <= -9999 else dtype_range(dtype_str)[0]
    if kind == "f":  # float
        return np.nan  # GeoTIFF supports NaN as NoData for float32/64
    return None

def coerce_nodata(dtype_str, nodata):
    """Return a nodata value guaranteed to be valid for dtype_str."""
    if nodata is None:
        return default_nodata_for_dtype(dtype_str)

    kind = np.dtype(dtype_str).kind
    mn, mx = dtype_range(dtype_str)

    # NaN is fine for float; for ints it's invalid
    if kind == "f":
        # allow NaN or any finite within range
        if isinstance(nodata, float) and np.isnan(nodata):
            return np.nan
        # clamp extreme float sentinels to something reasonable in-range
        if np.isfinite(nodata):
            # float nodata like -3.4e38 might overflow float32 max; switch to NaN
            if nodata < mn or nodata > mx:
                return np.nan
            return float(nodata)
        return np.nan

    # integer dtypes: nodata must be an integer in-range
    try:
        nodata_int = int(nodata)
    except Exception:
        return default_nodata_for_dtype(dtype_str)

    if nodata_int < mn or nodata_int > mx:
        return default_nodata_for_dtype(dtype_str)
    return nodata_int

def pick_resampling(dtype_str, is_categorical=False):
    if is_categorical:
        return Resampling.nearest
    return Resampling.nearest if np.dtype(dtype_str).kind in ("u", "i") else Resampling.bilinear

# ---------- reference grid ----------
with rasterio.open(reference_file) as ref:
    REF_TRANSFORM = ref.transform
    REF_CRS       = ref.crs
    REF_WIDTH     = ref.width
    REF_HEIGHT    = ref.height
    REF_PROFILE   = ref.profile

def align_to_reference(in_path, out_path, is_categorical=False):
    with rasterio.open(in_path) as src:
        src_dtype   = src.dtypes[0]
        raw_nodata  = (src.nodatavals[0] if src.nodatavals and src.nodatavals[0] is not None
                       else src.nodata)
        nodata_val  = coerce_nodata(src_dtype, raw_nodata)

        out_profile = REF_PROFILE.copy()
        out_profile.update({
            "width": REF_WIDTH,
            "height": REF_HEIGHT,
            "transform": REF_TRANSFORM,
            "crs": REF_CRS,
            "count": src.count,
            "dtype": src_dtype,
            "compress": "deflate",
            "tiled": True,
            "nodata": nodata_val
        })

        resampling = pick_resampling(src_dtype, is_categorical=is_categorical)

        with rasterio.open(out_path, "w", **out_profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=REF_TRANSFORM,
                    dst_crs=REF_CRS,
                    resampling=resampling,
                    src_nodata=nodata_val,   # use coerced nodata consistently
                    dst_nodata=nodata_val,
                    init_dest_nodata=True
                )

# ---------- batch process ----------
for name in os.listdir(input_folder):
    if not name.lower().endswith(".tif"):
        continue
    in_path  = os.path.join(input_folder, name)
    if os.path.abspath(in_path) == os.path.abspath(reference_file):
        continue
    out_path = os.path.join(output_folder, name)
    print(f"Aligning: {name}")
    # Set True for LULC/classes; False for continuous rasters.
    align_to_reference(in_path, out_path, is_categorical=False)

print("✅ Finished aligning all rasters.")
