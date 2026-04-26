import os
import numpy as np
import pandas as pd
import rasterio

# ============================================================
# PATHS (EDIT)
# ============================================================
c2000_path = r"D:\Khalil\PHD_doc\software\GeoSOS-FLUS\Python\Newcode\Class\c2000.tif"
c2010_path = r"D:\Khalil\PHD_doc\software\GeoSOS-FLUS\Python\Newcode\Class\c2010.tif"
out_excel  = r"D:\Khalil\PHD_doc\software\GeoSOS-FLUS\Python\Newcode\change_2000_2010_and_2050.xlsx"

# Markov settings
step_years = 10
target_year = 2050
base_year = 2010
n_steps = (target_year - base_year) // step_years
if (target_year - base_year) % step_years != 0:
    raise ValueError("target_year-base_year must be divisible by step_years (e.g., 2010->2050 with step 10).")

# ============================================================
# Helpers
# ============================================================
def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.nodata
    return arr, nodata

def valid_mask(arr, nodata):
    if nodata is None:
        return np.ones(arr.shape, dtype=bool)
    if isinstance(nodata, float) and np.isnan(nodata):
        return ~np.isnan(arr)
    return arr != nodata

def class_counts(arr, mask):
    vals = arr[mask].astype(np.int64, copy=False)
    u, c = np.unique(vals, return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))

def transition_counts(a_from, a_to, mask, classes):
    """Counts transitions from class i -> class j (matrix)."""
    idx = {cls: i for i, cls in enumerate(classes)}
    mat = np.zeros((len(classes), len(classes)), dtype=np.int64)

    f = a_from[mask].astype(np.int64, copy=False)
    t = a_to[mask].astype(np.int64, copy=False)

    ok = np.isin(f, classes) & np.isin(t, classes)
    f = f[ok]
    t = t[ok]

    for ff, tt in zip(f, t):
        mat[idx[ff], idx[tt]] += 1
    return mat

def row_normalize(mat_counts):
    """Convert count matrix to transition probability matrix P (rows sum to 1)."""
    mat = mat_counts.astype(np.float64)
    row_sums = mat.sum(axis=1, keepdims=True)
    # Avoid division by zero (classes not present)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums != 0)
    return P

def mat_power(P, n):
    """Matrix power for transition matrix."""
    return np.linalg.matrix_power(P, n)

# ============================================================
# 1) Observed change 2000 -> 2010
# ============================================================
c2000, nodata_2000 = read_raster(c2000_path)
c2010, nodata_2010 = read_raster(c2010_path)

if c2000.shape != c2010.shape:
    raise ValueError(f"Raster shapes differ: 2000={c2000.shape} vs 2010={c2010.shape}")

mask = valid_mask(c2000, nodata_2000) & valid_mask(c2010, nodata_2010)

total_valid = int(mask.sum())
changed_2000_2010 = int(((c2000 != c2010) & mask).sum())
unchanged_2000_2010 = total_valid - changed_2000_2010
pct_changed_2000_2010 = (changed_2000_2010 / total_valid * 100.0) if total_valid else np.nan

counts_2000 = class_counts(c2000, mask)
counts_2010 = class_counts(c2010, mask)

classes = np.unique(np.concatenate([c2000[mask].ravel(), c2010[mask].ravel()])).astype(np.int64)
classes = np.sort(classes)

T_counts_2000_2010 = transition_counts(c2000, c2010, mask, classes)

# ============================================================
# 2) Markov projection to 2050 (using 2000->2010 transitions)
#    - Build P from counts
#    - Compute P^(n_steps) for 2010->2050 (40 years = 4 steps)
#    - Expected counts2050 = counts2010_vector @ P^(n_steps)
#    - Expected changed pixels (2010->2050) = total - expected unchanged
#       where expected unchanged = sum_i counts2010[i] * (P^n)[i,i]
# ============================================================
P_10yr = row_normalize(T_counts_2000_2010)
P_40yr = mat_power(P_10yr, n_steps)

# counts vector at 2010 aligned with classes
v2010 = np.array([counts_2010.get(int(cls), 0) for cls in classes], dtype=np.float64)

v2050_expected = v2010 @ P_40yr
expected_unchanged_2010_2050 = float(np.sum(v2010 * np.diag(P_40yr)))
expected_changed_2010_2050 = float(total_valid - expected_unchanged_2010_2050)
expected_pct_changed_2010_2050 = (expected_changed_2010_2050 / total_valid * 100.0) if total_valid else np.nan

# ============================================================
# Build DataFrames
# ============================================================
df_summary = pd.DataFrame([{
    "valid_pixels_total": total_valid,
    "changed_pixels_2000_to_2010": changed_2000_2010,
    "unchanged_pixels_2000_to_2010": unchanged_2000_2010,
    "changed_percent_2000_to_2010": pct_changed_2000_2010,
    "markov_steps_(10yr_each)_2010_to_2050": n_steps,
    "expected_changed_pixels_2010_to_2050_(Markov)": expected_changed_2010_2050,
    "expected_changed_percent_2010_to_2050_(Markov)": expected_pct_changed_2010_2050,
}])

df_counts = pd.DataFrame([{
    "class": int(cls),
    "count_2000": int(counts_2000.get(int(cls), 0)),
    "count_2010": int(counts_2010.get(int(cls), 0)),
    "net_change_2010_minus_2000": int(counts_2010.get(int(cls), 0) - counts_2000.get(int(cls), 0)),
    "expected_count_2050_(Markov)": float(v2050_expected[i]),
    "expected_net_change_2050_minus_2010": float(v2050_expected[i] - v2010[i]),
} for i, cls in enumerate(classes)])

df_trans_counts = pd.DataFrame(
    T_counts_2000_2010,
    index=[f"from_{c}" for c in classes],
    columns=[f"to_{c}" for c in classes],
)
df_trans_counts.index.name = "from_class"

df_P10 = pd.DataFrame(
    P_10yr,
    index=[f"from_{c}" for c in classes],
    columns=[f"to_{c}" for c in classes],
)
df_P10.index.name = "from_class"

df_P40 = pd.DataFrame(
    P_40yr,
    index=[f"from_{c}" for c in classes],
    columns=[f"to_{c}" for c in classes],
)
df_P40.index.name = "from_class"

# ============================================================
# Save Excel
# ============================================================
os.makedirs(os.path.dirname(out_excel), exist_ok=True)

with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
    df_summary.to_excel(writer, sheet_name="summary", index=False)
    df_counts.to_excel(writer, sheet_name="class_counts_2000_2010_2050", index=False)
    df_trans_counts.to_excel(writer, sheet_name="transition_counts_2000_2010")
    df_P10.to_excel(writer, sheet_name="P_10yr_from_2000_2010")
    df_P40.to_excel(writer, sheet_name="P_40yr_2010_to_2050")

print("DONE. Saved Excel:", out_excel)
