import os
import glob
import time
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import convolve
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import cv2

# =============================================================================
# USER CONTROLS
# =============================================================================
RNG_SEED = 42

ITERS_PER_YEAR = 8
RECOMPUTE_POTENTIAL_EVERY = 5

NB_ALPHA = 0.5
MIN_FORCE_THRESHOLD = 0.80

HARDNESS_STRENGTH = 0.70
USE_THRESHOLD_FILTER = False

# If True, classes with zero or negative Markov demand will not expand.
BLOCK_EXPANSION_WHEN_NO_POSITIVE_DEMAND = True
# it set the spead of expansion and contraction, but the final class counts are controlled by Markov demand, not speed.
CLASS_SPEED = {
    0: 0.05, #Building area
    1: 0.70, #Grassland
    2: 0.50, #Deciduous forest
    3: 0.55, #Coniferous forest
    4: 0.55, #Mixed forest
    5: 0.50, #Transitional forest
    6: 0.45, #Clearings
    7: 0.30, #Dwarf pine
    8: 0.30, #Peatland
    9: 0.15, #Rocks, stone seas
    10: 0.15 #Water bodies
}
DEFAULT_SPEED = 0.45

#It is important to note that speed is not directly controlling the final counts, but it can influence the order and likelihood of pixel changes during allocation.
# protection values for pixels currently in each class (when they are considered for removal) and for pixels being added to each class (when they are considered for addition).
#This value can be changeable for making diffrent scinarios, but it is important to note that these values are not controlling the final counts, but they can influence the order and likelihood of pixel changes during allocation.
FROM_PROTECT = {
    
    0: 0.95, #Building area
    1: 0.40, #Grassland
    2: 0.60, #Deciduous forest
    3: 0.70, #Coniferous forest
    4: 0.65, #Mixed forest
    5: 0.40, #Transitional forest
    6: 0.25, #Clearings   
    7: 0.80, #Dwarf pine   
    8: 0.85, #Peatland
    9: 0.95, #Rocks, stone seas
    10: 0.90 #Water bodies
}

TO_PROTECT = {
    0: 0.95, #Building area
    1: 0.40, #Grassland
    2: 0.50, #Deciduous forest
    3: 0.60, #Coniferous forest
    4: 0.55, #Mixed forest
    5: 0.25, #Transitional forest
    6: 0.20, #Clearings
    7: 0.80, #Dwarf pine
    8: 0.80, #Peatland
    9: 0.98, #Rocks, stone seas
    10: 0.95 #Water bodies
}

DEFAULT_FROM = 0.40
DEFAULT_TO = 0.40


def from_protect(code: int) -> float:
    return float(np.clip(FROM_PROTECT.get(int(code), DEFAULT_FROM), 0.0, 1.0))


def to_protect(code: int) -> float:
    return float(np.clip(TO_PROTECT.get(int(code), DEFAULT_TO), 0.0, 1.0))


def class_threshold(cls: int) -> float:
    s = float(CLASS_SPEED.get(int(cls), DEFAULT_SPEED))
    return float(np.clip(0.90 - 0.30 * s, 0.0, 1.0))


# =============================================================================
# HELPERS
# =============================================================================
def make_nodata_mask(arr, nodata_value):
    if nodata_value is None:
        return np.isnan(arr)
    if isinstance(nodata_value, float) and np.isnan(nodata_value):
        return np.isnan(arr)
    return (arr == nodata_value) | np.isnan(arr)


def sanitize_prob_map(prob_map):
    prob_map = np.nan_to_num(prob_map, nan=0.0, posinf=0.0, neginf=0.0)
    prob_map = np.clip(prob_map, 0.0, 1.0)
    return prob_map.astype(np.float32)


def safe_read_prob_map(path, height, width):
    with rasterio.open(path) as src:
        arr = src.read(1)

    if arr.shape != (height, width):
        raise ValueError(f"Shape mismatch: {arr.shape} != {(height, width)} for {path}")

    return sanitize_prob_map(arr)


def find_prob_map_file_by_code(class_code: int):
    class_code = int(class_code)

    candidates = [
        os.path.join(PROB_MAPS_DIR, f"probability_class_{class_code}.tif"),
        os.path.join(PROB_MAPS_DIR, f"probability_gain_to_{class_code}.tif"),
        os.path.join(PROB_MAPS_DIR, f"probability_class_{class_code}.tiff"),
        os.path.join(PROB_MAPS_DIR, f"probability_gain_to_{class_code}.tiff"),
    ]

    for fp in candidates:
        if os.path.exists(fp):
            return fp

    hits = glob.glob(os.path.join(PROB_MAPS_DIR, f"*{class_code}*.tif"))
    if hits:
        hits = sorted(hits, key=lambda x: len(os.path.basename(x)))
        return hits[0]

    return None


def list_excel_sheets(path):
    try:
        return list(pd.ExcelFile(path).sheet_names)
    except Exception:
        return []


def try_load_cost_matrix(cost_matrix_path, sheet_name, unique_classes):
    n_classes = len(unique_classes)

    sheets = list_excel_sheets(cost_matrix_path)
    if sheets and sheet_name not in sheets:
        print(f"[WARN] Cost sheet '{sheet_name}' not found.")
        print(f"[WARN] Available sheets: {sheets}")
        return None

    try:
        df = pd.read_excel(cost_matrix_path, sheet_name=sheet_name, index_col=0)

        def to_int_safe(x):
            try:
                return int(str(x).strip())
            except Exception:
                return None

        row_codes = [to_int_safe(x) for x in df.index]
        col_codes = [to_int_safe(x) for x in df.columns]

        if all(v is not None for v in row_codes) and all(v is not None for v in col_codes):
            M = np.ones((n_classes, n_classes), dtype=np.float32)
            df_numeric = df.apply(pd.to_numeric, errors="coerce")

            row_map = {int(c): i for i, c in enumerate(row_codes)}
            col_map = {int(c): j for j, c in enumerate(col_codes)}

            for i, r in enumerate(unique_classes):
                for j, c in enumerate(unique_classes):
                    if int(r) in row_map and int(c) in col_map:
                        val = df_numeric.iloc[row_map[int(r)], col_map[int(c)]]
                        if pd.notna(val):
                            M[i, j] = float(val)

            mn, mx = float(np.nanmin(M)), float(np.nanmax(M))
            if (mx - mn) > 1e-12 and (mn < 0 or mx > 1):
                M = (M - mn) / (mx - mn)

            return np.clip(M, 0.0, 1.0).astype(np.float32)

        df2 = pd.read_excel(cost_matrix_path, sheet_name=sheet_name, header=None)
        arr = df2.to_numpy(dtype=np.float32)

        if arr.shape == (n_classes, n_classes):
            mn, mx = float(np.min(arr)), float(np.max(arr))
            if (mx - mn) > 1e-12 and (mn < 0 or mx > 1):
                arr = (arr - mn) / (mx - mn)
            return np.clip(arr, 0.0, 1.0).astype(np.float32)

        print(f"[WARN] Cost matrix not usable. shape={arr.shape}, expected=({n_classes},{n_classes})")
        return None

    except Exception as e:
        print(f"[WARN] Cost matrix load failed: {e}")
        return None


# =============================================================================
# PATHS
# =============================================================================
WORKDIR = r"E:\Khalil\PHD_doc\software\Python\Article\Finall data and code for journal\Data\Results"
c2010 = r"E:\Khalil\PHD_doc\software\Python\Article\Finall data and code for journal\Data\Classification\c2010.tif"

cost_matrix_path = r"E:\Khalil\PHD_doc\software\Python\Article\Finall data and code for journal\Data\Classification\transition_matrix_2000_2010.xlsx"
cost_sheet_name = "Rio_2000to2010"

excel_path = r"E:\Khalil\PHD_doc\software\Python\Article\Finall data and code for journal\Data\Classification\change_2000_2010_and_2050.xlsx"
excel_sheet_name = "class_counts_2000_2010_2050"

PROB_MAPS_DIR = r"E:\Khalil\PHD_doc\software\Python\Article\Finall data and code for journal\Data\Results\transition_2000_2010_rf\probability_maps"

BASE_OUTPUT_DIR = os.path.join(WORKDIR, "outputs")
RASTER_OUT_DIR = os.path.join(BASE_OUTPUT_DIR, "predicted_rasters")
PNG_OUT_DIR = os.path.join(BASE_OUTPUT_DIR, "predicted_pngs")
VIDEO_OUT_DIR = os.path.join(BASE_OUTPUT_DIR, "video")

os.makedirs(RASTER_OUT_DIR, exist_ok=True)
os.makedirs(PNG_OUT_DIR, exist_ok=True)
os.makedirs(VIDEO_OUT_DIR, exist_ok=True)
os.chdir(WORKDIR)


# =============================================================================
# LOAD BASE LULC
# =============================================================================
with rasterio.open(c2010) as src:
    lulc_2010 = src.read(1)
    ref_transform = src.transform
    ref_crs = src.crs
    src_nodata = src.nodata
    height, width = lulc_2010.shape

nodata_value = src_nodata if src_nodata is not None else -9999

unique_classes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int16)
n_classes = len(unique_classes)

code_to_idx = {int(code): i for i, code in enumerate(unique_classes)}
idx_to_code = {i: int(code) for i, code in enumerate(unique_classes)}

valid_mask_2d = (~make_nodata_mask(lulc_2010, nodata_value)) & np.isin(lulc_2010, unique_classes)
total_valid_pixels = int(np.sum(valid_mask_2d))

if total_valid_pixels <= 0:
    raise ValueError("No valid pixels. Check nodata and class codes.")

current_counts = {
    int(c): int(np.sum((lulc_2010 == c) & valid_mask_2d))
    for c in unique_classes
}

print("unique_classes:", unique_classes)
print("n_classes:", n_classes)
print("Total valid pixels:", total_valid_pixels)
print("Counts 2010:", current_counts)


# =============================================================================
# LOAD PROBABILITY MAPS BY CLASS CODE
# =============================================================================
probability_maps = []
found_paths = []

for cls_code in unique_classes:
    fp = find_prob_map_file_by_code(int(cls_code))

    if fp is None:
        raise RuntimeError(f"Missing probability map for class {int(cls_code)}")

    probability_maps.append(safe_read_prob_map(fp, height, width))
    found_paths.append(fp)

probability_maps = np.stack(probability_maps, axis=0).astype(np.float32)

print("probability_maps.shape:", probability_maps.shape)
print("[INFO] Probability maps used:")
for j, cls_code in enumerate(unique_classes):
    print(f"  j={j}, class={int(cls_code)} -> {found_paths[j]}")


# =============================================================================
# LOAD TARGET COUNTS
# =============================================================================
df = pd.read_excel(excel_path, sheet_name=excel_sheet_name)
df["expected_count_2050_(Markov)"] = pd.to_numeric(
    df["expected_count_2050_(Markov)"],
    errors="coerce"
)

target_counts_2050 = dict(zip(df["class"], df["expected_count_2050_(Markov)"]))

for c in unique_classes:
    c = int(c)
    if c not in target_counts_2050 or pd.isna(target_counts_2050[c]):
        print(f"[WARN] target for class {c} missing -> using current {current_counts[c]}")
        target_counts_2050[c] = float(current_counts[c])

total_target = float(np.nansum(list(target_counts_2050.values())))
if total_target <= 0:
    raise ValueError("Invalid total target. Check Excel.")

if abs(total_valid_pixels - total_target) > 1:
    scale = total_valid_pixels / total_target
    target_counts_2050 = {
        int(k): float(v) * scale
        for k, v in target_counts_2050.items()
    }
    print("[INFO] Scaled targets:", {k: int(round(v)) for k, v in target_counts_2050.items()})


# =============================================================================
# YEARLY CHANGE RATES
# =============================================================================
start_year = 2010
end_year = 2050
n_years = end_year - start_year

yearly_change_rates = {}

for c in unique_classes:
    c = int(c)
    yearly_change_rates[c] = (
        float(target_counts_2050[c]) - float(current_counts[c])
    ) / float(n_years)

print("yearly_change_rates:", {k: round(v, 4) for k, v in yearly_change_rates.items()})


# =============================================================================
# LOAD COST MATRIX
# =============================================================================
cost_matrix = try_load_cost_matrix(
    cost_matrix_path,
    sheet_name=cost_sheet_name,
    unique_classes=unique_classes
)

USE_COST = cost_matrix is not None

if USE_COST:
    print("Loaded cost_matrix.shape:", cost_matrix.shape)
else:
    print("[WARN] Cost matrix disabled.")


# =============================================================================
# NEIGHBORHOOD KERNEL
# =============================================================================
neighborhood_kernel = np.array(
    [[0.05, 0.10, 0.05],
     [0.10, 0.40, 0.10],
     [0.05, 0.10, 0.05]],
    dtype=np.float32
)
neighborhood_kernel /= neighborhood_kernel.sum()


# =============================================================================
# COMPUTE POTENTIALS
# =============================================================================
def compute_flat_potential(current_lulc_2d: np.ndarray):
    nodata_mask_2d = ~valid_mask_2d
    valid_mask_flat = (~nodata_mask_2d).flatten()

    neighborhood_effect = np.zeros_like(probability_maps, dtype=np.float32)

    for cls_idx, cls_code in enumerate(unique_classes):
        binary = ((current_lulc_2d == cls_code) & (~nodata_mask_2d)).astype(np.float32)
        neighborhood_effect[cls_idx] = convolve(
            binary,
            neighborhood_kernel,
            mode="constant",
            cval=0.0
        )

    if USE_COST:
        transition_potential = np.zeros_like(probability_maps, dtype=np.float32)

        cur_code = current_lulc_2d.astype(np.int32)
        cur_idx = np.full(cur_code.shape, -1, dtype=np.int32)

        for code, idx in code_to_idx.items():
            cur_idx[cur_code == int(code)] = int(idx)

        ok = (cur_idx >= 0) & (~nodata_mask_2d)

        for j in range(n_classes):
            cost_factor = np.ones((height, width), dtype=np.float32)
            cost_factor[ok] = 1.0 - cost_matrix[cur_idx[ok], j]
            transition_potential[j] = probability_maps[j] * cost_factor
    else:
        transition_potential = probability_maps.copy()

    transition_potential = np.clip(transition_potential, 0.0, 1.0)

    # Block expansion into classes whose Markov demand is zero or negative.
    if BLOCK_EXPANSION_WHEN_NO_POSITIVE_DEMAND:
        for cls_code in unique_classes:
            cls_code = int(cls_code)
            if yearly_change_rates[cls_code] <= 0:
                j = code_to_idx[cls_code]
                transition_potential[j] = 0.0

    combined = transition_potential * (1.0 + NB_ALPHA * neighborhood_effect)

    s = np.sum(combined, axis=0, keepdims=True)
    combined = combined / (s + 1e-10)

    return combined.reshape(n_classes, -1), valid_mask_flat


# =============================================================================
# ALLOCATION
# =============================================================================
def allocate_class(target_class_code: int, k_int: int, counts: dict,
                   land_flat: np.ndarray, flat_potential: np.ndarray,
                   valid_mask_flat: np.ndarray):

    target_class_code = int(target_class_code)
    target_idx = code_to_idx[target_class_code]
    T_base = class_threshold(target_class_code)

    k = int(abs(k_int))
    if k <= 0:
        return land_flat, counts, 0

    # Do not add a class if Markov demand is zero or negative
    if k_int > 0 and BLOCK_EXPANSION_WHEN_NO_POSITIVE_DEMAND:
        if yearly_change_rates[target_class_code] <= 0:
            return land_flat, counts, 0

    valid_idx = np.where(valid_mask_flat)[0]
    cur_valid = land_flat[valid_mask_flat].astype(np.int32)

    # -------------------------------------------------------------------------
    # ADD pixels to target class
    # -------------------------------------------------------------------------
    if k_int > 0:
        mask = cur_valid != target_class_code

        if not np.any(mask):
            return land_flat, counts, 0

        cand = valid_idx[mask]
        prob = flat_potential[target_idx, cand].astype(np.float32)

        tp = to_protect(target_class_code)
        prob *= (1.0 - HARDNESS_STRENGTH * 0.60 * tp)

        from_codes = land_flat[cand].astype(np.int32)
        fp = np.array([from_protect(c) for c in from_codes], dtype=np.float32)
        prob *= (1.0 - HARDNESS_STRENGTH * 0.60 * fp)

        if USE_THRESHOLD_FILTER:
            T_eff = float(np.clip(T_base - 0.10 * tp, 0.01, 0.99))
            ok = prob >= T_eff
            cand = cand[ok]
            prob = prob[ok]

            if cand.size == 0:
                return land_flat, counts, 0

        pos = prob > 1e-8
        cand = cand[pos]
        prob = prob[pos]

        if cand.size == 0:
            return land_flat, counts, 0

        prob = prob + 1e-8 * np.random.random(prob.shape).astype(np.float32)

        top_k = min(k, cand.size)
        pick = cand[np.argpartition(-prob, top_k - 1)[:top_k]]

        if pick.size == 0:
            return land_flat, counts, 0

        old = land_flat[pick].astype(np.int32)
        land_flat[pick] = target_class_code

        for oc in np.unique(old):
            counts[int(oc)] -= int(np.sum(old == oc))

        counts[target_class_code] += int(pick.size)

        return land_flat, counts, +int(pick.size)

    # -------------------------------------------------------------------------
    # REMOVE pixels from target class
    # -------------------------------------------------------------------------
    else:
        mask = cur_valid == target_class_code

        if not np.any(mask):
            return land_flat, counts, 0

        cand = valid_idx[mask]

        probs = flat_potential[:, cand].copy()
        probs[target_idx, :] = -np.inf

        # Do not allow removed pixels to become classes with zero/negative demand
        if BLOCK_EXPANSION_WHEN_NO_POSITIVE_DEMAND:
            for cls_code in unique_classes:
                cls_code = int(cls_code)
                if yearly_change_rates[cls_code] <= 0:
                    probs[code_to_idx[cls_code], :] = -np.inf

        new_idx = np.argmax(probs, axis=0)
        best = probs[new_idx, np.arange(probs.shape[1])].astype(np.float32)

        fp_target = from_protect(target_class_code)
        best *= (1.0 - HARDNESS_STRENGTH * 0.60 * fp_target)

        if USE_THRESHOLD_FILTER:
            T_eff = float(np.clip(T_base - 0.10 * fp_target, 0.01, 0.99))
            ok = best >= T_eff
            cand2 = cand[ok]
            best2 = best[ok]
            new_idx2 = new_idx[ok]

            if cand2.size == 0:
                return land_flat, counts, 0
        else:
            cand2 = cand
            best2 = best
            new_idx2 = new_idx

        pos = best2 > 1e-8
        cand2 = cand2[pos]
        best2 = best2[pos]
        new_idx2 = new_idx2[pos]

        if cand2.size == 0:
            return land_flat, counts, 0

        best2 = best2 + 1e-8 * np.random.random(best2.shape).astype(np.float32)

        top_k = min(k, cand2.size)
        pick_pos = np.argpartition(-best2, top_k - 1)[:top_k]
        pick = cand2[pick_pos]

        if pick.size == 0:
            return land_flat, counts, 0

        new_codes = unique_classes[new_idx2[pick_pos]].astype(np.int16)
        land_flat[pick] = new_codes

        counts[target_class_code] -= int(pick.size)

        for nc in np.unique(new_codes):
            counts[int(nc)] += int(np.sum(new_codes == nc))

        return land_flat, counts, -int(pick.size)


# =============================================================================
# PLOTTING SETUP
# =============================================================================
land_use_classes = [
    "Building area",
    "Grassland",
    "Deciduous forest",
    "Coniferous forest",
    "Mixed forest",
    "Transitional forest",
    "Clearings",
    "Dwarf pine",
    "Peatland",
    "Rocks, stone seas",
    "Water bodies",
    "NoData"
]

cmap = mcolors.ListedColormap([
    "red",
    "yellow",
    "#006400",
    "#228B22",
    "#ADFF2F",
    "#b3c232",
    "#e0e0c9",
    "#bec29f",
    "#e0e0da",
    "black",
    "blue",
    "white"
])

ticks = np.arange(12)
bounds = np.arange(-0.5, 11.5, 1)
norm = mcolors.BoundaryNorm(bounds, cmap.N)


# =============================================================================
# SIMULATION
# =============================================================================
rng = np.random.default_rng(RNG_SEED)

current_lulc = lulc_2010.copy()
current_counts_year = current_counts.copy()

demand_carry = {int(c): 0.0 for c in unique_classes}

for year in range(start_year + 1, end_year + 1):
    print("\n==============================")
    print(f"Year {year}")

    t0 = time.time()
    prev = current_lulc.copy()

    land_flat = current_lulc.flatten().copy()
    counts = current_counts_year.copy()

    flat_potential = None
    valid_mask_flat = None
    applied_this_year = 0

    for it in range(ITERS_PER_YEAR):
        if flat_potential is None or (it % RECOMPUTE_POTENTIAL_EVERY == 0):
            temp_map = land_flat.reshape(height, width)
            flat_potential, valid_mask_flat = compute_flat_potential(temp_map)

        for cls_code in rng.permutation(unique_classes):
            cls_code = int(cls_code)

            # IMPORTANT:
            # Speed is NOT used here.
            # Markov demand must control final class counts.
            delta = yearly_change_rates[cls_code] / float(ITERS_PER_YEAR)
            demand_carry[cls_code] += delta

            k_want = int(np.trunc(demand_carry[cls_code]))

            if k_want != 0:
                land_flat, counts, applied = allocate_class(
                    cls_code,
                    k_want,
                    counts,
                    land_flat,
                    flat_potential,
                    valid_mask_flat
                )

                demand_carry[cls_code] -= applied
                applied_this_year += abs(int(applied))

    temp_map = land_flat.reshape(height, width)
    flat_potential, valid_mask_flat = compute_flat_potential(temp_map)

    for cls_code in unique_classes:
        cls_code = int(cls_code)
        carry = float(demand_carry[cls_code])

        if carry >= MIN_FORCE_THRESHOLD:
            land_flat, counts, applied = allocate_class(
                cls_code,
                +1,
                counts,
                land_flat,
                flat_potential,
                valid_mask_flat
            )
            demand_carry[cls_code] -= applied
            applied_this_year += abs(int(applied))

        elif carry <= -MIN_FORCE_THRESHOLD:
            land_flat, counts, applied = allocate_class(
                cls_code,
                -1,
                counts,
                land_flat,
                flat_potential,
                valid_mask_flat
            )
            demand_carry[cls_code] -= applied
            applied_this_year += abs(int(applied))

    current_lulc = land_flat.reshape(height, width)
    current_counts_year = counts.copy()

    changed_pixels = int(np.sum((current_lulc != prev) & valid_mask_2d))

    print(f"[DIAG] pixels changed this year: {changed_pixels}")
    print(f"[DIAG] applied counter: {applied_this_year}")

    out_tif = os.path.join(RASTER_OUT_DIR, f"land_use_{year}.tif")

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": rasterio.int16,
        "crs": ref_crs,
        "transform": ref_transform,
        "nodata": int(nodata_value),
        "compress": "lzw"
    }

    with rasterio.open(out_tif, "w", **profile) as dst:
        arr_to_write = np.where(valid_mask_2d, current_lulc, int(nodata_value))
        dst.write(arr_to_write.astype(np.int16), 1)

    display = current_lulc.astype(np.int32).copy()
    display[~valid_mask_2d] = 11

    out_png = os.path.join(PNG_OUT_DIR, f"land_use_{year}.png")

    plt.figure(figsize=(12, 10))
    im = plt.imshow(display, cmap=cmap, norm=norm, interpolation="nearest")
    cbar = plt.colorbar(im, ticks=ticks, orientation="horizontal", fraction=0.05, pad=0.05)
    cbar.set_label("Class")
    cbar.set_ticklabels(land_use_classes)

    for lab in cbar.ax.get_xticklabels():
        lab.set_rotation(45)
        lab.set_fontsize(8)

    plt.title(f"Predicted Land Use {year} - Krkonoše Protected Area", pad=20)
    plt.axis("on")
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close()

    carry_sorted = sorted(
        [(k, float(v)) for k, v in demand_carry.items()],
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    print("[DIAG] top carry:", [(k, round(v, 3)) for k, v in carry_sorted])
    print("[OK] TIFF:", out_tif)
    print("[OK] PNG :", out_png)
    print("[OK] Time:", round(time.time() - t0, 2), "sec")
    print("[OK] Counts:", current_counts_year)

print("\nYearly land use simulation completed.")


# =============================================================================
# VIDEO
# =============================================================================
output_video = os.path.join(VIDEO_OUT_DIR, "land_use_timelapse.mp4")
frame_rate = 1

image_files = sorted(glob.glob(os.path.join(PNG_OUT_DIR, "land_use_*.png")))

if not image_files:
    print("[WARN] No PNGs found for video.")
else:
    first = cv2.imread(image_files[0])

    if first is None:
        raise ValueError("Could not read first PNG for video sizing.")

    h, w = first.shape[:2]
    frame_size = (w, h)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(output_video, fourcc, frame_rate, frame_size)

    for fp in image_files:
        img = cv2.imread(fp)

        if img is None:
            print(f"[WARN] Could not read {fp}, skipping.")
            continue

        if (img.shape[1], img.shape[0]) != frame_size:
            img = cv2.resize(img, frame_size)

        vw.write(img)

    vw.release()
    print("[OK] Video saved:", output_video)