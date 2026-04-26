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

# more iterations = more chance to satisfy demand gradually
ITERS_PER_YEAR = 20

# recompute potentials more often (important when changes occur)
RECOMPUTE_POTENTIAL_EVERY = 1

# neighborhood influence (0=no neighborhood, 1=strong)
NB_ALPHA = 0.5

# if many years become "stuck", these help push small changes
MIN_FORCE_THRESHOLD = 0.80   # if carry exceeds this, force 1 pixel attempt

# -----------------------------------------------------------------------------
# SPEED (0..1): affects how fast demand is consumed for each class
#   IMPORTANT:
#   - This does NOT protect a class from being "stolen" by other classes.
#   - Protection is handled by FROM_PROTECT / TO_PROTECT below.
# -----------------------------------------------------------------------------
CLASS_SPEED = {
    0: 0.05,    # Building: very slow (recommended)
    1: 0.70,     # Grassland: easy donor (others can take it)
    2: 0.50,     # Deciduous
    3: 0.55,    # Coniferous
    4: 0.55,    # Mixed
    5: 0.50,    # Transitional
    6: 0.45,    # Clearings
    7: 0.40,    # Dwarf pine
    8: 0.30,    # Peatland
    9: 0.15,    # Rocks
    10: 0.15    # Water
}
DEFAULT_SPEED = 0.45

def class_threshold(cls: int) -> float:
    """
    Base threshold from SPEED:
    Higher speed => lower threshold.
    Too-high thresholds freeze the system.
    """
    s = float(CLASS_SPEED.get(int(cls), DEFAULT_SPEED))
    return float(np.clip(0.90 - 0.30 * s, 0.0, 1.0))

# -----------------------------------------------------------------------------
# HARDNESS / PRIORITY (0..1)
# These are the knobs you asked for:
#   FROM_PROTECT: protects pixels of this class from being taken by others.
#   TO_PROTECT  : makes it hard to create/expand into this class.
#
# Interpretation:
#   - Building should be HIGH in both => last priority to change.
#   - Grassland should be LOW in FROM_PROTECT => easiest donor.
# -----------------------------------------------------------------------------
FROM_PROTECT = {
    0: 0.98,  # Building: very protected (hard to be converted away)
    1: 0.10,  # Grassland: easy donor (others can take it)
    2: 0.60,  # Deciduous
    3: 0.70,  # Coniferous
    4: 0.65,  # Mixed
    5: 0.40,  # Transitional
    6: 0.25,  # Clearings
    7: 0.55,  # Dwarf pine
    8: 0.85,  # Peatland
    9: 0.95,  # Rocks
    10: 0.90  # Water
}

TO_PROTECT = {
    0: 0.95,  # Building: hard to expand into
    1: 0.10,  # Grassland: easy to create
    2: 0.50,
    3: 0.60,
    4: 0.55,
    5: 0.25,
    6: 0.20,
    7: 0.45,
    8: 0.80,
    9: 0.98,
    10: 0.95
}

DEFAULT_FROM = 0.40
DEFAULT_TO   = 0.40

def from_protect(code: int) -> float:
    return float(np.clip(FROM_PROTECT.get(int(code), DEFAULT_FROM), 0.0, 1.0))

def to_protect(code: int) -> float:
    return float(np.clip(TO_PROTECT.get(int(code), DEFAULT_TO), 0.0, 1.0))

# Strength of hardness penalties (0..1)
# If you still see buildings disappear, increase HARDNESS_STRENGTH to 0.95.
HARDNESS_STRENGTH = 0.85


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
    try:
        with rasterio.open(path) as src:
            arr = src.read(1)
        if arr.shape != (height, width):
            raise ValueError(f"Shape mismatch: {arr.shape} != {(height, width)}")
        return sanitize_prob_map(arr)
    except Exception as e:
        print(f"[WARN] Could not load prob map: {path}\n       {e}\n       -> using zeros")
        return np.zeros((height, width), dtype=np.float32)

def try_load_cost_matrix(cost_matrix_path, sheet_name, unique_classes):
    """
    Robust loader:
    - If the Excel sheet has row/col labels (class codes), align/reorder to unique_classes.
    - If raw numeric matrix, use only if shape matches (n_classes,n_classes).
    Returns: cost_matrix (float32 0..1) or None
    """
    n_classes = len(unique_classes)

    try:
        # Try labeled matrix
        df = pd.read_excel(cost_matrix_path, sheet_name=sheet_name, index_col=0)
        row_labels = list(df.index)
        col_labels = list(df.columns)

        def to_int_safe(x):
            try:
                return int(str(x).strip())
            except Exception:
                return None

        row_codes = [to_int_safe(x) for x in row_labels]
        col_codes = [to_int_safe(x) for x in col_labels]

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

        # Try raw numeric
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
# PATHS (EDIT)
# =============================================================================
WORKDIR = r"D:\Khalil\PHD_doc\software\GeoSOS-FLUS\Python\Newcode\Variable\transition_2000_2010_rf\probability_maps"
c2010 = r"d:\Khalil\PHD_doc\software\GeoSOS-FLUS\Python\Newcode\Class\c2014utm.tif"
cost_matrix_path = r"D:\Khalil\PHD_doc\software\GeoSOS-FLUS\Python\Newcode\Class\transition_matrix_1990_2010.xlsx"
excel_path = r"D:\Khalil\PHD_doc\software\GeoSOS-FLUS\Python\Newcode\change_2000_2010_and_2050.xlsx"
PROB_MAPS_DIR = r"D:\Khalil\PHD_doc\software\GeoSOS-FLUS\Python\Newcode\Variable\transition_2000_2010_rf\probability_maps"

BASE_OUTPUT_DIR = os.path.join(WORKDIR, "outputs")
RASTER_OUT_DIR = os.path.join(BASE_OUTPUT_DIR, "predicted_rasters")
PNG_OUT_DIR    = os.path.join(BASE_OUTPUT_DIR, "predicted_pngs")
VIDEO_OUT_DIR  = os.path.join(BASE_OUTPUT_DIR, "video")
os.makedirs(RASTER_OUT_DIR, exist_ok=True)
os.makedirs(PNG_OUT_DIR, exist_ok=True)
os.makedirs(VIDEO_OUT_DIR, exist_ok=True)

os.chdir(WORKDIR)

# =============================================================================
# LOAD 2010 LULC
# =============================================================================
with rasterio.open(c2010) as src:
    lulc_2010 = src.read(1)
    ref_transform = src.transform
    ref_crs = src.crs
    src_nodata = src.nodata
    height, width = lulc_2010.shape

nodata_value = src_nodata if src_nodata is not None else -9999

# Define your model classes explicitly
unique_classes = np.array([0,1,2,3,4,5,6,7,8,9,10], dtype=np.int16)
n_classes = len(unique_classes)

# CODE<->IDX maps
code_to_idx = {int(code): i for i, code in enumerate(unique_classes)}
idx_to_code = {i: int(code) for i, code in enumerate(unique_classes)}

valid_mask_2d = ~make_nodata_mask(lulc_2010, nodata_value)
total_valid_pixels = int(np.sum(valid_mask_2d))

current_counts = {int(c): int(np.sum((lulc_2010 == c) & valid_mask_2d)) for c in unique_classes}

print("unique_classes:", unique_classes)
print("n_classes:", n_classes)
print("Total valid pixels:", total_valid_pixels)
print("Counts 2010:", current_counts)

# =============================================================================
# LOAD PROBABILITY MAPS
# =============================================================================
probability_maps = []
for idx in range(n_classes):
    fp = os.path.join(PROB_MAPS_DIR, f"probability_class_{idx}.tif")
    probability_maps.append(safe_read_prob_map(fp, height, width))
probability_maps = np.stack(probability_maps, axis=0)  # (n_classes, H, W)
print("probability_maps.shape:", probability_maps.shape)

# =============================================================================
# LOAD TARGET COUNTS (2050)
# =============================================================================
df = pd.read_excel(excel_path, sheet_name="class_counts_2000_2010_2050")
df["expected_count_2050_(Markov)"] = pd.to_numeric(df["expected_count_2050_(Markov)"], errors="coerce")
target_counts_2050 = dict(zip(df["class"], df["expected_count_2050_(Markov)"]))

for c in unique_classes:
    c = int(c)
    if c not in target_counts_2050 or pd.isna(target_counts_2050[c]):
        print(f"[WARN] target for class {c} missing -> using current {current_counts[c]}")
        target_counts_2050[c] = float(current_counts[c])

# Normalize targets to match valid pixels
total_target = float(np.nansum(list(target_counts_2050.values())))
if total_target <= 0:
    raise ValueError("Invalid total target (check Excel).")

if abs(total_valid_pixels - total_target) > 1:
    scale = total_valid_pixels / total_target
    target_counts_2050 = {int(k): float(v) * scale for k, v in target_counts_2050.items()}
    print("[INFO] Scaled targets (rounded):", {k: int(round(v)) for k, v in target_counts_2050.items()})

# =============================================================================
# YEARLY BASE CHANGE RATES
# =============================================================================
start_year = 2010
end_year = 2050
n_years = end_year - start_year

yearly_change_rates = {}
for c in unique_classes:
    c = int(c)
    yearly_change_rates[c] = (float(target_counts_2050[c]) - float(current_counts[c])) / float(n_years)
print("yearly_change_rates:", {k: round(v, 4) for k, v in yearly_change_rates.items()})

# =============================================================================
# LOAD COST MATRIX (optional)
# =============================================================================
cost_matrix = try_load_cost_matrix(cost_matrix_path, sheet_name="Rio_1990to2010", unique_classes=unique_classes)
USE_COST = cost_matrix is not None
if USE_COST:
    print("Loaded cost_matrix.shape:", cost_matrix.shape)
else:
    print("[WARN] Cost matrix disabled (not loaded/aligned).")

# =============================================================================
# NEIGHBORHOOD KERNEL
# =============================================================================
neighborhood_kernel = np.array(
    [[0.05, 0.10, 0.05],
     [0.10, 0.40, 0.10],
     [0.05, 0.10, 0.05]], dtype=np.float32
)
neighborhood_kernel /= neighborhood_kernel.sum()

# =============================================================================
# COMPUTE POTENTIALS
# =============================================================================
def compute_flat_potential(current_lulc_2d: np.ndarray):
    nodata_mask_2d = make_nodata_mask(current_lulc_2d, nodata_value)
    valid_mask_flat = (~nodata_mask_2d).flatten() & np.isin(current_lulc_2d.flatten(), unique_classes)

    # neighborhood effect
    neighborhood_effect = np.zeros_like(probability_maps, dtype=np.float32)
    for cls_idx, cls_code in enumerate(unique_classes):
        binary = ((current_lulc_2d == cls_code) & (~nodata_mask_2d)).astype(np.float32)
        neighborhood_effect[cls_idx] = convolve(binary, neighborhood_kernel, mode="constant", cval=0.0)

    # transition potential
    transition_potential = np.zeros_like(probability_maps, dtype=np.float32)

    if USE_COST:
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

    # combine + normalize
    combined = transition_potential * (1.0 + NB_ALPHA * neighborhood_effect)
    s = np.sum(combined, axis=0, keepdims=True)
    combined = combined / (s + 1e-10)

    return combined.reshape(n_classes, -1), valid_mask_flat

# =============================================================================
# ALLOCATION with HARDNESS PRIORITY
# =============================================================================
def allocate_class(target_class_code: int, k_int: int, counts: dict,
                   land_flat: np.ndarray, flat_potential: np.ndarray, valid_mask_flat: np.ndarray):
    """
    k_int > 0 : add pixels to target class (enter)
    k_int < 0 : remove pixels from target class (exit)
    Returns: (land_flat, counts, applied)
      applied: signed int number of changed pixels (+added, -removed)
    """
    target_class_code = int(target_class_code)
    target_idx = code_to_idx[target_class_code]
    T_base = class_threshold(target_class_code)

    k = int(abs(k_int))
    if k <= 0:
        return land_flat, counts, 0

    valid_idx = np.where(valid_mask_flat)[0]
    cur_valid = land_flat[valid_mask_flat].astype(np.int32)

    # ---------------- ADD (ENTER target class) ----------------
    if k_int > 0:
        mask = cur_valid != target_class_code
        if not np.any(mask):
            return land_flat, counts, 0

        cand = valid_idx[mask]
        prob = flat_potential[target_idx, cand].astype(np.float32)

        # 1) penalty for entering protected target classes (e.g., building)
        tp = to_protect(target_class_code)
        prob *= (1.0 - HARDNESS_STRENGTH * tp)
        T_eff = min(0.995, T_base + 0.20 * tp)

        # 2) penalty for stealing from protected donor classes (KEY for building stability)
        from_codes = land_flat[cand].astype(np.int32)
        fp = np.array([from_protect(c) for c in from_codes], dtype=np.float32)
        prob *= (1.0 - fp)

        ok = prob >= T_eff
        if np.any(ok):
            cand = cand[ok]
            prob = prob[ok]

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

    # ---------------- REMOVE (EXIT target class) ----------------
    else:
        mask = cur_valid == target_class_code
        if not np.any(mask):
            return land_flat, counts, 0

        cand = valid_idx[mask]
        probs = flat_potential[:, cand].copy()
        probs[target_idx, :] = -np.inf

        new_idx = np.argmax(probs, axis=0)
        best = probs[new_idx, np.arange(probs.shape[1])].astype(np.float32)

        # penalty for leaving protected classes (e.g., building)
        fp_target = from_protect(target_class_code)
        best *= (1.0 - HARDNESS_STRENGTH * fp_target)
        T_eff = min(0.995, T_base + 0.20 * fp_target)

        ok = best >= T_eff
        if np.any(ok):
            cand2 = cand[ok]
            best2 = best[ok]
            new_idx2 = new_idx[ok]
        else:
            cand2 = cand
            best2 = best
            new_idx2 = new_idx

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
    "Building area", "Grassland", "Deciduous forest", "Coniferous forest",
    "Mixed forest", "Transitional forest", "Clearings", "Dwarf pine",
    "Peatland", "Rocks, stone seas", "Water bodies", "NoData"
]
cmap = mcolors.ListedColormap([
    "red", "yellow", "#006400", "#228B22", "#ADFF2F", "#b3c232",
    "#e0e0c9", "#bec29f", "#e0e0da", "black", "blue", "white"
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

# IMPORTANT: carry must NOT be destroyed when allocation fails
demand_carry = {int(c): 0.0 for c in unique_classes}

for year in range(start_year + 1, end_year + 1):
    print(f"\n==============================")
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
            speed = float(CLASS_SPEED.get(cls_code, DEFAULT_SPEED))

            delta = (yearly_change_rates[cls_code] * speed) / float(ITERS_PER_YEAR)
            demand_carry[cls_code] += delta

            k_want = int(np.trunc(demand_carry[cls_code]))
            if k_want != 0:
                land_flat, counts, applied = allocate_class(
                    cls_code, k_want, counts, land_flat, flat_potential, valid_mask_flat
                )
                demand_carry[cls_code] -= applied  # subtract ONLY what happened
                applied_this_year += abs(int(applied))

    # force attempts if stuck (still do NOT destroy carry)
    temp_map = land_flat.reshape(height, width)
    flat_potential, valid_mask_flat = compute_flat_potential(temp_map)

    for cls_code in unique_classes:
        cls_code = int(cls_code)
        carry = float(demand_carry[cls_code])

        if carry >= MIN_FORCE_THRESHOLD:
            land_flat, counts, applied = allocate_class(
                cls_code, +1, counts, land_flat, flat_potential, valid_mask_flat
            )
            demand_carry[cls_code] -= applied
            applied_this_year += abs(int(applied))

        elif carry <= -MIN_FORCE_THRESHOLD:
            land_flat, counts, applied = allocate_class(
                cls_code, -1, counts, land_flat, flat_potential, valid_mask_flat
            )
            demand_carry[cls_code] -= applied
            applied_this_year += abs(int(applied))

    current_lulc = land_flat.reshape(height, width)
    current_counts_year = counts.copy()

    changed_pixels = int(np.sum((current_lulc != prev) & valid_mask_2d))
    print(f"[DIAG] pixels changed this year: {changed_pixels} (attempted/applied counter={applied_this_year})")

    # SAVE TIFF
    out_tif = os.path.join(RASTER_OUT_DIR, f"land_use_{year}.tif")
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": rasterio.int16,
        "crs": ref_crs,
        "transform": ref_transform,
        "nodata": int(nodata_value)
    }
    with rasterio.open(out_tif, "w", **profile) as dst:
        arr_to_write = np.where(make_nodata_mask(current_lulc, nodata_value), int(nodata_value), current_lulc)
        dst.write(arr_to_write.astype(np.int16), 1)

    # SAVE PNG
    display = current_lulc.astype(np.int32).copy()
    display[make_nodata_mask(display, nodata_value)] = 11
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

    carry_sorted = sorted([(k, float(v)) for k, v in demand_carry.items()],
                          key=lambda x: abs(x[1]), reverse=True)[:5]
    print("[DIAG] top carry (abs):", [(k, round(v, 3)) for k, v in carry_sorted])

    print("[OK] TIFF:", out_tif)
    print("[OK] PNG :", out_png)
    print("[OK] Time:", round(time.time() - t0, 2), "sec")
    print("[OK] Counts:", current_counts_year)

print("\nYearly land use simulation completed.")

# =============================================================================
# VIDEO (optional)
# =============================================================================
output_video = os.path.join(VIDEO_OUT_DIR, "land_use_timelapse.mp4")
frame_rate = 1
frame_size = (1200, 1500)

image_files = sorted(glob.glob(os.path.join(PNG_OUT_DIR, "land_use_*.png")))
if not image_files:
    print("[WARN] No PNGs found for video.")
else:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(output_video, fourcc, frame_rate, frame_size)
    for fp in image_files:
        img = cv2.imread(fp)
        if img is None:
            print(f"[WARN] Could not read {fp}, skipping...")
            continue
        img = cv2.resize(img, frame_size)
        vw.write(img)
    vw.release()
    print("[OK] Video saved:", output_video)
