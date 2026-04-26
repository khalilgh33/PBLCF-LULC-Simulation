import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ============================================================
# PATHS (EDIT)
# ============================================================
c2000_path = r"D:\Khalil\PHD_doc\software\GeoSOS-FLUS\Python\Newcode\Class\c2000.tif"
c2010_path = r"D:\Khalil\PHD_doc\software\GeoSOS-FLUS\Python\Newcode\Class\c2010.tif"

variables_dir = r"D:\Khalil\PHD_doc\software\GeoSOS-FLUS\Python\Newcode\Variable"

out_root = os.path.join(variables_dir, "transition_2000_2010_rf")
change_dir = os.path.join(out_root, "change_maps")
prob_dir   = os.path.join(out_root, "probability_maps")
os.makedirs(change_dir, exist_ok=True)
os.makedirs(prob_dir, exist_ok=True)

variables = [
    'aspect.tif', 'building.tif', 'dis_to_church.tif', 'dis_to_hotels.tif',
    'dis_to_natural_place.tif', 'dis_to_rivers.tif', 'dis_to_roads.tif',
    'dis_to_staion.tif', 'dis_to_wpp.tif',
    'population.tif', 'rain.tif', 'slope.tif'
]

# ============================================================
# SETTINGS
# ============================================================
LULC_NODATA = -9999
PRED_NODATA_VALUES = [-9999, -32768]
FORCE_CRS = "EPSG:32633"

CLASSES = np.array([0,1,2,3,4,5,6,7,8,9,10], dtype=np.int16)

CLASS_NAMES = {
    0: "Building area",
    1: "Grassland",
    2: "Deciduous forest",
    3: "Coniferous forest",
    4: "Mixed forest",
    5: "Transitional forest",
    6: "Clearings, areas with low",
    7: "Dwarf pine",
    8: "Peatland",
    9: "Rocks, stone seas",
    10: "Water bodies and streams"
}

RANDOM_STATE = 42
N_JOBS = 4

TRAIN_POS_CAP = 30000
NEG_RATIO = 2.0
TEST_FRAC = 0.2
PRED_CHUNK = 200000

# Stable classes use presence/persistence ROC label instead of "gain"
STABLE_CLASSES = {0, 10}
ROC_MODE_FOR_STABLE = "presence"  # "presence" or "persistence"

# --- critical: separate thresholds ---
MIN_POS_FOR_TRAIN = 50   # produce probability map only if >= 50 positives (honest)
MIN_POS_FOR_ROC   = 5    # allow ROC curve if >= 5 positives (still weak, but possible)

# Pixels accepted if at least this fraction of predictors is finite (water often fails here)
MIN_VALID_FRAC_PRED = 0.50

rng = np.random.default_rng(RANDOM_STATE)

# ============================================================
# HELPERS
# ============================================================
def get_crs_or_force(src_crs, force_crs, where):
    if src_crs is not None:
        return src_crs
    if force_crs is None:
        raise ValueError(f"CRS missing in: {where}. Set FORCE_CRS.")
    return CRS.from_string(force_crs)

def read_to_template(path, template_crs, template_transform, template_shape,
                     resampling, force_crs=None, fill_value=None):
    with rasterio.open(path) as src:
        src_crs = get_crs_or_force(src.crs, force_crs, where=path)
        arr = src.read(1)
        template_crs = get_crs_or_force(template_crs, force_crs, where="TEMPLATE")

        need_warp = (src_crs != template_crs) or (src.transform != template_transform) or (arr.shape != template_shape)
        if not need_warp:
            return arr

        if fill_value is None:
            fill_value = 0
        dst = np.full(template_shape, fill_value, dtype=arr.dtype)

        reproject(
            source=arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=template_transform,
            dst_crs=template_crs,
            resampling=resampling
        )
        return dst

def save_uint8_map(path, arr_uint8, ref_profile):
    profile = ref_profile.copy()
    profile.update({"dtype": "uint8", "count": 1, "nodata": 255, "compress": "lzw"})
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr_uint8.astype(np.uint8), 1)

def balanced_subsample_indices(y01, pos_cap, neg_ratio, rng):
    pos = np.flatnonzero(y01 == 1)
    neg = np.flatnonzero(y01 == 0)
    if pos.size == 0 or neg.size == 0:
        return None
    n_pos = min(pos_cap, pos.size)
    pos_sel = rng.choice(pos, size=n_pos, replace=False) if pos.size > n_pos else pos
    n_neg = min(int(n_pos * neg_ratio), neg.size)
    neg_sel = rng.choice(neg, size=n_neg, replace=False) if neg.size > n_neg else neg
    sel = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(sel)
    return sel

def make_binary_label(y2000_flat, y2010_flat, class_code):
    if class_code in STABLE_CLASSES:
        if ROC_MODE_FOR_STABLE == "presence":
            return (y2010_flat == class_code).astype(np.uint8), "presence"
        elif ROC_MODE_FOR_STABLE == "persistence":
            return ((y2000_flat == class_code) & (y2010_flat == class_code)).astype(np.uint8), "persistence"
        else:
            raise ValueError("ROC_MODE_FOR_STABLE must be 'presence' or 'persistence'")
    else:
        return ((y2010_flat == class_code) & (y2000_flat != class_code)).astype(np.uint8), "gain"

# ============================================================
# LOAD LULC 2010 TEMPLATE + WARP 2000
# ============================================================
with rasterio.open(c2010_path) as src:
    lulc2010 = src.read(1).astype(np.int32)
    ref_transform = src.transform
    ref_crs = get_crs_or_force(src.crs, FORCE_CRS, where=c2010_path)
    H, W = lulc2010.shape
    ref_profile = src.profile

lulc2000 = read_to_template(
    c2000_path, ref_crs, ref_transform, (H, W),
    resampling=Resampling.nearest,
    force_crs=FORCE_CRS,
    fill_value=LULC_NODATA
).astype(np.int32)

valid_lulc = (lulc2000 != LULC_NODATA) & (lulc2010 != LULC_NODATA)

classes = CLASSES
print("Classes:", classes)
print("Valid LULC pixels:", int(valid_lulc.sum()), "/", valid_lulc.size)

# ============================================================
# LOAD PREDICTORS (partial nodata allowed + impute)
# ============================================================
stack = []
for f in variables:
    p = os.path.join(variables_dir, f)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing predictor: {p}")

    arr = read_to_template(
        p, ref_crs, ref_transform, (H, W),
        resampling=Resampling.bilinear,
        force_crs=FORCE_CRS,
        fill_value=PRED_NODATA_VALUES[0]
    ).astype(np.float32)

    for nd in PRED_NODATA_VALUES:
        arr[arr == nd] = np.nan

    stack.append(arr)

Xpred = np.stack(stack, axis=0)  # (F,H,W)
F = Xpred.shape[0]

finite_cnt = np.isfinite(Xpred).sum(axis=0)
min_needed = int(np.ceil(MIN_VALID_FRAC_PRED * F))
pred_ok = finite_cnt >= min_needed
ok_mask = valid_lulc & pred_ok

print("Predictor finite threshold:", min_needed, "of", F)
print("Final usable pixels:", int(ok_mask.sum()))

# Water diagnostic
print("Water raw:", int((lulc2010 == 10).sum()))
print("Water ok_mask:", int((ok_mask & (lulc2010 == 10)).sum()))

if ok_mask.sum() == 0:
    raise RuntimeError("No usable pixels after masks. Lower MIN_VALID_FRAC_PRED or fix predictors.")

# Impute NaNs by median over ok_mask
for i in range(F):
    band = Xpred[i]
    med = np.nanmedian(band[ok_mask])
    if not np.isfinite(med):
        raise RuntimeError(f"Predictor band {variables[i]} has no finite values in ok_mask.")
    band[~np.isfinite(band)] = med
    Xpred[i] = band

Xpred = np.moveaxis(Xpred, 0, -1).astype(np.float32)  # (H,W,F)
X_flat = Xpred.reshape(-1, F)
ok_flat = ok_mask.ravel()

# ============================================================
# 1) CHANGE MAPS
# ============================================================
print("\n--- Creating change maps (2000->2010) ---")
for c in classes:
    c = int(c)
    gain    = ok_mask & (lulc2010 == c) & (lulc2000 != c)
    persist = ok_mask & (lulc2010 == c) & (lulc2000 == c)
    loss    = ok_mask & (lulc2000 == c) & (lulc2010 != c)

    save_uint8_map(os.path.join(change_dir, f"gain_to_{c}.tif"),
                   np.where(ok_mask, gain.astype(np.uint8), 255), ref_profile)
    save_uint8_map(os.path.join(change_dir, f"persist_{c}.tif"),
                   np.where(ok_mask, persist.astype(np.uint8), 255), ref_profile)
    save_uint8_map(os.path.join(change_dir, f"loss_from_{c}.tif"),
                   np.where(ok_mask, loss.astype(np.uint8), 255), ref_profile)

print("Saved change maps to:", change_dir)

# ============================================================
# 2) PER-CLASS RF + PROB MAPS + ROC
# ============================================================
print("\n--- Training RF per class and writing probabilities ---")

roc_store = {}  # class_code -> (fpr,tpr,auc,mode_tag)
pool_idx = np.flatnonzero(ok_flat)

X_pool = X_flat[pool_idx]
y2000_pool = lulc2000.ravel()[pool_idx]
y2010_pool = lulc2010.ravel()[pool_idx]

for c in classes:
    c = int(c)
    cname = CLASS_NAMES.get(c, f"Class {c}")

    y_label, mode_tag = make_binary_label(y2000_pool, y2010_pool, c)
    n_pos = int(y_label.sum())
    n_neg = int(y_label.size - n_pos)
    print(f"\nClass {c} ({cname}): label={mode_tag} positives={n_pos} negatives={n_neg}")

    # Probability map is only produced when training threshold is met
    proba_map = np.zeros((H, W), dtype=np.float32)

    # Need both classes present for any ROC/training
    if n_pos < 2 or n_neg < 2:
        print("  Skip: not enough positives/negatives for any evaluation.")
    else:
        sel = balanced_subsample_indices(y_label, TRAIN_POS_CAP, NEG_RATIO, rng)
        if sel is None:
            print("  Skip: cannot sample both classes.")
        else:
            X_sub = X_pool[sel]
            y_sub = y_label[sel]

            # Split (stratify if possible)
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_sub, y_sub, test_size=TEST_FRAC,
                    random_state=RANDOM_STATE, stratify=y_sub
                )
            except ValueError:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_sub, y_sub, test_size=TEST_FRAC,
                    random_state=RANDOM_STATE
                )

            # --- ROC model (evaluation-only allowed) ---
            # If positives are tiny, we allow training ONLY to get a ROC curve,
            # but we clearly do NOT generate a probability map.
            can_do_roc = (n_pos >= MIN_POS_FOR_ROC)

            if can_do_roc:
                roc_clf = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=N_JOBS,
                    random_state=RANDOM_STATE
                )
                roc_clf.fit(X_tr, y_tr)

                if np.unique(y_te).size == 2:
                    y_score = roc_clf.predict_proba(X_te)[:, 1]
                    auc_val = roc_auc_score(y_te, y_score)
                    fpr, tpr, _ = roc_curve(y_te, y_score)
                    roc_store[c] = (fpr, tpr, auc_val, mode_tag)
                    print(f"  ROC-AUC:{auc_val:.3f}  (note: positives={n_pos})")
                else:
                    print("  ROC skipped: y_te has one class.")
            else:
                print(f"  ROC skipped: positives<{MIN_POS_FOR_ROC}.")

            # --- Map model (only if enough positives) ---
            if n_pos >= MIN_POS_FOR_TRAIN:
                map_clf = RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=N_JOBS,
                    random_state=RANDOM_STATE
                )
                map_clf.fit(X_tr, y_tr)

                pred_te = map_clf.predict(X_te)
                acc  = accuracy_score(y_te, pred_te)
                prec = precision_score(y_te, pred_te, zero_division=0)
                rec  = recall_score(y_te, pred_te, zero_division=0)
                f1   = f1_score(y_te, pred_te, zero_division=0)
                print(f"  Test -> Acc:{acc:.3f} Prec:{prec:.3f} Rec:{rec:.3f} F1:{f1:.3f}")

                proba_flat = np.zeros(lulc2010.size, dtype=np.float32)
                for start in range(0, pool_idx.size, PRED_CHUNK):
                    chunk = pool_idx[start:start + PRED_CHUNK]
                    proba_flat[chunk] = map_clf.predict_proba(X_flat[chunk])[:, 1].astype(np.float32)
                proba_map = proba_flat.reshape(H, W)
            else:
                print(f"  Not training map model (positives<{MIN_POS_FOR_TRAIN}). Probability map will be zeros.")

    # Save probability map
    out_tif = os.path.join(prob_dir, f"probability_class_{c}.tif")
    profile = ref_profile.copy()
    profile.update({"dtype": "float32", "count": 1, "nodata": 0.0, "compress": "lzw"})
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(proba_map.astype(np.float32), 1)
    print("  Saved probability:", out_tif)

print("\nDONE.")
print("Change maps:", change_dir)
print("Probability maps:", prob_dir)

# ============================================================
# 3) ROC CURVES FIGURE
# ============================================================
roc_png = os.path.join(out_root, "roc_curves_all_classes.png")

if len(roc_store) == 0:
    print("\nNo ROC curves available.")
else:
    plt.figure(figsize=(12, 9))
    for c, (fpr, tpr, auc_val, mode_tag) in roc_store.items():
        name = CLASS_NAMES.get(c, f"Class {c}")
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} [{mode_tag}] (AUC={auc_val:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves per Class (2000->2010 Transition Detection)")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(roc_png, dpi=200)
    plt.close()
    print("\nSaved ROC figure:", roc_png)
