import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

# CLUEE = E:\Khalil\PHD_doc\software\Python\Final\Class\DYNA_CLUE\CLUE_2020_INT.tif
# ML    = E:\Khalil\PHD_doc\software\Python\Final\Variable\transition_2000_2010_rf\probability_maps\outputs\predicted_rasters\land_use_2020.tif

# Baseline map (needed for FoM and change-based validation)
base_path   = r"E:\Khalil\PHD_doc\software\Python\Final\Class\c2010.tif"

# Observed future map
actual_path = r"E:\Khalil\PHD_doc\software\Python\Final\Class\c2020.tif"

# Simulated future map
sim_path    = r"E:\Khalil\PHD_doc\software\Python\Final\Variable\transition_2000_2010_rf\probability_maps\outputs\predicted_rasters\land_use_2020.tif"

out_dir = r"E:\Khalil\PHD_doc\software\Python\Final\Variable\validation_rf_2020"
os.makedirs(out_dir, exist_ok=True)

out_csv_metrics   = os.path.join(out_dir, "validation_report_rf.csv")
out_csv_confusion = os.path.join(out_dir, "confusion_matrix_rf.csv")
out_csv_disagree  = os.path.join(out_dir, "disagreement_metrics_rf.csv")
out_xlsx          = os.path.join(out_dir, "validation_outputs_rf.xlsx")

out_png_cm        = os.path.join(out_dir, "confusion_matrix_counts_rf.png")
out_png_cm_norm   = os.path.join(out_dir, "confusion_matrix_row_normalized_rf.png")
out_png_spatial   = os.path.join(out_dir, "spatial_error_codes_rf.png")

# Spatial error maps
out_map_error_codes = os.path.join(out_dir, "spatial_error_codes_rf.tif")
out_map_obs_change  = os.path.join(out_dir, "observed_change_rf.tif")
out_map_sim_change  = os.path.join(out_dir, "simulated_change_rf.tif")

classes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int16)

FORCE_NODATA = -9999
FORCE_CRS = "EPSG:5514"   # set your true CRS

land_use_class_names = {
    0: "Building area",
    1: "Grassland",
    2: "Deciduous forest",
    3: "Coniferous forest",
    4: "Mixed forest",
    5: "Transitional forest",
    6: "Clearings",
    7: "Dwarf pine",
    8: "Peatland",
    9: "Rocks, stone seas",
    10: "Water bodies",
}

# Spatial error code legend
ERROR_CODE_NAMES = {
    0: "Correct persistence",
    1: "Correct change (hit)",
    2: "Missed change",
    3: "False change",
    4: "Wrong persistence class",
    5: "Wrong change type",
    255: "NoData"
}

# ============================================================
# HELPERS
# ============================================================
def make_nodata_mask(arr, nodata_value):
    if nodata_value is None:
        return np.zeros(arr.shape, dtype=bool)
    if isinstance(nodata_value, float) and np.isnan(nodata_value):
        return np.isnan(arr)
    return (arr == nodata_value) | np.isnan(arr)

def confusion_matrix_from_arrays(y_true, y_pred, classes):
    k = len(classes)
    idx = {int(c): i for i, c in enumerate(classes)}
    t = np.array([idx[int(v)] for v in y_true], dtype=np.int32)
    p = np.array([idx[int(v)] for v in y_pred], dtype=np.int32)
    flat = t * k + p
    counts = np.bincount(flat, minlength=k * k)
    return counts.reshape(k, k)

def compute_metrics(cm):
    total = cm.sum()
    correct = np.diag(cm).sum()
    oa = correct / total if total > 0 else np.nan

    row_sum = cm.sum(axis=1)  # true totals
    col_sum = cm.sum(axis=0)  # pred totals

    recall = np.divide(
        np.diag(cm), row_sum,
        out=np.full_like(row_sum, np.nan, dtype=float),
        where=row_sum != 0
    )
    precision = np.divide(
        np.diag(cm), col_sum,
        out=np.full_like(col_sum, np.nan, dtype=float),
        where=col_sum != 0
    )

    pe = (row_sum * col_sum).sum() / (total * total) if total > 0 else np.nan
    kappa = (oa - pe) / (1 - pe) if (total > 0 and pe < 1) else np.nan

    f1 = np.divide(
        2 * precision * recall, (precision + recall),
        out=np.full_like(precision, np.nan, dtype=float),
        where=(precision + recall) != 0
    )

    macro_precision = np.nanmean(precision)
    macro_recall = np.nanmean(recall)
    macro_f1 = np.nanmean(f1)

    weights = row_sum.astype(float)
    if np.nansum(weights) > 0:
        weighted_precision = np.nansum(np.nan_to_num(precision) * weights) / np.nansum(weights)
        weighted_recall = np.nansum(np.nan_to_num(recall) * weights) / np.nansum(weights)
        weighted_f1 = np.nansum(np.nan_to_num(f1) * weights) / np.nansum(weights)
    else:
        weighted_precision = weighted_recall = weighted_f1 = np.nan

    micro_f1 = oa

    return dict(
        oa=oa,
        kappa=kappa,
        precision=precision,
        recall=recall,
        f1=f1,
        support_true=row_sum,
        support_pred=col_sum,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_precision=weighted_precision,
        weighted_recall=weighted_recall,
        weighted_f1=weighted_f1,
        micro_f1=micro_f1
    )

def warp_to_template(src_arr, src_transform, src_crs, template_transform, template_crs, out_shape):
    dst = np.empty(out_shape, dtype=src_arr.dtype)
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=template_transform,
        dst_crs=template_crs,
        resampling=Resampling.nearest
    )
    return dst

def plot_confusion_matrix(cm, classes, class_name_map=None, normalize=None, title=None, save_path=None):
    cm_plot = cm.astype(float).copy()

    if normalize == "true":
        row_sum = cm_plot.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm_plot, row_sum, out=np.zeros_like(cm_plot), where=row_sum != 0)
        default_title = "Confusion Matrix (row-normalized)"
        annotation_values = cm_plot
        annotation_fmt = "{:.2f}"
    elif normalize == "pred":
        col_sum = cm_plot.sum(axis=0, keepdims=True)
        cm_plot = np.divide(cm_plot, col_sum, out=np.zeros_like(cm_plot), where=col_sum != 0)
        default_title = "Confusion Matrix (column-normalized)"
        annotation_values = cm_plot
        annotation_fmt = "{:.2f}"
    elif normalize == "all":
        total = cm_plot.sum()
        cm_plot = cm_plot / total if total != 0 else cm_plot
        default_title = "Confusion Matrix (normalized)"
        annotation_values = cm_plot
        annotation_fmt = "{:.2f}"
    else:
        default_title = "Confusion Matrix (counts)"
        annotation_values = cm
        annotation_fmt = "{:d}"

    title = default_title if title is None else title

    if class_name_map is None:
        labels = [str(int(c)) for c in classes]
    else:
        labels = [f"{int(c)}: {class_name_map.get(int(c), str(int(c)))}" for c in classes]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_plot, interpolation="nearest", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    # Major ticks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_yticklabels(labels, fontsize=14)

    # Minor ticks for grid lines between cells
    ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)

    # Cell borders
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Axis labels and title
    ax.set_ylabel("True class", fontsize=14)
    ax.set_xlabel("Predicted class", fontsize=14)
    ax.set_title(title, fontsize=13, pad=12)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Outer border
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    thresh = cm_plot.max() * 0.6 if cm_plot.size and np.nanmax(cm_plot) > 0 else 0

    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            if normalize is None:
                txt = annotation_fmt.format(int(annotation_values[i, j]))
            else:
                txt = annotation_fmt.format(annotation_values[i, j])

            ax.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=14,
                color="black" if cm_plot[i, j] > thresh else "white"
            )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Saved confusion matrix figure to:", save_path)

    plt.close(fig)

def save_raster(path, arr, ref_profile, dtype, nodata):
    profile = ref_profile.copy()
    profile.update(dtype=dtype, count=1, nodata=nodata, compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)

def quantity_allocation_disagreement(cm):
    """
    Pontius-style disagreement decomposition for a square confusion matrix.
    total disagreement = 1 - OA
    quantity disagreement = 0.5 * sum(|row proportion - col proportion|)
    allocation disagreement = total disagreement - quantity disagreement
    """
    total = cm.sum()
    if total == 0:
        return np.nan, np.nan, np.nan

    row_prop = cm.sum(axis=1) / total
    col_prop = cm.sum(axis=0) / total
    oa = np.trace(cm) / total
    total_disagreement = 1.0 - oa
    quantity = 0.5 * np.sum(np.abs(row_prop - col_prop))
    allocation = total_disagreement - quantity
    return total_disagreement, quantity, allocation

def compute_change_maps(base_arr, actual_arr, sim_arr, valid_mask):
    """
    Returns:
    obs_change : observed change mask
    sim_change : simulated change mask
    error_code : spatial error categories
    """
    obs_change = (base_arr != actual_arr) & valid_mask
    sim_change = (base_arr != sim_arr) & valid_mask

    error_code = np.full(base_arr.shape, 255, dtype=np.uint8)

    # correct persistence
    m0 = (~obs_change) & (~sim_change) & (actual_arr == sim_arr) & valid_mask
    error_code[m0] = 0

    # correct change (hit)
    m1 = obs_change & sim_change & (actual_arr == sim_arr) & valid_mask
    error_code[m1] = 1

    # missed change: observed changed but simulated stayed baseline
    m2 = obs_change & (~sim_change) & valid_mask
    error_code[m2] = 2

    # false change: observed persistence but simulated changed
    m3 = (~obs_change) & sim_change & valid_mask
    error_code[m3] = 3

    # wrong persistence class
    m4 = (~obs_change) & (~sim_change) & (actual_arr != sim_arr) & valid_mask
    error_code[m4] = 4

    # wrong change type
    m5 = obs_change & sim_change & (actual_arr != sim_arr) & valid_mask
    error_code[m5] = 5

    return obs_change.astype(np.uint8), sim_change.astype(np.uint8), error_code

def figure_of_merit(base_arr, actual_arr, sim_arr, valid_mask):
    """
    FoM = Hits / (Hits + Misses + False Alarms + Wrong Hits)
    """
    obs_change = (base_arr != actual_arr) & valid_mask
    sim_change = (base_arr != sim_arr) & valid_mask

    hits = np.sum(obs_change & sim_change & (actual_arr == sim_arr))
    misses = np.sum(obs_change & (~sim_change))
    false_alarms = np.sum((~obs_change) & sim_change)
    wrong_hits = np.sum(obs_change & sim_change & (actual_arr != sim_arr))

    denom = hits + misses + false_alarms + wrong_hits
    fom = hits / denom if denom > 0 else np.nan

    return {
        "hits": int(hits),
        "misses": int(misses),
        "false_alarms": int(false_alarms),
        "wrong_hits": int(wrong_hits),
        "fom": float(fom) if not np.isnan(fom) else np.nan
    }

# ============================================================
# LOAD RASTERS
# ============================================================
with rasterio.open(base_path) as b_src:
    base = b_src.read(1).astype(np.int32)
    tr_base = b_src.transform
    h_base, w_base = b_src.height, b_src.width
    nodata_base = b_src.nodata
    ref_profile = b_src.profile

with rasterio.open(actual_path) as a_src:
    actual_raw = a_src.read(1).astype(np.int32)
    tr_actual = a_src.transform
    h_actual, w_actual = a_src.height, a_src.width
    nodata_actual = a_src.nodata

with rasterio.open(sim_path) as s_src:
    sim_raw = s_src.read(1).astype(np.int32)
    tr_sim = s_src.transform
    h_sim, w_sim = s_src.height, s_src.width
    nodata_sim = s_src.nodata

# ============================================================
# CRS
# ============================================================
if FORCE_CRS is None:
    raise ValueError("Set FORCE_CRS because CRS is missing in your rasters.")

crs_base = CRS.from_string(FORCE_CRS)
crs_actual = CRS.from_string(FORCE_CRS)
crs_sim = CRS.from_string(FORCE_CRS)

# ============================================================
# ALIGN TO BASE GRID
# ============================================================
def align_if_needed(arr, transform, h, w, template_transform, template_crs, src_crs):
    same_grid = (transform == template_transform) and (h == h_base) and (w == w_base)
    if same_grid:
        return arr
    return warp_to_template(
        src_arr=arr,
        src_transform=transform,
        src_crs=src_crs,
        template_transform=template_transform,
        template_crs=template_crs,
        out_shape=(h_base, w_base)
    )

actual = align_if_needed(actual_raw, tr_actual, h_actual, w_actual, tr_base, crs_base, crs_actual)
sim    = align_if_needed(sim_raw, tr_sim, h_sim, w_sim, tr_base, crs_base, crs_sim)

# ============================================================
# VALID MASK
# ============================================================
nodata_value_base   = FORCE_NODATA if FORCE_NODATA is not None else nodata_base
nodata_value_actual = FORCE_NODATA if FORCE_NODATA is not None else nodata_actual
nodata_value_sim    = FORCE_NODATA if FORCE_NODATA is not None else nodata_sim

mask = (
    ~make_nodata_mask(base, nodata_value_base) &
    ~make_nodata_mask(actual, nodata_value_actual) &
    ~make_nodata_mask(sim, nodata_value_sim) &
    np.isin(base, classes) &
    np.isin(actual, classes) &
    np.isin(sim, classes)
)

y_true = actual[mask].astype(np.int16)
y_pred = sim[mask].astype(np.int16)

print("Valid pixels used for validation:", y_true.size)
if y_true.size == 0:
    raise ValueError("No valid pixels after masking. Check nodata and class codes.")

# ============================================================
# CONFUSION + METRICS
# ============================================================
cm = confusion_matrix_from_arrays(y_true, y_pred, classes)
m = compute_metrics(cm)

print("\n=== Overall Metrics ===")
print(f"Overall Accuracy (OA): {m['oa']:.6f}")
print(f"Cohen's Kappa:         {m['kappa']:.6f}")
print(f"Micro F1:              {m['micro_f1']:.6f}")
print(f"Macro Precision:       {m['macro_precision']:.6f}")
print(f"Macro Recall:          {m['macro_recall']:.6f}")
print(f"Macro F1:              {m['macro_f1']:.6f}")
print(f"Weighted F1:           {m['weighted_f1']:.6f}")

# ============================================================
# QUANTITY / ALLOCATION DISAGREEMENT
# ============================================================
total_disagreement, quantity_disagreement, allocation_disagreement = quantity_allocation_disagreement(cm)

print("\n=== Disagreement Metrics ===")
print(f"Total disagreement:     {total_disagreement:.6f}")
print(f"Quantity disagreement:  {quantity_disagreement:.6f}")
print(f"Allocation disagreement:{allocation_disagreement:.6f}")

# ============================================================
# CHANGE-BASED VALIDATION (FoM)
# ============================================================
fom_stats = figure_of_merit(base, actual, sim, mask)

print("\n=== Change-Based Validation ===")
print(f"Hits:                  {fom_stats['hits']}")
print(f"Misses:                {fom_stats['misses']}")
print(f"False alarms:          {fom_stats['false_alarms']}")
print(f"Wrong hits:            {fom_stats['wrong_hits']}")
print(f"Figure of Merit (FoM): {fom_stats['fom']:.6f}")

# ============================================================
# REPORT TABLES
# ============================================================
report = pd.DataFrame({
    "class": classes,
    "class_name": [land_use_class_names.get(int(c), "") for c in classes],
    "true_pixels": m["support_true"],
    "pred_pixels": m["support_pred"],
    "correct_pixels": np.diag(cm),
    "precision_user_accuracy": m["precision"],
    "recall_producer_accuracy": m["recall"],
    "f1": m["f1"],
})

summary_df = pd.DataFrame([{
    "overall_accuracy": m["oa"],
    "kappa": m["kappa"],
    "micro_f1": m["micro_f1"],
    "macro_precision": m["macro_precision"],
    "macro_recall": m["macro_recall"],
    "macro_f1": m["macro_f1"],
    "weighted_precision": m["weighted_precision"],
    "weighted_recall": m["weighted_recall"],
    "weighted_f1": m["weighted_f1"],
    "total_disagreement": total_disagreement,
    "quantity_disagreement": quantity_disagreement,
    "allocation_disagreement": allocation_disagreement,
    "fom": fom_stats["fom"],
    "hits": fom_stats["hits"],
    "misses": fom_stats["misses"],
    "false_alarms": fom_stats["false_alarms"],
    "wrong_hits": fom_stats["wrong_hits"],
}])

confusion_df = pd.DataFrame(
    cm,
    index=[f"true_{c}_{land_use_class_names.get(int(c), '')}" for c in classes],
    columns=[f"pred_{c}_{land_use_class_names.get(int(c), '')}" for c in classes]
)

disagree_df = pd.DataFrame([
    {"metric": "Total disagreement", "value": total_disagreement},
    {"metric": "Quantity disagreement", "value": quantity_disagreement},
    {"metric": "Allocation disagreement", "value": allocation_disagreement},
    {"metric": "FoM", "value": fom_stats["fom"]},
    {"metric": "Hits", "value": fom_stats["hits"]},
    {"metric": "Misses", "value": fom_stats["misses"]},
    {"metric": "False alarms", "value": fom_stats["false_alarms"]},
    {"metric": "Wrong hits", "value": fom_stats["wrong_hits"]},
])

with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    report.to_excel(writer, sheet_name="class_metrics", index=False)
    summary_df.to_excel(writer, sheet_name="summary", index=False)
    confusion_df.to_excel(writer, sheet_name="confusion_matrix")
    disagree_df.to_excel(writer, sheet_name="disagreement", index=False)

report.to_csv(out_csv_metrics, index=False)
confusion_df.to_csv(out_csv_confusion)
disagree_df.to_csv(out_csv_disagree, index=False)

print("\nSaved tables to:")
print(out_csv_metrics)
print(out_csv_confusion)
print(out_csv_disagree)
print(out_xlsx)

# ============================================================
# PLOTS
# ============================================================
plot_confusion_matrix(
    cm,
    classes=classes,
    class_name_map=land_use_class_names,
    normalize=None,
    title="Confusion Matrix (counts)",
    save_path=out_png_cm
)

plot_confusion_matrix(
    cm,
    classes=classes,
    class_name_map=land_use_class_names,
    normalize="true",
    title="Confusion Matrix (row-normalized / recall view)",
    save_path=out_png_cm_norm
)

# ============================================================
# SPATIAL ERROR ANALYSIS
# ============================================================
obs_change, sim_change, error_code = compute_change_maps(base, actual, sim, mask)

save_raster(out_map_obs_change, obs_change, ref_profile, dtype="uint8", nodata=255)
save_raster(out_map_sim_change, sim_change, ref_profile, dtype="uint8", nodata=255)
save_raster(out_map_error_codes, error_code, ref_profile, dtype="uint8", nodata=255)

print("\nSaved spatial error maps:")
print(out_map_obs_change)
print(out_map_sim_change)
print(out_map_error_codes)

# Optional quick visualization
plt.figure(figsize=(10, 8))
plt.imshow(np.where(error_code == 255, np.nan, error_code), interpolation="nearest")
plt.title("Spatial Error Categories")
cbar = plt.colorbar()
cbar.set_label("Error code")
plt.tight_layout()
plt.savefig(out_png_spatial, dpi=300, bbox_inches="tight")
plt.close()

print("\nDone.")