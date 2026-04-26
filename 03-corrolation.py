import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

# ==========================
# PATH
# ==========================
folder = r"E:\Khalil\PHD_doc\software\Python\Final\Variables\Equal"

# ==========================
# RENAME MAP
# revise these names if needed
# ==========================
rename_dict = {
    "aspect": "Aspect",
    "building": "Building density",      
    "dem": "Elevation",
    "dis_to_church": "Distance to Parking",
    "dis_to_hotels": "Distance to hotel",
    "dis_to_natural_place": "Distance to natural area",
    "dis_to_rivers": "Distance to river",
    "dis_to_roads": "Distance to road",
    "dis_to_staion": "Distance to station",
    "dis_to_wpp": "Distance to waterbody",    
    "population": "Population density",  #
    "rain": "Precipitation",
    "slope": "Slope",
    "temperature": "Temperature"
}

# ==========================
# READ RASTERS
# ==========================
data = {}
mask_all = None
used_variables = []

for file in sorted(os.listdir(folder)):
    if not file.lower().endswith(".tif"):
        continue

    path = os.path.join(folder, file)
    old_name = os.path.splitext(file)[0]
    new_name = rename_dict.get(old_name, old_name)

    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")

        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan

        # also remove inf values if present
        arr[np.isinf(arr)] = np.nan

        arr_flat = arr.flatten()

        valid = ~np.isnan(arr_flat)
        mask_all = valid if mask_all is None else (mask_all & valid)

        data[new_name] = arr_flat
        used_variables.append([old_name, new_name])

# ==========================
# BUILD DATAFRAME
# ==========================
df = pd.DataFrame(data)
df = df.loc[mask_all].copy()

print("Final valid pixel count:", len(df))
print("\nVariables used:")
for old_name, new_name in used_variables:
    print(f"{old_name}  -->  {new_name}")

# optional: sample if dataset is too large
# df = df.sample(n=min(50000, len(df)), random_state=42)

# ==========================
# CORRELATION MATRIX
# ==========================
corr = df.corr(method="pearson")

# save variable name list
var_df = pd.DataFrame(used_variables, columns=["original_name", "revised_name"])
var_df.to_csv(os.path.join(folder, "variable_name_list.csv"), index=False)

# save valid pixel values table if needed
# df.to_csv(os.path.join(folder, "raster_values_table.csv"), index=False)

# save correlation matrix
corr.to_csv(os.path.join(folder, "correlation_matrix.csv"))

# ==========================
# PLOT CORRELATION MATRIX (WITH GRID)
# ==========================
fig, ax = plt.subplots(figsize=(12, 10))

im = ax.imshow(corr.values, interpolation="none", aspect="auto")

# ticks
ax.set_xticks(np.arange(len(corr.columns)))
ax.set_yticks(np.arange(len(corr.index)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.index)

# ==========================
# ADD CELL GRID LINES  ✅
# ==========================
ax.set_xticks(np.arange(-.5, len(corr.columns), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(corr.index), 1), minor=True)

ax.grid(which="minor", color="white", linestyle='-', linewidth=0.8)

# remove minor tick marks
ax.tick_params(which="minor", bottom=False, left=False)

# ==========================
# ADD VALUES INSIDE CELLS
# ==========================
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=10, color= "white" if abs(corr.iloc[i, j]) > 0.5 else "black")

# colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Pearson correlation coefficient")

# title
ax.set_title("Correlation Matrix of Driving Factors")

plt.tight_layout()

png_path = os.path.join(folder, "correlation_matrix.png")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.show()
