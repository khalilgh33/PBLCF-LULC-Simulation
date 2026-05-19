# 🌍 PBLCF: Python-Based Land Use / Land Cover Framework

PBLCF is a Python-based framework for simulating Land Use/Land Cover (LULC) change using a hybrid modelling approach that integrates Random Forest (RF) transition potential modelling with Cellular Automata–Markov (CA–Markov) spatial allocation.

The framework generates spatially explicit future LULC scenarios by combining data-driven transition probability estimation with rule-based spatial simulation, allowing users to explore alternative land-use trajectories under different scenario assumptions.
---

## 🔍 Overview

PBLCF follows three main modeling components:

1. **Demand Estimation**
   - Markov Chain estimates future land-use demand.

2. **Transition Potential Modeling**
   - Random Forest learns transition suitability from historical LULC changes.

3. **Spatial Allocation**
   - Cellular Automata allocates future changes using neighborhood effects, transition rules, and conversion resistance.

---

## ⚙️ Features

- Hybrid **RF + CA–Markov** workflow  
- Pixel-level spatial simulation  
- Scenario-based LULC modeling  
- Integration of environmental and socio-economic drivers  
- Transition probability mapping  
- Accuracy assessment and change-based validation  
- Reproducible step-by-step Python workflow  

---

## 📊 Input Data

- Multi-temporal LULC maps, e.g. 2000, 2010, 2020  
- Driving factor rasters:
  - DEM, slope, aspect  
  - Distance to roads, rivers, settlements  
  - Climate variables  
  - Population or socio-economic indicators  

---

## 📈 Outputs

- Aligned raster datasets  
- Transition matrices  
- Markov-based future land demand  
- RF-based transition probability maps  
- Simulated future LULC maps, e.g. 2030–2050  
- Accuracy metrics:
  - Overall Accuracy
  - Kappa
  - F1-score
  - ROC-AUC
  - Figure of Merit
  - Quantity and allocation disagreement  

---

## 📁 Workflow Overview

The model runs in **7 sequential steps**.  
Before running the simulation, define the input and output paths inside each script.

---

## 🧩 Step 00 — Raster Alignment

**Script:** `00-align_rasters.py`

Aligns all raster layers to a common reference grid.

It standardizes:

- CRS  
- Spatial resolution  
- Extent  
- Row and column size  
- Pixel alignment  

**Output:**

- Aligned LULC and driving factor rasters ready for modeling

---

## ⚙️ Step 01 — Pixel Statistics & Markov Demand

**Script:** `01-count_pixel.py`

- Computes class-wise pixel counts
- Detects observed LULC changes
- Builds Markov transition probabilities
- Projects future land demand, e.g. 2050

**Output:**

- Excel file containing:
  - Class counts
  - Transition matrix
  - Markov-based future demand

---

## 🔁 Step 02 — Transition Matrix

**Script:** `02-transional_matrix.py`

- Generates class-to-class transition matrix
- Produces:
  - Transition counts
  - Transition ratios
  - Area matrix

**Usage:**

- Can be used as an optional cost matrix for allocation

---

## 📊 Step 03 — Driving Factor Correlation Analysis

**Script:** `03.0-corrolation.py`

- Reads raster-based driving factors
- Removes NoData and invalid values
- Computes Pearson correlation matrix

**Output:**

- Correlation matrix CSV  
- Correlation matrix figure  
- Variable name list  

---

## 🤖 Step 04 — Transition Potential Modeling

**Script:** `03-probabilitymappingPlus_accuracy_RF_2000_to_2010+roc.py`

- Generates class-specific change maps:
  - Gain
  - Loss
  - Persistence
- Trains Random Forest models for each LULC class
- Produces transition probability maps
- Evaluates model performance using ROC-AUC and other metrics

**Output:**

- RF transition probability maps  
- Change maps  
- ROC curve figure  
- Accuracy results  

---

## 🧠 Step 05 — Spatial Allocation / CA Simulation

**Script:** `05- Allocation_2010to2050new+iteraion.py`

This is the main simulation engine.

It uses:

- RF probability maps  
- Markov-based demand  
- Neighborhood effects  
- Optional cost matrix  
- Class-specific transition rules  

**Key controls:**

- `CLASS_SPEED` → controls class growth or conversion speed  
- `FROM_PROTECT` → controls resistance of existing classes to conversion  
- `TO_PROTECT` → controls resistance to expansion into a target class  
- `NB_ALPHA` → controls neighborhood influence  

**Output:**

- Yearly predicted LULC maps as GeoTIFF  
- PNG visualizations  
- Optional simulation video  

---

## 📈 Step 06 — Model Validation

**Script:** `05-Accuracy_Simulation_model.py`

- Compares simulated LULC map with observed future LULC map
- Calculates:
  - Confusion matrix
  - Overall Accuracy
  - Kappa
  - Precision
  - Recall
  - F1-score
  - Quantity disagreement
  - Allocation disagreement
  - Figure of Merit

**Output:**

- CSV and Excel validation reports  
- Confusion matrix plots  
- Spatial error maps  

---
Reference:
Gholamnia et al. (2026). Environmental Modelling & Software. https://doi.org/10.1016/j.envsoft.2026.107032
---
## ▶️ How to Run

Run the scripts in this order:

```bash
python 00-align_rasters.py
python 01-count_pixel.py
python 02-transional_matrix.py
python 03-corrolation.py
python 04-probabilitymappingPlus_accuracy_RF_2000_to_2010+roc.py
python 05- Allocation_2010to2050new+iteraion.py
python 06-Accuracy_Simulation_model.py
