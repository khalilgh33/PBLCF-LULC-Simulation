# 🌍 PBLCF: Python-Based Land Use / Land Cover Framework

PBLCF is a Python-based framework for simulating **Land Use / Land Cover (LULC) changes** using a hybrid approach that integrates **Random Forest (RF)** and **Cellular Automata–Markov (CA–Markov)**.

It generates **spatially explicit future land-use scenarios** by combining data-driven transition modeling with rule-based spatial allocation.

---

## 🔍 Overview

PBLCF follows a three-stage workflow:

1. **Transition Learning (Machine Learning)**
   - RF learns transition suitability from historical LULC changes

2. **Demand Estimation (Markov Chain)**
   - Estimates future land-use demand using transition probabilities

3. **Spatial Allocation (Cellular Automata)**
   - Allocates changes based on:
     - Neighborhood effects
     - Transition rules
     - Conversion resistance

---

## ⚙️ Features

- Hybrid **ML + CA–Markov** framework  
- Pixel-level spatial simulation  
- Scenario-based modeling (BAU, Conservation, Sustainable Development)  
- Integration of environmental & socio-economic drivers  
- SHAP-based model interpretability  
- Reproducible Python workflow  

---

## 📊 Input Data

- Multi-temporal LULC maps (e.g., 2000, 2010, 2020)
- Environmental variables:
  - DEM, slope, aspect  
  - Distance to roads, rivers, settlements  
  - Climate data (temperature, precipitation)  
  - Socio-economic indicators  

---

## 📈 Outputs

- Transition probability maps  
- Future LULC maps (2030–2050)  
- Scenario simulations  
- Feature importance (RF + SHAP)  
- Validation metrics:
  - Overall Accuracy (OA)
  - Kappa coefficient
  - F1-score
  - ROC-AUC
  - Figure of Merit (FoM)

---

## 🧪 Validation

- Confusion matrix  
- ROC curve and AUC  
- Spatial validation (FoM, allocation disagreement)

---

## 🧰 Tech Stack

- Python 3.x  
- NumPy, Pandas  
- Rasterio  
- scikit-learn  
- SHAP  
- Matplotlib / Seaborn  

---

## 📁 Project Structure
# 🌍 PBLCF: Land Use / Land Cover Simulation Framework

A Python-based framework for simulating future land-use/land-cover (LULC) changes using a combination of:

- Markov Chain (demand estimation)
- Random Forest (transition potential modeling)
- Cellular Automata (spatial allocation)

---

## 📁 Workflow Overview

The model runs in **6 sequential steps**.  
Each step is implemented as a separate script.

---

## ⚙️ Step 0 — Pixel Statistics & Markov Demand

**Script:** `01-count_pixel.py`

- Computes class-wise pixel counts (2000, 2010)
- Detects observed changes
- Builds transition probability matrix
- Projects future land demand (e.g., 2050)

**Output:**
- Excel file with:
  - Class counts
  - Transition matrix
  - Markov-based future demand

---

## 🔁 Step 1 — Transition Matrix (Optional)

**Script:** `02-transional_matrix.py`

- Generates transition matrix (1990 → 2010)
- Produces:
  - Counts
  - Ratios
  - Area matrix

**Usage:**
- Optional cost matrix for simulation

---

## 📊 Step 2 — Driving Factor Analysis

**Script:** `03.0-corrolation.py`

- Reads raster variables (e.g., slope, population, distance layers)
- Removes NoData and invalid values
- Builds dataset
- Computes Pearson correlation matrix

**Output:**
- Correlation matrix (CSV + PNG)
- Variable list

---

## 🤖 Step 3 — Transition Potential Modeling (Random Forest)

**Script:**  
`03-probabilitymappingPlus_accuracy_RF_2000_to_2010+roc.py`

- Generates change maps:
  - Gain
  - Loss
  - Persistence
- Trains Random Forest per class
- Produces:
  - Probability maps
  - Accuracy metrics
  - ROC curves

**Output:**
- Probability maps (GeoTIFF)
- Model performance results

---

## 🧠 Step 4 — Spatial Allocation (CA Simulation)

**Script:** `05- Allocation_2010to2050new+iteraion.py`

Core simulation engine:

- Inputs:
  - Probability maps (Step 3)
  - Markov demand (Step 0)
  - Cost matrix (optional)
- Applies:
  - Neighborhood effect
  - Class conversion rules
  - Iterative allocation

**Key Controls:**
- `CLASS_SPEED` → growth rate
- `FROM_PROTECT` → resistance to change
- `TO_PROTECT` → resistance to expansion

**Output:**
- Predicted LULC maps (GeoTIFF)
- PNG visualizations
- Optional simulation video

---

## 📈 Step 5 — Model Validation

**Script:** `05-Accuracy_Simulation_model.py`

- Compares simulated vs observed map
- Computes:
  - Confusion matrix
  - Overall Accuracy (OA)
  - Kappa
  - F1-score
  - Quantity & allocation disagreement
  - Figure of Merit (FoM)

**Output:**
- CSV / Excel reports
- Confusion matrix plots
- Spatial error maps

---

## ▶️ How to Run

1. Define file paths in each script
2. Run scripts sequentially:

```bash
python 01-count_pixel.py
python 02-transional_matrix.py
python 03.0-corrolation.py
python 03-probabilitymappingPlus_accuracy_RF_2000_to_2010+roc.py
python 05- Allocation_2010to2050new+iteraion.py
python 05-Accuracy_Simulation_model.py
