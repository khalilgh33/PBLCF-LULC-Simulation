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
