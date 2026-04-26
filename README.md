# 🌍 PBLCF: Python-Based Land Use / Land Cover Framework

PBLCF is a Python-based framework for simulating **Land Use / Land Cover (LULC) changes** using a hybrid modeling approach that integrates **Random Forest (RF)** and **Cellular Automata–Markov (CA–Markov)**.

The framework combines **data-driven transition modeling** with **spatial allocation mechanisms** to produce realistic, scenario-based future land-use maps.

---

## 🔍 Overview

PBLCF follows a three-stage workflow:

1. **Transition Learning (Machine Learning)**
   - Random Forest is used to learn transition suitability from historical LULC changes

2. **Demand Estimation (Markov Chain)**
   - Transition probabilities are used to estimate future land-use demand

3. **Spatial Allocation (Cellular Automata)**
   - Land-use changes are allocated spatially based on:
     - Neighborhood effects
     - Transition rules
     - Conversion resistance

---

## ⚙️ Features

- Hybrid **ML + CA–Markov** framework  
- Pixel-level spatial simulation  
- Scenario-based modeling (e.g., BAU, Conservation, Sustainable Development)  
- Integration of environmental and socio-economic drivers  
- Model interpretability using SHAP  
- Reproducible Python workflow  

---

## 📊 Input Data

- Multi-temporal LULC maps (e.g., 2000, 2010, 2020)
- Environmental variables:
  - DEM, slope, aspect  
  - Distance to roads, rivers, settlements  
  - Climate data (temperature, precipitation)  
  - Socio-economic indicators (e.g., population density)

---

## 📈 Outputs

- Transition probability maps  
- Future LULC maps (e.g., 2030, 2040, 2050)  
- Scenario-based simulations  
- Feature importance (RF + SHAP)  
- Validation metrics:
  - Overall Accuracy (OA)
  - Kappa coefficient
  - F1-score
  - ROC-AUC
  - Figure of Merit (FoM)

---

## 🧪 Validation

The framework supports:

- Confusion matrix evaluation  
- ROC curve and AUC analysis  
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
PBLCF/
│── data/ # Input datasets
│── preprocessing/ # Data preparation scripts
│── modeling/ # Random Forest modeling
│── allocation/ # CA-Markov allocation
│── validation/ # Accuracy assessment
│── outputs/ # Results and maps
│── config.yaml # Configuration file
│── main.py # Main execution script


---

## 🚀 Installation

```bash
git clone https://github.com/your-username/PBLCF.git
cd PBLCF
pip install -r requirements.txt

---

---
Gholamnia, K., et al. (2026).
Comparative Evaluation of DYNA-CLUE and a Python-Based Machine Learning Framework for Long-Term LULC Simulation.
(Under review)

