<h1 align="center">Flipkart Gridlock Hackathon 2.0 — Traffic Demand Prediction</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/LightGBM-40%25-brightgreen.svg" alt="LightGBM Weight">
  <img src="https://img.shields.io/badge/XGBoost-30%25-red.svg" alt="XGBoost Weight">
  <img src="https://img.shields.io/badge/CatBoost-30%25-yellow.svg" alt="CatBoost Weight">
  <img src="https://img.shields.io/badge/OOF_R²-0.947-success.svg" alt="R2 Score">
</p>

---

## 🎯 Problem Statement
The challenge requires forecasting ride-hailing traffic demand across **1,249 unique geohash zones** for a specific target day (Day 49), provided 48 days of historical training data. Traffic demand is presented as a normalized continuous value between `0.0` and `1.0`. The primary evaluation metric is **R² (Coefficient of Determination)**.

The dataset includes 41,778 location-time pairs per day encompassing:
- **Spatio-Temporal Features:** `geohash` (6-character geographic cell), `day` (1-48), `timestamp` (15-minute intervals, e.g., "23:45").
- **Road Topology:** `RoadType` (Highway, Arterial, Residential, etc.), `NumberofLanes`, presence of `LargeVehicles`, and `Landmarks`.
- **Environmental Factors:** Ambient `Temperature` and categorical `Weather` conditions.

---

<p align="center">
  <img src="assets/flowchart.png" alt="Flowchart">
</p>

## 📌 Overview
This repository contains a robust, fully-generalized machine learning pipeline developed for the **Flipkart Gridlock Hackathon 2.0**. The solution achieves a highly competitive **~94.7% Out-Of-Fold (OOF) R²** utilizing advanced feature engineering and an optimized gradient-boosting ensemble.

## 📖 Table of Contents
- [Problem Statement](#-problem-statement)
- [Overview](#-overview)
- [Dataset Integrity & Leakage Mitigation](#️-dataset-integrity--leakage-mitigation)
- [Architecture & Feature Engineering](#️-architecture--feature-engineering)
- [Ensemble Models & Regularization](#-ensemble-models--regularization)
- [Repository Structure](#-repository-structure)
- [Installation & Execution](#-installation--execution)

---

## 🛡️ Dataset Integrity & Leakage Mitigation
During the Exploratory Data Analysis (EDA) phase, an exact match was identified between the hackathon dataset and the open-source "Grab Traffic Demand" dataset (2019) available on Kaggle. While a pure table-join lookup would theoretically yield a perfect score on the public leaderboard, **this approach was strictly avoided** to ensure architectural integrity.

A lookup-based strategy presents zero generalization capability and fails entirely when exposed to an unseen or altered private test set. Instead, this solution implements a rigorous, leakage-free machine learning methodology focused on legitimate pattern recognition, ensuring high performance regardless of underlying data origin.

---

## ⚙️ Architecture & Feature Engineering

![Pipeline Diagram](assets/pipeline_diagram.png)

Raw features such as timestamp strings and categorical geohashes require significant transformation for optimal tree-model consumption. The pipeline executes the following core engineering steps:

### 1. Temporal Cyclical Encoding
Treating timestamps linearly misrepresents the proximity of late-night and early-morning hours (e.g., "23:45" vs. "00:00"). To preserve the continuous loop of time, hours, minutes, and days were mapped onto a unit circle using Sine and Cosine transformations:
```python
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
```

### 2. Spatial Target Encoding (5-Fold OOF)
With 1,249 distinct geohash codes, standard one-hot encoding introduces severe sparsity, degrading model performance. Instead, geohashes were aggregated by their 4- and 5-character prefixes (approximating district and neighborhood hierarchies) and encoded with their historical mean demand.
> [!IMPORTANT]
> To strictly prevent data leakage, target encoding was computed utilizing robust **5-Fold Out-of-Fold (OOF) cross-validation**.

### 3. Autoregressive Features (Lag-1 Demand)
Traffic exhibits strong 24-hour periodicity. The demand at a specific geohash at 8:30 AM today correlates heavily with the demand at that same location at 8:30 AM yesterday. 
The pipeline maps the actual recorded demand from **Day 48** directly into **Day 49** as a `lag_1_demand` feature. This autoregressive component was the most impactful transformation, elevating the OOF R² from ~0.85 to **0.947**.

### 4. Heuristic Interaction Flags
Custom boolean interaction features were engineered to assist gradient boosters in immediately partitioning high-variance traffic regimes:
- `is_rush_hour`: Active between 7:00–9:00 AM and 5:00–7:00 PM.
- `is_night`: Active between 10:00 PM and 5:00 AM, signifying baseline low demand.
- `highway_rush`: Logical conjunction of `RoadType == Highway` and `is_rush_hour`.

---

## 🧠 Ensemble Models & Regularization
Final predictions are generated via a weighted ensemble of three distinct gradient-boosting frameworks, rigorously evaluated inside a 5-fold cross-validation loop to prevent overfitting:

| Model | Weight | Characteristics |
|-------|--------|-----------------|
| **LightGBM** | 40% | Ultra-fast, leaf-wise growth. Excels at large-scale non-linear pattern capture. |
| **XGBoost** | 30% | Depth-wise growth with strict L2 regularization. Provides a highly stable backbone. |
| **CatBoost** | 30% | Ordered boosting algorithm. Natively optimizes categorical interactions (`RoadType`, `Weather`). |

### Hyperparameter Regularization Strategy
To prevent the models from memorizing training noise, aggressive regularization parameters were enforced:
- **Shallow Trees:** `max_depth = 6` to `8`
- **Conservative Leaf Sizes:** `min_child_samples >= 25`
- **L2 Weight Penalties:** `reg_lambda` scaled appropriately
- **Early Stopping:** Monitored against the validation fold to halt training dynamically.

---

## 📂 Repository Structure

```text
📦 flipkart-gridlock-hackathon
 ┣ 📂 assets/
 ┃  ┣ 📜 flowchart.png
 ┃  ┗ 📜 pipeline_diagram.png
 ┣ 📂 dataset/                  # (Place train.csv and test.csv here)
 ┣ 📜 detailed_approach.pdf     # Comprehensive 5-page technical report
 ┣ 📜 feature_engineering.py    # Standalone data transformation module
 ┣ 📜 traffic_demand_prediction.ipynb # Interactive EDA and step-by-step pipeline
 ┣ 📜 train_predict.py          # Main orchestration script
 ┗ 📜 README.md                 # Project documentation
```

---

## 🚀 Installation & Execution

### Prerequisites
Ensure you have Python 3.8+ installed. The required dependencies include:
- `numpy`
- `pandas`
- `scikit-learn`
- `lightgbm`
- `xgboost`
- `catboost`

You can install the dependencies via pip:
```bash
pip install numpy pandas scikit-learn lightgbm xgboost catboost
```

### Execution Steps
1. Place the competition data (`train.csv` and `test.csv`) into the `dataset/` directory.
2. Execute the main training orchestrator:
```bash
python train_predict.py
```
3. The script will automatically perform 5-fold cross-validation, train the ensemble, and output the final predictions to `submission.csv`.
