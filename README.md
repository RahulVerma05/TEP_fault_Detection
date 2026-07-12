# Chemical Process Fault Detection
### Tennessee Eastman Process — Isolation Forest vs Autoencoder

![Python](https://img.shields.io/badge/Python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-latest-red)
![Status](https://img.shields.io/badge/Status-Complete-green)

---

## Project Overview

This project builds an **unsupervised fault detection system** for a simulated chemical plant using the **Tennessee Eastman Process (TEP)** — an industry-standard benchmark dataset in chemical engineering and process control research, originally developed by Eastman Chemical Company.

Two anomaly detection models are compared:
- **Isolation Forest** — a tree-based method that detects anomalies by isolating outliers using random decision trees
- **Autoencoder** — a neural network trained to reconstruct normal process data; faults are detected via high reconstruction error

Both models are trained **exclusively on normal operation data** — no fault labels are used during training. This is a semi-supervised anomaly detection approach: unsupervised training, supervised evaluation.

---

## Dataset

**Source:** Tennessee Eastman Process Simulation Dataset (Rieth et al., 2017) — Harvard Dataverse
Download from Kaggle: [Tennessee Eastman Process — CSV Format](https://www.kaggle.com/datasets/afrniomelo/tep-csv)
- `TEP_FaultFree_Training.csv`
- `TEP_FaultFree_Testing.csv`
- `TEP_Faulty_Testing.csv`
**Industry:** Chemical manufacturing / continuous process industries  
**Process:** Simulated chemical plant with reactor, condenser, separator, stripper, and recycle stream

| File | Description | Rows Used |
|---|---|---|
| TEP_FaultFree_Training.csv | Normal operation — model training | 10,000 |
| TEP_FaultFree_Testing.csv | Normal operation — false alarm evaluation | 500 |
| TEP_Faulty_Testing.csv | 20 fault types — detection evaluation | 10,000 |

### Features — 52 Process Variables
- **XMEAS 1–22:** Continuous process measurements — temperatures, pressures, flow rates, liquid levels across reactor, separator, stripper, compressor
- **XMEAS 23–41:** Composition measurements — percentage of chemical components (A, B, C, D, E, F, G, H) in various streams
- **XMV 1–11:** Manipulated variables — valve positions and control signals (feed flows, cooling water, purge valve)

> **Subsampling note:** The full dataset contains 500 simulation runs per fault type. First 20 runs are used to keep computation manageable on standard hardware, which is statistically sufficient for a comparative study.

---

## Methodology

### Preprocessing Pipeline
1. Load 3 CSV files, subsample to first 20 simulation runs out of 500
2. Label normal samples as 0, faulty samples as 1 (for evaluation only — never used in training)
3. Balance test set — 500 samples per fault type via stratified sampling → 10,500 test rows total
4. Scale all 52 features using StandardScaler — fit on training data only to prevent data leakage

### Model 1 — Isolation Forest
Isolation Forest detects anomalies by building random decision trees and measuring how quickly each data point gets isolated. Normal points sit close together in feature space and require many splits to isolate. Anomalies sit far from the crowd and are isolated in very few splits.

- `n_estimators=100` — 100 isolation trees for stable results
- `contamination=0.3` — flags ~30% of test data as anomalous
- Output converted from sklearn's -1/+1 convention to 0/1 to match labels

### Model 2 — Autoencoder
The Autoencoder is a neural network trained to compress and reconstruct normal process data. When shown a fault pattern it has never seen, reconstruction quality drops — high reconstruction error signals an anomaly.

- **Architecture:** 52 → 32 → 16 → 8 (bottleneck) → 16 → 32 → 52
- **Activation:** ReLU after each layer except the final decoder output
- **Optimizer:** Adam, learning rate = 0.001
- **Loss:** Mean Squared Error (MSELoss)
- **Epochs:** 50
- **Threshold:** 95th percentile of log-transformed training reconstruction error

> **Note on log transformation:** Raw reconstruction errors ranged from 0.52 (hard faults like IDV(3)) to 337 (IDV(6)). Log transformation via `np.log1p` was applied to compress this range so a single threshold could work effectively across all fault types.

---

## Results

### Overall Performance

| Model | Precision | Recall | F1 Score | False Alarm Rate |
|---|---|---|---|---|
| Isolation Forest | 0.977 | 0.632 | 0.767 | 0.300 |
| Autoencoder | 0.996 | 0.574 | 0.728 | 0.058 |

### Per-Fault Detection Rate — Isolation Forest

| IDV(1) | IDV(2) | IDV(3) | IDV(4) | IDV(5) | IDV(6) | IDV(7) | IDV(8) | IDV(9) | IDV(10) |
|---|---|---|---|---|---|---|---|---|---|
| 0.88 | 0.87 | 0.30 | 0.50 | 0.50 | 0.88 | 0.86 | 0.30 | 0.65 | 0.53 |

| IDV(11) | IDV(12) | IDV(13) | IDV(14) | IDV(15) | IDV(16) | IDV(17) | IDV(18) | IDV(19) | IDV(20) |
|---|---|---|---|---|---|---|---|---|---|
| 0.88 | 0.86 | 0.64 | 0.32 | 0.53 | 0.68 | 0.85 | 0.41 | 0.60 | 0.60 |

### Per-Fault Detection Rate — Autoencoder

| IDV(1) | IDV(2) | IDV(3) | IDV(4) | IDV(5) | IDV(6) | IDV(7) | IDV(8) | IDV(9) | IDV(10) |
|---|---|---|---|---|---|---|---|---|---|
| 0.85 | 0.84 | 0.06 | 0.85 | 0.23 | 0.85 | 0.85 | 0.83 | 0.06 | 0.29 |

| IDV(11) | IDV(12) | IDV(13) | IDV(14) | IDV(15) | IDV(16) | IDV(17) | IDV(18) | IDV(19) | IDV(20) |
|---|---|---|---|---|---|---|---|---|---|
| 0.65 | 0.84 | 0.81 | 0.85 | 0.06 | 0.23 | 0.77 | 0.80 | 0.30 | 0.45 |

---

## Key Findings

- **Isolation Forest** detects more faults overall (63.2% recall) but raises significantly more false alarms (30%) — it frequently flags normal operation as faulty. Better suited when missing a fault is more costly than false alarms.

- **Autoencoder** catches slightly fewer faults (57.4% recall) but is far more conservative with false alarms (5.8%) — making it more practical for industrial deployment where unnecessary shutdowns are expensive.

- **Both models struggle with IDV(3), IDV(9), IDV(15)** — these are notoriously difficult faults in TEP that even published research papers report low detection rates for. Consistency across both models validates our results.

- **Precision is near-identical** (~97–99%) for both models — when either model flags an anomaly, it is almost always correct. The key engineering tradeoff is recall vs false alarm rate.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.13 | Core language |
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| scikit-learn | Isolation Forest, StandardScaler, metrics |
| PyTorch | Autoencoder neural network |
| matplotlib / seaborn | Visualizations |

---

## Project Structure
