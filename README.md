# 🛡️ FraudGuard 2.0 — Credit Card Fraud Detection with MLOps

A three-version machine learning project that evolves from a baseline Random Forest classifier to a production-style XGBoost pipeline with experiment tracking, health guardrails, and a champion model registry.

The goal was not just to build a model — but to build the infrastructure around a model that lets you iterate safely, compare versions objectively, and deploy with confidence.

## 🗂️ Project Structure
```text
fraudguard-fraud-detection/
│
├── v1_random_forest/          # Baseline — Random Forest + SMOTE comparison
├── v2_xgboost_baseline/       # Upgrade — XGBoost + first feature engineering
├── v3_xgboost/                  # Final — XGBoost + new feauture engineered + light MLOps pipeline
│   ├── v3_xgboost_model.ipynb
│   ├── experiment_tracking_v3.csv
│   ├── model_registry.json
│   └── registered_models/
└── README.md

## 🔁 Version Evolution

### V1 — Random Forest Baseline
The starting point. Built to understand the dataset, handle class imbalance, and establish a performance benchmark.

- **Algorithm:** Random Forest (100 estimators, max_depth=20)
- **Imbalance handling:** GridSearchCV compared SMOTE, class_weight=balanced, and no weighting — found that unweighted RF performed best on this PCA dataset
- **Key finding:** SMOTE reduced precision by blurring the fraud/normal boundary. class_weight=balanced caused overfitting on the minority class.
- **AUPRC:** 0.8016 | **Recall:** 74.7% | **FP Rate:** 0.002%


### V2 — XGBoost Baseline
Replaced Random Forest with XGBoost and introduced the first feature engineering on the raw `Time` and `Amount` columns.

- **Algorithm:** XGBoost with `scale_pos_weight` (replaces SMOTE — cleaner and more effective at 582:1 class ratio)
- **New features:** `log_amount`, `Hour_of_Day`, `IS_NIGHT` (11pm–6am risk window)
- **Key finding:** `scale_pos_weight` at the raw class ratio (582) caused the model to peak at iteration 4 due to aggressive early overfitting. Required careful tuning.
- **AUPRC:** ~0.81

### V3- XGBOOST UPDATED  ← Current Champion
Added a fourth engineered feature, fixed the training pipeline, and built full MLOps infrastructure around the model.

- **Algorithm:** XGBoost with early stopping (eval_set wired correctly)
- **New feature:** `AMOUNT_TIER` — buckets transactions into micro/small/medium/large/whale
- **MLOps layer:** Experiment tracker (CSV), 3-guardrail health check, champion registry (JSON), production-style inference function
- **AUPRC:** 0.8427 | **Recall:** 77.9% | **Net ROI:** $10,095 (test set)

---

## 🧬 MLOps Pipeline

The V3 training process follows this automated flow:

`Train XGBoost → Guardrail Checks → Register if Champion → Log Run Data`

While data prep is done manually in the notebook, everything after training is automated. The system evaluates the model's health, blocks bad models from being registered, and writes the experiment details to the tracker.

**Champion Registry** — `model_registry.json` always points to the best model seen so far. The serving layer loads from the registry, not from whatever happens to be in memory.

**Experiment Tracker** — every run appends to `experiment_tracking_v3.csv` with full hyperparameters, metrics, and business ROI:

| Version | n_estimators | max_depth | AUPRC | Net ROI | Status |
|---------|-------------|-----------|-------|---------|--------|
| RF V1 | 100 | 20 | 0.8016 | $10,709 | Healthy |
| XGBoost V2 | 200 | 5 | ~0.81 | $10,187 | Healthy |
| XGBoost V3 | 300 | 6 | 0.8427 | $10,095 | Healthy ← Champion |

---

## ⚙️ Feature Engineering

| Feature | Source | Reasoning |
|---------|--------|-----------|
| `Hour_of_Day` | Time | Converts elapsed seconds to 24-hour clock |
| `IS_NIGHT` | Hour_of_Day | Binary flag — 11pm to 6am is a higher-risk window |
| `LOG_AMOUNT` | Amount | log1p squashes high-value outliers, handles zero safely |
| `AMOUNT_TIER` | Amount | Micro/small/medium/large/whale bucket — fraudsters often test with tiny amounts first |

---

## 📊 Why AUPRC over Accuracy?

With 0.17% fraud rate, a model predicting "normal" for every transaction gets 99.83% accuracy while catching zero fraud. AUPRC measures performance across every possible decision threshold — which matters because the dashboard sensitivity slider adjusts the threshold at runtime.

---

## 🖥️ Dashboard

Built with Streamlit. Three pages:

- **Overview** — project summary, feature engineering explanation, model evolution
- **Live Monitor** — simulate transactions, observe feature deviation profile, adjust sensitivity threshold
- **MLOps Console** — experiment log, AUPRC trend, guardrail status, ROI by version, fraud rate by hour of day

```bash
cd v3_mlops
streamlit run v3_dashboard.py
```

> **Note:** Place `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) in the `v3_xgboost/` directory before running.

---

## 🗃️ Dataset

**Credit Card Fraud Detection** — Kaggle (ULB Machine Learning Group)
- 284,807 transactions | 492 fraud cases | 0.17% fraud rate
- Features V1–V28: PCA-transformed for anonymity
- Raw features: `Time`, `Amount`, `Class`

Not included due to file size. Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Modelling | XGBoost, Scikit-learn, Random Forest |
| Feature Engineering | Pandas, NumPy |
| MLOps | Custom tracker (CSV), registry (JSON), joblib |
| Dashboard | Streamlit, Plotly |
| Environment | Python 3.13, Jupyter |

---

## 📦 Setup

```bash
pip install xgboost scikit-learn pandas numpy streamlit plotly joblib imbalanced-learn
```

Run notebooks in order: `v1_random_forest` → `v2_xgboost_baseline` → `v3_xgboost`

---

## 💡 Key Takeaways

- **SMOTE vs scale_pos_weight:** At 582:1 class ratio, `scale_pos_weight` is cleaner and more effective. SMOTE blurs the decision boundary and reduces precision.
- **eval_set matters:** Setting `eval_metric='aucpr'` without passing `eval_set` to `.fit()` is decorative — the metric never actually runs. V2 had this bug; V3 fixed it.
- **ROI ≠ model quality:** V1 RF had slightly higher net ROI despite lower AUPRC. ROI depends on which specific transactions are missed and their dollar value. AUPRC measures consistent ranking ability across all thresholds.
- **PCA limits feature engineering:** With V1–V28 already anonymised, only `Time` and `Amount` are available for engineering. On raw transaction data, the performance gap between RF and XGBoost would be significantly wider.

---

*Built by Vishwal| April 2026*
