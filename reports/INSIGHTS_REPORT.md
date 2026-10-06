# ChurnShield — Insights Report

## Demo artifact
This packaged report uses a **synthetic demonstration dataset matching the IBM Telco schema** so the complete project is runnable immediately.

### Demo model performance

| Metric | Result | Brief target |
|---|---:|---:|
| Accuracy | 0.946 | > 0.80 |
| Recall | 0.904 | > 0.75 |
| F1 | 0.904 | Reported |
| ROC-AUC | 0.989 | > 0.85 |

The demo deployment model is **XGBoost** and exceeds the brief's stated demonstration thresholds.

### Demo churn drivers
The synthetic data intentionally creates relationships around contract type, tenure, internet service, monthly charges, payment method, and support/security services. These are **demonstration relationships only**, not official IBM dataset findings.

### Official internship run
Download the official IBM dataset and execute:

```powershell
python download_dataset.py
python train.py --data data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Then replace the demo metrics and insights with the official-data results.

### Explainability
A SHAP global feature-importance export is included when SHAP is available.
