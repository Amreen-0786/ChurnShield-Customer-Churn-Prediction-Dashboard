# ChurnShield — Customer Churn Prediction Dashboard

This project follows the supplied Code A Nova Data Science brief: clean a telecom dataset, perform EDA, engineer features, compare classifiers, evaluate them, explain predictions with SHAP, persist the model, and provide a Streamlit dashboard. fileciteturn1file0L7-L16

## Stack
Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-Learn, XGBoost, SHAP, Streamlit, Joblib and Jupyter, matching the specified technology stack. fileciteturn1file0L17-L28

## Run immediately
The ZIP contains a synthetic demo CSV with the same schema so you can test the complete project offline.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

## Official IBM/Kaggle dataset
The brief specifies the IBM Telco Churn Dataset. fileciteturn1file0L17-L28

Download and verify it:

```powershell
python download_dataset.py
```

Then train:

```powershell
python train.py --data data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The public IBM repository contains the corresponding CSV, while Kaggle documents the 7,043-customer/21-column dataset. urlIBM Telco dataset repositoryturn0search3 urlKaggle dataset pageturn0search8

## Requirements covered
The brief calls for Logistic Regression → Random Forest → XGBoost, evaluation with confusion matrix/ROC-AUC/Precision-Recall/F1, SHAP explanations, Joblib persistence and a Streamlit dashboard. fileciteturn1file0L34-L48

The deliverables are notebooks, model artifact, `app.py`, insights report, SHAP exports and README. fileciteturn1file0L49-L55

The model pipeline keeps feature engineering and preprocessing inside an sklearn Pipeline to address the no-leakage requirement. Training is local and no paid API is needed, matching the constraints. fileciteturn1file0L85-L89

## Evaluation target
The brief specifies Accuracy >80%, Recall >75% and ROC-AUC >0.85, with the best model justified. fileciteturn1file0L56-L66

**Important:** the included metrics are from the synthetic demo data. Do not submit them as IBM-dataset results. Run the official-data training command before using final metrics.

## 7-day mapping
Day 1 data checks → Day 2 EDA → Day 3 feature engineering → Day 4 model training → Day 5 evaluation/tuning → Day 6 SHAP/model/dashboard → Day 7 dashboard polish/report/demo, matching the brief. fileciteturn1file0L40-L48
