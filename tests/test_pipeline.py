import pandas as pd
from src.churnshield_pipeline import FeatureEngineer
def test_features():
    x=pd.DataFrame([{"customerID":"x","gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":5,"PhoneService":"Yes","MultipleLines":"No","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":55,"TotalCharges":"250"}])
    out=FeatureEngineer().fit_transform(x)
    assert "customerID" not in out and "ServiceCount" in out and "TenureBand" in out
