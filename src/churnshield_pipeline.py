import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES=["SeniorCitizen","tenure","MonthlyCharges","TotalCharges","ServiceCount","HasInternet","HasPhone","MonthlyToTotalRatio"]
CATEGORICAL_FEATURES=["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","TenureBand"]

class FeatureEngineer(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None): return self
    def transform(self,X):
        X=X.copy().drop(columns=[c for c in ["customerID","Churn"] if c in X.columns])
        X["TotalCharges"]=pd.to_numeric(X["TotalCharges"],errors="coerce")
        service_cols=["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
        X["ServiceCount"]=sum((X[c].eq("Yes").astype(int) for c in service_cols),start=0)
        X["HasInternet"]=(X["InternetService"]!="No").astype(int)
        X["HasPhone"]=(X["PhoneService"]=="Yes").astype(int)
        X["MonthlyToTotalRatio"]=X["MonthlyCharges"]/(X["TotalCharges"]+1)
        X["TenureBand"]=pd.cut(X["tenure"],[-1,12,24,48,72],labels=["0-12","13-24","25-48","49-72"]).astype(str)
        return X

def make_preprocessor():
    return ColumnTransformer([
        ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),NUMERIC_FEATURES),
        ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),CATEGORICAL_FEATURES)
    ])

def make_pipeline(model):
    return Pipeline([("features",FeatureEngineer()),("preprocessor",make_preprocessor()),("model",model)])

def clean_target(y):
    return y.astype(str).str.strip().map({"Yes":1,"No":0}).astype(int)
