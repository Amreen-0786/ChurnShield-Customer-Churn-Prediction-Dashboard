import argparse, json
from pathlib import Path
import joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,f1_score,roc_auc_score
from sklearn.model_selection import train_test_split,GridSearchCV
from xgboost import XGBClassifier
from src.churnshield_pipeline import clean_target,make_pipeline

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    ap.add_argument("--output",default="models/churnshield_model.joblib")
    args=ap.parse_args()
    p=Path(args.data)
    if not p.exists(): raise FileNotFoundError(f"{p} not found. Run download_dataset.py first.")
    df=pd.read_csv(p).drop_duplicates()
    if "Churn" not in df: raise ValueError("Expected Churn column.")
    df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
    X=df.drop(columns="Churn"); y=clean_target(df.Churn)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    candidates={
      "Logistic Regression":make_pipeline(LogisticRegression(max_iter=1500,class_weight="balanced",random_state=42)),
      "Random Forest":make_pipeline(RandomForestClassifier(n_estimators=250,max_depth=12,min_samples_leaf=2,class_weight="balanced",random_state=42,n_jobs=-1))}
    fitted={}; rows=[]
    for name,m in candidates.items():
        m.fit(Xtr,ytr); fitted[name]=m; pr=m.predict(Xte); pp=m.predict_proba(Xte)[:,1]
        rows.append({"model":name,"accuracy":accuracy_score(yte,pr),"recall":recall_score(yte,pr),"f1":f1_score(yte,pr),"roc_auc":roc_auc_score(yte,pp)})
    xgb=make_pipeline(XGBClassifier(objective="binary:logistic",eval_metric="logloss",random_state=42,n_jobs=2))
    grid=GridSearchCV(xgb,{"model__n_estimators":[150,250],"model__max_depth":[3,4],"model__learning_rate":[.05,.1],"model__subsample":[.85],"model__colsample_bytree":[.85]},scoring="roc_auc",cv=3,n_jobs=1)
    grid.fit(Xtr,ytr); fitted["XGBoost"]=grid.best_estimator_; pr=grid.predict(Xte); pp=grid.predict_proba(Xte)[:,1]
    rows.append({"model":"XGBoost","accuracy":accuracy_score(yte,pr),"recall":recall_score(yte,pr),"f1":f1_score(yte,pr),"roc_auc":roc_auc_score(yte,pp)})
    out=pd.DataFrame(rows).sort_values("roc_auc",ascending=False); best_name=out.iloc[0].model; best=fitted[best_name]
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); joblib.dump(best,args.output)
    Path("reports").mkdir(exist_ok=True); out.to_csv("reports/model_comparison.csv",index=False)
    Path("reports/metrics.json").write_text(json.dumps({"best_model":best_name,**out.iloc[0].to_dict()},indent=2),encoding="utf-8")
    print(out.to_string(index=False)); print("Saved:",args.output)
if __name__=="__main__": main()
