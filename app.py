from pathlib import Path
import joblib,numpy as np,pandas as pd,streamlit as st,plotly.express as px

MODEL=Path("models/churnshield_model.joblib")
DATA=Path("data/sample/WA_Fn-UseC_-Telco-Customer-Churn_sample.csv")
st.set_page_config(page_title="ChurnShield",page_icon="🛡️",layout="wide")
st.title("🛡️ ChurnShield — Customer Churn Prediction Dashboard")
st.caption("Telecom churn risk prediction with SHAP-based explanation")
if not MODEL.exists(): st.error("Model not found. Run train.py."); st.stop()
model=joblib.load(MODEL)
st.sidebar.header("Customer Parameters")
gender=st.sidebar.selectbox("Gender",["Female","Male"]); senior=st.sidebar.selectbox("Senior Citizen",[0,1])
partner=st.sidebar.selectbox("Partner",["Yes","No"]); dependents=st.sidebar.selectbox("Dependents",["Yes","No"])
tenure=st.sidebar.slider("Tenure (months)",0,72,12); phone=st.sidebar.selectbox("Phone Service",["Yes","No"])
multiple=st.sidebar.selectbox("Multiple Lines",["Yes","No","No phone service"])
internet=st.sidebar.selectbox("Internet Service",["DSL","Fiber optic","No"])
def opts(v): return ["No internet service"] if v=="No" else ["Yes","No"]
security=st.sidebar.selectbox("Online Security",opts(internet)); backup=st.sidebar.selectbox("Online Backup",opts(internet))
device=st.sidebar.selectbox("Device Protection",opts(internet)); tech=st.sidebar.selectbox("Tech Support",opts(internet))
tv=st.sidebar.selectbox("Streaming TV",opts(internet)); movies=st.sidebar.selectbox("Streaming Movies",opts(internet))
contract=st.sidebar.selectbox("Contract",["Month-to-month","One year","Two year"])
paperless=st.sidebar.selectbox("Paperless Billing",["Yes","No"])
payment=st.sidebar.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
monthly=st.sidebar.number_input("Monthly Charges",18.0,130.0,70.0); total=st.sidebar.number_input("Total Charges",0.0,10000.0,800.0)
customer=pd.DataFrame([{"customerID":"dashboard-input","gender":gender,"SeniorCitizen":senior,"Partner":partner,"Dependents":dependents,"tenure":tenure,"PhoneService":phone,"MultipleLines":multiple,"InternetService":internet,"OnlineSecurity":security,"OnlineBackup":backup,"DeviceProtection":device,"TechSupport":tech,"StreamingTV":tv,"StreamingMovies":movies,"Contract":contract,"PaperlessBilling":paperless,"PaymentMethod":payment,"MonthlyCharges":monthly,"TotalCharges":total}])
p=float(model.predict_proba(customer)[0,1]); label="HIGH CHURN RISK" if p>=.5 else "LOWER CHURN RISK"
a,b,c=st.columns(3); a.metric("Churn Probability",f"{p:.1%}"); b.metric("Prediction",label); c.metric("Tenure",f"{tenure} months")
st.progress(float(np.clip(p,0,1)))
st.subheader("Prediction Explanation")
try:
    import shap
    clf=model.named_steps["model"]; feats=model.named_steps["features"]; prep=model.named_steps["preprocessor"]
    if hasattr(clf,"get_booster"):
        Xt=prep.transform(feats.transform(customer)); names=list(prep.get_feature_names_out())
        vals=np.asarray(shap.TreeExplainer(clf).shap_values(Xt))
        if vals.ndim==3: vals=vals[:,:,1]
        local=pd.DataFrame({"Feature":names,"Contribution":vals[0]}); local["Abs"]=local.Contribution.abs(); local=local.nlargest(10,"Abs")
        fig=px.bar(local.sort_values("Contribution"),x="Contribution",y="Feature",orientation="h",title="Top SHAP Contributions")
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Positive contributions increase predicted churn; negative contributions reduce it.")
except Exception as e: st.info(f"SHAP explanation unavailable: {e}")
st.divider(); st.subheader("Interactive EDA")
if DATA.exists():
    d=pd.read_csv(DATA)
    l,r=st.columns(2)
    with l: st.plotly_chart(px.histogram(d,x="Churn",color="Churn",title="Churn Distribution"),use_container_width=True)
    with r:
        cr=d.groupby("Contract")["Churn"].apply(lambda s:(s=="Yes").mean()).reset_index(name="ChurnRate")
        fig=px.bar(cr,x="Contract",y="ChurnRate",title="Churn Rate by Contract"); fig.update_yaxes(tickformat=".0%"); st.plotly_chart(fig,use_container_width=True)
    st.dataframe(d.head(10),use_container_width=True)
