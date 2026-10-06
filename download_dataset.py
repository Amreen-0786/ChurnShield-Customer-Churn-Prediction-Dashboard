from pathlib import Path
import hashlib,urllib.request
URL="https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUT=Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
SHA256="16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
OUT.parent.mkdir(parents=True,exist_ok=True)
urllib.request.urlretrieve(URL,OUT)
digest=hashlib.sha256(OUT.read_bytes()).hexdigest()
if digest!=SHA256: raise RuntimeError("SHA-256 mismatch")
print("Downloaded and verified:",OUT)
