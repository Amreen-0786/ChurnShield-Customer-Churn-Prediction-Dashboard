from pathlib import Path
import joblib
def test_model_artifact():
    p=Path('models/churnshield_model.joblib'); assert p.exists(); assert hasattr(joblib.load(p),'predict_proba')
