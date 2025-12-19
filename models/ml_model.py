"""ML Model for adaptive learning"""
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os

model = None

def load_ml_model():
    """Load or create ML model"""
    global model
    model_path = 'data/ml_model.pkl'
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("✅ ML model loaded")
    else:
        # Create simple model
        model = LogisticRegression()
        print("📈 New ML model created")

def predict_forget_probability(attempts: int, correct: int) -> float:
    """Predict if user will forget (placeholder)"""
    if model is None:
        return 0.5
    score = correct / max(attempts, 1)
    return model.predict_proba([[attempts, score]])[0][1] if hasattr(model, 'predict_proba') else 0.5
