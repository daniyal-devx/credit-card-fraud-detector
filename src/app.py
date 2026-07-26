from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load your real trained model + scaler
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/")
def home():
    return {"message": "Fraud detection API is running"}

@app.post("/predict")
def predict(transaction: Transaction):
    # Convert incoming data into a DataFrame, same column order as training
    data = pd.DataFrame([transaction.dict()])
    
    # Scale Time and Amount, same as training
    data[['Time', 'Amount']] = scaler.transform(data[['Time', 'Amount']])
    
    # Predict
    proba = model.predict_proba(data)[0][1]
    is_fraud = bool(proba >= 0.5)
    
    return {
        "fraud_probability": round(float(proba), 4),
        "is_fraud": is_fraud
    }