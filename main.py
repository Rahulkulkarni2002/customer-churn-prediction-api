from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="Customer Churn Prediction API")

# Load and prepare data, train the model once at startup
data = pd.read_csv("cleaned_retail.csv")
rfm = pd.read_csv("rfm_table.csv")

data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
rfm["Churned"] = (rfm["Recency"] > 90).astype(int)

tenure = data.groupby("Customer ID")["InvoiceDate"].agg(["min", "max"])
tenure["Tenure"] = (tenure["max"] - tenure["min"]).dt.days
tenure = tenure[["Tenure"]].reset_index()
rfm = rfm.merge(tenure, on="Customer ID", how="left")

rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]
rfm["Frequency_log"] = np.log1p(rfm["Frequency"])
rfm["Monetary_log"] = np.log1p(rfm["Monetary"])
rfm["Freq_Monetary_Interaction"] = rfm["Frequency"] * rfm["Monetary"]

feature_cols = [
    "Frequency",
    "Monetary",
    "Tenure",
    "AvgOrderValue",
    "Frequency_log",
    "Monetary_log",
    "Freq_Monetary_Interaction",
]
X = rfm[feature_cols]
y = rfm["Churned"]

model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X, y)


# Define the expected input shape for predictions
class CustomerFeatures(BaseModel):
    Frequency: float
    Monetary: float
    Tenure: float


@app.get("/")
def read_root():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict_churn(customer: CustomerFeatures):
    if customer.Frequency <= 0:
        raise HTTPException(status_code=400, detail="Frequency must be greater than 0")
    if customer.Monetary < 0:
        raise HTTPException(status_code=400, detail="Monetary cannot be negative")
    if customer.Tenure < 0:
        raise HTTPException(status_code=400, detail="Tenure cannot be negative")

    avg_order_value = customer.Monetary / customer.Frequency
    frequency_log = np.log1p(customer.Frequency)
    monetary_log = np.log1p(customer.Monetary)
    interaction = customer.Frequency * customer.Monetary

    features = pd.DataFrame(
        [
            {
                "Frequency": customer.Frequency,
                "Monetary": customer.Monetary,
                "Tenure": customer.Tenure,
                "AvgOrderValue": avg_order_value,
                "Frequency_log": frequency_log,
                "Monetary_log": monetary_log,
                "Freq_Monetary_Interaction": interaction,
            }
        ]
    )

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "churned_prediction": int(prediction),
        "churn_probability": float(probability),
    }
