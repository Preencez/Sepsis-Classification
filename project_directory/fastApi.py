# fastapi_app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load the trained model
model_filepath = "D:/Projects/Sepsis Classification/ML components/rf_model.joblib"
model = joblib.load(model_filepath)

# Define the input data model
class InputData(BaseModel):
    PRG: float
    PL: float
    PR: float
    SK: float
    TS: float
    M11: float
    BD2: float
    Age: float
    Insurance: int

# Define the prediction endpoint
@app.post("/predict/")
def predict_sepsis(input_data: InputData):
    try:
        input_data_dict = input_data.dict()
        input_data_list = [list(input_data_dict.values())]

        # Make the prediction
        prediction = model.predict(input_data_list)[0]
        probabilities = model.predict_proba(input_data_list)[0]
        sepsis_status = "Positive" if prediction == 1 else "Negative"

        return {
            "Prediction": sepsis_status,
            "Negative Probability": probabilities[0],
            "Positive Probability": probabilities[1]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
