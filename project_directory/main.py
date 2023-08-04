from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load the numerical imputer, scaler, and model
num_imputer_filepath = "D:\Projects\Sepsis Classification\ML components"
num_imputer = joblib.load(num_imputer_filepath)

scaler_filepath = "D:/Projects/Sepsis Classification/ML components/scaler.joblib"
scaler = joblib.load(scaler_filepath)

model_filepath = "D:/Projects/Sepsis Classification/ML components/rf_model.joblib"
model = joblib.load(model_filepath)

# Define the input data model for the API request
class InputData(BaseModel):
    PRG: float
    PL: float
    PR: int
    SK: float
    TS: float
    M11: float
    BD2: float
    Age: int
    Insurance: int

# Define a function to preprocess the input data
def preprocess_input_data(input_data):
    input_data_df = pd.DataFrame([input_data.dict()])
    num_columns = input_data_df.select_dtypes(include='number').columns

    input_data_imputed_num = num_imputer.transform(input_data_df[num_columns])
    input_scaled_df = pd.DataFrame(scaler.transform(input_data_imputed_num), columns=num_columns)

    return input_scaled_df

# Define a function to make the sepsis prediction
def predict_sepsis(input_data):
    input_scaled_df = preprocess_input_data(input_data)
    prediction = model.predict(input_scaled_df)[0]
    probabilities = model.predict_proba(input_scaled_df)[0]
    sepsis_status = "Positive" if prediction == 1 else "Negative"

    output_dict = input_data.dict()
    output_dict['Prediction'] = sepsis_status
    output_dict['Negative Probability'] = probabilities[0]
    output_dict['Positive Probability'] = probabilities[1]

    return output_dict

# Define the API endpoint for the sepsis prediction
@app.post("/predict_sepsis/")
def predict_sepsis_api(input_data: InputData):
    output_dict = predict_sepsis(input_data)
    return output_dict
