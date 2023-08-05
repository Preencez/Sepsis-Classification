from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import uvicorn

app = FastAPI()

# Load the numerical imputer, scaler, and model
num_imputer_filepath = "/com.docker.devenvironments.code/project_directory/ML components/numerical_imputer.joblib"
scaler_filepath = "/com.docker.devenvironments.code/project_directory/ML components/scaler.joblib"
model_filepath = "/com.docker.devenvironments.code/project_directory/ML components/rf_model.joblib"

num_imputer = joblib.load(num_imputer_filepath)
scaler = joblib.load(scaler_filepath)
model = joblib.load(model_filepath)

class PatientData(BaseModel):
    PRG: float
    PL: float
    PR: float
    SK: float
    TS: float
    M11: float
    BD2: float
    Age: float
    Insurance: int

def preprocess_input_data(input_data):
    input_data_df = pd.DataFrame([input_data])
    num_columns = [col for col in input_data_df.columns if input_data_df[col].dtype != 'object']
    input_data_imputed_num = num_imputer.transform(input_data_df[num_columns])
    input_scaled_df = pd.DataFrame(scaler.transform(input_data_imputed_num), columns=num_columns)
    return input_scaled_df

@app.get("/")
def read_root():
    return "Sepsis Prediction App"

@app.post("/sepsis/predict")
def predict_sepsis_endpoint(data: PatientData):
    input_data = data.dict()
    input_scaled_df = preprocess_input_data(input_data)
    
    probabilities = model.predict_proba(input_scaled_df)[0]
    prediction = np.argmax(probabilities)

    sepsis_status = "Positive" if prediction == 1 else "Negative"
    probability = probabilities[1] if prediction == 1 else probabilities[0]

    if prediction == 1:
        status_icon = "✔"
        sepsis_explanation = "Sepsis is a life-threatening condition caused by an infection. A positive prediction suggests that the patient might be exhibiting sepsis symptoms and requires immediate medical attention."
    else:
        status_icon = "✘"
        sepsis_explanation = "Sepsis is a life-threatening condition caused by an infection. A negative prediction suggests that the patient is not currently exhibiting sepsis symptoms."

    statement = f"The patient's sepsis status is {sepsis_status} {status_icon} with a probability of {probability:.2f}. {sepsis_explanation}"

    user_input_statement = "Please note this is the user-inputted data: "
    output_df = pd.DataFrame([input_data])

    result = {'predicted_sepsis': sepsis_status, 'statement': statement, 'user_input_statement': user_input_statement, 'input_data_df': output_df.to_dict('records')}
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
