import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import time
import os 

# Load the numerical imputer
num_imputer_filepath = "D:/Projects/Sepsis Classification/ML components/numerical_imputer.joblib"
num_imputer = joblib.load(num_imputer_filepath)

# Load the scaler
scaler_filepath = "D:/Projects/Sepsis Classification/ML components/scaler.joblib"
scaler = joblib.load(scaler_filepath)

# Load the Random Forest model
model_filepath = "D:/Projects/Sepsis Classification/ML components/rf_model.joblib"
model = joblib.load(model_filepath)

# Define a function to preprocess the input data
def preprocess_input_data(input_data):
    input_data_df = pd.DataFrame(input_data, columns=['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age', 'Insurance'])
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

    output_df = pd.DataFrame(input_data, columns=['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age', 'Insurance'])
    output_df['Prediction'] = sepsis_status
    output_df['Negative Probability'] = probabilities[0]
    output_df['Positive Probability'] = probabilities[1]

    return output_df, probabilities

# Create a Streamlit app
# Create a Streamlit app
def main():
    st.title('Sepsis Prediction App')

    # Display the image using HTML
    image_url = "https://lh3.googleusercontent.com/-UU_-cM2FZnI/YLgc3z-EFCI/AAAAAAAAAuo/sORie7aJNgsM8UY7_qAUTZUSeSxKtA7UQCLcBGAsYHQ/s16000/streamlit_log.png"
    st.markdown(f'<img src="{image_url}" alt="Streamlit Logo" style="width: 300px;">', unsafe_allow_html=True)


    # How to use
    st.sidebar.title('How to Use')
    st.sidebar.markdown('Follow these steps to predict sepsis:')
    st.sidebar.markdown('1. Enter the patient\'s medical data on the left sidebar.')
    st.sidebar.markdown('2. Click the "Predict" button to analyze the patient\'s condition.')
    st.sidebar.markdown('3. Wait for the app to process the information.')
    st.sidebar.markdown('4. View the prediction result and the likelihood of sepsis.')
    st.sidebar.markdown('5. Interpret the result: "Positive" indicates sepsis, and "Negative" means no sepsis.')
    st.sidebar.markdown('6. Check the probability bars for the likelihood of sepsis.')
    #st.sidebar.markdown('7. If feature importance is available, view the impact of each input on the prediction.')
  
    st.sidebar.title('Input Parameters')

    # Input parameter explanations

    # Plasma Glucose - Slider
    st.sidebar.markdown('**PRG:** Plasma Glucose')
    PRG = st.sidebar.slider('PRG', min_value=0.0, max_value=200.0, value=100.0)

    # Blood Work Result 1 - Text Input
    st.sidebar.markdown('**PL:** Blood Work Result 1')
    PL = st.sidebar.number_input('PL', value=0.0)

    # Blood Pressure Measured - Slider
    st.sidebar.markdown('**PR:** Blood Pressure Measured')
    PR = st.sidebar.slider('PR', min_value=60, max_value=180, value=120)

    # Blood Work Result 2 - Text Input
    st.sidebar.markdown('**SK:** Blood Work Result 2')
    SK = st.sidebar.number_input('SK', value=0.0)

    # Blood Work Result 3 - Slider
    st.sidebar.markdown('**TS:** Blood Work Result 3')
    TS = st.sidebar.slider('TS', min_value=0.0, max_value=100.0, value=50.0)

    # BMI - Text Input
    st.sidebar.markdown('**M11:** BMI')
    M11 = st.sidebar.number_input('M11', value=0.0)

    # Blood Work Result 4 - Slider
    st.sidebar.markdown('**BD2:** Blood Work Result 4')
    BD2 = st.sidebar.slider('BD2', min_value=0.0, max_value=10.0, value=5.0)

    # Age of the Patient - Slider
    st.sidebar.markdown('**Age:** What is the Age of the Patient: ')
    Age = st.sidebar.slider('Age', min_value=0, max_value=120, value=30)

    # Insurance - Radio Buttons
    st.sidebar.markdown('**Insurance:** Does the patient have Insurance?')
    insurance_options = {0: 'NO', 1: 'YES'}
    Insurance = st.sidebar.radio('Insurance', list(insurance_options.keys()), format_func=lambda x: insurance_options[x])

    input_data = [[PRG, PL, PR, SK, TS, M11, BD2, Age, Insurance]]

    if st.sidebar.button('Predict'):
        with st.spinner("Predicting..."):
            # Simulate a long-running process
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.1)
                progress_bar.progress(i + 1)

            output_df, probabilities = predict_sepsis(input_data)

            st.subheader('Prediction Result')
            st.write(output_df)

            # Plot the probabilities as a pie chart with black and red colors
            fig, ax = plt.subplots()
            colors = ['#FF0000', '#000000']  # Red and Black colors
            ax.pie(probabilities, labels=['Negative', 'Positive'], autopct='%1.1f%%', startangle=90, colors=colors)
            ax.set_title('Sepsis Prediction Probabilities')
            st.pyplot(fig)
                          
if __name__ == '__main__':
    main()