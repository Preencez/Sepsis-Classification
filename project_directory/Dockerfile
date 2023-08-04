# Use the official Python image as the base image
FROM python:3.9

# Copy the Streamlit app file to the /app/ directory inside the container
COPY streamlit_app.py /app/

# Copy the FastAPI app file to the /app/ directory inside the container
COPY main.py /app/

# Set the working directory
WORKDIR /app

# Install required dependencies for both Streamlit and FastAPI
RUN pip install pandas joblib scikit-learn streamlit fastapi uvicorn

# Expose the ports used by Streamlit and FastAPI
EXPOSE 8501 8000

# Start the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
