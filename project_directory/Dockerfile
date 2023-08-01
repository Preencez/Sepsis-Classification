# Dockerfile

# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the FastAPI app and ML components to the container
COPY fastapi_app.py /app/
COPY ML_components /app/ML_components

# Install dependencies
RUN pip install fastapi uvicorn joblib pandas scikit-learn matplotlib

# Expose the port on which FastAPI will run (change if needed)
EXPOSE 8000

# Command to run the FastAPI application with Uvicorn
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
