FROM python:3.10

# Set working directory
WORKDIR /app

# Install required packages
RUN pip install fastapi uvicorn mlflow pandas scikit-learn numpy

# Copy the custom server script
COPY model_server.py /app/

# Copy data for column transformer
COPY data/01_raw/dataset_id_T01_V3_96.csv /app/data/

# Expose the port the app runs on
EXPOSE 5002

# Command to run the custom Flask server
CMD ["uvicorn", "model_server:app", "--host", "0.0.0.0", "--port", "5002"]