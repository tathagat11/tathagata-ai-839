FROM python:3.9-slim

# Install required packages
RUN pip install fastapi uvicorn mlflow pandas scikit-learn

# Set working directory
WORKDIR /app

# Copy the custom server script
COPY model_server.py /app/

# Expose the port the app runs on
EXPOSE 5002

# Command to run the custom Flask server
CMD ["uvicorn", "model_server:app", "--host", "0.0.0.0", "--port", "5002"]