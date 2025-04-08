# Use an official lightweight Python image.
FROM python:3.9-slim

# Set working directory.
WORKDIR /app

# Copy requirements and install dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY src/ ./src/

# Expose the port that the app runs on.
EXPOSE 8000

# Command to run the app with Uvicorn.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
