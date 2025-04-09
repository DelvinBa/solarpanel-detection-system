FROM python:3.9-slim

WORKDIR /app/solarpanel_detection_system

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire project folder into the container
COPY . .

# Tell Python where your package is (optional if /app is already in PYTHONPATH)
ENV PYTHONPATH=/app

# Point Uvicorn to the correct module path
CMD ["uvicorn", "src.api_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
