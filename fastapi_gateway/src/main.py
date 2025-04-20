from fastapi import FastAPI, HTTPException
import requests
import os
from .schemas import PipelineRequest, MultipleVidRequest, PipelineResponse

app = FastAPI(title="Solar Panel Detection API Gateway")

# In containerized environments like Docker, each container has its own network namespace
# So we use the service name from docker-compose as the hostname, not localhost
# This allows containers to communicate with each other
DETECTION_SERVICE_URL = os.getenv('DETECTION_SERVICE_URL', 'http://solarpanel_detection_service:8000')

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/trigger_ingestion_by_city", response_model=PipelineResponse)
def trigger_ingestion_by_city(request: PipelineRequest):
    """
    Trigger the ingestion pipeline for a given city using the internal API endpoint.
    """
    # Use environment variable for service URL
    internal_url = f"{DETECTION_SERVICE_URL}/run_ingestion_by_city"
    try:
        response = requests.post(internal_url, json=request.model_dump())
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Internal API call failed: {e}")

@app.post("/trigger_ingestion_by_vids", response_model=PipelineResponse)
def trigger_ingestion_by_vids(request: MultipleVidRequest):
    internal_url = f"{DETECTION_SERVICE_URL}/run_ingestion_by_vids"
    try:
        response = requests.post(internal_url, json=request.dict())
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Internal API call failed: {e}")
