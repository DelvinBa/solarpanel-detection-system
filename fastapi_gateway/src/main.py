from fastapi import FastAPI, HTTPException
import requests
from .schemas import PipelineRequest, MultipleVidRequest, PipelineResponse

app = FastAPI(title="Solar Panel Detection API Gateway")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/trigger_ingestion_by_city", response_model=PipelineResponse)
def trigger_ingestion_by_city(request: PipelineRequest):
    """
    Trigger the ingestion pipeline for a given city using the internal API endpoint.
    """
    # This internal URL should match the internal API endpoint for city ingestion.
    internal_url = "http://solarpanel_detection_system:8000/run_ingestion_by_city"
    try:
        response = requests.post(internal_url, json=request.model_dump())
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Internal API call failed: {e}")

@app.post("/trigger_ingestion_by_vids", response_model=PipelineResponse)
def trigger_ingestion_by_vids(request: MultipleVidRequest):
    internal_url = "http://solarpanel_detection_system:8000/run_ingestion_by_vids"
    try:
        response = requests.post(internal_url, json=request.dict())
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Internal API call failed: {e}")
