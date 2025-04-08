from fastapi import FastAPI, HTTPException
from .schemas import PipelineRequest, PipelineResponse
import requests

app = FastAPI(title="Solar Panel Detection API Gateway")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/trigger_pipeline", response_model=PipelineResponse)
def trigger_pipeline(request: PipelineRequest):
    # Note: internal_url must match the internal API endpoint.
    internal_url = "http://solarpanel_detection_system:8000/run_pipeline"
    try:
        response = requests.post(internal_url, json=request.model_dump())
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Internal API call failed: {e}")
