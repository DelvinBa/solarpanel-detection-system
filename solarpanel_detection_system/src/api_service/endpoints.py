from fastapi import APIRouter, HTTPException, BackgroundTasks
from .schemas import PipelineRequest, PipelineResponse, MultipleVidRequest
# Use alias to avoid shadowing
from solarpanel_detection_system.src.data_ingestion_service.run_ingestion_inference import (
    run_ingestion_by_city as ingestion_by_city,
    run_ingestion_by_vids as ingestion_by_vids
)

router = APIRouter()

@router.post("/run_ingestion_by_city", response_model=PipelineResponse)
def run_ingestion_by_city_endpoint(request: PipelineRequest, background_tasks: BackgroundTasks):
    """
    Triggers the image detection/scraping pipeline for all houses in a specific city.
    Processes are handled as background tasks based on the provided gemeentecode (city code).
    """
    try:
        # Pass parameters positionally to the ingestion function.
        background_tasks.add_task(ingestion_by_city, request.gemeentecode, request.limit)
        return PipelineResponse(
            message=f"Pipeline triggered for city code '{request.gemeentecode}'. Processing in the background."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/run_ingestion_by_vids", response_model=PipelineResponse)
def trigger_ingestion_by_vids(request: MultipleVidRequest, background_tasks: BackgroundTasks):
    """
    Triggers the ingestion pipeline for a list of VIDs.
    """
    try:
        background_tasks.add_task(ingestion_by_vids, request.vids)
        vids_str = ", ".join(request.vids)
        return PipelineResponse(
            message=f"Pipeline triggered for VIDs '{vids_str}'. Processing in the background."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))