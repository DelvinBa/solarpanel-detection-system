from fastapi import APIRouter, HTTPException, BackgroundTasks
from .schemas import PipelineRequest, PipelineResponse, MultipleVidRequest
# Use alias to avoid shadowing
from solarpanel_detection_service.src.data_collection_service.run_collection_inference import (
    run_collection_by_city as collection_by_city,
    run_collection_by_vids as collection_by_vids
)

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/run_collection_by_city", response_model=PipelineResponse)
def run_collection_by_city_endpoint(request: PipelineRequest, background_tasks: BackgroundTasks):
    """
    Triggers the image detection/scraping pipeline for all houses in a specific city.
    Processes are handled as background tasks based on the provided gemeentecode (city code).
    """
    try:
        # Pass parameters positionally to the collection function.
        background_tasks.add_task(collection_by_city, request.gemeentecode, request.limit)
        return PipelineResponse(
            message=f"Pipeline triggered for city code '{request.gemeentecode}'. Processing in the background."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/run_collection_by_vids", response_model=PipelineResponse)
def trigger_collection_by_vids(request: MultipleVidRequest, background_tasks: BackgroundTasks):
    """
    Triggers the collection pipeline for a list of VIDs.
    """
    try:
        background_tasks.add_task(collection_by_vids, request.vids)
        vids_str = ", ".join(request.vids)
        return PipelineResponse(
            message=f"Pipeline triggered for VIDs '{vids_str}'. Processing in the background."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))