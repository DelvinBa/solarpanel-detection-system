from fastapi import APIRouter, HTTPException, BackgroundTasks
from .schemas import PipelineRequest, PipelineResponse
# Adjust the import path to where your function is defined.
from solarpanel_detection_system.src.data_ingestion_service.run_ingestion_inference import get_images_by_citycode

router = APIRouter()

@router.post("/run_pipeline", response_model=PipelineResponse)
def run_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """
    Triggers the image detection/scraping pipeline.
    The pipeline runs as a background task.
    """
    try:
        background_tasks.add_task(get_images_by_citycode, request.gemeentecode, request.limit, request.demo)
        return PipelineResponse(message=f"Pipeline triggered for gemeente {request.gemeentecode}. Processing in background.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
