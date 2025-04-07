from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from .schemas.detection import DetectionRequest, DetectionResponse, DetectionResult

app = FastAPI(title="Solar Panel Detection API")

@app.get("/health")
def health_check():
    return {"status": "healthy"}
@app.post("/detect", response_model=DetectionResponse)
async def detect(request: DetectionRequest):
    # Example logic: loop over IDs, build results
    results = []
    for detection_id in request.ids:
        # You'd call your real detection pipeline here
        dummy_image = "base64-encoded-image"
        results.append(
            DetectionResult(
                id=detection_id,
                detection_image=dummy_image,
                message="Detection successful"
            )
        )

    return DetectionResponse(results=results)
