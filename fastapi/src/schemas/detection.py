from typing import List
from pydantic import BaseModel

class DetectionRequest(BaseModel):
    ids: List[int]

class DetectionResult(BaseModel):
    id: int
    detection_image: str  # e.g., base64 string or a URL
    message: str

class DetectionResponse(BaseModel):
    results: List[DetectionResult]