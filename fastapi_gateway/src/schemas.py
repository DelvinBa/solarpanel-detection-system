# api_gateway/schemas.py
from pydantic import BaseModel
from typing import List

class PipelineRequest(BaseModel):
    gemeentecode: str = "GM0153"
    limit: int = 10
class PipelineResponse(BaseModel):
    message: str

class MultipleVidRequest(BaseModel):
    vids: List[str]
