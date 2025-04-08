from pydantic import BaseModel

class PipelineRequest(BaseModel):
    gemeentecode: str = "GM0153"
    limit: int = 10
    demo: bool = False

class PipelineResponse(BaseModel):
    message: str
