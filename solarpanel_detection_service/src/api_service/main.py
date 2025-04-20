from fastapi import FastAPI
from .endpoints import router as pipeline_router

app = FastAPI(title="Internal Solar Detection API")
app.include_router(pipeline_router)
