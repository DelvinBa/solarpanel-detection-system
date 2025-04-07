from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI(title="Solar Panel Detection API")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Placeholder: Call your detection logic here.
    # For now, we simply return the filename.
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Here you would integrate your detection logic
    # For example: result = detect_solar_panel(await file.read())
    result = {"filename": file.filename, "message": "Detection triggered (placeholder)"}
    
    return JSONResponse(content=result)
