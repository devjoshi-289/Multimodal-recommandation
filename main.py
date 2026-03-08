from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn
import shutil
import os
import logging
from search import multimodal_search

app = FastAPI(title="Multimodal Product Search API")

from fastapi.staticfiles import StaticFiles

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount images directory to serve directly to UI
app.mount("/images", StaticFiles(directory="data/filtered_images"), name="images")

TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
os.makedirs("dataset", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

@app.get("/")
def read_root():
    return {"message": "Multimodal Search Backend Running"}

@app.post("/search")
async def search_endpoint(
    text_query: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None),
    category: Optional[str] = Form(None),
    limit: int = Form(20),
    offset: int = Form(0)
):
    logger.info(f"Received search request. Text: {text_query}, Image: {image_file.filename if image_file else None}, Category: {category}, Limit: {limit}, Offset: {offset}")
    
    if not text_query and not image_file and not category:
        raise HTTPException(status_code=400, detail="Must provide at least text_query, image_file, or category.")
    
    image_path = None
    if image_file:
        image_path = os.path.join(TEMP_UPLOAD_DIR, image_file.filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image_file.file, buffer)
            
    try:
        results = multimodal_search(
            image_path=image_path, 
            text_query=text_query,
            category=category,
            limit=limit,
            offset=offset
        )
        return results
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during search.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
