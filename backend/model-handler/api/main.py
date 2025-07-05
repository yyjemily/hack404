from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import io
from PIL import Image
import base64
from typing import Optional
import logging
import os
from models.model_handler import DentalAIPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dental AI Assistant API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[DentalAIPredictor] = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global predictor
    try:
        # Configure your model path here
        model_path = os.getenv("MODEL_PATH", "models/dental_model.pth")  # or .h5
        
        # If using PyTorch with custom architecture, pass model_class
        # model_class = YourResNetClass  # Import your ResNet class
        predictor = DentalAIPredictor(model_path)
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Dental AI Assistant API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if predictor and predictor.model_loaded else "not loaded"
    return {
        "status": "healthy",
        "model_status": model_status
    }

@app.post("/predict/upload")
async def predict_from_upload(file: UploadFile = File(...)):
    """Predict dental condition from uploaded image"""
    if not predictor or not predictor.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Preprocess for model (resize, normalize, etc.)
        processed_image = preprocess_image(image_array)
        
        # Make prediction
        result = predictor.predict_dental_condition(processed_image)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/base64")
async def predict_from_base64(data: dict):
    """Predict dental condition from base64 encoded image"""
    if not predictor or not predictor.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Preprocess for model
        processed_image = preprocess_image(image_array)
        
        # Make prediction
        result = predictor.predict_dental_condition(processed_image)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/reload")
async def reload_model(data: dict):
    """Reload model with new path"""
    global predictor
    try:
        model_path = data.get('model_path')
        if not model_path:
            raise HTTPException(status_code=400, detail="model_path required")
        
        predictor = DentalAIPredictor(model_path)
        
        return {"message": "Model reloaded successfully", "model_path": model_path}
        
    except Exception as e:
        logger.error(f"Model reload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def preprocess_image(image_array: np.ndarray) -> np.ndarray:
    """Preprocess image for model input"""
    # Convert to RGB if needed
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Resize to model input size (adjust as needed)
    target_size = (224, 224)  # Common ResNet input size
    resized_image = cv2.resize(image_array, target_size)
    
    # Normalize pixel values
    normalized_image = resized_image.astype(np.float32) / 255.0
    
    # Add batch dimension
    batched_image = np.expand_dims(normalized_image, axis=0)
    
    return batched_image

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)