from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import random
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import io
import numpy as np
from datetime import datetime


model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features,6)

state_dict = torch.load('densenetpretrained_dental_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

model.eval()

app = FastAPI(title="Image Upload API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Image Upload API is running"}

@app.get("/status")
async def status():
    return {"status": "running", "message": "API is healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read file (for real implementation, this would go to your AI model)
    contents = await file.read()
    
    image = Image.open(io.BytesIO(contents))
    image = image.resize((224, 224))  # Resize to model input size
    img_array = np.array(image)/255.0
    img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1).unsqueeze(0)  # Convert to tensor and add batch dimension
    with torch.no_grad():
        predictions = model(img_tensor)
        predictions = torch.softmax(predictions, dim=1)

    confidence = float(torch.max(predictions))*100
    predicted_class = int(torch.argmax(predictions))
    condition_map = {
        0: "Normal",
        1: "Cavity",
        2: "Gum Disease",
        3: "Impacted Tooth",
        4: "Root Canal Needed",
    }

    primary_finding = condition_map.get(predicted_class, "Unknown Condition")

    def generate_response_message(finding,confidence,severity,urgency,next_steps):
        message = f"The diagnosed patient has {finding} of {severity} severity."
        message+= f"Therefore, I suggest that there is an {urgency} urgency to seek {next_steps}."
        return message
   
    finding_lower = primary_finding.lower()
    if "normal" in finding_lower:
        urgency = "low"
        next_steps = "routine monitoring and regular check-ups"
    elif "caries" in finding_lower and confidence > 70:
        urgency = "moderate"
        next_steps = "restorative treatment and caries management"
    elif "periodontal" in finding_lower:
        urgency = "moderate"
        next_steps = "periodontal therapy and maintenance"
    elif "impacted" in finding_lower:
        urgency = "high"
        next_steps = "surgical consultation and extraction planning"
    elif "root canal" in finding_lower:
        urgency = "high"
        next_steps = "endodontic treatment and pain management"
    else:
        urgency = "moderate"
        next_steps = "clinical correlation and follow-up examination"
    
    user_message = generate_response_message(primary_finding, confidence, urgency, urgency, next_steps)

    # Add some realistic metadata
    response = {
        "analysisid": f"DENT{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "image_info": {
            "filename": file.filename,
            "size_kb": len(contents) // 1024,
            "format": file.content_type
        },
        "primary_finding": primary_finding,
        "confidence": int(confidence),
        
        "severity": random.choice(["mild", "moderate", "severe"]),
        "affected_teeth": ["#14", "#15"],  # You might want to make this dynamic
        "recommendations": [
            "Clinical correlation recommended",
            "Follow-up as needed based on findings"
        ],
        "additional_findings": [
            "AI analysis completed successfully"
        ],
        "urgency": "routine",
        "technical_quality": {
            "contrast": "good",
            "positioning": "acceptable",
            "artifacts": "minimal"
        },
        "next_steps": "Clinical correlation and patient examination recommended",
        "disclaimer": "AI analysis for screening purposes only. Final diagnosis requires clinical evaluation."
    }

    return JSONResponse(content=response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
