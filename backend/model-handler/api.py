from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import random
from datetime import datetime

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
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read file (for real implementation, this would go to your AI model)
    contents = await file.read()
    
    # Enhanced mock responses with realistic dental findings
    conditions = [
        {
            "primary_finding": "Periapical radiolucency consistent with apical periodontitis",
            "confidence": random.randint(82, 95),
            "severity": "moderate",
            "affected_teeth": ["#14", "#15"],
            "recommendations": [
                "Endodontic evaluation recommended",
                "Consider root canal therapy",
                "Follow-up radiograph in 3-6 months",
                "Monitor for symptoms of acute infection"
            ],
            "additional_findings": [
                "Slight widening of periodontal ligament space",
                "Cortical bone integrity maintained"
            ],
            "urgency": "routine",
            "icd_codes": ["K04.5", "K04.4"]
        },
        {
            "primary_finding": "Moderate horizontal bone loss suggestive of chronic periodontitis",
            "confidence": random.randint(78, 91),
            "severity": "moderate",
            "affected_teeth": ["#18", "#19", "#30", "#31"],
            "recommendations": [
                "Periodontal therapy indicated",
                "Scaling and root planing recommended",
                "Evaluate for surgical intervention",
                "Establish 3-month maintenance schedule"
            ],
            "additional_findings": [
                "Calculus deposits visible",
                "Furcation involvement grade I-II",
                "Lamina dura discontinuity"
            ],
            "urgency": "routine",
            "icd_codes": ["K05.3", "K05.4"]
        },
        {
            "primary_finding": "Impacted third molar with associated dentigerous cyst",
            "confidence": random.randint(85, 96),
            "severity": "significant",
            "affected_teeth": ["#32"],
            "recommendations": [
                "Oral surgery consultation urgently needed",
                "CT scan for surgical planning",
                "Extraction with cyst enucleation",
                "Histopathological examination recommended"
            ],
            "additional_findings": [
                "Cortical expansion present",
                "Adjacent root resorption possible",
                "Cystic radiolucency 2.5cm diameter"
            ],
            "urgency": "urgent",
            "icd_codes": ["K09.0", "K01.1"]
        }
    ]
    
    # Randomly select a condition for demo
    selected_condition = random.choice(conditions)
    
    # Add some realistic metadata
    response = {
        "analysis_id": f"DENT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "image_info": {
            "filename": file.filename,
            "size_kb": len(contents) // 1024,
            "format": file.content_type
        },
        **selected_condition,
        "technical_quality": {
            "contrast": random.choice(["excellent", "good", "adequate"]),
            "positioning": random.choice(["optimal", "acceptable", "suboptimal"]),
            "artifacts": random.choice(["none", "minimal", "moderate"])
        },
        "next_steps": "Clinical correlation and patient examination recommended",
        "disclaimer": "AI analysis for screening purposes only. Final diagnosis requires clinical evaluation."
    }
    
    return JSONResponse(content=response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)