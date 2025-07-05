class DentalAIClient {
    constructor(apiUrl = 'http://localhost:8000') {
        this.apiUrl = apiUrl;
    }

    async predictFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${this.apiUrl}/predict`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        return await response.json();
    }

    async predictBase64(base64Image) {
        const response = await fetch(`${this.apiUrl}/predict_base64`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({image: base64Image})
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        return await response.json();
    }

    async reloadModel(modelPath) {
        const response = await fetch(`${this.apiUrl}/reload_model`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model_path: modelPath})
        });
        
        return await response.json();
    }

    async getStatus() {
        const response = await fetch(`${this.apiUrl}/status`);
        return await response.json();
    }

    // Utility: Convert file to base64
    fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    // Utility: Resize image before sending
    async resizeImage(file, maxWidth = 224, maxHeight = 224) {
        return new Promise((resolve) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            
            img.onload = () => {
                canvas.width = maxWidth;
                canvas.height = maxHeight;
                ctx.drawImage(img, 0, 0, maxWidth, maxHeight);
                
                canvas.toBlob(resolve, 'image/jpeg', 0.8);
            };
            
            img.src = URL.createObjectURL(file);
        });
    }
}

// Simple usage wrapper
class DentalAI {
    constructor() {
        this.client = new DentalAIClient();
    }

    async analyze(imageFile) {
        try {
            // Resize image first
            const resizedFile = await this.client.resizeImage(imageFile);
            
            // Get prediction
            const result = await this.client.predictFile(resizedFile);
            
            // Parse results
            const prediction = result.predictions[0];
            const maxIdx = prediction.indexOf(Math.max(...prediction));
            const confidence = result.confidence;
            
            return {
                prediction: maxIdx,
                confidence: confidence,
                modelType: result.model_type,
                rawResult: result
            };
        } catch (error) {
            console.error('Analysis failed:', error);
            throw error;
        }
    }
}

// Export for use
if (typeof module !== 'undefined') {
    module.exports = { DentalAIClient, DentalAI };
}