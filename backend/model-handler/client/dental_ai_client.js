// client/dental_ai_client.js
class DentalAIClient {
    constructor(apiBaseUrl = 'http://localhost:8000') {
        this.apiBaseUrl = apiBaseUrl;
        this.isLoading = false;
    }

    async healthCheck() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            return await response.json();
        } catch (error) {
            console.error('Health check failed:', error);
            throw error;
        }
    }

    async predictFromFile(file) {
        this.isLoading = true;
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${this.apiBaseUrl}/predict/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            return result;
        } catch (error) {
            console.error('Prediction from file failed:', error);
            throw error;
        } finally {
            this.isLoading = false;
        }
    }

    async predictFromBase64(base64Image) {
        this.isLoading = true;
        try {
            const response = await fetch(`${this.apiBaseUrl}/predict/base64`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: base64Image })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            return result;
        } catch (error) {
            console.error('Prediction from base64 failed:', error);
            throw error;
        } finally {
            this.isLoading = false;
        }
    }

    async reloadModel(modelPath) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/model/reload`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ model_path: modelPath })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Model reload failed:', error);
            throw error;
        }
    }

    fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
        });
    }

    // Image preprocessing utilities
    resizeImage(canvas, targetWidth, targetHeight) {
        const ctx = canvas.getContext('2d');
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        
        tempCanvas.width = targetWidth;
        tempCanvas.height = targetHeight;
        
        tempCtx.drawImage(canvas, 0, 0, targetWidth, targetHeight);
        
        return tempCanvas.toDataURL('image/jpeg', 0.8);
    }

    async processImageFile(file, targetSize = { width: 224, height: 224 }) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            img.onload = () => {
                canvas.width = targetSize.width;
                canvas.height = targetSize.height;
                
                ctx.drawImage(img, 0, 0, targetSize.width, targetSize.height);
                
                const base64 = canvas.toDataURL('image/jpeg', 0.8);
                resolve(base64);
            };
            
            img.onerror = reject;
            img.src = URL.createObjectURL(file);
        });
    }
}

// Usage example and integration
class DentalAIApp {
    constructor() {
        this.client = new DentalAIClient();
        this.currentResult = null;
        this.init();
    }

    async init() {
        try {
            const health = await this.client.healthCheck();
            console.log('API Health:', health);
        } catch (error) {
            console.error('API not available:', error);
        }
    }

    async analyzeDentalImage(file) {
        try {
            // Show loading state
            this.showLoading(true);
            
            // Process and predict
            const result = await this.client.predictFromFile(file);
            
            // Store result
            this.currentResult = result;
            
            // Display result
            this.displayResult(result);
            
            return result;
        } catch (error) {
            console.error('Analysis failed:', error);
            this.showError(error.message);
        } finally {
            this.showLoading(false);
        }
    }

    displayResult(result) {
        if (result.error) {
            this.showError(result.error);
            return;
        }

        const { dental_analysis, confidence, model_type } = result;
        
        console.log('Dental Analysis Result:', {
            condition: dental_analysis?.condition || 'Unknown',
            confidence: dental_analysis?.confidence_score || 0,
            modelType: model_type,
            allProbabilities: dental_analysis?.all_probabilities || {}
        });

        // Emit custom event for UI updates
        document.dispatchEvent(new CustomEvent('dentalAnalysisComplete', {
            detail: result
        }));
    }

    showLoading(show) {
        document.dispatchEvent(new CustomEvent('loadingStateChange', {
            detail: { isLoading: show }
        }));
    }

    showError(message) {
        document.dispatchEvent(new CustomEvent('dentalAnalysisError', {
            detail: { message }
        }));
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DentalAIClient, DentalAIApp };
}

// Example usage in browser
if (typeof window !== 'undefined') {
    window.DentalAIClient = DentalAIClient;
    window.DentalAIApp = DentalAIApp;
}