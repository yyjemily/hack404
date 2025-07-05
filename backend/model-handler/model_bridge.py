import torch
import tensorflow as tf
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Union
from PIL import Image

class ModelBridge:
    """Universal model bridge for H5 and PTH formats"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.model_type = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """Auto-detect and load model based on file extension"""
        if self.model_path.suffix.lower() == '.pth':
            self._load_pytorch_model()
        elif self.model_path.suffix.lower() == '.h5':
            self._load_tensorflow_model()
        else:
            raise ValueError(f"Unsupported model format: {self.model_path.suffix}")
    
    def _load_pytorch_model(self):
        """Load PyTorch model"""
        try:
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()
            self.model_type = 'pytorch'
            logging.info(f"PyTorch model loaded: {self.model_path}")
        except Exception as e:
            logging.error(f"Failed to load PyTorch model: {e}")
            raise
    
    def _load_tensorflow_model(self):
        """Load TensorFlow model"""
        try:
            self.model = tf.keras.models.load_model(str(self.model_path))
            self.model_type = 'tensorflow'
            logging.info(f"TensorFlow model loaded: {self.model_path}")
        except Exception as e:
            logging.error(f"Failed to load TensorFlow model: {e}")
            raise
    
    def predict(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Universal prediction method"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Convert PIL image to np.array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        processed_image = self._preprocess_image(image)
        
        if self.model_type == 'pytorch':
            return self._predict_pytorch(processed_image)
        elif self.model_type == 'tensorflow':
            return self._predict_tensorflow(processed_image)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input"""
        img = Image.fromarray(image)
        img = img.resize((224, 224))
        image = np.array(img).astype(np.float32) / 255.0

        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)  # [H, W, C] -> [1, H, W, C]

        return image
    
    def _postprocess_output(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Convert raw model output to structured dental findings"""
        class_labels = {
            0: {
                "primary_finding": "Healthy",
                "severity": "mild",
                "recommendations": ["Routine check-up"],
                "urgency": "routine"
            },
            1: {
                "primary_finding": "Cavity",
                "severity": "moderate",
                "recommendations": ["Filling required"],
                "urgency": "routine"
            },
            2: {
                "primary_finding": "Impacted wisdom tooth",
                "severity": "significant",
                "recommendations": ["Surgical extraction"],
                "urgency": "urgent"
            }
        }

        class_idx = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        result = class_labels.get(class_idx, {
            "primary_finding": "Unknown",
            "severity": "moderate",
            "recommendations": ["Refer to dentist"],
            "urgency": "routine"
        })

        result.update({
            "confidence": round(confidence * 100, 2),
            "affected_teeth": ["#14"]  # You can update this later dynamically
        })

        return result

    def _predict_pytorch(self, image: np.ndarray) -> Dict[str, Any]:
        """PyTorch prediction"""
        with torch.no_grad():
            input_tensor = torch.FloatTensor(image).permute(0, 3, 1, 2).to(self.device)  # NHWC → NCHW
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            predictions = probs.cpu().numpy()

        return self._postprocess_output(predictions)
    
    def _predict_tensorflow(self, image: np.ndarray) -> Dict[str, Any]:
        """TensorFlow prediction"""
        predictions = self.model.predict(image, verbose=0)
        return self._postprocess_output(predictions)
    
    def reload_model(self, new_model_path: str):
        """Reload with a different model"""
        self.model_path = Path(new_model_path)
        self.model = None
        self.model_type = None
        self._load_model()
        return f"Model reloaded: {self.model_type} from {self.model_path}"
