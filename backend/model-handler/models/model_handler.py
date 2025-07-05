import torch
import tensorflow as tf
from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Dict, Any
import os
import logging

class ModelHandler(ABC):
    """Abstract base class for model handlers"""
    
    @abstractmethod
    def load_model(self, model_path: str):
        pass
    
    @abstractmethod
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def preprocess(self, input_data: Any) -> np.ndarray:
        pass

class PyTorchModelHandler(ModelHandler):
    """Handler for PyTorch models (.pth files)"""
    
    def __init__(self, model_class=None, device='cpu'):
        self.model = None
        self.model_class = model_class
        self.device = device
        
    def load_model(self, model_path: str):
        """Load PyTorch model from .pth file"""
        try:
            if self.model_class:
                # If model architecture is provided
                self.model = self.model_class()
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                # Load entire model
                self.model = torch.load(model_path, map_location=self.device)
            
            self.model.eval()
            logging.info(f"PyTorch model loaded from {model_path}")
            return True
        except Exception as e:
            logging.error(f"Error loading PyTorch model: {str(e)}")
            return False
    
    def preprocess(self, input_data: Any) -> np.ndarray:
        """Preprocess input for PyTorch model"""
        if isinstance(input_data, np.ndarray):
            return input_data
        # Add your specific preprocessing logic here
        return np.array(input_data)
    
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Make prediction using PyTorch model"""
        try:
            processed_input = self.preprocess(input_data)
            input_tensor = torch.FloatTensor(processed_input).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                
            # Convert to numpy for consistency
            predictions = output.cpu().numpy()
            
            return {
                'predictions': predictions.tolist(),
                'confidence': float(np.max(predictions)),
                'model_type': 'pytorch'
            }
        except Exception as e:
            logging.error(f"PyTorch prediction error: {str(e)}")
            return {'error': str(e)}

class TensorFlowModelHandler(ModelHandler):
    """Handler for TensorFlow models (.h5 files)"""
    
    def __init__(self):
        self.model = None
        
    def load_model(self, model_path: str):
        """Load TensorFlow model from .h5 file"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            logging.info(f"TensorFlow model loaded from {model_path}")
            return True
        except Exception as e:
            logging.error(f"Error loading TensorFlow model: {str(e)}")
            return False
    
    def preprocess(self, input_data: Any) -> np.ndarray:
        """Preprocess input for TensorFlow model"""
        if isinstance(input_data, np.ndarray):
            return input_data
        # Add your specific preprocessing logic here
        return np.array(input_data)
    
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Make prediction using TensorFlow model"""
        try:
            processed_input = self.preprocess(input_data)
            predictions = self.model.predict(processed_input)
            
            return {
                'predictions': predictions.tolist(),
                'confidence': float(np.max(predictions)),
                'model_type': 'tensorflow'
            }
        except Exception as e:
            logging.error(f"TensorFlow prediction error: {str(e)}")
            return {'error': str(e)}

class ModelFactory:
    """Factory class to create appropriate model handlers"""
    
    @staticmethod
    def create_handler(model_path: str, model_class=None) -> ModelHandler:
        """Create appropriate model handler based on file extension"""
        file_extension = os.path.splitext(model_path)[1].lower()
        
        if file_extension == '.pth':
            return PyTorchModelHandler(model_class)
        elif file_extension == '.h5':
            return TensorFlowModelHandler()
        else:
            raise ValueError(f"Unsupported model format: {file_extension}")

class DentalAIPredictor:
    """Main predictor class for dental AI assistant"""
    
    def __init__(self, model_path: str, model_class=None):
        self.model_handler = ModelFactory.create_handler(model_path, model_class)
        self.model_loaded = self.model_handler.load_model(model_path)
        
    def predict_dental_condition(self, image_data: np.ndarray) -> Dict[str, Any]:
        """Predict dental condition from image data"""
        if not self.model_loaded:
            return {'error': 'Model not loaded properly'}
            
        result = self.model_handler.predict(image_data)
        
        # Add dental-specific post-processing
        if 'predictions' in result:
            result['dental_analysis'] = self._interpret_predictions(result['predictions'])
            
        return result
    
    def _interpret_predictions(self, predictions: list) -> Dict[str, Any]:
        """Interpret model predictions for dental context"""
        # Add your dental-specific interpretation logic here
        dental_conditions = ['healthy', 'cavity', 'gingivitis', 'plaque', 'other']
        
        if len(predictions) > 0:
            max_idx = np.argmax(predictions)
            confidence = predictions[max_idx]
            
            return {
                'condition': dental_conditions[max_idx] if max_idx < len(dental_conditions) else 'unknown',
                'confidence_score': float(confidence),
                'all_probabilities': dict(zip(dental_conditions, predictions[:len(dental_conditions)]))
            }
        
        return {'condition': 'unknown', 'confidence_score': 0.0}