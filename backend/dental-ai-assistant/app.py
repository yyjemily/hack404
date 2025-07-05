from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import uuid
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b0
import google.generativeai as genai
import warnings 
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

load_dotenv()
# Gemini API Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Set your API key as environment variable
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-pro')
    gemini_vision_model = genai.GenerativeModel('gemini-1.5-pro')
else:
    print("⚠ Warning: GEMINI_API_KEY not found. Using fallback responses.")
    gemini_model = None
    gemini_vision_model = None

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global storage (replace with database in production)
chat_sessions = {}
xray_analyses = {}

class DentalAIAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.load_models()
        
    def load_models(self):
        """Load pre-trained dental AI models"""
        try:
            # Load TensorFlow model (if available)
            # self.tf_model = load_model('models/dental_classifier.h5')
            
            # Load PyTorch model (if available)
            # self.torch_model = torch.load('models/dental_detector.pth', map_location=self.device)
            
            # For now, we'll use a pre-trained ResNet as base and fine-tune logic
            self.classification_model = resnet50(pretrained=True)
            self.classification_model.eval()
            
            # Define image transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            print("✓ AI models loaded successfully")
            
        except Exception as e:
            print(f"⚠ Warning: Could not load AI models: {e}")
            print("Using rule-based analysis as fallback")
    
    def preprocess_image(self, image_path):
        """Preprocess dental X-ray image"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply dental-specific preprocessing
            image_array = np.array(image)
            
            # Enhance contrast for better analysis
            image_array = cv2.equalizeHist(cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY))
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            
            # Convert back to PIL
            processed_image = Image.fromarray(image_array)
            
            return processed_image
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return Image.open(image_path).convert('RGB')
    
    def analyze_dental_conditions(self, image_path):
        """Analyze dental X-ray using Gemini Vision API"""
        try:
            if gemini_vision_model and GEMINI_API_KEY:
                return self.analyze_with_gemini(image_path)
            else:
                return self.fallback_analysis(image_path)
                
        except Exception as e:
            print(f"Error in dental analysis: {e}")
            return self.fallback_analysis(image_path)
    
    def analyze_with_gemini(self, image_path):
        """Use Gemini Vision API for dental X-ray analysis"""
        try:
            # Load and prepare image
            image = Image.open(image_path)
            
            # Create detailed prompt for dental analysis
            prompt = """You are an expert dental radiologist AI. Analyze this dental X-ray image and provide a comprehensive assessment.

Please provide your analysis in the following structured format:

FINDINGS:
- List specific dental conditions you can identify
- Include confidence levels (High/Medium/Low)
- Specify anatomical locations
- Note any abnormalities or pathology

RECOMMENDATIONS:
- Treatment suggestions based on findings
- Urgency level (Immediate/Urgent/Routine/Monitor)
- Follow-up requirements

TECHNICAL ASSESSMENT:
- Image quality evaluation
- Visibility of structures
- Any limitations in assessment

Focus on common dental conditions like:
- Caries (tooth decay)
- Periodontal disease
- Root canal treatments
- Dental restorations (crowns, fillings)
- Bone loss or abnormalities
- Impacted teeth
- Fractures or trauma

Be specific about tooth numbering when possible and provide clinical recommendations appropriate for dental professionals."""

            # Generate analysis using Gemini
            response = gemini_vision_model.generate_content([prompt, image])
            
            # Parse Gemini response into structured format
            analysis_text = response.text
            
            # Convert to structured findings format
            findings = self.parse_gemini_response(analysis_text)
            
            return findings
            
        except Exception as e:
            print(f"Error with Gemini analysis: {e}")
            return self.fallback_analysis(image_path)
    
    def parse_gemini_response(self, analysis_text):
        """Parse Gemini response into structured findings"""
        findings = []
        
        # Extract findings from Gemini response
        # This is a simplified parser - you can make it more sophisticated
        lines = analysis_text.split('\n')
        
        current_finding = {}
        for line in lines:
            line = line.strip()
            
            if 'FINDINGS:' in line.upper():
                continue
            elif line.startswith('- ') and any(condition in line.lower() for condition in ['caries', 'decay', 'root canal', 'crown', 'filling', 'bone', 'perio']):
                # Extract condition from line
                condition_text = line[2:].strip()
                
                # Determine confidence level
                confidence = 0.7  # default
                if 'high confidence' in condition_text.lower():
                    confidence = 0.9
                elif 'medium confidence' in condition_text.lower():
                    confidence = 0.7
                elif 'low confidence' in condition_text.lower():
                    confidence = 0.5
                
                # Determine condition type
                condition_type = "General Finding"
                if 'caries' in condition_text.lower() or 'decay' in condition_text.lower():
                    condition_type = "Dental Caries"
                elif 'root canal' in condition_text.lower():
                    condition_type = "Root Canal Treatment"
                elif 'crown' in condition_text.lower():
                    condition_type = "Dental Crown"
                elif 'filling' in condition_text.lower():
                    condition_type = "Dental Filling"
                elif 'bone' in condition_text.lower():
                    condition_type = "Bone Assessment"
                elif 'perio' in condition_text.lower():
                    condition_type = "Periodontal Condition"
                
                findings.append({
                    'condition': condition_type,
                    'confidence': confidence,
                    'description': condition_text,
                    'location': 'As specified in analysis',
                    'severity': 'See detailed analysis',
                    'full_analysis': analysis_text
                })
        
        # If no specific findings extracted, include the full analysis
        if not findings:
            findings.append({
                'condition': 'Comprehensive Analysis',
                'confidence': 0.8,
                'description': 'Detailed analysis provided by AI',
                'location': 'Full mouth assessment',
                'severity': 'See full analysis',
                'full_analysis': analysis_text
            })
        
        return findings
    
    def fallback_analysis(self, image_path):
        """Fallback analysis when AI models are not available"""
        return [
            {
                'condition': 'General Assessment',
                'confidence': 0.60,
                'description': 'X-ray image processed successfully. Please consult with a dental professional for detailed analysis.',
                'location': 'Full mouth',
                'severity': 'Assessment needed'
            }
        ]
    
    def generate_recommendations(self, findings):
        """Generate treatment recommendations based on findings"""
        recommendations = []
        
        for finding in findings:
            condition = finding['condition']
            severity = finding['severity']
            
            if 'Caries' in condition:
                if severity == 'Severe':
                    recommendations.append("Immediate dental intervention required - possible extraction or root canal")
                elif severity == 'Moderate':
                    recommendations.append("Schedule dental appointment for filling or crown")
                else:
                    recommendations.append("Monitor and maintain good oral hygiene")
            
            elif 'Root Canal' in condition:
                recommendations.append("Follow-up with endodontist for evaluation")
            
            elif 'Crown' in condition:
                recommendations.append("Regular check-ups to monitor crown integrity")
            
            elif 'Bone' in condition:
                recommendations.append("Consider periodontal evaluation")
        
        if not recommendations:
            recommendations.append("Continue regular dental check-ups and maintain oral hygiene")
        
        return recommendations

# Initialize AI analyzer
ai_analyzer = DentalAIAnalyzer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_session_id():
    return request.headers.get('X-Session-ID', 'default')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '').strip()
        session_id = get_session_id()
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Initialize session if not exists
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        # Add user message
        user_msg = {
            'id': str(uuid.uuid4()),
            'type': 'user',
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        chat_sessions[session_id].append(user_msg)
        
        # Generate AI response (enhanced with dental knowledge)
        ai_response = generate_dental_response(message)
        
        # Add AI response
        ai_msg = {
            'id': str(uuid.uuid4()),
            'type': 'assistant',
            'message': ai_response,
            'timestamp': datetime.now().isoformat()
        }
        chat_sessions[session_id].append(ai_msg)
        
        return jsonify({
            'response': ai_response,
            'session_id': session_id,
            'message_id': ai_msg['id']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# MISSING UPLOAD ROUTE - THIS IS THE FIX!
@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        session_id = get_session_id()
        
        # If user does not select file, browser also submits empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Generate unique filename to prevent conflicts
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save file
            file.save(file_path)
            
            # Analyze the uploaded X-ray
            try:
                findings = ai_analyzer.analyze_dental_conditions(file_path)
                recommendations = ai_analyzer.generate_recommendations(findings)
                
                # Store analysis results
                analysis_id = str(uuid.uuid4())
                analysis_data = {
                    'id': analysis_id,
                    'filename': filename,
                    'filepath': file_path,
                    'findings': findings,
                    'recommendations': recommendations,
                    'timestamp': datetime.now().isoformat(),
                    'session_id': session_id
                }
                
                xray_analyses[analysis_id] = analysis_data
                
                return jsonify({
                    'message': 'File uploaded and analyzed successfully',
                    'analysis_id': analysis_id,
                    'filename': filename,
                    'findings': findings,
                    'recommendations': recommendations
                })
                
            except Exception as e:
                # Clean up file if analysis fails
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
        
        return jsonify({'error': 'File type not allowed. Please upload an image file.'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/analysis/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    try:
        if analysis_id not in xray_analyses:
            return jsonify({'error': 'Analysis not found'}), 404
        
        analysis = xray_analyses[analysis_id]
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyses', methods=['GET'])
def get_analyses():
    try:
        session_id = get_session_id()
        
        # Filter analyses by session
        session_analyses = {
            aid: analysis for aid, analysis in xray_analyses.items()
            if analysis.get('session_id') == session_id
        }
        
        return jsonify(session_analyses)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
def generate_dental_response(message):
    try:
        print("[DEBUG] Calling Gemini with message:", message)
        model = genai.GenerativeModel('gemini-1.5-pro')

        response = model.generate_content(message)

        print("[DEBUG] Gemini responded:", response.text)
        return response.text
    except Exception as e:
        print("[ERROR in generate_dental_response]:", repr(e))
        return "Sorry, I encountered an error generating the response."

#def generate_dental_response(message):
 #   """Generate dental response using Gemini AI"""
  #  try:
   #     if gemini_model and GEMINI_API_KEY:
    #        return generate_gemini_response(message)
     #   else:
      #      return generate_fallback_response(message)
    #except Exception as e:
     #   print(f"Error generating response: {e}")
      #  return generate_fallback_response(message)

def generate_gemini_response(message):
    """Use Gemini AI for dental chat responses"""
    try:
        # Create a specialized dental prompt
        dental_prompt = f"""You are an expert dental AI assistant helping dental professionals and students. 
        
Your role is to provide accurate, professional dental information while emphasizing that your responses are educational and should not replace professional dental diagnosis or treatment.

Key guidelines:
- Provide comprehensive, evidence-based dental information
- Use proper dental terminology
- Include relevant clinical considerations
- Suggest when professional consultation is needed
- Focus on preventive care when appropriate
- Be specific about procedures, materials, and techniques

User question: {message}

Please provide a detailed, professional response that would be helpful for dental education and practice."""

        # Generate response using Gemini
        response = gemini_model.generate_content(dental_prompt)
        
        # Add professional disclaimer
        gemini_response = response.text
        
        return f"{gemini_response}\n\n💡 *Note: This information is for educational purposes. Always consult with a qualified dental professional for diagnosis and treatment planning.*"
        
    except Exception as e:
        print(f"Error with Gemini response: {e}")
        return generate_fallback_response(message)

def generate_fallback_response(message):
    """Fallback response when Gemini is not available"""
    message_lower = message.lower()
    
    # Basic dental responses (your original logic)
    if any(word in message_lower for word in ['root canal', 'endodontic']):
        return """Root canal treatment (endodontic therapy) is performed when the tooth's pulp becomes infected or severely damaged. The procedure involves:

1. **Diagnosis**: X-rays and tests to confirm pulp damage
2. **Access**: Creating an opening in the crown  
3. **Cleaning**: Removing infected pulp and cleaning canals
4. **Shaping**: Preparing canals for filling
5. **Sealing**: Filling canals with gutta-percha
6. **Restoration**: Crown placement for protection

**Post-treatment care**: Avoid hard foods, take prescribed antibiotics, and follow up as scheduled. Success rate is typically 85-95%.

💡 *Note: Please consult with a qualified dental professional for proper diagnosis and treatment.*"""
    
    elif any(word in message_lower for word in ['caries', 'cavity', 'tooth decay']):
        return """Dental caries (tooth decay) is a multifactorial disease process. Key information:

**Stages of Caries**:
- **Initial**: White spot lesions (reversible)
- **Moderate**: Enamel cavitation
- **Advanced**: Dentin involvement
- **Severe**: Pulp exposure

**Treatment Options**:
- Fluoride therapy (early stages)
- Composite fillings (moderate)
- Crowns (extensive decay)
- Root canal + crown (pulp involvement)

**Prevention**: Regular brushing, flossing, fluoride use, dietary modifications, and professional cleanings.

💡 *Note: This information is for educational purposes. Always consult with a qualified dental professional.*"""
    
    else:
        return f"""I'm here to help with dental questions! I can provide information about:

- Dental procedures (root canals, extractions, crowns)
- Oral diseases and conditions
- Treatment planning considerations
- Preventive care strategies
- Dental materials and techniques

For more accurate and detailed responses, please ensure the Gemini AI integration is properly configured.

Your question: "{message}"

💡 *Note: This information is for educational purposes. Always consult with a qualified dental professional for diagnosis and treatment planning.*"""

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')