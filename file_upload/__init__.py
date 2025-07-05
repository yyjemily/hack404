from flask import Flask
import os

def create_app():
    # Tell Flask where to find templates
    app = Flask(__name__, template_folder='../templates')
    app.config['SECRET_KEY'] = 'your-secret-key'
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    
    # Create upload folder
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Register routes
    from .routes import upload_bp
    app.register_blueprint(upload_bp)
    
    return app