from flask import Blueprint, request, render_template, flash, redirect, url_for, current_app
import os
from werkzeug.utils import secure_filename
from .utils import allowed_file, get_file_type

upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/')
def index():
    return render_template('upload.html')

@upload_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('upload.index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('upload.index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        file_type = get_file_type(filename)
        flash(f'{file_type.title()} file "{filename}" uploaded successfully!')
        return redirect(url_for('upload.index'))
    else:
        flash('Invalid file type. Please upload data files, PDFs, or images.')
        return redirect(url_for('upload.index'))