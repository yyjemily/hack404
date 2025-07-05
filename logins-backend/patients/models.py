from datetime import datetime
from backend import db
from backend.logins_backend.dentists.models import Dentist

class Patient(db.Model):
    __tablename__ = 'patients'

    id = db.Column(db.Integer, primary_key=True)
    dentist_id = db.Column(db.Integer, db.ForeignKey('dentists.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    dob = db.Column(db.Date)
    phone = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    dentist = db.relationship('Dentist', backref='patients')
    diagnostics = db.relationship('Diagnostic', backref='patient')

class Diagnostic(db.Model):
    __tablename__ = 'diagnostics'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    diagnostic_date = db.Column(db.DateTime, default=datetime.utcnow)
    diagnosis_text = db.Column(db.Text)
    image_url = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
