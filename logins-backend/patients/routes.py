from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from backend import db
from .models import Patient, Diagnostic

patients_bp = Blueprint('patients', __name__, url_prefix='/patients')

@patients_bp.route('/', methods=['GET'])
@jwt_required()
def get_patients():
    dentist_id = get_jwt_identity()
    patients = Patient.query.filter_by(dentist_id=dentist_id).all()
    patients_list = [{
        'id': p.id,
        'name': p.name,
        'dob': p.dob.isoformat() if p.dob else None,
        'phone': p.phone
    } for p in patients]
    return jsonify(patients_list)

@patients_bp.route('/', methods=['POST'])
@jwt_required()
def add_patient():
    dentist_id = get_jwt_identity()
    data = request.get_json()
    new_patient = Patient(
        dentist_id=dentist_id,
        name=data.get('name'),
        dob=data.get('dob'),
        phone=data.get('phone')
    )
    db.session.add(new_patient)
    db.session.commit()
    return jsonify({'msg': 'Patient added', 'id': new_patient.id}), 201
