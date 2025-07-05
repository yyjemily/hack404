from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token
from backend import db
from .models import Dentist

dentists_bp = Blueprint('dentists', __name__, url_prefix='/auth')

@dentists_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    name = data.get('name')

    if Dentist.query.filter_by(email=email).first():
        return jsonify({'msg': 'User already exists'}), 400

    new_dentist = Dentist(email=email, name=name)
    new_dentist.set_password(password)
    db.session.add(new_dentist)
    db.session.commit()
    return jsonify({'msg': 'User created successfully'}), 201

@dentists_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = Dentist.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({'msg': 'Bad email or password'}), 401

    access_token = create_access_token(identity=user.id)
    return jsonify({'access_token': access_token})
