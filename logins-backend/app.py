from flask import Flask
from backend import db, jwt

def create_app():
    app = Flask(__name__)

    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dental_app.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['JWT_SECRET_KEY'] = 'your-secret-key'

    db.init_app(app)
    jwt.init_app(app)

    from backend.logins_backend.dentists import dentists_bp
    app.register_blueprint(dentists_bp)

    from backend.logins_backend.patients import patients_bp
    app.register_blueprint(patients_bp)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
