from flask import Blueprint

logins_bp = Blueprint('logins', __name__, url_prefix='/auth')

from . import routes  # Import routes to register them with the blueprint
