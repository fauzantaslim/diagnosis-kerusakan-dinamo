from flask import Blueprint
from flask_login import login_required
from app.controllers import auth_controller

auth_bp = Blueprint('auth_bp', __name__)

auth_bp.route('/login', methods=['GET', 'POST'])(auth_controller.login)
auth_bp.route('/register', methods=['GET', 'POST'])(auth_controller.register)
auth_bp.route('/logout', methods=['GET', 'POST'])(login_required(auth_controller.logout))
auth_bp.route('/me', methods=['GET'])(login_required(auth_controller.me))
