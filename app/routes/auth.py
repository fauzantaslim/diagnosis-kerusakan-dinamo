from flask import Blueprint
from app.controllers import auth_controller

auth_bp = Blueprint('auth_bp', __name__)

# Hanya POST — tidak ada render halaman dari route ini
auth_bp.route('/login',    methods=['POST'])(auth_controller.login)
auth_bp.route('/register', methods=['POST'])(auth_controller.register)

# Logout tidak butuh JWT
auth_bp.route('/logout', methods=['POST'])(auth_controller.logout)

# Me: butuh JWT (decorator ada di controller)
auth_bp.route('/me', methods=['GET'])(auth_controller.me)
