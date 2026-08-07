from flask import Blueprint
from app.controllers import dashboard_controller

dashboard_bp = Blueprint('dashboard_bp', __name__)

# JWT decorator ada di controller (@jwt_required)
dashboard_bp.route('/')(dashboard_controller.index)
dashboard_bp.route('/api/history')(dashboard_controller.api_history)
