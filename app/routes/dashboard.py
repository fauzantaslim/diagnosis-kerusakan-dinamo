from flask import Blueprint
from flask_login import login_required
from app.controllers import dashboard_controller

dashboard_bp = Blueprint('dashboard_bp', __name__)

dashboard_bp.route('/')(login_required(dashboard_controller.index))
dashboard_bp.route('/api/history')(login_required(dashboard_controller.api_history))
