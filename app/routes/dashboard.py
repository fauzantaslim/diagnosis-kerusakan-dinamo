from flask import Blueprint, render_template
from flask_login import login_required, current_user
from app.services import dashboard_service

dashboard_bp = Blueprint('dashboard_bp', __name__)

@dashboard_bp.route('/')
@login_required
def index():
    # If using actual DB history:
    # history_data = dashboard_service.get_recent_history(current_user.id)
    
    dummy_history = dashboard_service.get_dummy_history()
    return render_template('pages/app/index.html', dummy_history=dummy_history)
