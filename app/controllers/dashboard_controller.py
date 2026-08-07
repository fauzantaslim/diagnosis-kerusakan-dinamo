from flask import render_template, jsonify
from app.utils.jwt_utils import jwt_required, get_current_user_id
from app.services import dashboard_service


@jwt_required
def index():
    """Controller untuk dashboard index."""
    user_id      = get_current_user_id()
    history_data = dashboard_service.get_recent_history(user_id, limit=5)
    recent_history = [item.to_dict() for item in history_data]
    return render_template('pages/app/index.html', recent_history=recent_history)


@jwt_required
def api_history():
    """API endpoint untuk mengambil riwayat terbaru dalam format JSON."""
    user_id      = get_current_user_id()
    history_data = dashboard_service.get_recent_history(user_id, limit=20)
    return jsonify([item.to_dict() for item in history_data])
