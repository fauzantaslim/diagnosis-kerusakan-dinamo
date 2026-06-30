from flask import render_template, jsonify
from flask_login import current_user
from app.services import dashboard_service

def index():
    """Controller untuk dashboard index."""
    # Ambil history asli dari database
    history_data = dashboard_service.get_recent_history(current_user.id, limit=5)
    recent_history = [item.to_dict() for item in history_data]
    
    return render_template('pages/app/index.html', recent_history=recent_history)

def api_history():
    """API endpoint untuk mengambil riwayat terbaru dalam format JSON"""
    history_data = dashboard_service.get_recent_history(current_user.id, limit=20)
    return jsonify([item.to_dict() for item in history_data])
