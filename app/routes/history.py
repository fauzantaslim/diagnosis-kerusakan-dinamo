from flask import Blueprint
from flask_login import login_required
from app.controllers import history_controller

history_bp = Blueprint('history_bp', __name__)

history_bp.route('/', methods=['GET'])(login_required(history_controller.history_page))
history_bp.route('/api', methods=['GET'])(login_required(history_controller.get_history))
history_bp.route('/api/<int:history_id>', methods=['GET'])(login_required(history_controller.get_history_detail))
history_bp.route('/api/<int:history_id>', methods=['DELETE'])(login_required(history_controller.delete_history))
