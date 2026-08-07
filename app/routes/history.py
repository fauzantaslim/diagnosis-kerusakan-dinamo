from flask import Blueprint
from app.controllers import history_controller

history_bp = Blueprint('history_bp', __name__)

# JWT decorator ada di controller (@jwt_required)
history_bp.route('/',                           methods=['GET']   )(history_controller.history_page)
history_bp.route('/api',                        methods=['GET']   )(history_controller.get_history)
history_bp.route('/api/<int:history_id>',       methods=['GET']   )(history_controller.get_history_detail)
history_bp.route('/api/<int:history_id>',       methods=['DELETE'])(history_controller.delete_history)
