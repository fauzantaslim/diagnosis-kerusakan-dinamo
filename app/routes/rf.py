from flask import Blueprint
from app.controllers import rf_controller

rf_bp = Blueprint('rf_bp', __name__)

# GET /api/rf/tree/<session_id>/<tree_id> — React Flow data untuk 1 tree
rf_bp.route(
    '/tree/<string:session_id>/<int:tree_id>',
    methods=['GET'],
)(rf_controller.get_tree)

# GET /api/rf/trees/<session_id> — list semua vote per tree dalam session
rf_bp.route(
    '/trees/<string:session_id>',
    methods=['GET'],
)(rf_controller.get_trees)
