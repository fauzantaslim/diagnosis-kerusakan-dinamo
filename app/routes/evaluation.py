from flask import Blueprint
from app.controllers import evaluation_controller

evaluation_bp = Blueprint('evaluation_bp', __name__)

# API endpoint untuk menjalankan pengujian
evaluation_bp.route('/run', methods=['GET'])(evaluation_controller.run_evaluation)
evaluation_bp.route('/shap-plot', methods=['GET'])(evaluation_controller.serve_shap_plot)
