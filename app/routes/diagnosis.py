from flask import Blueprint
from app.controllers import diagnosis_controller

diagnosis_bp = Blueprint('diagnosis_bp', __name__)

# Halaman form — publik (tidak butuh JWT)
diagnosis_bp.route('/',             methods=['GET'], endpoint='identify_page'    )(diagnosis_controller.identify_page)
diagnosis_bp.route('/identify-page',methods=['GET'], endpoint='identify_page_alt')(diagnosis_controller.identify_page)

# API prediksi — @jwt_required ada di controller
diagnosis_bp.route('/identify', methods=['POST'])(diagnosis_controller.identify)
