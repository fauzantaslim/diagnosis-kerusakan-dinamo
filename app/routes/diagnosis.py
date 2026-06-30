from flask import Blueprint
from flask_login import login_required
from app.controllers import diagnosis_controller

diagnosis_bp = Blueprint('diagnosis_bp', __name__)

diagnosis_bp.route('/', methods=['GET'], endpoint='identify_page')(login_required(diagnosis_controller.identify_page))
diagnosis_bp.route('/identify-page', methods=['GET'], endpoint='identify_page_alt')(login_required(diagnosis_controller.identify_page))

diagnosis_bp.route('/identify', methods=['POST'])(login_required(diagnosis_controller.identify))
