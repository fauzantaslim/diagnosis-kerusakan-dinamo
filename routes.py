from flask import Blueprint
from flask_login import login_required
import controllers

main_bp = Blueprint('main_bp', __name__)

@main_bp.route('/')
@login_required
def index():
    return controllers.index_controller()

@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    return controllers.login_controller()

@main_bp.route('/register', methods=['GET', 'POST'])
def register():
    return controllers.register_controller()

@main_bp.route('/logout')
@login_required
def logout():
    return controllers.logout_controller()
