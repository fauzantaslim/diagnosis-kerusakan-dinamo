from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import current_user
from app.services import auth_service

auth_bp = Blueprint('auth_bp', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard_bp.index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        success, user, message = auth_service.authenticate_user(username, password)
        
        if success:
            return redirect(url_for('dashboard_bp.index'))
        else:
            flash(message, 'error')
            
    return render_template('pages/auth/login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard_bp.index'))
        
    if request.method == 'POST':
        nama_lengkap = request.form.get('nama_lengkap')
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        success, message = auth_service.register_new_user(nama_lengkap, username, password, confirm_password)
        
        if success:
            flash(message, 'success')
            return redirect(url_for('dashboard_bp.index'))
        else:
            flash(message, 'error')
            
    return render_template('pages/auth/register.html')

@auth_bp.route('/logout')
def logout():
    auth_service.logout_current_user()
    return redirect(url_for('auth_bp.login'))
