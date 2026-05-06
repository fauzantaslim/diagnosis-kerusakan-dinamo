from flask import render_template, request, redirect, url_for, flash
from flask_login import current_user
import services

def index_controller():
    # If using actual DB history:
    # history_data = services.get_recent_history(current_user.id)
    
    dummy_history = services.get_dummy_history()
    return render_template('pages/app/index.html', dummy_history=dummy_history)

def login_controller():
    if current_user.is_authenticated:
        return redirect(url_for('main_bp.index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        success, user, message = services.authenticate_user(username, password)
        
        if success:
            return redirect(url_for('main_bp.index'))
        else:
            flash(message, 'error')
            
    return render_template('pages/auth/login.html')

def register_controller():
    if current_user.is_authenticated:
        return redirect(url_for('main_bp.index'))
        
    if request.method == 'POST':
        nama_lengkap = request.form.get('nama_lengkap')
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        success, message = services.register_new_user(nama_lengkap, username, password, confirm_password)
        
        if success:
            flash(message, 'success')
            return redirect(url_for('main_bp.index'))
        else:
            flash(message, 'error')
            
    return render_template('pages/auth/register.html')

def logout_controller():
    services.logout_current_user()
    return redirect(url_for('main_bp.login'))
