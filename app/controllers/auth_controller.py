from flask import request, jsonify, redirect, url_for, render_template
from flask_login import current_user
from app.services import auth_service

def login():
    """Controller untuk login."""
    if request.method == 'GET':
        if current_user.is_authenticated:
            return jsonify({"success": True, "message": "Sudah login.", "redirect": url_for('dashboard_bp.index')})
        return render_template('pages/auth/login.html')

    data = request.get_json(silent=True) or request.form
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({"success": False, "message": "Username dan password wajib diisi."}), 400

    success, user, message = auth_service.authenticate_user(username, password)

    if success:
        return jsonify({
            "success": True,
            "message": message,
            "user": {
                "id": user.id,
                "username": user.username,
                "nama_lengkap": user.nama_lengkap,
            },
            "redirect": url_for('dashboard_bp.index'),
        }), 200
    else:
        return jsonify({"success": False, "message": message}), 401


def register():
    """Controller untuk register."""
    if request.method == 'GET':
        if current_user.is_authenticated:
            return jsonify({"success": True, "message": "Sudah login.", "redirect": url_for('dashboard_bp.index')})
        return render_template('pages/auth/register.html')

    data = request.get_json(silent=True) or request.form
    nama_lengkap    = data.get('nama_lengkap', '').strip()
    username        = data.get('username', '').strip()
    password        = data.get('password', '')
    confirm_password = data.get('confirm_password', '')

    if not all([nama_lengkap, username, password, confirm_password]):
        return jsonify({"success": False, "message": "Semua field wajib diisi."}), 400

    success, message = auth_service.register_new_user(nama_lengkap, username, password, confirm_password)

    if success:
        return jsonify({
            "success": True,
            "message": message,
            "redirect": url_for('dashboard_bp.index'),
        }), 201
    else:
        return jsonify({"success": False, "message": message}), 422


def logout():
    """Controller untuk logout."""
    auth_service.logout_current_user()
    return jsonify({
        "success": True,
        "message": "Logout berhasil.",
        "redirect": url_for('auth_bp.login'),
    }), 200


def me():
    """Controller untuk me."""
    return jsonify({
        "success": True,
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "nama_lengkap": current_user.nama_lengkap,
        }
    }), 200
