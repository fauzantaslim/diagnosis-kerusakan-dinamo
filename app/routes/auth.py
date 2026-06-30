from flask import Blueprint, request, jsonify, redirect, url_for
from flask_login import current_user, login_required
from app.services import auth_service

auth_bp = Blueprint('auth_bp', __name__)


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """
    GET  → Redirect ke halaman login (untuk non-API flow).
    POST → JSON API login.

    Request JSON:
        { "username": "...", "password": "..." }

    Response JSON (sukses):
        { "success": true, "message": "Login berhasil.", "user": { "id": 1, "username": "...", "nama_lengkap": "..." } }

    Response JSON (gagal):
        { "success": false, "message": "Login gagal. ..." }
    """
    if request.method == 'GET':
        # Jika sudah login, arahkan ke dashboard
        if current_user.is_authenticated:
            return jsonify({"success": True, "message": "Sudah login.", "redirect": url_for('dashboard_bp.index')})
        from flask import render_template
        return render_template('pages/auth/login.html')

    # POST → JSON
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


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """
    GET  → Render halaman register.
    POST → JSON API register.

    Request JSON:
        { "nama_lengkap": "...", "username": "...", "password": "...", "confirm_password": "..." }

    Response JSON (sukses):
        { "success": true, "message": "Registrasi berhasil! ..." }

    Response JSON (gagal):
        { "success": false, "message": "..." }
    """
    if request.method == 'GET':
        if current_user.is_authenticated:
            return jsonify({"success": True, "message": "Sudah login.", "redirect": url_for('dashboard_bp.index')})
        from flask import render_template
        return render_template('pages/auth/register.html')

    # POST → JSON
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


@auth_bp.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    """
    Logout user yang sedang aktif.

    Response JSON:
        { "success": true, "message": "Logout berhasil." }
    """
    auth_service.logout_current_user()
    return jsonify({
        "success": True,
        "message": "Logout berhasil.",
        "redirect": url_for('auth_bp.login'),
    }), 200


@auth_bp.route('/me', methods=['GET'])
@login_required
def me():
    """
    Mengambil info user yang sedang login.

    Response JSON:
        { "success": true, "user": { "id": ..., "username": ..., "nama_lengkap": ... } }
    """
    return jsonify({
        "success": True,
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "nama_lengkap": current_user.nama_lengkap,
        }
    }), 200
