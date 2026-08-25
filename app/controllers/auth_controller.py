from flask import request, jsonify, make_response, url_for
from app.services import auth_service
from app.utils.jwt_utils import jwt_required, get_jwt_payload


def login():
    """POST /auth/login — autentikasi user, return JWT."""
    data     = request.get_json(silent=True) or request.form
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({"success": False, "message": "Username dan password wajib diisi."}), 400

    success, user, token, message = auth_service.authenticate_user(username, password)

    if success:
        resp = make_response(jsonify({
            "success" : True,
            "message" : message,
            "token"   : token,
            "user"    : {
                "id"          : user.id,
                "username"    : user.username,
                "nama_lengkap": user.nama_lengkap,
                "role"        : user.role,
            },
        }), 200)
        resp.set_cookie(
            "access_token", token,
            httponly=True, samesite="Lax",
            max_age=3600,
        )
        return resp

    return jsonify({"success": False, "message": message}), 401


def register():
    """POST /auth/register — registrasi user baru, return JWT."""
    data             = request.get_json(silent=True) or request.form
    nama_lengkap     = data.get('nama_lengkap', '').strip()
    username         = data.get('username', '').strip()
    password         = data.get('password', '')
    confirm_password = data.get('confirm_password', '')

    if not all([nama_lengkap, username, password, confirm_password]):
        return jsonify({"success": False, "message": "Semua field wajib diisi."}), 400

    success, user, token, message = auth_service.register_new_user(
        nama_lengkap, username, password, confirm_password
    )

    if success:
        resp = make_response(jsonify({
            "success" : True,
            "message" : message,
            "token"   : token,
        }), 201)
        resp.set_cookie(
            "access_token", token,
            httponly=True, samesite="Lax",
            max_age=3600,
        )
        return resp

    return jsonify({"success": False, "message": message}), 422


def logout():
    """Controller untuk logout — hapus cookie access_token."""
    resp = make_response(jsonify({
        "success" : True,
        "message" : "Logout berhasil.",
        "redirect": url_for('auth_bp.login'),
    }), 200)
    resp.delete_cookie("access_token")
    return resp


@jwt_required
def me():
    """Controller untuk me — info user dari JWT payload."""
    payload = get_jwt_payload()
    return jsonify({
        "success": True,
        "user"   : {
            "id"          : payload.get("sub"),
            "username"    : payload.get("username"),
            "nama_lengkap": payload.get("nama_lengkap"),
            "role"        : payload.get("role", "user"),
        },
    }), 200
