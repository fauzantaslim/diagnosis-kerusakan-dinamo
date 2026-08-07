"""
JWT utility — generate & validate token, decorator jwt_required.

Token:
  - Payload: sub (user_id), username, nama_lengkap, iat, exp
  - exp: 1 jam dari waktu issue
  - Algorithm: HS256
  - Secret: dari app.config['SECRET_KEY']

Cara pakai di controller:
  from app.utils.jwt_utils import jwt_required, get_current_user

  @jwt_required
  def my_view():
      user = get_current_user()  # User object
      ...

Token diterima dari:
  1. Header: Authorization: Bearer <token>
  2. Cookie: access_token=<token>  (untuk halaman HTML)
"""

import jwt
from datetime import datetime, timezone, timedelta
from functools import wraps
from flask import request, jsonify, g, current_app, redirect, url_for


# ================================================================ #
#  Generate                                                        #
# ================================================================ #

def generate_token(user) -> str:
    """
    Buat JWT untuk user. Expired 1 jam.

    Args:
        user: User SQLAlchemy object (harus punya .id, .username, .nama_lengkap)

    Returns:
        JWT string
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub"         : str(user.id),   # PyJWT v2 requires sub as string
        "user_id"     : user.id,         # int copy untuk kemudahan
        "username"    : user.username,
        "nama_lengkap": user.nama_lengkap,
        "iat"         : now,
        "exp"         : now + timedelta(hours=1),
    }
    return jwt.encode(payload, current_app.config["SECRET_KEY"], algorithm="HS256")


# ================================================================ #
#  Decode                                                          #
# ================================================================ #

def decode_token(token: str) -> dict:
    """
    Decode & validasi JWT.

    Returns:
        payload dict jika valid

    Raises:
        jwt.ExpiredSignatureError  — token expired
        jwt.InvalidTokenError      — token tidak valid
    """
    return jwt.decode(
        token,
        current_app.config["SECRET_KEY"],
        algorithms=["HS256"],
        options={"verify_sub": False},  # sub bisa int atau str
    )


def _extract_token() -> str | None:
    """Ambil raw token dari Authorization header atau cookie."""
    # 1. Authorization: Bearer <token>
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()

    # 2. Cookie access_token (untuk SSR pages)
    return request.cookies.get("access_token")


# ================================================================ #
#  Decorator                                                       #
# ================================================================ #

def jwt_required(f):
    """
    Decorator: pastikan request memiliki JWT yang valid.
    Set g.jwt_payload dan g.current_user_id untuk dipakai di controller.
    Jika request Accept: text/html → redirect ke login page.
    Jika Accept: application/json → return 401 JSON.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token = _extract_token()

        if not token:
            return _unauthorized("Token tidak ditemukan. Silakan login terlebih dahulu.")

        try:
            payload = decode_token(token)
        except jwt.ExpiredSignatureError:
            return _unauthorized("Token expired. Silakan login ulang.")
        except jwt.InvalidTokenError:
            return _unauthorized("Token tidak valid.")

        # Simpan payload ke flask.g
        g.jwt_payload     = payload
        g.current_user_id = payload.get("user_id") or int(payload["sub"])

        return f(*args, **kwargs)

    return decorated


def _unauthorized(message: str):
    """Kembalikan 401 atau redirect berdasarkan Accept header."""
    accept = request.headers.get("Accept", "")
    wants_html = "text/html" in accept and "application/json" not in accept

    if wants_html:
        return redirect(url_for("auth_bp.login"))

    return jsonify({"success": False, "message": message}), 401


# ================================================================ #
#  Helper untuk controller                                         #
# ================================================================ #

def get_current_user():
    """
    Ambil User object dari DB berdasarkan user_id di JWT payload.
    Harus dipanggil di dalam fungsi yang sudah di-wrap @jwt_required.

    Returns:
        User object atau None
    """
    from app.models.user import User
    user_id = getattr(g, "current_user_id", None)
    if user_id is None:
        return None
    return User.query.get(int(user_id))


def get_current_user_id() -> int | None:
    """Ambil user_id dari JWT payload tanpa query DB."""
    return getattr(g, "current_user_id", None)


def get_jwt_payload() -> dict:
    """Ambil full payload JWT (tanpa query DB)."""
    return getattr(g, "jwt_payload", {})
