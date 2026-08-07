from app.models.user import User
from app.models import db
from app.utils.jwt_utils import generate_token


def authenticate_user(username: str, password: str):
    """
    Validasi kredensial user dan buat JWT jika valid.

    Returns:
        (True, user, token, message)  — sukses
        (False, None, None, message)  — gagal
    """
    user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
        token = generate_token(user)
        return True, user, token, "Login berhasil."

    return False, None, None, "Login gagal. Periksa kembali username dan password Anda."


def register_new_user(nama_lengkap: str, username: str, password: str, confirm_password: str):
    """
    Registrasi user baru dan langsung buat JWT.

    Returns:
        (True, user, token, message)  — sukses
        (False, None, None, message)  — gagal
    """
    if len(password) < 8:
        return False, None, None, "Password minimal 8 karakter."

    if password != confirm_password:
        return False, None, None, "Password dan Konfirmasi Password tidak cocok."

    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return False, None, None, "Username sudah digunakan."

    new_user = User(nama_lengkap=nama_lengkap, username=username)
    new_user.set_password(password)

    db.session.add(new_user)
    db.session.commit()

    token = generate_token(new_user)
    return True, new_user, token, "Registrasi berhasil! Selamat datang di sistem."
