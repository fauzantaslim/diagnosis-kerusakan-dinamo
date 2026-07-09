from app.models.user import User
from app.models import db
from flask_login import login_user, logout_user

def authenticate_user(username, password):
    user = User.query.filter_by(username=username).first()
    
    if user and user.check_password(password):
        login_user(user)
        return True, user, "Login berhasil."
    
    return False, None, "Login gagal. Periksa kembali username dan password Anda."

def register_new_user(nama_lengkap, username, password, confirm_password):
    if len(password) < 8:
        return False, "Password minimal 8 karakter."

    if password != confirm_password:
        return False, "Password dan Konfirmasi Password tidak cocok."
        
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return False, "Username sudah digunakan."
        
    new_user = User(nama_lengkap=nama_lengkap, username=username)
    new_user.set_password(password)
    
    db.session.add(new_user)
    db.session.commit()
    
    login_user(new_user)
    
    return True, "Registrasi berhasil! Selamat datang di sistem."

def logout_current_user():
    logout_user()
