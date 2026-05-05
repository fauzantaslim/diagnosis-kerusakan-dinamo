from models import db, User, History
from flask_login import login_user, logout_user

def authenticate_user(username, password):
    """
    Authenticates a user based on username and password.
    Returns (success_boolean, user_object_or_none, message)
    """
    user = User.query.filter_by(username=username).first()
    
    if user and user.check_password(password):
        login_user(user)
        return True, user, "Login berhasil."
    
    return False, None, "Login gagal. Periksa kembali username dan password Anda."

def register_new_user(nama_lengkap, username, password, confirm_password):
    """
    Registers a new user.
    Returns (success_boolean, message)
    """
    if password != confirm_password:
        return False, "Password dan Konfirmasi Password tidak cocok."
        
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return False, "Username sudah digunakan."
        
    new_user = User(nama_lengkap=nama_lengkap, username=username)
    new_user.set_password(password)
    
    db.session.add(new_user)
    db.session.commit()
    
    # Auto-login after registration
    login_user(new_user)
    
    return True, "Registrasi berhasil! Selamat datang di sistem."

def logout_current_user():
    """
    Logs out the current user.
    """
    logout_user()
    
def get_recent_history(user_id, limit=5):
    """
    Retrieves recent diagnostic history for a given user.
    Returns a list of History objects.
    """
    return History.query.filter_by(user_id=user_id).order_by(History.tanggal.desc()).limit(limit).all()

def get_dummy_history():
    """
    Returns dummy data for UI visualization.
    """
    return [
        {"id": 1, "tanggal": "05 Mei 2026 09:14", "dinamo": "Dinamo A-04\nUnit Press", "diagnosis": "Bearing Aus", "confidence": "96.2%"},
        {"id": 2, "tanggal": "05 Mei 2026 08:52", "dinamo": "Dinamo B-11\nKonveyor", "diagnosis": "Normal", "confidence": "99.1%"},
        {"id": 3, "tanggal": "04 Mei 2026 15:30", "dinamo": "Dinamo C-02\nPompa Air", "diagnosis": "Gulungan Terbakar", "confidence": "91.8%"},
        {"id": 4, "tanggal": "04 Mei 2026 11:05", "dinamo": "Dinamo D-07\nFan Industri", "diagnosis": "Tegangan Tidak Stabil", "confidence": "87.4%"},
        {"id": 5, "tanggal": "03 Mei 2026 14:20", "dinamo": "Dinamo E-01\nKompressor", "diagnosis": "Normal", "confidence": "98.6%"}
    ]
