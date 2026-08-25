from app.models import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    nama_lengkap  = db.Column(db.String(150), nullable=False)
    username      = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role          = db.Column(db.String(20), nullable=False, default='user')  # 'admin' | 'user'

    # ------------------------------------------------------------------ #
    #  Auth helpers                                                        #
    # ------------------------------------------------------------------ #

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def is_admin(self) -> bool:
        """Return True jika user memiliki role admin."""
        return self.role == 'admin'

    def to_dict(self) -> dict:
        return {
            'id'          : self.id,
            'nama_lengkap': self.nama_lengkap,
            'username'    : self.username,
            'role'        : self.role,
        }

