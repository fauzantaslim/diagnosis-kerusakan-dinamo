"""
Seeder: buat user admin default jika belum ada.

Usage:
    .\.venv\Scripts\python.exe seed_admin.py
"""
from app import create_app, db
from app.models.user import User

app = create_app()

with app.app_context():
    existing = User.query.filter_by(username='admin123').first()
    if existing:
        print(f'[skip] User "admin123" sudah ada (role={existing.role}).')
    else:
        admin = User(
            nama_lengkap='Administrator',
            username='admin123',
            role='admin',
        )
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print(f'[ok] User admin berhasil dibuat: username=admin123, role=admin')
