# Deploy Flask ke cPanel (CloudLinux Python App)

Panduan ini dibuat berdasarkan proses deploy aplikasi **Flask + SQLAlchemy + Random Forest** ke hosting cPanel menggunakan **Python App (CloudLinux)**.

---

# 1. Persiapan

## Struktur Project

Pastikan struktur project seperti berikut:

```text
diagnosis_kerusakan_dinamo/
│
├── app/
├── dataset/
├── tests/
├── .env
├── config.py
├── requirements.txt
├── run.py
└── README.md
```

---

# 2. Membuat Subdomain

Masuk ke cPanel

```
Domains
```

Buat subdomain

```
dinamo.fauzantaslim.biz.id
```

Setelah dibuat biasanya folder otomatis menjadi

```text
/home/username/dinamo.fauzantaslim.biz.id/
```

---

# 3. Membuat Database

Masuk

```
MySQL Database Wizard
```

Contoh

Database

```
fauzanta_dinamo_db
```

User

```
fauzanta_dinamo_user
```

Password

```
********
```

Berikan

```
ALL PRIVILEGES
```

---

# 4. Import Database

Masuk

```
phpMyAdmin
```

Import

```
database.sql
```

---

## Jika muncul error

```
#1273 Unknown collation utf8mb4_0900_ai_ci
```

Artinya database berasal dari MySQL 8 sedangkan hosting memakai MariaDB/MySQL lama.

Solusi:

Cari

```sql
utf8mb4_0900_ai_ci
```

Ganti menjadi

```sql
utf8mb4_unicode_ci
```

Lalu import kembali.

---

# 5. Upload Source Code

Upload seluruh project ke

```text
/home/username/dinamo.fauzantaslim.biz.id/
```

Pastikan **bukan**

```text
/home/username/public_html/
```

---

# 6. Konfigurasi Python App

Masuk

```
Setup Python App
```

Isi

Python Version

```
3.10
```

Application Root

```
dinamo.fauzantaslim.biz.id
```

Application URL

```
dinamo.fauzantaslim.biz.id
```

Startup File

```
run.py
```

Entry Point

```
app
```

---

# 7. Install Dependency

Masuk Terminal

Aktifkan Virtual Environment

```bash
source /home/fauzanta/virtualenv/dinamo.fauzantaslim.biz.id/3.10/bin/activate
cd /home/fauzanta/dinamo.fauzantaslim.biz.id
```

Install dependency

```bash
pip install -r requirements.txt
```

Jika berhasil akan muncul

```
Successfully installed ...
```

---

# 8. Konfigurasi Environment

File

```
.env
```

Contoh

```env
SECRET_KEY=random_secret_key

DB_HOST=localhost
DB_PORT=3306
DB_USER=fauzanta_dinamo_user
DB_PASSWORD=********
DB_NAME=fauzanta_dinamo_db
```

---

# 9. Config Flask

config.py

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:

    SECRET_KEY = os.environ.get("SECRET_KEY")

    DB_USER = os.environ.get("DB_USER")
    DB_PASSWORD = os.environ.get("DB_PASSWORD")
    DB_HOST = os.environ.get("DB_HOST")
    DB_PORT = os.environ.get("DB_PORT")
    DB_NAME = os.environ.get("DB_NAME")

    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False
```

---

# 10. Jangan Gunakan run.py Sebagai Startup File

CloudLinux akan membuat file default

```python
def app(environ,start_response):
    ...
```

yang menghasilkan tampilan

```
It works!

Python v3.10.20
```

Artinya aplikasi Flask belum dijalankan.

---

# 11. Jika Website Menampilkan

```
It works!

Python v3.10.20
```

Hal tersebut menandakan bahwa file `run.py` telah **di-override** oleh file bawaan CloudLinux saat pembuatan Python App.

Solusinya adalah mengembalikan isi file `run.py` menjadi seperti semula:

```python
from app import create_app
from app.models import db

app = create_app()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
```

Setelah itu simpan perubahan, kemudian lakukan **Restart** atau **Save** pada Python App dan akses kembali website.