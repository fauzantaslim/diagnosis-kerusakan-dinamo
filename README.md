# 🔧 Diagnosis Kerusakan Dinamo dengan Random Forest

Aplikasi web berbasis **Flask** untuk mendiagnosis kerusakan dinamo (motor listrik) menggunakan algoritma **Random Forest**. Proyek ini dikembangkan sebagai bagian dari penelitian **Skripsi**.

---

## 📋 Deskripsi

Sistem ini memungkinkan pengguna untuk:

- **Registrasi & Login** — Sistem autentikasi lengkap dengan hashing password
- **Diagnosis Kerusakan** — Mengidentifikasi jenis kerusakan dinamo
- **Riwayat Diagnosis** — Menyimpan dan menampilkan riwayat hasil diagnosis per pengguna

---

## 🏗️ Arsitektur

Proyek ini menggunakan pola arsitektur **Routes → Controllers → Services** untuk memisahkan tanggung jawab antar layer:

| Layer | File | Tanggung Jawab |
|---|---|---|
| **Routes** | `routes.py` | Mendefinisikan endpoint URL dan HTTP method |
| **Controllers** | `controllers.py` | Menangani request/response dan render template |
| **Services** | `services.py` | Business logic (autentikasi, registrasi, query data) |
| **Models** | `models.py` | Definisi tabel database (ORM SQLAlchemy) |

---

## 📁 Struktur Folder

```
diagnosis_kerusakan_dinamo/
│
├── app.py                  # Entry point aplikasi Flask
├── routes.py               # Definisi routes/endpoint (Blueprint)
├── controllers.py          # Controller untuk setiap route
├── services.py             # Business logic & data access
├── models.py               # Model database (User, History)
│
├── .env                    # Environment variables (tidak di-commit)
├── .env.example            # Template environment variables
├── .gitignore              # File yang diabaikan Git
│
├── static/                 # File statis
│   └── images/
│       └── dinamo.png      # Gambar aset dinamo
│
├── templates/              # Template HTML (Jinja2)
│   ├── layouts/            # Layout utama
│   │   ├── base.html       # Base layout (head, meta, CSS/JS global)
│   │   ├── app_layout.html # Layout untuk halaman aplikasi (dengan navbar)
│   │   └── auth_layout.html# Layout untuk halaman autentikasi
│   │
│   ├── pages/              # Halaman-halaman aplikasi
│   │   ├── app/
│   │   │   └── index.html  # Dashboard utama
│   │   └── auth/
│   │       ├── login.html  # Halaman login
│   │       └── register.html # Halaman registrasi
│   │
│   └── components/         # Komponen UI reusable
│       └── ui.html         # Komponen UI bersama (navbar, footer, dll.)
│
└── .venv/                  # Virtual environment Python (tidak di-commit)
```

---

## ⚙️ Tech Stack

| Machine Learning | Random Forest (Scikit-Learn) |
| Backend | Python 3, Flask 3.1 |
| Database | MySQL (via PyMySQL) |
| ORM | Flask-SQLAlchemy (SQLAlchemy 2.0) |
| Autentikasi | Flask-Login |
| Password Hashing | Werkzeug |
| Templating | Jinja2 |
| Environment | python-dotenv |

---

## 🚀 Cara Menjalankan

### Prasyarat

- **Python 3.10+** sudah terinstall
- **MySQL** sudah terinstall dan berjalan
- **Git** (opsional, untuk clone repository)

### 1. Clone Repository

```bash
git clone https://github.com/<username>/diagnosis-kerusakan-dinamo.git
cd diagnosis-kerusakan-dinamo
```

### 2. Buat Virtual Environment

```bash
# Buat virtual environment
python -m venv .venv

# Aktifkan virtual environment
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (CMD)
.\.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependensi

```bash
pip install flask flask-sqlalchemy flask-login pymysql python-dotenv werkzeug scikit-learn pandas
```

### 4. Konfigurasi Environment Variables

Salin file `.env.example` menjadi `.env` dan sesuaikan nilainya:

```bash
cp .env.example .env
```

Isi file `.env`:

```env
SECRET_KEY=ganti_dengan_secret_key_yang_aman
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=password_mysql_anda
DB_NAME=dinamo_db
```

### 5. Buat Database MySQL

Buat database di MySQL sesuai dengan nama yang ada di `.env`:

```sql
CREATE DATABASE dinamo_db;
```

> **Note:** Tabel akan dibuat otomatis oleh SQLAlchemy saat aplikasi pertama kali dijalankan.

### 6. Jalankan Aplikasi

```bash
python app.py
```

Aplikasi akan berjalan di: **http://localhost:5000**

---

## 📝 Catatan Pengembangan

- Proyek ini menggunakan algoritma **Random Forest** untuk klasifikasi jenis kerusakan berdasarkan fitur teknis dinamo.
- Aplikasi menggunakan **Blueprint** Flask untuk modularitas routing.
- Password disimpan dalam bentuk hash menggunakan `werkzeug.security`.
- Saat ini dashboard menggunakan **dummy data** untuk visualisasi riwayat diagnosis.
- Tabel database (`User`, `History`) dibuat otomatis via `db.create_all()` saat startup.

---
