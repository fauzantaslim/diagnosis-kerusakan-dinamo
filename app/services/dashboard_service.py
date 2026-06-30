from app.models.history import History

def get_recent_history(user_id, limit=5):
    return History.query.filter_by(user_id=user_id).order_by(History.tanggal.desc()).limit(limit).all()

def get_dummy_history():
    return [
        {"id": 1, "tanggal": "05 Mei 2026 09:14", "jenis_mesin": "Dinamo A-04\nUnit Press", "diagnosis": "Bearing Aus", "confidence": "96.2%"},
        {"id": 2, "tanggal": "05 Mei 2026 08:52", "jenis_mesin": "Dinamo B-11\nKonveyor", "diagnosis": "Normal", "confidence": "99.1%"},
        {"id": 3, "tanggal": "04 Mei 2026 15:30", "jenis_mesin": "Dinamo C-02\nPompa Air", "diagnosis": "Gulungan Terbakar", "confidence": "91.8%"},
        {"id": 4, "tanggal": "04 Mei 2026 11:05", "jenis_mesin": "Dinamo D-07\nFan Industri", "diagnosis": "Tegangan Tidak Stabil", "confidence": "87.4%"},
        {"id": 5, "tanggal": "03 Mei 2026 14:20", "jenis_mesin": "Dinamo E-01\nKompressor", "diagnosis": "Normal", "confidence": "98.6%"}
    ]
