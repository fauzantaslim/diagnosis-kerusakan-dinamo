from app.models import db
from datetime import datetime

class History(db.Model):
    __tablename__ = 'histories'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    tanggal = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Spesifikasi Mesin
    jenis_mesin = db.Column(db.String(100), nullable=True)
    daya = db.Column(db.Float, nullable=True)
    jumlah_pole = db.Column(db.Integer, nullable=True)
    
    # Gejala (Symptoms)
    suara_bising_abnormal = db.Column(db.Boolean, nullable=True)
    getaran_berlebih = db.Column(db.Boolean, nullable=True)
    ampere_stabil = db.Column(db.Boolean, nullable=True)
    tegangan_stabil = db.Column(db.Boolean, nullable=True)
    sulit_start = db.Column(db.Boolean, nullable=True)
    bau_hangus = db.Column(db.Boolean, nullable=True)
    warna_gulungan_berubah = db.Column(db.Boolean, nullable=True)
    kipas_pendingin_rusak = db.Column(db.Boolean, nullable=True)
    terminal_terbakar = db.Column(db.Boolean, nullable=True)
    bearing_aus_pecah = db.Column(db.Boolean, nullable=True)
    housing_bearing_aus = db.Column(db.Boolean, nullable=True)
    kebocoran_pelumas = db.Column(db.Boolean, nullable=True)
    keretakan_dudukan = db.Column(db.Boolean, nullable=True)
    lubang_spi_aus = db.Column(db.Boolean, nullable=True)
    resistansi_isolasi_normal = db.Column(db.Boolean, nullable=True)
    
    # Pengukuran
    kecepatan_putaran_rpm = db.Column(db.Float, nullable=True)
    
    # Hasil Prediksi (Label)
    diagnosis = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)

    SYMPTOM_LABEL_MAP = {
        "suara_bising_abnormal"   : "Suara Bising Abnormal",
        "getaran_berlebih"        : "Getaran Berlebih",
        "ampere_stabil"           : "Ampere Stabil",
        "tegangan_stabil"         : "Tegangan Stabil",
        "sulit_start"             : "Sulit Start",
        "bau_hangus"              : "Bau Hangus",
        "warna_gulungan_berubah"  : "Warna Gulungan Berubah",
        "kipas_pendingin_rusak"   : "Kipas Pendingin Rusak",
        "terminal_terbakar"       : "Terminal Terbakar",
        "bearing_aus_pecah"       : "Bearing Aus/Pecah",
        "housing_bearing_aus"     : "Housing Bearing Aus",
        "kebocoran_pelumas"       : "Kebocoran Pelumas",
        "keretakan_dudukan"       : "Keretakan Dudukan",
        "lubang_spi_aus"          : "Lubang Spi (Keyway) Aus",
        "resistansi_isolasi_normal": "Resistansi Isolasi Normal",
    }

    def active_symptoms(self) -> list:
        """Mengembalikan list nama gejala yang aktif (True)."""
        return [
            label
            for field, label in self.SYMPTOM_LABEL_MAP.items()
            if getattr(self, field, False)
        ]

    def to_dict(self) -> dict:
        """Serialisasi objek History ke dictionary (untuk JSON response)."""
        return {
            "id"          : self.id,
            "user_id"     : self.user_id,
            "tanggal"     : self.tanggal.strftime("%d %b %Y %H:%M") if self.tanggal else None,
            "tanggal_iso" : self.tanggal.isoformat() if self.tanggal else None,
            # Spesifikasi
            "jenis_mesin" : self.jenis_mesin,
            "daya"  : self.daya,
            "jumlah_pole" : self.jumlah_pole,
            # Pengukuran
            "kecepatan_putaran_rpm"          : self.kecepatan_putaran_rpm,
            # Gejala aktif
            "gejala_aktif": self.active_symptoms(),
            # Hasil
            "diagnosis"     : self.diagnosis,
            "confidence"    : round(self.confidence, 4),
            "confidence_pct": f"{self.confidence * 100:.1f}%",
        }
