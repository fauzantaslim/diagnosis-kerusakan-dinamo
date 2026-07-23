from app.models import db
from datetime import datetime

class History(db.Model):
    __tablename__ = 'histories'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    tanggal = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Spesifikasi
    daya = db.Column(db.Float, nullable=True)
    jumlah_pole = db.Column(db.Integer, nullable=True)
    kecepatan_putaran_rpm = db.Column(db.Float, nullable=True)
    # Gejala (Symptoms)
    suara_bising_abnormal = db.Column(db.Boolean, nullable=True)
    bau_hangus = db.Column(db.Boolean, nullable=True)
    indikasi_overheating = db.Column(db.Boolean, nullable=True)
    putaran_poros_seret = db.Column(db.Boolean, nullable=True)
    getaran_berlebih = db.Column(db.Boolean, nullable=True)
    terminal_overheating = db.Column(db.Boolean, nullable=True)
    kipas_pendingin_rusak = db.Column(db.Boolean, nullable=True)
    cooling_duct_tersumbat = db.Column(db.Boolean, nullable=True)
    resistansi_isolasi_tidak_seimbang = db.Column(db.Boolean, nullable=True)
    resistansi_winding_tidak_seimbang = db.Column(db.Boolean, nullable=True)
    arus_antar_fasa_tidak_seimbang = db.Column(db.Boolean, nullable=True)
    
    # Hasil Prediksi (Label)
    diagnosis = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)

    SYMPTOM_LABEL_MAP = {
        "suara_bising_abnormal": "Suara Bising Abnormal",
        "bau_hangus": "Bau Hangus",
        "indikasi_overheating": "Indikasi Overheating",
        "putaran_poros_seret": "Putaran Poros Seret",
        "getaran_berlebih": "Getaran Berlebih",
        "terminal_overheating": "Terminal Overheating",
        "kipas_pendingin_rusak": "Kipas Pendingin Rusak",
        "cooling_duct_tersumbat": "Cooling Duct Tersumbat",
        "resistansi_isolasi_tidak_seimbang": "Resistansi Isolasi Tidak Seimbang",
        "resistansi_winding_tidak_seimbang": "Resistansi Winding Tidak Seimbang",
        "arus_antar_fasa_tidak_seimbang": "Arus Antar Fasa Tidak Seimbang",
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
            # Gejala aktif
            "gejala_aktif": self.active_symptoms(),
            # Spesifikasi
            "daya"          : self.daya,
            "jumlah_pole"   : self.jumlah_pole,
            "kecepatan_putaran_rpm": self.kecepatan_putaran_rpm,
            # Hasil
            "diagnosis"     : self.diagnosis,
            "confidence"    : round(self.confidence, 4),
            "confidence_pct": f"{self.confidence * 100:.1f}%",
        }
