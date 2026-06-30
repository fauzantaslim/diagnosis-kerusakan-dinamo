from app.models import db
from datetime import datetime

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    tanggal = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Spesifikasi Mesin
    jenis_mesin = db.Column(db.String(100), nullable=True)
    daya_hp_kw = db.Column(db.Float, nullable=True)
    jumlah_pole = db.Column(db.Integer, nullable=True)
    
    # Gejala (Symptoms)
    suara_bising_abnormal = db.Column(db.Boolean, nullable=True)
    getaran_berlebih = db.Column(db.Boolean, nullable=True)
    motor_cepat_panas = db.Column(db.Boolean, nullable=True)
    arus_melebihi_normal = db.Column(db.Boolean, nullable=True)
    tegangan_tidak_stabil = db.Column(db.Boolean, nullable=True)
    putaran_menurun = db.Column(db.Boolean, nullable=True)
    sulit_start = db.Column(db.Boolean, nullable=True)
    sering_trip_mcb = db.Column(db.Boolean, nullable=True)
    trip_overload_relay = db.Column(db.Boolean, nullable=True)
    efisiensi_menurun = db.Column(db.Boolean, nullable=True)
    bau_hangus = db.Column(db.Boolean, nullable=True)
    intermittent_stopping = db.Column(db.Boolean, nullable=True)
    warna_gulungan_berubah = db.Column(db.Boolean, nullable=True)
    kipas_pendingin_rusak = db.Column(db.Boolean, nullable=True)
    terminal_terbakar = db.Column(db.Boolean, nullable=True)
    bearing_aus_pecah = db.Column(db.Boolean, nullable=True)
    housing_bearing_aus = db.Column(db.Boolean, nullable=True)
    poros_shaft_aus = db.Column(db.Boolean, nullable=True)
    kebocoran_pelumas = db.Column(db.Boolean, nullable=True)
    keretakan_dudukan = db.Column(db.Boolean, nullable=True)
    sumbatan_sirip_pendingin = db.Column(db.Boolean, nullable=True)
    lubang_spi_aus = db.Column(db.Boolean, nullable=True)
    
    # Pengukuran
    temperatur_c = db.Column(db.Float, nullable=True)
    arus_a = db.Column(db.Float, nullable=True)
    tegangan_v = db.Column(db.Float, nullable=True)
    resistansi_isolasi_mohm = db.Column(db.Float, nullable=True)
    kecepatan_putaran_rpm = db.Column(db.Float, nullable=True)
    ketidakseimbangan_arus_pct = db.Column(db.Float, nullable=True)
    ketidakseimbangan_tegangan_pct = db.Column(db.Float, nullable=True)
    faktor_daya = db.Column(db.Float, nullable=True)
    
    # Hasil Prediksi (Label)
    diagnosis = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)

    SYMPTOM_LABEL_MAP = {
        "suara_bising_abnormal"  : "Suara Bising Abnormal",
        "getaran_berlebih"       : "Getaran Berlebih",
        "motor_cepat_panas"      : "Motor Cepat Panas",
        "arus_melebihi_normal"   : "Arus Melebihi Normal",
        "tegangan_tidak_stabil"  : "Tegangan Tidak Stabil",
        "putaran_menurun"        : "Putaran Menurun",
        "sulit_start"            : "Sulit Start",
        "sering_trip_mcb"        : "Sering Trip MCB/MCCB",
        "trip_overload_relay"    : "Trip Overload Relay",
        "efisiensi_menurun"      : "Efisiensi Menurun",
        "bau_hangus"             : "Bau Hangus",
        "intermittent_stopping"  : "Intermittent Stopping",
        "warna_gulungan_berubah" : "Warna Gulungan Berubah",
        "kipas_pendingin_rusak"  : "Kipas Pendingin Rusak",
        "terminal_terbakar"      : "Terminal Terbakar",
        "bearing_aus_pecah"      : "Bearing Aus/Pecah",
        "housing_bearing_aus"    : "Housing Bearing Aus",
        "poros_shaft_aus"        : "Poros (Shaft) Aus",
        "kebocoran_pelumas"      : "Kebocoran Pelumas",
        "keretakan_dudukan"      : "Keretakan Dudukan",
        "sumbatan_sirip_pendingin": "Sumbatan Sirip Pendingin",
        "lubang_spi_aus"         : "Lubang Spi (Keyway) Aus",
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
            "daya_hp_kw"  : self.daya_hp_kw,
            "jumlah_pole" : self.jumlah_pole,
            # Pengukuran
            "temperatur_c"                   : self.temperatur_c,
            "arus_a"                         : self.arus_a,
            "tegangan_v"                     : self.tegangan_v,
            "resistansi_isolasi_mohm"        : self.resistansi_isolasi_mohm,
            "kecepatan_putaran_rpm"          : self.kecepatan_putaran_rpm,
            "ketidakseimbangan_arus_pct"     : self.ketidakseimbangan_arus_pct,
            "ketidakseimbangan_tegangan_pct" : self.ketidakseimbangan_tegangan_pct,
            "faktor_daya"                    : self.faktor_daya,
            # Gejala aktif
            "gejala_aktif": self.active_symptoms(),
            # Hasil
            "diagnosis"     : self.diagnosis,
            "confidence"    : round(self.confidence, 4),
            "confidence_pct": f"{self.confidence * 100:.1f}%",
        }
