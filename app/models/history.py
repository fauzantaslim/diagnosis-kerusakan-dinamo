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
