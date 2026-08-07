from app.models import db


class Dataset(db.Model):
    __tablename__ = 'datasets'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # Metadata (bukan fitur)
    daya        = db.Column(db.Float, nullable=True)
    jumlah_pole = db.Column(db.Integer, nullable=True)

    # Fitur (0 = TIDAK, 1 = YA)
    suara_bising_abnormal            = db.Column(db.Integer, nullable=False, default=0)
    bau_hangus                       = db.Column(db.Integer, nullable=False, default=0)
    indikasi_overheating             = db.Column(db.Integer, nullable=False, default=0)
    putaran_poros_seret              = db.Column(db.Integer, nullable=False, default=0)
    getaran_berlebih                 = db.Column(db.Integer, nullable=False, default=0)
    terminal_overheating             = db.Column(db.Integer, nullable=False, default=0)
    kipas_pendingin_rusak            = db.Column(db.Integer, nullable=False, default=0)
    cooling_duct_tersumbat           = db.Column(db.Integer, nullable=False, default=0)
    resistansi_isolasi_tidak_seimbang   = db.Column(db.Integer, nullable=False, default=0)
    resistansi_winding_tidak_seimbang   = db.Column(db.Integer, nullable=False, default=0)
    arus_antar_fasa_tidak_seimbang      = db.Column(db.Integer, nullable=False, default=0)

    # Target / Label
    label = db.Column(db.String(100), nullable=False)

    # ------------------------------------------------------------------ #
    #  Kolom fitur (urut sesuai dataset) — dipakai Random Forest manual   #
    # ------------------------------------------------------------------ #
    FEATURE_COLUMNS = [
        'suara_bising_abnormal',
        'bau_hangus',
        'indikasi_overheating',
        'putaran_poros_seret',
        'getaran_berlebih',
        'terminal_overheating',
        'kipas_pendingin_rusak',
        'cooling_duct_tersumbat',
        'resistansi_isolasi_tidak_seimbang',
        'resistansi_winding_tidak_seimbang',
        'arus_antar_fasa_tidak_seimbang',
    ]

    def to_dict(self) -> dict:
        """Serialisasi objek Dataset ke dictionary (untuk JSON response)."""
        return {
            'id'                                : self.id,
            'daya'                              : self.daya,
            'jumlah_pole'                       : self.jumlah_pole,
            'suara_bising_abnormal'             : self.suara_bising_abnormal,
            'bau_hangus'                        : self.bau_hangus,
            'indikasi_overheating'              : self.indikasi_overheating,
            'putaran_poros_seret'               : self.putaran_poros_seret,
            'getaran_berlebih'                  : self.getaran_berlebih,
            'terminal_overheating'              : self.terminal_overheating,
            'kipas_pendingin_rusak'             : self.kipas_pendingin_rusak,
            'cooling_duct_tersumbat'            : self.cooling_duct_tersumbat,
            'resistansi_isolasi_tidak_seimbang' : self.resistansi_isolasi_tidak_seimbang,
            'resistansi_winding_tidak_seimbang' : self.resistansi_winding_tidak_seimbang,
            'arus_antar_fasa_tidak_seimbang'    : self.arus_antar_fasa_tidak_seimbang,
            'label'                             : self.label,
        }
