"""
Preprocessing module untuk pipeline Machine Learning Diagnosis Kerusakan Dinamo.
Bertugas membaca dataset Excel, membersihkan data, dan melakukan encoding.
Catatan: StandardScaler dihapus karena Random Forest tidak membutuhkan scaling.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Mapping nama kolom Excel -> nama atribut Python/model
COLUMN_RENAME_MAP = {
    "Jenis Mesin": "jenis_mesin",
    "Daya (HP/kW)": "daya_hp_kw",
    "Jumlah Pole": "jumlah_pole",
    "Suara bising abnormal": "suara_bising_abnormal",
    "Getaran berlebih": "getaran_berlebih",
    "Motor cepat panas": "motor_cepat_panas",
    "Arus melebihi normal": "arus_melebihi_normal",
    "Tegangan tidak stabil": "tegangan_tidak_stabil",
    "Putaran menurun": "putaran_menurun",
    "Sulit start": "sulit_start",
    "Sering trip MCB/MCCB": "sering_trip_mcb",
    "Trip Overload Relay": "trip_overload_relay",
    "Efisiensi Menurun": "efisiensi_menurun",
    "Bau hangus": "bau_hangus",
    "Intermittent Stopping": "intermittent_stopping",
    "Warna gulungan berubah": "warna_gulungan_berubah",
    "Kipas pendingin rusak": "kipas_pendingin_rusak",
    "Terminal terbakar": "terminal_terbakar",
    "Bearing aus/pecah": "bearing_aus_pecah",
    "Housing bearing aus": "housing_bearing_aus",
    "Poros (shaft) aus": "poros_shaft_aus",
    "Kebocoran Pelumas": "kebocoran_pelumas",
    "Keretakan Dudukan": "keretakan_dudukan",
    "Sumbatan Sirip Pendingin": "sumbatan_sirip_pendingin",
    "Lubang spi (keyway) aus": "lubang_spi_aus",
    "Temperatur (\u00b0C)": "temperatur_c",
    "Arus (A)": "arus_a",
    "Tegangan (V)": "tegangan_v",
    "Resistansi isolasi (M\u03a9)": "resistansi_isolasi_mohm",
    "Kecepatan Putaran (RPM)": "kecepatan_putaran_rpm",
    "Ketidakseimbangan Arus (%)": "ketidakseimbangan_arus_pct",
    "Ketidakseimbangan Tegangan (%)": "ketidakseimbangan_tegangan_pct",
    "Faktor Daya": "faktor_daya",
    "Label": "label",
}

# Kolom gejala (nilai: "Ya"/"Tidak" -> 1/0)
SYMPTOM_COLS = [
    "suara_bising_abnormal", "getaran_berlebih", "motor_cepat_panas",
    "arus_melebihi_normal", "tegangan_tidak_stabil", "putaran_menurun",
    "sulit_start", "sering_trip_mcb", "trip_overload_relay",
    "efisiensi_menurun", "bau_hangus", "intermittent_stopping",
    "warna_gulungan_berubah", "kipas_pendingin_rusak", "terminal_terbakar",
    "bearing_aus_pecah", "housing_bearing_aus", "poros_shaft_aus",
    "kebocoran_pelumas", "keretakan_dudukan", "sumbatan_sirip_pendingin",
    "lubang_spi_aus",
]

# Kolom numerik (tanpa scaling — RF tidak membutuhkan)
NUMERIC_COLS = [
    "daya_hp_kw", "jumlah_pole", "temperatur_c", "arus_a", "tegangan_v",
    "resistansi_isolasi_mohm", "kecepatan_putaran_rpm",
    "ketidakseimbangan_arus_pct", "ketidakseimbangan_tegangan_pct",
    "faktor_daya",
]

# Kolom kategorikal (Jenis Mesin -> LabelEncoder)
CATEGORICAL_COLS = ["jenis_mesin"]

# Semua kolom fitur dalam urutan tetap (penting untuk konsistensi prediksi)
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS + SYMPTOM_COLS

TARGET_COL = "label"


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Membaca dataset Excel dan melakukan rename kolom."""
    df = pd.read_excel(dataset_path)
    df = df.rename(columns=COLUMN_RENAME_MAP)
    df.columns = [c.strip() for c in df.columns]

    # Bersihkan kolom daya: "7.5 HP" -> 7.5
    if "daya_hp_kw" in df.columns:
        df["daya_hp_kw"] = (
            df["daya_hp_kw"]
            .astype(str)
            .str.extract(r"([\d.]+)", expand=False)
            .astype(float)
        )

    # Pastikan jumlah_pole numerik
    if "jumlah_pole" in df.columns:
        df["jumlah_pole"] = pd.to_numeric(df["jumlah_pole"], errors="coerce").fillna(0)

    return df


def encode_symptoms(df: pd.DataFrame) -> pd.DataFrame:
    """Mengubah nilai 'Ya'/'Tidak' menjadi 1/0 pada kolom gejala."""
    for col in SYMPTOM_COLS:
        if col in df.columns:
            df[col] = df[col].map({"Ya": 1, "Tidak": 0}).fillna(0).astype(int)
    return df


def fit_preprocessors(df: pd.DataFrame) -> dict:
    """
    Membuat dan menyesuaikan (fit) LabelEncoder untuk kolom kategorikal.
    StandardScaler dihapus karena Random Forest tidak membutuhkan scaling.
    Returns:
        label_encoders: dict {col_name: LabelEncoder} yang sudah di-fit
    """
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        label_encoders[col] = le

    return label_encoders


def transform_features(df: pd.DataFrame, label_encoders: dict) -> pd.DataFrame:
    """
    Menerapkan transformasi pada fitur:
    - Gejala: Ya/Tidak -> 1/0
    - Numerik: dibiarkan asli (tanpa scaling)
    - Kategorikal: LabelEncoder
    """
    df = encode_symptoms(df)

    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))

    return df


def preprocess_input(input_dict: dict, label_encoders: dict) -> pd.DataFrame:
    """
    Menerima satu baris input dari form user (dict),
    melakukan preprocessing, dan mengembalikan DataFrame siap-prediksi.
    StandardScaler dihapus karena Random Forest tidak membutuhkan scaling.
    """
    row = {}

    # Gejala
    for col in SYMPTOM_COLS:
        val = input_dict.get(col, "Tidak")
        row[col] = 1 if str(val).lower() in ("ya", "1", "true") else 0

    # Numerik (nilai asli, tanpa scaling)
    for col in NUMERIC_COLS:
        row[col] = float(input_dict.get(col, 0))

    # Kategorikal
    for col in CATEGORICAL_COLS:
        raw = str(input_dict.get(col, ""))
        le = label_encoders[col]
        if raw in le.classes_:
            row[col] = int(le.transform([raw])[0])
        else:
            row[col] = 0

    df_input = pd.DataFrame([row], columns=FEATURE_COLS)

    return df_input
