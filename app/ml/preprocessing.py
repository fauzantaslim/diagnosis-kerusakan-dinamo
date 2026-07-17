"""
Preprocessing module untuk pipeline Machine Learning Diagnosis Kerusakan Dinamo.
Bertugas membaca dataset Excel, membersihkan data, dan melakukan encoding.
Catatan: StandardScaler dihapus karena Random Forest tidak membutuhkan scaling.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# =========================================================
# INPUT (baris 11-54)
# Definisi struktur data sebagai konstanta modul:
#   - COLUMN_RENAME_MAP : mapping nama kolom Excel → Python
#   - SYMPTOM_COLS      : 15 kolom gejala (Ya/Tidak)
#   - NUMERIC_COLS      : 3 kolom numerik (tanpa scaling)
#   - CATEGORICAL_COLS  : 1 kolom kategorikal (jenis_mesin)
#   - FEATURE_COLS      : urutan fitur tetap (gabungan semua)
#   - TARGET_COL        : kolom label/kelas target
# =========================================================

# Mapping nama kolom Excel -> nama atribut Python/model
COLUMN_RENAME_MAP = {
    "Jenis Mesin": "jenis_mesin",
    "Daya (HP/kW)": "daya",
    "Jumlah Pole": "jumlah_pole",
    "Kecepatan Putaran (RPM)": "kecepatan_putaran_rpm",
    "Suara Bising Abnormal": "suara_bising_abnormal",
    "Getaran Berlebih": "getaran_berlebih",
    "Ampere Stabil": "ampere_stabil",
    "Tegangan Stabil": "tegangan_stabil",
    "Sulit Start": "sulit_start",
    "Bau Hangus": "bau_hangus",
    "Warna Gulungan Berubah": "warna_gulungan_berubah",
    "Kipas Pendingin Rusak": "kipas_pendingin_rusak",
    "Terminal Terbakar": "terminal_terbakar",
    "Bearing Aus/Pecah": "bearing_aus_pecah",
    "Housing Bearing Aus": "housing_bearing_aus",
    "Kebocoran Pelumas": "kebocoran_pelumas",
    "Keretakan Dudukan": "keretakan_dudukan",
    "Lubang Spi (Keyway) Aus": "lubang_spi_aus",
    "Resistansi Isolasi Normal": "resistansi_isolasi_normal",
    "Label": "label",
}

# Kolom gejala (nilai: "Ya"/"Tidak" -> 1/0)
SYMPTOM_COLS = [
    "suara_bising_abnormal", "getaran_berlebih", "ampere_stabil",
    "tegangan_stabil", "sulit_start", "bau_hangus",
    "warna_gulungan_berubah", "kipas_pendingin_rusak", "terminal_terbakar",
    "bearing_aus_pecah", "housing_bearing_aus", "kebocoran_pelumas",
    "keretakan_dudukan", "lubang_spi_aus", "resistansi_isolasi_normal",
]

# Kolom numerik (tanpa scaling — RF tidak membutuhkan)
NUMERIC_COLS = [
    "daya", "jumlah_pole", "kecepatan_putaran_rpm",
]

# Kolom kategorikal (Jenis Mesin -> LabelEncoder)
CATEGORICAL_COLS = ["jenis_mesin"]

# Semua kolom fitur dalam urutan tetap (penting untuk konsistensi prediksi)
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS + SYMPTOM_COLS

TARGET_COL = "label"


# =========================================================
# PROSES (baris 70-145)
# Fungsi-fungsi transformasi data:
#   - load_dataset()       : baca Excel, bersihkan & rename kolom
#   - encode_symptoms()    : konversi Ya/Tidak → 1/0
#   - fit_preprocessors()  : fit LabelEncoder pada data latih
#   - transform_features() : terapkan encoding ke seluruh dataset
# =========================================================

def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Membaca dataset Excel dan melakukan rename kolom."""
    df = pd.read_excel(dataset_path)
    df = df.rename(columns=COLUMN_RENAME_MAP)
    df.columns = [c.strip() for c in df.columns]

    # Bersihkan kolom daya: "7.5 HP" -> 7.5
    if "daya" in df.columns:
        df["daya"] = (
            df["daya"]
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


# =========================================================
# OUTPUT (baris 130-148)
# Fungsi yang menghasilkan data siap prediksi dari input user:
#   - preprocess_input() : menerima dict dari form, mengembalikan
#                          DataFrame 1 baris siap masuk model RF
# =========================================================

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
