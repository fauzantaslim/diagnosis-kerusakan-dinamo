"""
Preprocessing module untuk pipeline Machine Learning Diagnosis Kerusakan Dinamo.
Bertugas membaca dataset Excel, membersihkan data, dan melakukan encoding.

Dataset: 11 kolom gejala (YA/TIDAK) + 1 kolom Label.
Tidak ada kolom numerik maupun kategorikal tambahan.
StandardScaler tidak digunakan karena Random Forest tidak membutuhkan scaling.
"""

import pandas as pd

# =========================================================
# INPUT (baris 18-57)
# Definisi struktur data sebagai konstanta modul:
#   - COLUMN_RENAME_MAP : mapping nama kolom Excel → Python
#   - SYMPTOM_COLS      : 11 kolom gejala (YA/TIDAK)
#   - FEATURE_COLS      : urutan fitur tetap (semua kolom gejala)
#   - TARGET_COL        : kolom label/kelas target
# =========================================================

# Mapping nama kolom Excel -> nama atribut Python/model
COLUMN_RENAME_MAP = {
    "Suara Bising Abnormal":            "suara_bising_abnormal",
    "Bau Hangus":                       "bau_hangus",
    "Indikasi Overheating":             "indikasi_overheating",
    "Putaran Poros Seret":              "putaran_poros_seret",
    "Getaran Berlebih":                 "getaran_berlebih",
    "Terminal Overheating":             "terminal_overheating",
    "Kipas Pendingin Rusak":            "kipas_pendingin_rusak",
    "Cooling Duct Tersumbat":           "cooling_duct_tersumbat",
    "Resistansi Isolasi Tidak Seimbang":"resistansi_isolasi_tidak_seimbang",
    "Resistansi Winding Tidak Seimbang":"resistansi_winding_tidak_seimbang",
    "Arus Antar Fasa Tidak Seimbang":   "arus_antar_fasa_tidak_seimbang",
    "Label":                            "label",
}

# Kolom gejala (nilai: "YA"/"TIDAK" -> 1/0)
SYMPTOM_COLS = [
    "suara_bising_abnormal",
    "bau_hangus",
    "indikasi_overheating",
    "putaran_poros_seret",
    "getaran_berlebih",
    "terminal_overheating",
    "kipas_pendingin_rusak",
    "cooling_duct_tersumbat",
    "resistansi_isolasi_tidak_seimbang",
    "resistansi_winding_tidak_seimbang",
    "arus_antar_fasa_tidak_seimbang",
]

# Semua kolom fitur dalam urutan tetap (penting untuk konsistensi prediksi)
# Dataset ini hanya memiliki kolom gejala — tidak ada numerik/kategorikal tambahan
FEATURE_COLS = SYMPTOM_COLS

TARGET_COL = "label"


# =========================================================
# PROSES (baris 60-120)
# Fungsi-fungsi transformasi data:
#   - load_dataset()       : baca Excel, bersihkan & rename kolom
#   - encode_symptoms()    : konversi YA/TIDAK → 1/0 (case-insensitive)
#   - fit_preprocessors()  : kompatibilitas — mengembalikan dict kosong
#                            (tidak ada LabelEncoder yang dibutuhkan)
#   - transform_features() : terapkan encoding gejala ke seluruh dataset
# =========================================================

def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Membaca dataset Excel dan melakukan rename kolom."""
    df = pd.read_excel(dataset_path)
    df = df.rename(columns=COLUMN_RENAME_MAP)
    df.columns = [c.strip() for c in df.columns]

    # Normalisasi label: hapus duplikat/typo jika ada
    # (misal: "Kerusakan Kerusakan Terminal" -> "Kerusakan Terminal")
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
        df[TARGET_COL] = df[TARGET_COL].str.replace(
            r"\bKerusakan Kerusakan\b", "Kerusakan", regex=True
        )

    return df


def encode_symptoms(df: pd.DataFrame) -> pd.DataFrame:
    """Mengubah nilai 'YA'/'TIDAK' menjadi 1/0 pada kolom gejala (case-insensitive)."""
    for col in SYMPTOM_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.upper()
                .str.strip()
                .map({"YA": 1, "TIDAK": 0})
                .fillna(0)
                .astype(int)
            )
    return df


def fit_preprocessors(df: pd.DataFrame) -> dict:
    """
    Placeholder untuk menjaga kompatibilitas dengan train.py.
    Dataset ini hanya memiliki kolom gejala biner — tidak ada
    LabelEncoder yang perlu di-fit.

    Returns:
        label_encoders: dict kosong {}
    """
    return {}


def transform_features(df: pd.DataFrame, label_encoders: dict) -> pd.DataFrame:
    """
    Menerapkan transformasi pada fitur:
    - Gejala: YA/TIDAK -> 1/0 (case-insensitive)
    - Tidak ada encoding kategorikal maupun scaling numerik.
    """
    df = encode_symptoms(df)
    return df


# =========================================================
# OUTPUT (baris 120-145)
# Fungsi yang menghasilkan data siap prediksi dari input user:
#   - preprocess_input() : menerima dict dari form, mengembalikan
#                          DataFrame 1 baris siap masuk model RF
# =========================================================

def preprocess_input(input_dict: dict, label_encoders: dict) -> pd.DataFrame:
    """
    Menerima satu baris input dari form user (dict),
    melakukan preprocessing, dan mengembalikan DataFrame siap-prediksi.

    Setiap nilai gejala diterima sebagai: "YA"/"ya"/"1"/True -> 1,
    selain itu -> 0.
    """
    row = {}

    for col in SYMPTOM_COLS:
        val = input_dict.get(col, "TIDAK")
        row[col] = 1 if str(val).upper().strip() in ("YA", "1", "TRUE") else 0

    df_input = pd.DataFrame([row], columns=FEATURE_COLS)
    return df_input
