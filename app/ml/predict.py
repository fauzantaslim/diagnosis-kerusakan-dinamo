"""
Modul prediksi untuk Diagnosis Kerusakan Dinamo.
Memuat model yang sudah di-train dan menyediakan fungsi predict() 
yang dapat dipanggil dari route/controller Flask.
"""

import os
import sys
import joblib

# Tambahkan root project ke path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.ml.preprocessing import preprocess_input

# --- Paths ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")

# Cache: model dan preprocessors dimuat sekali saja saat modul pertama kali di-import
_model = None
_scaler = None
_label_encoders = None


def _load_artifacts():
    """Memuat model dan preprocessors dari disk (lazy loading)."""
    global _model, _scaler, _label_encoders

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model belum ditemukan di '{MODEL_PATH}'. "
                "Jalankan terlebih dahulu: python -m app.ml.train"
            )
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _label_encoders = joblib.load(ENCODERS_PATH)


def predict(input_dict: dict) -> dict:
    """
    Melakukan prediksi diagnosis kerusakan dinamo berdasarkan input dari user.

    Args:
        input_dict: Dictionary dengan key nama fitur (sesuai FEATURE_COLS)
                    dan nilainya dari form user.
                    Contoh gejala: {"suara_bising_abnormal": "Ya", ...}
                    Contoh numerik: {"temperatur_c": 95.5, "arus_a": 14.2, ...}
                    Contoh kategorikal: {"jenis_mesin": "Motor Induksi 3 Fasa"}

    Returns:
        dict dengan key:
            - "diagnosis"    : str, label kelas hasil prediksi
            - "confidence"   : float, probabilitas tertinggi (0.0 - 1.0)
            - "probabilities": dict {label: probability} untuk semua kelas
    """
    _load_artifacts()

    df_input = preprocess_input(input_dict, _scaler, _label_encoders)

    # Prediksi
    predicted_label = _model.predict(df_input)[0]
    proba_array = _model.predict_proba(df_input)[0]
    class_labels = _model.classes_

    probabilities = {label: float(prob) for label, prob in zip(class_labels, proba_array)}
    confidence = float(max(proba_array))

    return {
        "diagnosis": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def get_model_info() -> dict:
    """Mengembalikan informasi ringkas tentang model yang sedang dimuat."""
    _load_artifacts()
    return {
        "n_estimators": _model.n_estimators,
        "n_classes": _model.n_classes_,
        "classes": list(_model.classes_),
        "n_features": _model.n_features_in_,
    }
