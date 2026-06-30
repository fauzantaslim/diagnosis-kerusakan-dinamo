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

from app.ml.preprocessing import preprocess_input, FEATURE_COLS, SYMPTOM_COLS, NUMERIC_COLS

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

    # --- Local Feature Importance (Leave-One-Out) ---
    c_idx = list(class_labels).index(predicted_label)
    base_prob = proba_array[c_idx]
    
    local_importances = []
    
    # Hanya hitung untuk gejal  a dan fitur numerik, abaikan kategorikal (jenis_mesin)
    target_cols = SYMPTOM_COLS + NUMERIC_COLS
    
    for col in target_cols:
        if col not in df_input.columns:
            continue
            
        original_val = df_input.at[0, col]
        
        # Jika nilai fitur sudah 0 (gejala 'Tidak' atau numerik sama dengan mean), lewati
        if original_val == 0.0:
            continue
            
        # Buat copy dan set ke baseline (0.0)
        df_temp = df_input.copy()
        df_temp.at[0, col] = 0.0
        
        # Prediksi ulang
        new_prob = _model.predict_proba(df_temp)[0][c_idx]
        
        # Drop probabilitas (seberapa besar fitur ini mendongkrak probabilitas)
        drop = base_prob - new_prob
        
        # Jika drop > 0, artinya fitur ini membantu menaikkan probabilitas diagnosis
        if drop > 0:
            local_importances.append({"feature": col, "drop": drop})
            
    # Normalisasi agar total = 100%
    total_drop = sum(item["drop"] for item in local_importances)
    if total_drop > 0:
        for item in local_importances:
            item["importance"] = round((item["drop"] / total_drop) * 100, 2)
            del item["drop"]
    else:
        local_importances = []
        
    local_importances.sort(key=lambda x: x["importance"], reverse=True)

    return {
        "diagnosis": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "local_importances": local_importances[:5], # Ambil Top 5
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
