"""
Modul untuk membaca feature importance dari model Random Forest yang sudah di-train.
Digunakan untuk menampilkan grafik/tabel pada dashboard.
"""

import os
import sys
import joblib
from sklearn.inspection import permutation_importance

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.ml.preprocessing import FEATURE_COLS, TARGET_COL, load_dataset, transform_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "dataset_dummy.xlsx")

_model = None
_scaler = None
_label_encoders = None

def _load_artifacts():
    global _model, _scaler, _label_encoders
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            return False
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _label_encoders = joblib.load(ENCODERS_PATH)
    return True


def get_feature_importances(top_n: int = 10) -> list:
    """
    Mengembalikan daftar fitur terpenting dari model menggunakan Permutation Feature Importance.

    Args:
        top_n: Jumlah fitur teratas yang ingin ditampilkan.

    Returns:
        List of dict: [{"feature": str, "importance": float}, ...]
        Sudah diurutkan dari yang paling penting, dan sudah dinormalisasi 
        dalam persen (0-100).
    """
    if not _load_artifacts():
        return []
        
    try:
        # Load data untuk permutasi
        df = load_dataset(DATASET_PATH)
        X_raw = df[FEATURE_COLS].copy()
        y = df[TARGET_COL].copy()
        
        # Transform data agar sesuai input model
        X = transform_features(X_raw, _scaler, _label_encoders)
        
        # Pisahkan test set dengan random_state yang persis sama dengan train.py
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Hitung permutation importance pada TEST SET
        result = permutation_importance(
            _model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
        )
        
        # Ambil rata-rata importance
        importances = result.importances_mean
        
        # Ubah nilai negatif (jika ada karena noise) menjadi 0 alih-alih di-absolute
        importances_clean = [max(0, imp) for imp in importances]
        total = sum(importances_clean)
        
        if total > 0:
            feature_imp = [
                {
                    "feature": feat,
                    "importance": round(float(imp / total) * 100, 2),
                }
                for feat, imp in zip(FEATURE_COLS, importances_clean)
            ]
        else:
            feature_imp = [
                {
                    "feature": feat,
                    "importance": 0.0,
                }
                for feat in FEATURE_COLS
            ]

        # Urutkan descending
        feature_imp.sort(key=lambda x: x["importance"], reverse=True)
        return feature_imp[:top_n]
        
    except Exception as e:
        print(f"Error computing permutation importance: {e}")
        return []
