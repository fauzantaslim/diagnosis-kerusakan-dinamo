"""
Modul untuk membaca feature importance dari model Random Forest yang sudah di-train.
Digunakan untuk menampilkan grafik/tabel pada dashboard.
"""

import os
import sys
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.ml.preprocessing import FEATURE_COLS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")

_model = None


def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            return None
        _model = joblib.load(MODEL_PATH)
    return _model


def get_feature_importances(top_n: int = 10) -> list:
    """
    Mengembalikan daftar fitur terpenting dari model Random Forest.

    Args:
        top_n: Jumlah fitur teratas yang ingin ditampilkan.

    Returns:
        List of dict: [{"feature": str, "importance": float}, ...]
        Sudah diurutkan dari yang paling penting, dan sudah dinormalisasi 
        dalam persen (0-100).
    """
    model = _load_model()
    if model is None:
        return []

    importances = model.feature_importances_
    total = sum(importances)

    feature_imp = [
        {
            "feature": feat,
            "importance": round(float(imp / total) * 100, 2),
        }
        for feat, imp in zip(FEATURE_COLS, importances)
    ]

    # Urutkan descending
    feature_imp.sort(key=lambda x: x["importance"], reverse=True)

    return feature_imp[:top_n]
