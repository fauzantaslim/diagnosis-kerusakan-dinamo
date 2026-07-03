"""
Modul untuk membaca feature importance dari model Random Forest yang sudah di-train.
Digunakan untuk menampilkan grafik/tabel pada dashboard.
Sekarang menggunakan SHAP untuk menghasilkan penjelasan lokal spesifik pada input.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

import shap

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.ml.preprocessing import FEATURE_COLS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")

_model = None
_explainer = None

def _load_artifacts():
    global _model, _explainer
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            return False
        _model = joblib.load(MODEL_PATH)
        # Inisialisasi SHAP TreeExplainer
        _explainer = shap.TreeExplainer(_model)
    return True


def get_local_shap_importances(df_input: pd.DataFrame, predicted_label: str, top_n: int = 5) -> list:
    """
    Menghitung feature importance lokal menggunakan SHAP untuk satu baris input.

    Args:
        df_input: DataFrame Pandas (1 baris) yang sudah melalui tahap preprocessing 
                  (encoding, scaling) dan siap diprediksi.
        predicted_label: Label string (nama kelas) yang merupakan hasil prediksi model.
        top_n: Jumlah fitur teratas yang ingin ditampilkan.

    Returns:
        List of dict: [{"feature": str, "importance": float}, ...]
    """
    if not _load_artifacts():
        return []
        
    try:
        shap_values = _explainer.shap_values(df_input)
        
        class_idx = list(_model.classes_).index(predicted_label)
        
        if isinstance(shap_values, list):
            local_shap_values = shap_values[class_idx][0]
        else:
            if len(shap_values.shape) == 3:
                local_shap_values = shap_values[0, :, class_idx]
            else:
                local_shap_values = shap_values[0]

        abs_shap = np.abs(local_shap_values)
        total_shap = np.sum(abs_shap)
        
        if total_shap > 0:
            feature_imp = [
                {
                    "feature": feat,
                    "importance": round(float(val / total_shap) * 100, 2)
                }
                for feat, val in zip(FEATURE_COLS, abs_shap)
            ]
        else:
            feature_imp = [
                {"feature": feat, "importance": 0.0}
                for feat in FEATURE_COLS
            ]
            
        feature_imp.sort(key=lambda x: x["importance"], reverse=True)
        return feature_imp[:top_n]

    except Exception as e:
        print(f"Error computing SHAP importance: {e}")
        return []
