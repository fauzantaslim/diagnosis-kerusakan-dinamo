"""
Modul prediksi untuk Diagnosis Kerusakan Dinamo.
Memuat model yang sudah di-train dan menyediakan fungsi predict() 
yang dapat dipanggil dari route/controller Flask.
"""

import os
import sys
import joblib
import shap

# Tambahkan root project ke path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.ml.preprocessing import preprocess_input, FEATURE_COLS, SYMPTOM_COLS
from app.ml.feature_importance import get_local_shap_importances

# --- Paths ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")

# Cache: model, preprocessors, dan SHAP explainer dimuat sekali saja saat modul pertama kali di-import
_model = None
_label_encoders = None
_explainer = None  # TreeExplainer di-cache agar tidak dibuat ulang setiap request


def _load_artifacts():
    global _model, _label_encoders, _explainer

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model belum ditemukan di '{MODEL_PATH}'. "
                "Jalankan terlebih dahulu: python -m app.ml.train"
            )
        _model = joblib.load(MODEL_PATH)
        _label_encoders = joblib.load(ENCODERS_PATH)
        # Buat TreeExplainer sekali saja — ini operasi berat, jangan diulang tiap request
        _explainer = shap.TreeExplainer(_model)


def predict(input_dict: dict) -> dict:
    _load_artifacts()

    df_input = preprocess_input(input_dict, _label_encoders)

    # Prediksi
    predicted_label = _model.predict(df_input)[0]
    proba_array = _model.predict_proba(df_input)[0]
    class_labels = _model.classes_

    probabilities = {label: float(prob) for label, prob in zip(class_labels, proba_array)}
    confidence = float(max(proba_array))

    local_importances = get_local_shap_importances(
        df_input,
        predicted_label,
        top_n=5,
        model=_model,
        explainer=_explainer,
    )

    return {
        "diagnosis": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "local_importances": local_importances,
    }


def get_model_info() -> dict:
    _load_artifacts()
    return {
        "n_estimators": _model.n_estimators,
        "n_classes": _model.n_classes_,
        "classes": list(_model.classes_),
        "n_features": _model.n_features_in_,
    }
