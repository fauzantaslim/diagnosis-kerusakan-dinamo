"""
Generate local SHAP waterfall plot dan simpan ke app/ml/models/local_waterfall.png
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app.ml.predict import _load_artifacts, _model, _explainer
from app.ml.preprocessing import preprocess_input
from app.ml.feature_importance import plot_local_waterfall

MODEL_DIR = os.path.join(BASE_DIR, "app", "ml", "models")

# Load model + explainer
_load_artifacts()

# Refresh setelah import (supaya dapat nilai yang sudah di-load)
from app.ml import predict as _p
model    = _p._model
explainer = _p._explainer

# Input contoh
input_data = {
    "suara_bising_abnormal":            "YA",
    "bau_hangus":                       "YA",
    "indikasi_overheating":             "TIDAK",
    "putaran_poros_seret":              "YA",
    "getaran_berlebih":                 "TIDAK",
    "terminal_overheating":             "TIDAK",
    "kipas_pendingin_rusak":            "TIDAK",
    "cooling_duct_tersumbat":           "TIDAK",
    "resistansi_isolasi_tidak_seimbang":"YA",
    "resistansi_winding_tidak_seimbang":"TIDAK",
    "arus_antar_fasa_tidak_seimbang":   "TIDAK",
}

df_input = preprocess_input(input_data, {})

# Prediksi label
predicted_label = model.predict(df_input)[0]
confidence = max(model.predict_proba(df_input)[0]) * 100
print(f"Prediksi : {predicted_label}  ({confidence:.1f}%)")

# Generate waterfall plot
out = plot_local_waterfall(
    df_input=df_input,
    predicted_label=predicted_label,
    output_dir=MODEL_DIR,
    model=model,
    explainer=explainer,
)
print(f"Gambar disimpan : {out}")
