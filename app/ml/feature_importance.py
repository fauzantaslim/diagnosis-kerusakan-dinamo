"""
Modul untuk membaca feature importance dari model Random Forest yang sudah di-train.
Dimodifikasi untuk mendukung:
  - Global SHAP (mean |SHAP|)
  - Local SHAP (per prediksi)
  - SHAP Summary Plot (beeswarm)
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

# =========================================================
# INPUT (baris 23-27)
# Path sumber model terlatih dan cache objek di memori:
#   - MODEL_PATH  : file rf_model.pkl (model Random Forest)
#   - _model      : cache model setelah dimuat dari disk
#   - _explainer  : cache SHAP TreeExplainer
# =========================================================
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


# =========================================================
# PROSES (baris 45-141)
# Fungsi-fungsi perhitungan SHAP:
#   - get_local_shap_importances() : SHAP lokal untuk 1 input prediksi
#                                    → normalisasi ke persentase, top-N fitur
#   - compute_global_shap()        : SHAP global dari seluruh test set
#                                    → mean |SHAP| antar kelas, cetak ranking
# =========================================================

def get_local_shap_importances(df_input: pd.DataFrame, predicted_label: str, top_n: int = 5) -> list:

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


def compute_global_shap(model, X_test, feature_cols: list):

    print("\n[SHAP] Menghitung Global Feature Importance...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            mean_abs_shap = np.zeros(X_test.shape[1])
            for sv in shap_values:
                mean_abs_shap += np.abs(sv).mean(axis=0)
            mean_abs_shap /= len(shap_values)
        else:
            if len(shap_values.shape) == 3:
                mean_abs_shap = np.abs(shap_values).mean(axis=0).mean(axis=1)
            else:
                mean_abs_shap = np.abs(shap_values).mean(axis=0)

        global_importances = [
            {"feature": feat, "importance": imp}
            for feat, imp in zip(feature_cols, mean_abs_shap)
        ]
        global_importances.sort(key=lambda x: x["importance"], reverse=True)

        print("\n--- Global Feature Importance (SHAP) ---")
        for i, item in enumerate(global_importances[:15]):
            print(f"  {i+1}. {item['feature']:<30} : {item['importance']:.4f}")

        return global_importances

    except Exception as e:
        print(f"      Gagal menghitung SHAP: {e}")
        return []


# =========================================================
# OUTPUT (baris 151-211)
# Fungsi yang menghasilkan visualisasi SHAP sebagai file PNG:
#   - plot_shap_summary() : render beeswarm plot dari SHAP values
#                           → disimpan ke shap_summary_plot.png
# =========================================================

def plot_shap_summary(model, X_test: pd.DataFrame, feature_cols: list, output_dir: str):

    print("\n[SHAP] Membuat SHAP Summary Plot (beeswarm)...")
    try:
        import matplotlib
        matplotlib.use("Agg")  # Backend non-interaktif, aman di server
        import matplotlib.pyplot as plt

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Untuk multiclass, shap_values adalah list (1 array per kelas).
        # Gabungkan dengan rata-rata absolut agar ringkasan bersifat global.
        if isinstance(shap_values, list):
            # Stack: shape (n_classes, n_samples, n_features) → rata-rata |SHAP| per kelas
            combined_shap = np.mean(
                [np.abs(sv) for sv in shap_values], axis=0
            )
        elif len(shap_values.shape) == 3:
            # shape: (n_samples, n_features, n_classes)
            combined_shap = np.abs(shap_values).mean(axis=2)
        else:
            combined_shap = shap_values

        # --- Beeswarm / dot summary plot ---
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            combined_shap,
            X_test,
            feature_names=feature_cols,
            plot_type="dot",   # beeswarm
            show=False,
            max_display=15,
        )
        plt.title("SHAP Summary Plot - Random Forest Diagnosis Kerusakan Dinamo",
                  fontsize=13, pad=12)
        plt.tight_layout()

        out_path = os.path.join(output_dir, "shap_summary_plot.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"      SHAP Summary Plot disimpan ke: {out_path}")
        return out_path

    except ImportError as e:
        print(f"      Dependensi tidak ditemukan: {e}")
        print("      Pastikan matplotlib terinstal: pip install matplotlib")
        return None
    except Exception as e:
        print(f"      Gagal membuat SHAP Summary Plot: {e}")
        return None
