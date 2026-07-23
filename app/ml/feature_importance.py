"""
Modul untuk membaca feature importance dari model Random Forest yang sudah di-train.
Dimodifikasi untuk mendukung:
  - Global SHAP (mean |SHAP|)
  - Local SHAP (per prediksi)
  - SHAP Summary Plot (beeswarm)
  - Local SHAP Waterfall Plot (per prediksi → PNG)
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

def get_local_shap_importances(
    df_input: pd.DataFrame,
    predicted_label: str,
    top_n: int = 5,
    model=None,
    explainer=None,
) -> list:
    """
    Hitung SHAP lokal untuk satu prediksi.

    Args:
        df_input: DataFrame 1 baris (hasil preprocess_input).
        predicted_label: Label kelas yang diprediksi.
        top_n: Jumlah fitur teratas yang dikembalikan.
        model: (Opsional) Model RF yang sudah di-cache; jika None akan di-load dari disk.
        explainer: (Opsional) TreeExplainer yang sudah di-cache; jika None akan dibuat baru.
    """
    # Gunakan model/explainer dari cache eksternal bila tersedia;
    # jika tidak, fallback ke _load_artifacts() internal modul.
    if model is not None and explainer is not None:
        _eff_model = model
        _eff_explainer = explainer
    else:
        if not _load_artifacts():
            return []
        _eff_model = _model
        _eff_explainer = _explainer

    try:
        shap_values = _eff_explainer.shap_values(df_input, approximate=True, check_additivity=False)
        
        class_idx = list(_eff_model.classes_).index(predicted_label)
        
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


# =========================================================
# LOCAL WATERFALL PLOT (baris 216+)
# Fungsi yang menghasilkan waterfall plot SHAP lokal
# untuk SATU prediksi sebagai file PNG:
#   - plot_local_waterfall() : menampilkan kontribusi tiap fitur
#                              terhadap prediksi spesifik satu sampel
#                              → disimpan ke local_waterfall.png
# =========================================================

def plot_local_waterfall(
    df_input: pd.DataFrame,
    predicted_label: str,
    output_dir: str,
    model=None,
    explainer=None,
) -> str | None:
    """
    Membuat SHAP Waterfall Plot lokal untuk satu baris prediksi dan menyimpannya ke PNG.

    Waterfall plot menunjukkan kontribusi positif/negatif setiap fitur terhadap
    output model dibandingkan nilai baseline (expected value), sehingga bisa
    dijelaskan *mengapa* model memberikan prediksi tersebut.

    Args:
        df_input       : DataFrame 1 baris (output preprocess_input).
        predicted_label: Label kelas yang diprediksi model.
        output_dir     : Direktori tujuan penyimpanan PNG.
        model          : (Opsional) Model RF dari cache; jika None di-load dari disk.
        explainer      : (Opsional) TreeExplainer dari cache; jika None dibuat baru.

    Returns:
        str  : Path absolut file PNG yang disimpan.
        None : Jika terjadi error.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        # --- Pilih model/explainer ---
        if model is not None and explainer is not None:
            _eff_model    = model
            _eff_explainer = explainer
        else:
            if not _load_artifacts():
                print("      [Waterfall] Model belum tersedia.")
                return None
            _eff_model    = _model
            _eff_explainer = _explainer

        # --- Hitung SHAP values ---
        shap_values    = _eff_explainer.shap_values(df_input, approximate=True, check_additivity=False)
        expected_value = _eff_explainer.expected_value
        class_idx      = list(_eff_model.classes_).index(predicted_label)

        if isinstance(shap_values, list):
            local_shap = np.array(shap_values[class_idx][0], dtype=float)
            base_val   = float(expected_value[class_idx]) if hasattr(expected_value, "__len__") else float(expected_value)
        else:
            if shap_values.ndim == 3:
                local_shap = np.array(shap_values[0, :, class_idx], dtype=float)
                base_val   = float(expected_value[class_idx]) if hasattr(expected_value, "__len__") else float(expected_value)
            else:
                local_shap = np.array(shap_values[0], dtype=float)
                base_val   = float(expected_value) if not hasattr(expected_value, "__len__") else float(expected_value[0])

        feature_names = list(df_input.columns)
        feature_vals  = df_input.values[0]

        # --- Coba shap.plots.waterfall (native SHAP) ---
        try:
            explanation = shap.Explanation(
                values        = local_shap,
                base_values   = base_val,
                data          = feature_vals,
                feature_names = feature_names,
            )

            fig, ax = plt.subplots(figsize=(10, 7))
            shap.plots.waterfall(explanation, max_display=len(feature_names), show=False)
            fig = plt.gcf()
            fig.suptitle(
                f"SHAP Waterfall Plot — {predicted_label}",
                fontsize=13, fontweight="bold", y=1.01,
            )
            plt.tight_layout()

        except Exception:
            # --- Fallback: waterfall manual dengan matplotlib ---
            plt.close("all")

            # Urutkan berdasarkan nilai absolut SHAP (terkecil di atas → terbesar di bawah)
            order      = np.argsort(np.abs(local_shap))
            sv_sorted  = local_shap[order]
            fn_sorted  = [feature_names[i] for i in order]
            fv_sorted  = [feature_vals[i]  for i in order]

            n       = len(sv_sorted)
            running = base_val
            starts  = []
            for v in sv_sorted:
                starts.append(running)
                running += v
            final_val = running

            fig, ax = plt.subplots(figsize=(10, max(5, n * 0.55 + 1.5)))

            BLUE = "#2563EB"
            RED  = "#DC2626"
            GRAY = "#94A3B8"

            for i, (start, sv, fn, fv) in enumerate(zip(starts, sv_sorted, fn_sorted, fv_sorted)):
                color  = BLUE if sv >= 0 else RED
                bar    = ax.barh(i, sv, left=start, color=color, height=0.55,
                                 edgecolor="white", linewidth=0.8, zorder=3)
                label  = f"+{sv:.4f}" if sv >= 0 else f"{sv:.4f}"
                x_pos  = start + sv + (0.001 if sv >= 0 else -0.001)
                ha     = "left" if sv >= 0 else "right"
                ax.text(x_pos, i, label, va="center", ha=ha,
                        fontsize=8.5, fontweight="bold", color=color)

            # Nilai fitur di label y
            y_labels = [
                f"{fn.replace('_', ' ').upper()}  [{'YA' if fv == 1 else 'TIDAK'}]"
                for fn, fv in zip(fn_sorted, fv_sorted)
            ]
            ax.set_yticks(range(n))
            ax.set_yticklabels(y_labels, fontsize=9)

            # Garis baseline
            ax.axvline(base_val,  color=GRAY, linestyle="--", linewidth=1, label=f"Base: {base_val:.4f}")
            ax.axvline(final_val, color=BLUE, linestyle="-",  linewidth=1.5, label=f"Output: {final_val:.4f}")

            ax.set_xlabel("SHAP Value (kontribusi terhadap prediksi)", fontsize=10)
            ax.set_title(f"SHAP Waterfall Plot — {predicted_label}", fontsize=13, fontweight="bold", pad=12)
            ax.legend(fontsize=9, loc="lower right")
            ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)

            # Anotasi base & final di sumbu x
            ax.annotate(f"E[f(x)] = {base_val:.4f}", xy=(base_val, -0.7),
                        fontsize=8, color=GRAY, ha="center")
            ax.annotate(f"f(x) = {final_val:.4f}", xy=(final_val, -0.7),
                        fontsize=8, color=BLUE, ha="center")

            plt.tight_layout()

        # --- Simpan ---
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "local_waterfall.png")
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"      Waterfall Plot disimpan ke: {out_path}")
        return out_path

    except Exception as e:
        import traceback
        print(f"      Gagal membuat Waterfall Plot: {e}")
        traceback.print_exc()
        return None

