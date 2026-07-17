"""
Training script untuk model Random Forest Diagnosis Kerusakan Dinamo.
Jalankan sekali untuk melatih model dan menyimpannya ke app/ml/models/.

Penggunaan:
    python -m app.ml.train
    atau dari direktori proyek: .venv/Scripts/python.exe -m app.ml.train
"""

import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Tambahkan root project ke path agar bisa import modul lokal
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.ml.preprocessing import (
    load_dataset,
    fit_preprocessors,
    transform_features,
    FEATURE_COLS,
    TARGET_COL,
)
from app.ml.evaluate import evaluate_model, export_metrics_to_excel
from app.ml.feature_importance import compute_global_shap, plot_shap_summary
from app.ml.visualize import plot_tree, plot_confusion_matrix

# =========================================================
# INPUT (baris 34-37)
# Deklarasi path sumber data dan lokasi penyimpanan artefak:
#   - DATASET_PATH : file Excel dataset mentah (input utama)
#   - MODEL_DIR    : direktori output untuk model & hasil
#   - MODEL_PATH   : path output model terlatih (.pkl)
#   - ENCODERS_PATH: path output preprocessor (.pkl)
# =========================================================
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "dataset_dummy.xlsx")  # INPUT
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")


def train():
    """Melatih model Random Forest dan menyimpan artefak ke MODEL_DIR."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("  TRAINING: Random Forest - Diagnosis Kerusakan Dinamo")
    print("=" * 60)

    # 1. Load dataset (membaca file dari DATASET_PATH)
    print(f"\n[1/5] Membaca dataset dari: {DATASET_PATH}")
    df = load_dataset(DATASET_PATH)
    print(f"      Total data: {len(df)} baris, {len(df.columns)} kolom")

    # 2. Pisahkan fitur dan target
    print("\n[2/5] Memisahkan fitur dan target...")
    X_raw = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    print(f"      Label unik ({len(y.unique())}): {sorted(y.unique())}")

    # =========================================================
    # PROSES
    # Encoding, pembagian data, dan pelatihan model Random Forest
    # =========================================================

    # 3. Fit preprocessors & transform fitur
    print("\n[3/5] Melakukan preprocessing (encoding)...")
    label_encoders = fit_preprocessors(X_raw)                # PROSES: fit LabelEncoder
    X = transform_features(X_raw.copy(), label_encoders)     # PROSES: encode fitur
    print("      Preprocessing selesai (tanpa scaling).")

    # 4. Train / Test split
    print("\n[4/5] Membagi data training dan testing (80:20)...")
    X_train, X_test, y_train, y_test = train_test_split(     # PROSES: split 80:20
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Training: {len(X_train)} | Testing: {len(X_test)}")

    # 5. Training model
    print("\n[5/5] Melatih model Random Forest...")
    model = RandomForestClassifier(                          # PROSES: inisialisasi RF
        n_estimators=50,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        oob_score=True,
    )
    model.fit(X_train, y_train)                              # PROSES: latih model

    # =========================================================
    # OUTPUT (baris 93-112)
    # Hasil evaluasi, artefak model (.pkl), laporan metrik (.xlsx),
    # dan visualisasi (PNG confusion matrix, SHAP summary, tree)
    # =========================================================

    # --- Evaluasi ---
    y_pred, acc = evaluate_model(model, X_test, y_test)      # OUTPUT: metrik ke konsol
    export_metrics_to_excel(y_test, y_pred, MODEL_DIR)       # OUTPUT: evaluation_metrics.xlsx

    # --- SHAP Global Feature Importance ---
    compute_global_shap(model, X_test, FEATURE_COLS)         # OUTPUT: ranking SHAP ke konsol

    # --- SHAP Summary Plot ---
    plot_shap_summary(model, X_test, FEATURE_COLS, MODEL_DIR) # OUTPUT: shap_summary_plot.png

    # --- Simpan model & preprocessors ---
    joblib.dump(model, MODEL_PATH)                           # OUTPUT: rf_model.pkl
    joblib.dump(label_encoders, ENCODERS_PATH)               # OUTPUT: label_encoders.pkl
    print(f"\nModel disimpan ke  : {MODEL_PATH}")
    print(f"Encoders disimpan  : {ENCODERS_PATH}")

    # --- Visualisasi ---
    class_names = sorted(y.unique().astype(str))
    plot_tree(model, FEATURE_COLS, class_names, MODEL_DIR)           # OUTPUT: rf_tree_viz.png
    plot_confusion_matrix(y_test, y_pred, class_names, MODEL_DIR)    # OUTPUT: confusion_matrix.png

    print("\nTraining selesai!")
    return model, label_encoders, acc


if __name__ == "__main__":
    train()
