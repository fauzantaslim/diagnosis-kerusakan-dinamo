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
from imblearn.over_sampling import SMOTE

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
from app.ml.feature_importance import compute_global_shap
from app.ml.visualize import plot_tree, plot_confusion_matrix

# --- Paths ---
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "dataset_dummy.xlsx")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")


def train():
    """Melatih model Random Forest dan menyimpan artefak ke MODEL_DIR."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("  TRAINING: Random Forest - Diagnosis Kerusakan Dinamo")
    print("=" * 60)

    # 1. Load dataset
    print(f"\n[1/5] Membaca dataset dari: {DATASET_PATH}")
    df = load_dataset(DATASET_PATH)
    print(f"      Total data: {len(df)} baris, {len(df.columns)} kolom")

    # 2. Pisahkan fitur dan target
    print("\n[2/5] Memisahkan fitur dan target...")
    X_raw = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    print(f"      Label unik ({len(y.unique())}): {sorted(y.unique())}")

    # 3. Fit preprocessors & transform fitur
    print("\n[3/5] Melakukan preprocessing (encoding)...")
    label_encoders = fit_preprocessors(X_raw)
    X = transform_features(X_raw.copy(), label_encoders)
    print("      Preprocessing selesai (tanpa scaling).")

    # 4. Train / Test split + SMOTE
    print("\n[4/5] Membagi data training dan testing (80:20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Training sebelum SMOTE: {len(X_train)} | Testing: {len(X_test)}")

    print("      Menerapkan SMOTE pada data training...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"      Training sesudah SMOTE: {len(X_train)}")

    # 5. Training model
    print("\n[5/5] Melatih model Random Forest...")
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        oob_score=True,
    )
    model.fit(X_train, y_train)

    # --- Evaluasi ---
    y_pred, acc = evaluate_model(model, X_test, y_test)
    export_metrics_to_excel(y_test, y_pred, MODEL_DIR)

    # --- SHAP Global Feature Importance ---
    compute_global_shap(model, X_test, FEATURE_COLS)

    # --- Simpan model & preprocessors ---
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoders, ENCODERS_PATH)
    print(f"\nModel disimpan ke  : {MODEL_PATH}")
    print(f"Encoders disimpan  : {ENCODERS_PATH}")

    # --- Visualisasi ---
    class_names = sorted(y.unique().astype(str))
    plot_tree(model, FEATURE_COLS, class_names, MODEL_DIR)
    plot_confusion_matrix(y_test, y_pred, class_names, MODEL_DIR)

    print("\nTraining selesai!")
    return model, label_encoders, acc


if __name__ == "__main__":
    train()
