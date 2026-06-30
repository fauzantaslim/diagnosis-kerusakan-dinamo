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
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Tambahkan root project ke path agar bisa import preprocessing
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

# --- Paths ---
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "dataset_dummy.xlsx")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
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
    print("\n[3/5] Melakukan preprocessing (encoding + scaling)...")
    scaler, label_encoders = fit_preprocessors(X_raw)
    X = transform_features(X_raw.copy(), scaler, label_encoders)
    print("      Preprocessing selesai.")

    # 4. Train / Test split
    print("\n[4/5] Membagi data training dan testing (80:20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Training: {len(X_train)} | Testing: {len(X_test)}")

    # 5. Training model
    print("\n[5/5] Melatih model Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"  AKURASI TEST SET: {acc * 100:.2f}%")
    print(f"{'='*60}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Simpan model & preprocessors
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoders, ENCODERS_PATH)

    print(f"\nModel disimpan ke  : {MODEL_PATH}")
    print(f"Scaler disimpan ke : {SCALER_PATH}")
    print(f"Encoders disimpan  : {ENCODERS_PATH}")
    print("\nTraining selesai!")

    return model, scaler, label_encoders, acc


if __name__ == "__main__":
    train()
