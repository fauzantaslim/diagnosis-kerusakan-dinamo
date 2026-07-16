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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

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

    # 4. Train / Test split
    print("\n[4/5] Membagi data training dan testing (80:20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Training sebelum SMOTE: {len(X_train)} | Testing: {len(X_test)}")

    # 4.5. SMOTE
    print("      Menerapkan SMOTE pada data training...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"      Training sesudah SMOTE: {len(X_train)}")

    # 5. Training model
    print("\n[5/5] Melatih model Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        oob_score=True,
    )
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"  OOB SCORE       : {model.oob_score_ * 100:.2f}%")
    print(f"  AKURASI TEST SET: {acc * 100:.2f}%")
    print(f"{'='*60}")
    print("\nClassification Report:")
    report_str = classification_report(y_test, y_pred)
    print(report_str)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- SHAP GLOBAL FEATURE IMPORTANCE ---
    print("\n[SHAP] Menghitung Global Feature Importance...")
    try:
        import shap
        import numpy as np
        
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
                
        global_shap_importances = [
            {"feature": feat, "importance": imp}
            for feat, imp in zip(FEATURE_COLS, mean_abs_shap)
        ]
        global_shap_importances.sort(key=lambda x: x["importance"], reverse=True)
        
        print("\n--- Global Feature Importance (SHAP) ---")
        for i, item in enumerate(global_shap_importances[:15]):
            print(f"  {i+1}. {item['feature']:<30} : {item['importance']:.4f}")
            
    except ImportError:
        print("      Modul shap tidak ditemukan. Jalankan 'pip install shap'.")
    except Exception as e:
        print(f"      Gagal menghitung SHAP: {e}")


    # Simpan metrik ke file Excel (.xlsx)
    try:
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        df_metrics = pd.DataFrame(report_dict).transpose()
        df_metrics = df_metrics.round(4) # Rapikan angka desimal
        excel_path = os.path.join(MODEL_DIR, "evaluation_metrics.xlsx")
        df_metrics.to_excel(excel_path, sheet_name="Metrics")
        print(f"      Metrik evaluasi berhasil diekspor ke: {excel_path}")
    except Exception as e:
        print(f"      Gagal mengekspor metrik ke Excel: {e}")

    # Simpan model & preprocessors
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoders, ENCODERS_PATH)

    print(f"\nModel disimpan ke  : {MODEL_PATH}")
    print(f"Encoders disimpan  : {ENCODERS_PATH}")

    # Visualisasi dengan Graphviz
    print("\n[6/5] Membuat visualisasi salah satu tree (Decision Tree)...")
    try:
        from sklearn.tree import export_graphviz
        import graphviz
        
        # Ekstrak tree pertama (indeks 0) dari Random Forest
        estimator = model.estimators_[0]
        
        # Ekspor menjadi string format dot
        dot_data = export_graphviz(
            estimator, 
            out_file=None, 
            feature_names=FEATURE_COLS,
            class_names=sorted(y.unique().astype(str)),
            filled=True, 
            rounded=True, 
            special_characters=True,
            # max_depth=4,  
        )
        
        graph = graphviz.Source(dot_data)
        
        # Render ke bentuk gambar PNG
        viz_path = os.path.join(MODEL_DIR, "rf_tree_viz")
        graph.render(viz_path, format="png", cleanup=True)
        print(f"      Visualisasi tree berhasil disimpan ke: {viz_path}.png")
    except ImportError:
        print("      Modul graphviz tidak ditemukan. Lewati visualisasi.")
        print("      Jalankan 'pip install graphviz' untuk mengaktifkannya.")
    except Exception as e:
        print(f"      Gagal membuat visualisasi graphviz: {e}")
        print("      Pastikan aplikasi Graphviz sudah diinstal di sistem (Windows/Linux) dan ditambahkan ke PATH.")

    # Visualisasi Confusion Matrix
    print("\n[7/5] Membuat visualisasi Confusion Matrix...")
    try:
        import matplotlib
        matplotlib.use('Agg') # Gunakan backend non-interaktif
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(y.unique().astype(str))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        
        # Plot dengan ukuran figure yang cukup besar
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
        
        plt.title('Confusion Matrix - Random Forest')
        plt.tight_layout()
        
        cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        
        print(f"      Visualisasi Confusion Matrix berhasil disimpan ke: {cm_path}")
    except ImportError:
        print("      Modul matplotlib tidak ditemukan. Lewati visualisasi confusion matrix.")
        print("      Jalankan 'pip install matplotlib' untuk mengaktifkannya.")
    except Exception as e:
        print(f"      Gagal membuat visualisasi confusion matrix: {e}")

    print("\nTraining selesai!")

    return model, label_encoders, acc


if __name__ == "__main__":
    train()
