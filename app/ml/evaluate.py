"""
Modul evaluasi model untuk Diagnosis Kerusakan Dinamo.
Berisi fungsi untuk menghitung metrik performa dan mengekspornya ke Excel.
"""

import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test):
    """
    Mengevaluasi performa model pada test set.

    Mencetak OOB score, akurasi, classification report, dan confusion matrix.

    Args:
        model: Model RandomForest yang sudah di-train (harus punya oob_score_).
        X_test: DataFrame fitur test set.
        y_test: Series label test set.

    Returns:
        tuple: (y_pred, accuracy) — prediksi dan skor akurasi.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print(f"  OOB SCORE       : {model.oob_score_ * 100:.2f}%")
    print(f"  AKURASI TEST SET: {acc * 100:.2f}%")
    print(f"{'=' * 60}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return y_pred, acc


def export_metrics_to_excel(y_test, y_pred, output_dir: str):
    """
    Mengekspor classification report ke file Excel (.xlsx).

    Args:
        y_test: Series label aktual.
        y_pred: Array label prediksi.
        output_dir: Direktori tujuan penyimpanan file Excel.
    """
    try:
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        df_metrics = pd.DataFrame(report_dict).transpose()
        df_metrics = df_metrics.round(4)

        excel_path = os.path.join(output_dir, "evaluation_metrics.xlsx")
        df_metrics.to_excel(excel_path, sheet_name="Metrics")
        print(f"      Metrik evaluasi berhasil diekspor ke: {excel_path}")
    except Exception as e:
        print(f"      Gagal mengekspor metrik ke Excel: {e}")
