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
    Mengekspor metrik evaluasi ke file Excel (.xlsx) dengan format tabel kustom.

    Format kolom: Jenis Kerusakan | TP | TN | FP | FN | Precision (%) | Recall (%) | F1-Score (%)
    Termasuk baris ringkasan: Accuracy Model, Macro Average, Weighted Average.

    Args:
        y_test: Series label aktual.
        y_pred: Array label prediksi.
        output_dir: Direktori tujuan penyimpanan file Excel.
    """
    try:
        from openpyxl import load_workbook
        from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
        from openpyxl.utils import get_column_letter

        labels = sorted(list(set(y_test)))
        n_total = len(y_test)

        report_dict = classification_report(y_test, y_pred, output_dict=True, labels=labels)
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        # Hitung TP, TN, FP, FN per kelas dari confusion matrix
        rows = []
        for i, label in enumerate(labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = n_total - tp - fp - fn

            precision = report_dict[label]["precision"] * 100
            recall    = report_dict[label]["recall"]    * 100
            f1        = report_dict[label]["f1-score"]  * 100

            rows.append({
                "Jenis Kerusakan": label,
                "TP": int(tp),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "Precision (%)": round(precision, 2),
                "Recall (%)":    round(recall,    2),
                "F1-Score (%)":  round(f1,        2),
            })

        # Hitung akurasi keseluruhan
        acc = accuracy_score(y_test, y_pred)
        correct = int(round(acc * n_total))
        macro    = report_dict["macro avg"]
        weighted = report_dict["weighted avg"]

        # Baris ringkasan
        summary_rows = [
            {
                "Jenis Kerusakan": "Accuracy Model",
                "TP": f"{correct}/{n_total}",
                "TN": "-", "FP": "-", "FN": "-",
                "Precision (%)": "-",
                "Recall (%)":    "-",
                "F1-Score (%)":  f"{acc * 100:.2f}%",
            },
            {
                "Jenis Kerusakan": "Macro Average",
                "TP": "-", "TN": "-", "FP": "-", "FN": "-",
                "Precision (%)": round(macro["precision"]   * 100, 2),
                "Recall (%)":    round(macro["recall"]      * 100, 2),
                "F1-Score (%)":  round(macro["f1-score"]    * 100, 2),
            },
            {
                "Jenis Kerusakan": "Weighted Average",
                "TP": "-", "TN": "-", "FP": "-", "FN": "-",
                "Precision (%)": round(weighted["precision"] * 100, 2),
                "Recall (%)":    round(weighted["recall"]    * 100, 2),
                "F1-Score (%)":  round(weighted["f1-score"]  * 100, 2),
            },
        ]

        df = pd.DataFrame(rows + summary_rows)

        excel_path = os.path.join(output_dir, "evaluation_metrics.xlsx")
        df.to_excel(excel_path, sheet_name="Metrics", index=False)

        # ── Styling dengan openpyxl ────────────────────────────────────────────
        wb = load_workbook(excel_path)
        ws = wb["Metrics"]

        # Definisi style
        header_font  = Font(name="Calibri", bold=True, size=11, color="FFFFFF")
        summary_font = Font(name="Calibri", bold=True, size=11)
        normal_font  = Font(name="Calibri", size=11)
        header_fill  = PatternFill("solid", fgColor="2E4057")   # biru gelap
        summary_fill = PatternFill("solid", fgColor="D9E1F2")   # biru muda
        center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
        left_align   = Alignment(horizontal="left",   vertical="center")

        thin   = Side(style="thin", color="BFBFBF")
        border = Border(left=thin, right=thin, top=thin, bottom=thin)

        n_cols = ws.max_column
        n_rows = ws.max_row

        # Lebar kolom
        col_widths = [28, 8, 8, 8, 8, 16, 14, 16]
        for col_idx, width in enumerate(col_widths, start=1):
            ws.column_dimensions[get_column_letter(col_idx)].width = width

        # Format header (baris 1)
        for col_idx in range(1, n_cols + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.font      = header_font
            cell.fill      = header_fill
            cell.alignment = center_align
            cell.border    = border

        # Format baris data kelas (baris 2 s.d. len(rows)+1)
        for row_idx in range(2, len(rows) + 2):
            for col_idx in range(1, n_cols + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.font   = normal_font
                cell.border = border
                if col_idx == 1:
                    cell.alignment = left_align
                else:
                    cell.alignment = center_align

                # Format angka desimal dengan format lokal (#,##0.00)
                if isinstance(cell.value, float):
                    cell.number_format = '#,##0.00'

        # Format baris ringkasan (3 baris terakhir)
        summary_start = len(rows) + 2
        for row_idx in range(summary_start, n_rows + 1):
            for col_idx in range(1, n_cols + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.font   = summary_font
                cell.fill   = summary_fill
                cell.border = border
                if col_idx == 1:
                    cell.alignment = left_align
                else:
                    cell.alignment = center_align

                if isinstance(cell.value, float):
                    cell.number_format = '#,##0.00'

        # Tinggi baris header
        ws.row_dimensions[1].height = 30

        wb.save(excel_path)
        print(f"      Metrik evaluasi berhasil diekspor ke: {excel_path}")

    except Exception as e:
        print(f"      Gagal mengekspor metrik ke Excel: {e}")
