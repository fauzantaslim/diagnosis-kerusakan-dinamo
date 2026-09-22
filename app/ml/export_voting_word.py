"""
Export Tabel Voting Mayoritas - dari file data_word.xlsx.

Format kolom:
  No | Nama Klien/Perusahaan | Gejala Kerusakan | P1 ... Pn | Hasil Voting | Aktual | Evaluasi

- Gejala Kerusakan = nama gejala yang bernilai 1, dipisah koma
- X1-X11 dipetakan ke nama gejala
- Y sudah berupa kode angka (0-4)
- Label kode: 0=Bearing, 1=Gulungan, 2=Housing, 3=Shaft, 4=Terminal

Penggunaan:
    python -m app.ml.export_voting_word --max-trees 10
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from collections import Counter

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# =========================================================
# Path
# =========================================================
DATA_WORD_PATH = r"C:\Users\fauza\Documents\Kuliah\semester-7\PPAS\app_skripsi\data_word.xlsx"
MODEL_DIR      = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH     = os.path.join(MODEL_DIR, "rf_model.pkl")
OUTPUT_PATH    = os.path.join(MODEL_DIR, "voting_table_word.xlsx")

# Mapping X1-X11 ke nama gejala
FEAT_MAP = {
    "X1":  "Suara Bising Abnormal",
    "X2":  "Bau Hangus",
    "X3":  "Indikasi Overheating",
    "X4":  "Putaran Poros Seret",
    "X5":  "Getaran Berlebih",
    "X6":  "Terminal Overheating",
    "X7":  "Kipas Pendingin Rusak",
    "X8":  "Cooling Duct Tersumbat",
    "X9":  "Res. Isolasi Tidak Seimbang",
    "X10": "Res. Winding Tidak Seimbang",
    "X11": "Arus Antar Fasa Tidak Seimbang",
}
FEAT_COLS_WORD = list(FEAT_MAP.keys())   # ["X1", ..., "X11"]

# Kode kelas
LABEL_LEGEND = [
    ("0", "Kerusakan Bearing"),
    ("1", "Kerusakan Gulungan Stator"),
    ("2", "Kerusakan Housing Bearing"),
    ("3", "Kerusakan Shaft"),
    ("4", "Kerusakan Terminal"),
]

# Urutan fitur yang dipakai model (dari preprocessing.py FEATURE_COLS)
MODEL_FEAT_ORDER = [
    "suara_bising_abnormal",
    "bau_hangus",
    "indikasi_overheating",
    "putaran_poros_seret",
    "getaran_berlebih",
    "terminal_overheating",
    "kipas_pendingin_rusak",
    "cooling_duct_tersumbat",
    "resistansi_isolasi_tidak_seimbang",
    "resistansi_winding_tidak_seimbang",
    "arus_antar_fasa_tidak_seimbang",
]


def load_word_data(path):
    """Membaca data_word.xlsx, bersihkan baris korup, kembalikan X dan y."""
    df = pd.read_excel(path)

    # Hapus baris yang kolom X1-nya bukan angka (baris korup/separator)
    df = df[pd.to_numeric(df["X1"], errors="coerce").notna()].copy()

    # Konversi semua kolom X ke int
    for col in FEAT_COLS_WORD:
        df[col] = df[col].astype(int)

    # Konversi Y ke string (kode kelas)
    df["Y"] = df["Y"].astype(int).astype(str)

    df = df.reset_index(drop=True)
    return df


def gejala_aktif(row):
    """Kembalikan nama gejala yang bernilai 1 untuk satu baris."""
    aktif = [FEAT_MAP[col] for col in FEAT_COLS_WORD if row[col] == 1]
    return ", ".join(aktif) if aktif else "-"


def build_voting_table(model, df, max_trees=None):
    """Bangun tabel voting untuk data_word.xlsx."""
    estimators = model.estimators_
    n_trees    = len(estimators)
    n_shown    = min(max_trees, n_trees) if max_trees else n_trees
    classes    = model.classes_

    print(f"  Jumlah pohon dalam model : {n_trees}")
    print(f"  Pohon yang ditampilkan   : {n_shown}")

    # Susun fitur sesuai urutan yang dipakai model
    X = df[FEAT_COLS_WORD].values   # shape (n, 11)

    # Mapping label penuh → kode angka (dipakai di kedua cabang)
    label_to_code = {label: code for code, label in LABEL_LEGEND}

    # Prediksi semua pohon → (n_trees, n_samples)
    tree_preds_raw = np.array([est.predict(X) for est in estimators])

    # Langkah 1: decode indeks numerik → label string penuh via classes_
    if tree_preds_raw.dtype.kind in ("i", "u", "f"):
        decoded = np.vectorize(lambda idx: classes[int(idx)])(tree_preds_raw)
    else:
        decoded = tree_preds_raw

    # Langkah 2: petakan label penuh → kode angka ("0","1",...)
    tree_preds = np.vectorize(lambda lbl: label_to_code.get(lbl, lbl))(decoded)

    rows = []
    for i in range(len(df)):
        row = {
            "No":                  i + 1,
            "Nama Klien":          df.iloc[i].get("Nama Klien/Perusahaan", "-"),
            "Gejala Kerusakan":    gejala_aktif(df.iloc[i]),
        }

        votes = []
        for t in range(n_shown):
            pred = tree_preds[t, i]
            row[f"P{t + 1}"] = pred
            votes.append(pred)
        for t in range(n_shown, n_trees):
            votes.append(tree_preds[t, i])

        hasil_voting = Counter(votes).most_common(1)[0][0]
        aktual       = df.iloc[i]["Y"]
        evaluasi     = "Benar" if hasil_voting == aktual else "Salah"

        row["Hasil Voting"] = hasil_voting
        row["Aktual"]       = aktual
        row["Evaluasi"]     = evaluasi
        rows.append(row)

    df_table = pd.DataFrame(rows).set_index("No")
    return df_table, n_shown


def export_to_excel(df_table, output_path, n_shown):
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_table.to_excel(writer, sheet_name="Voting Mayoritas", index=True)
        ws = writer.sheets["Voting Mayoritas"]

        fill_id   = PatternFill("solid", fgColor="1F4E79")
        fill_nm   = PatternFill("solid", fgColor="1F4E79")
        fill_feat = PatternFill("solid", fgColor="2E75B6")
        fill_tree = PatternFill("solid", fgColor="375623")
        fill_eval = PatternFill("solid", fgColor="7F6000")
        fill_ok   = PatternFill("solid", fgColor="E2EFDA")
        fill_ng   = PatternFill("solid", fgColor="FCE4D6")

        white_bold = Font(color="FFFFFF", bold=True, size=9)
        small_font = Font(size=8)
        center     = Alignment(horizontal="center", vertical="center", wrap_text=True)
        left_wrap  = Alignment(horizontal="left",   vertical="center", wrap_text=True)
        thin = Border(
            left=Side(style="thin"),  right=Side(style="thin"),
            top=Side(style="thin"),   bottom=Side(style="thin"),
        )

        max_col = ws.max_column
        max_row = ws.max_row

        # Header baris 1
        for col_idx in range(1, max_col + 1):
            cell = ws.cell(row=1, column=col_idx)
            val  = str(cell.value or "")
            if val in ("No", "Nama Klien"):
                cell.fill = fill_id
            elif val == "Gejala Kerusakan":
                cell.fill = fill_feat
            elif val.startswith("P") and val[1:].isdigit():
                cell.fill = fill_tree
            else:
                cell.fill = fill_eval
            cell.font      = white_bold
            cell.alignment = center
            cell.border    = thin

        # Kolom Evaluasi: 1(No)+1(NamaKlien)+1(Gejala)+n_shown(pohon)+3(Hasil,Aktual,Eval)
        eval_col_idx = 1 + 1 + 1 + n_shown + 3

        # Baris data
        for row_idx in range(2, max_row + 1):
            evaluasi = str(ws.cell(row=row_idx, column=eval_col_idx).value or "")
            row_fill = fill_ok if evaluasi == "Benar" else fill_ng

            for col_idx in range(1, max_col + 1):
                cell        = ws.cell(row=row_idx, column=col_idx)
                cell.fill   = row_fill
                cell.border = thin
                if col_idx == 3:     # Gejala Kerusakan (kolom ke-3)
                    cell.font      = Font(size=8)
                    cell.alignment = left_wrap
                else:
                    cell.font      = small_font
                    cell.alignment = center

        # Lebar kolom
        ws.column_dimensions[get_column_letter(1)].width = 5    # No
        ws.column_dimensions[get_column_letter(2)].width = 15   # Nama Klien
        ws.column_dimensions[get_column_letter(3)].width = 55   # Gejala Kerusakan
        for c in range(4, 4 + n_shown):
            ws.column_dimensions[get_column_letter(c)].width = 6  # pohon
        for c in range(4 + n_shown, max_col + 1):
            ws.column_dimensions[get_column_letter(c)].width = 12

        ws.row_dimensions[1].height = 40
        ws.freeze_panes = f"{get_column_letter(4)}2"

        # Sheet Keterangan Kelas
        ws_lg = writer.book.create_sheet(title="Keterangan Kelas")
        ws_lg.append(["Kode", "Kelas / Label"])
        for code, label in LABEL_LEGEND:
            ws_lg.append([code, label])

        fill_lg_h = PatternFill("solid", fgColor="1F4E79")
        for ci in range(1, 3):
            cell = ws_lg.cell(row=1, column=ci)
            cell.fill = fill_lg_h
            cell.font = Font(color="FFFFFF", bold=True, size=10)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thin
        for ri in range(2, 2 + len(LABEL_LEGEND)):
            for ci in range(1, 3):
                cell = ws_lg.cell(row=ri, column=ci)
                cell.alignment = Alignment(horizontal="left", vertical="center")
                cell.border = thin
                cell.font = Font(size=10)
        ws_lg.column_dimensions["A"].width = 8
        ws_lg.column_dimensions["B"].width = 30

    print(f"  File disimpan ke: {output_path}")


def main(max_trees=None):
    print("=" * 60)
    print("  EXPORT: Tabel Voting - data_word.xlsx")
    print("=" * 60)

    print(f"\n[1/4] Memuat model dari: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("  ERROR: Jalankan 'python -m app.ml.train' terlebih dahulu.")
        sys.exit(1)
    model = joblib.load(MODEL_PATH)
    print(f"  Model dimuat. Jumlah pohon: {len(model.estimators_)}")

    print(f"\n[2/4] Memuat data dari: {DATA_WORD_PATH}")
    df = load_word_data(DATA_WORD_PATH)
    print(f"  Data valid: {len(df)} baris")
    print(f"  Kolom Y unik: {sorted(df['Y'].unique())}")

    print(f"\n[3/4] Membangun tabel voting...")
    df_table, n_shown = build_voting_table(model, df, max_trees=max_trees)

    n_benar = (df_table["Evaluasi"] == "Benar").sum()
    n_salah = (df_table["Evaluasi"] == "Salah").sum()
    acc     = n_benar / len(df_table) * 100

    print(f"\n  Total data  : {len(df_table)} baris")
    print(f"  Akurasi     : {acc:.2f}%")
    print(f"  Benar       : {n_benar}")
    print(f"  Salah       : {n_salah}")

    preview_cols = (
        ["Nama Klien", "Gejala Kerusakan"]
        + [f"P{i}" for i in range(1, min(4, n_shown + 1))]
        + ["Hasil Voting", "Aktual", "Evaluasi"]
    )
    pd.set_option("display.max_colwidth", 50)
    print(f"\n  Preview (3 baris pertama):")
    print(df_table[preview_cols].head(3).to_string())

    print(f"\n[4/4] Mengekspor ke Excel...")
    export_to_excel(df_table, OUTPUT_PATH, n_shown)
    print("\nSelesai!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export tabel voting mayoritas dari data_word.xlsx."
    )
    parser.add_argument(
        "--max-trees", type=int, default=None, metavar="N",
        help="Jumlah kolom pohon (default: semua). Contoh: --max-trees 10",
    )
    args = parser.parse_args()
    main(max_trees=args.max_trees)
