"""
Modul visualisasi untuk Diagnosis Kerusakan Dinamo.
Berisi fungsi untuk membuat plot tree (Graphviz) dan confusion matrix (Matplotlib).
Dependency opsional — aman dijalankan meskipun graphviz/matplotlib belum terinstal.
"""

import os


def plot_tree(model, feature_cols, class_names, output_dir: str):
    """
    Membuat visualisasi salah satu Decision Tree dari model Random Forest.

    Menggunakan Graphviz untuk merender tree pertama (estimators_[0]) ke PNG.

    Args:
        model: Model RandomForest yang sudah di-train.
        feature_cols: List nama fitur.
        class_names: List nama kelas (sorted).
        output_dir: Direktori tujuan penyimpanan gambar.
    """
    print("\n[VIZ] Membuat visualisasi Decision Tree...")
    try:
        from sklearn.tree import export_graphviz
        import graphviz

        estimator = model.estimators_[0]

        dot_data = export_graphviz(
            estimator,
            out_file=None,
            feature_names=feature_cols,
            class_names=class_names,
            filled=False,       # Nonaktifkan warna pengisi node
            rounded=True,
            special_characters=True,
            max_depth=3,
        )

        # Hapus atribut warna sisa dari DOT string agar benar-benar hitam putih
        import re
        dot_data = re.sub(r'fillcolor="[^"]*",?\s*', '', dot_data)
        dot_data = re.sub(r'color="[^"]*",?\s*', '', dot_data)

        graph = graphviz.Source(dot_data)
        viz_path = os.path.join(output_dir, "rf_tree_viz")
        graph.render(viz_path, format="png", cleanup=True)
        print(f"      Visualisasi tree berhasil disimpan ke: {viz_path}.png")
    except ImportError:
        print("      Modul graphviz tidak ditemukan. Lewati visualisasi.")
        print("      Jalankan 'pip install graphviz' untuk mengaktifkannya.")
    except Exception as e:
        print(f"      Gagal membuat visualisasi graphviz: {e}")
        print("      Pastikan aplikasi Graphviz sudah diinstal di sistem dan ditambahkan ke PATH.")


def plot_confusion_matrix(y_test, y_pred, class_names, output_dir: str):
    """
    Membuat dan menyimpan plot Confusion Matrix ke file PNG.

    Args:
        y_test: Series label aktual.
        y_pred: Array label prediksi.
        class_names: List nama kelas (sorted).
        output_dir: Direktori tujuan penyimpanan gambar.
    """
    print("\n[VIZ] Membuat visualisasi Confusion Matrix...")
    try:
        import matplotlib
        matplotlib.use("Agg")  # Backend non-interaktif
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(cmap="Blues", ax=ax, xticks_rotation=45)

        plt.title("Confusion Matrix - Random Forest")
        plt.tight_layout()

        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()

        print(f"      Visualisasi Confusion Matrix berhasil disimpan ke: {cm_path}")
    except ImportError:
        print("      Modul matplotlib tidak ditemukan. Lewati visualisasi confusion matrix.")
        print("      Jalankan 'pip install matplotlib' untuk mengaktifkannya.")
    except Exception as e:
        print(f"      Gagal membuat visualisasi confusion matrix: {e}")
