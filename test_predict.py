"""
Script test kecepatan prediksi.
Jalankan: .venv\Scripts\python.exe test_predict.py
"""
import time

print("=== TEST PREDIKSI ===\n")

# --- 1. Load artifacts (model + TreeExplainer) ---
t0 = time.time()
from app.ml.predict import predict, _load_artifacts
_load_artifacts()
t1 = time.time()
print(f"[1] Load model + build TreeExplainer : {t1 - t0:.2f}s")

# --- 2. Input dummy ---
input_data = {
    "suara_bising_abnormal":            "YA",
    "bau_hangus":                       "YA",
    "indikasi_overheating":             "TIDAK",
    "putaran_poros_seret":              "TIDAK",
    "getaran_berlebih":                 "TIDAK",
    "terminal_overheating":             "TIDAK",
    "kipas_pendingin_rusak":            "TIDAK",
    "cooling_duct_tersumbat":           "TIDAK",
    "resistansi_isolasi_tidak_seimbang":"YA",
    "resistansi_winding_tidak_seimbang":"YA",
    "arus_antar_fasa_tidak_seimbang":   "TIDAK",
}

# --- 3. Prediksi pertama ---
t2 = time.time()
result = predict(input_data)
t3 = time.time()
print(f"[2] Prediksi pertama (cold - SHAP)   : {t3 - t2:.2f}s")

# --- 4. Prediksi kedua (cache sudah aktif) ---
t4 = time.time()
result2 = predict(input_data)
t5 = time.time()
print(f"[3] Prediksi kedua  (warm - cache)   : {t5 - t4:.2f}s")

# --- 5. Tampilkan hasil ---
print()
print("=== HASIL ===")
print(f"Diagnosis  : {result['diagnosis']}")
print(f"Confidence : {result['confidence'] * 100:.2f}%")
print(f"Top SHAP   : {result['local_importances']}")
