from app.ml import predict as ml_predict
from app.ml import feature_importance as ml_fi
from app.services import history_service

def predict_diagnosis(user_id, data):
    """
    Melakukan prediksi diagnosis dan menyimpan riwayat.
    Returns: dictionary hasil prediksi.
    """
    # Validasi minimal ada satu gejala yang dipilih
    has_symptom = any(value == "Ya" for key, value in data.items())
    if not has_symptom:
        return {
            "success": False,
            "message": "Validasi gagal: Mohon pilih minimal satu gejala yang teridentifikasi."
        }

    # Validasi jumlah pole: minimal 2 dan harus genap
    jumlah_pole = data.get("jumlah_pole", 0)
    try:
        jumlah_pole = int(jumlah_pole)
    except (ValueError, TypeError):
        jumlah_pole = 0

    if jumlah_pole < 2:
        return {
            "success": False,
            "message": "Validasi gagal: Jumlah Pole harus minimal 2."
        }
    if jumlah_pole % 2 != 0:
        return {
            "success": False,
            "message": "Validasi gagal: Jumlah Pole harus berupa angka genap (2, 4, 6, 8, ...)."
        }

    # Prediksi menggunakan ML
    result = ml_predict.predict(data)
    
    # Simpan ke history
    try:
        history_service.save_history(
            user_id=user_id,
            input_data=data,
            result=result,
        )
    except Exception:
        pass  # Jangan gagalkan response jika penyimpanan bermasalah
        
    # Gunakan local feature importance (dari predict.py)
    result["feature_importances"] = result.get("local_importances", [])
    
    # Hapus key yang tidak perlu di return ke HTTP
    if "local_importances" in result:
        del result["local_importances"]
        
    return result
