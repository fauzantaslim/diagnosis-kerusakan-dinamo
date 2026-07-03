from app.ml import predict as ml_predict
from app.ml import feature_importance as ml_fi
from app.services import history_service

def predict_diagnosis(user_id, data):
    """
    Melakukan prediksi diagnosis dan menyimpan riwayat.
    Returns: dictionary hasil prediksi.
    """
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
        
    # Tambahkan global permutation feature importance
    result["feature_importances"] = ml_fi.get_feature_importances(top_n=5)
    
    return result
