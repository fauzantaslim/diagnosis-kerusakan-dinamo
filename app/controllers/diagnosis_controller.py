from flask import request, jsonify, render_template
from app.utils.jwt_utils import jwt_required, get_current_user_id
from app.services import random_forest_service, history_service


def identify_page():
    """Render halaman form identifikasi (tidak butuh JWT — halaman publik)."""
    return render_template('pages/app/identify.html')


@jwt_required
def identify():
    """
    POST /diagnosis/identify
    Mendiagnosis kerusakan dinamo menggunakan Random Forest manual.
    Membutuhkan JWT yang valid.
    """
    data = request.get_json(silent=True) or request.form.to_dict()

    if not data:
        return jsonify({
            "success": False,
            "message": "Request body kosong. Kirim data dalam format JSON.",
        }), 400

    # Validasi: minimal satu gejala harus Ya
    SYMPTOM_FIELDS = [
        'suara_bising_abnormal', 'bau_hangus', 'indikasi_overheating',
        'putaran_poros_seret', 'getaran_berlebih', 'terminal_overheating',
        'kipas_pendingin_rusak', 'cooling_duct_tersumbat',
        'resistansi_isolasi_tidak_seimbang', 'resistansi_winding_tidak_seimbang',
        'arus_antar_fasa_tidak_seimbang',
    ]
    has_symptom = any(
        str(data.get(f, '')).strip().upper() in ('YA', '1', 'TRUE')
        for f in SYMPTOM_FIELDS
    )
    if not has_symptom:
        return jsonify({
            "success": False,
            "message": "Validasi gagal: Mohon pilih minimal satu gejala yang teridentifikasi.",
        }), 400

    # Validasi jumlah_pole (opsional)
    jumlah_pole_raw = data.get("jumlah_pole", "")
    if jumlah_pole_raw not in ("", None):
        try:
            jp = int(jumlah_pole_raw)
            if jp < 2:
                return jsonify({
                    "success": False,
                    "message": "Validasi gagal: Jumlah Pole harus minimal 2.",
                }), 400
            if jp % 2 != 0:
                return jsonify({
                    "success": False,
                    "message": "Validasi gagal: Jumlah Pole harus genap (2, 4, 6, 8, ...).",
                }), 400
        except (ValueError, TypeError):
            pass

    try:
        result = random_forest_service.predict_with_detail(data)
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 422
    except Exception as e:
        return jsonify({"success": False, "message": f"Kesalahan prediksi: {str(e)}"}), 500

    # Simpan ke history
    try:
        user_id = get_current_user_id()
        if user_id:
            history_service.save_history(
                user_id=user_id,
                input_data=data,
                result={
                    "diagnosis"    : result["diagnosis"],
                    "confidence"   : result["confidence"],
                    "probabilities": result["probabilities"],
                },
            )
    except Exception:
        pass

    return jsonify({
        "success"            : True,
        "diagnosis"          : result["diagnosis"],
        "confidence"         : result["confidence"],
        "confidence_pct"     : result["confidence_pct"],
        "probabilities"      : result["probabilities"],
        "feature_importances": result["feature_importances"],
        "tree_votes"         : [
            {"tree_id": tv["tree_id"], "vote": tv["vote"], "path_length": tv["path_length"]}
            for tv in result["tree_votes"]
        ],
        "calculation_detail" : result["calculation_detail"],
        "session_id"         : result["session_id"],
        "split_info"         : result["split_info"],
        "n_training_samples" : result["n_training_samples"],
    }), 200
