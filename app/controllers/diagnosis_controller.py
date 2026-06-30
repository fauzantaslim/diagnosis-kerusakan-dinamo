from flask import request, jsonify, render_template
from flask_login import current_user
from app.services import diagnosis_service

def identify_page():
    """Controller untuk render halaman form identifikasi kerusakan dinamo."""
    return render_template('pages/app/identify.html')

def identify():
    """Controller untuk mendiagnosis kerusakan dinamo berdasarkan input gejala & pengukuran."""
    data = request.get_json(silent=True) or request.form.to_dict()

    if not data:
        return jsonify({
            "success": False,
            "message": "Request body kosong. Kirim data dalam format JSON.",
        }), 400

    try:
        result = diagnosis_service.predict_diagnosis(current_user.id, data)
    except FileNotFoundError as e:
        return jsonify({
            "success": False,
            "message": str(e),
        }), 503
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Terjadi kesalahan saat prediksi: {str(e)}",
        }), 500

    return jsonify({
        "success"       : True,
        "diagnosis"     : result["diagnosis"],
        "confidence"    : round(result["confidence"], 4),
        "confidence_pct": f"{result['confidence'] * 100:.2f}%",
        "probabilities" : {
            label: round(prob, 4)
            for label, prob in sorted(
                result["probabilities"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        },
        "feature_importances": result.get("feature_importances", []),
    }), 200

