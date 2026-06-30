from flask import Blueprint, request, jsonify, render_template
from flask_login import login_required, current_user
from app.ml import predict as ml_predict
from app.ml import feature_importance as ml_fi
from app.services import history_service

diagnosis_bp = Blueprint('diagnosis_bp', __name__)


@diagnosis_bp.route('/', methods=['GET'])
@diagnosis_bp.route('/identify-page', methods=['GET'])
@login_required
def identify_page():
    """Render halaman form identifikasi kerusakan dinamo."""
    return render_template('pages/app/identify.html')


@diagnosis_bp.route('/identify', methods=['POST'])
@login_required
def identify():
    """
    API untuk mendiagnosis kerusakan dinamo berdasarkan input gejala & pengukuran.

    Request (JSON atau form-data):
    {
        "jenis_mesin"              : "Motor Induksi 3 Fasa",
        "daya_hp_kw"               : 7.5,
        "jumlah_pole"              : 4,

        // Gejala — nilai: "Ya" atau "Tidak"
        "suara_bising_abnormal"    : "Ya",
        "getaran_berlebih"         : "Tidak",
        "motor_cepat_panas"        : "Ya",
        "arus_melebihi_normal"     : "Tidak",
        "tegangan_tidak_stabil"    : "Tidak",
        "putaran_menurun"          : "Tidak",
        "sulit_start"              : "Tidak",
        "sering_trip_mcb"          : "Tidak",
        "trip_overload_relay"      : "Tidak",
        "efisiensi_menurun"        : "Tidak",
        "bau_hangus"               : "Tidak",
        "intermittent_stopping"    : "Tidak",
        "warna_gulungan_berubah"   : "Tidak",
        "kipas_pendingin_rusak"    : "Tidak",
        "terminal_terbakar"        : "Tidak",
        "bearing_aus_pecah"        : "Ya",
        "housing_bearing_aus"      : "Tidak",
        "poros_shaft_aus"          : "Tidak",
        "kebocoran_pelumas"        : "Tidak",
        "keretakan_dudukan"        : "Tidak",
        "sumbatan_sirip_pendingin" : "Tidak",
        "lubang_spi_aus"           : "Tidak",

        // Pengukuran numerik
        "temperatur_c"                   : 85.0,
        "arus_a"                         : 14.5,
        "tegangan_v"                     : 380.0,
        "resistansi_isolasi_mohm"        : 1.5,
        "kecepatan_putaran_rpm"          : 1450,
        "ketidakseimbangan_arus_pct"     : 3.2,
        "ketidakseimbangan_tegangan_pct" : 1.1,
        "faktor_daya"                    : 0.85
    }

    Response JSON (sukses):
    {
        "success"      : true,
        "diagnosis"    : "Bearing Rusak",
        "confidence"   : 0.96,
        "confidence_pct": "96.00%",
        "probabilities": {
            "Bearing Rusak"       : 0.96,
            "Motor Normal"        : 0.02,
            ...
        }
    }

    Response JSON (error):
    {
        "success": false,
        "message": "..."
    }
    """
    data = request.get_json(silent=True) or request.form.to_dict()

    if not data:
        return jsonify({
            "success": False,
            "message": "Request body kosong. Kirim data dalam format JSON.",
        }), 400

    try:
        result = ml_predict.predict(data)
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

    # Simpan ke history
    try:
        history_service.save_history(
            user_id=current_user.id,
            input_data=data,
            result=result,
        )
    except Exception:
        pass  # Jangan gagalkan response jika penyimpanan bermasalah

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
    }), 200


@diagnosis_bp.route('/labels', methods=['GET'])
@login_required
def get_labels():
    """
    Mengembalikan daftar semua label kelas yang dikenali model.

    Response JSON:
    {
        "success": true,
        "labels": ["Bearing Rusak", "Gangguan Isolasi", ...]
    }
    """
    try:
        info = ml_predict.get_model_info()
        return jsonify({
            "success": True,
            "labels" : info["classes"],
            "total"  : info["n_classes"],
        }), 200
    except FileNotFoundError as e:
        return jsonify({"success": False, "message": str(e)}), 503


@diagnosis_bp.route('/feature-importance', methods=['GET'])
@login_required
def get_feature_importance():
    """
    Mengembalikan daftar fitur terpenting (top-N) dari model Random Forest.

    Query param:
        top_n (opsional, default=10) — jumlah fitur yang dikembalikan

    Response JSON:
    {
        "success": true,
        "features": [
            { "feature": "temperatur_c", "importance": 18.52 },
            ...
        ]
    }
    """
    try:
        top_n = int(request.args.get('top_n', 10))
    except ValueError:
        top_n = 10

    features = ml_fi.get_feature_importances(top_n=top_n)

    if not features:
        return jsonify({
            "success": False,
            "message": "Model belum tersedia. Jalankan training terlebih dahulu.",
        }), 503

    return jsonify({
        "success" : True,
        "features": features,
    }), 200


@diagnosis_bp.route('/model-info', methods=['GET'])
@login_required
def get_model_info():
    """
    Mengembalikan info teknis model yang sedang aktif.

    Response JSON:
    {
        "success": true,
        "model": {
            "n_estimators": 200,
            "n_classes": 10,
            "n_features": 33,
            "classes": [...]
        }
    }
    """
    try:
        info = ml_predict.get_model_info()
        return jsonify({"success": True, "model": info}), 200
    except FileNotFoundError as e:
        return jsonify({"success": False, "message": str(e)}), 503
