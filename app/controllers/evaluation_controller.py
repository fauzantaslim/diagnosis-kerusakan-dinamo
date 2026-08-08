import os
from flask import jsonify, send_from_directory
from app.utils.jwt_utils import jwt_required
from app.services.evaluation_service import evaluate_model_pipeline

@jwt_required
def run_evaluation():
    """
    GET /api/evaluation/run
    Menjalankan pengujian model (GridSearchCV) dan mengembalikan metrik evaluasi.
    """
    try:
        result = evaluate_model_pipeline()
        if not result.get("success", False):
            return jsonify(result), 400
            
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Terjadi kesalahan saat pengujian model: {str(e)}"
        }), 500

def serve_shap_plot():
    """
    GET /api/evaluation/shap-plot
    Melayani file gambar SHAP summary plot.
    """
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml", "models"))
    return send_from_directory(model_dir, "shap_summary_plot.png")
