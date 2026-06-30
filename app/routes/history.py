from flask import Blueprint, request, jsonify, render_template
from flask_login import login_required, current_user
from app.services import history_service

history_bp = Blueprint('history_bp', __name__)


@history_bp.route('/', methods=['GET'])
@login_required
def history_page():
    """Render halaman riwayat diagnosis."""
    return render_template('pages/app/history.html')


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@history_bp.route('/api', methods=['GET'])
@login_required
def get_history():
    """
    Mengambil riwayat diagnosis milik user yang sedang login (dengan pagination).

    Query params:
        page     (int, default=1)   — halaman saat ini
        per_page (int, default=20)  — jumlah data per halaman

    Response JSON:
    {
        "success" : true,
        "data"    : {
            "items"   : [ { ...history_dict... }, ... ],
            "total"   : 42,
            "page"    : 1,
            "per_page": 20,
            "pages"   : 3
        }
    }
    """
    try:
        page     = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        per_page = min(per_page, 100)   # Batasi maksimal 100 per halaman
    except ValueError:
        page, per_page = 1, 20

    data = history_service.get_history_by_user(
        user_id=current_user.id,
        page=page,
        per_page=per_page,
    )
    return jsonify({"success": True, "data": data}), 200


@history_bp.route('/api/<int:history_id>', methods=['GET'])
@login_required
def get_history_detail(history_id: int):
    """
    Mengambil detail satu record riwayat berdasarkan ID.

    Response JSON:
    {
        "success": true,
        "data"   : { ...history_dict... }
    }
    """
    record = history_service.get_history_detail(history_id, current_user.id)
    if not record:
        return jsonify({"success": False, "message": "Data tidak ditemukan."}), 404
    return jsonify({"success": True, "data": record.to_dict()}), 200


@history_bp.route('/api/<int:history_id>', methods=['DELETE'])
@login_required
def delete_history(history_id: int):
    """
    Menghapus satu record riwayat.

    Response JSON:
    {
        "success": true,
        "message": "Riwayat berhasil dihapus."
    }
    """
    deleted = history_service.delete_history(history_id, current_user.id)
    if not deleted:
        return jsonify({"success": False, "message": "Data tidak ditemukan."}), 404
    return jsonify({"success": True, "message": "Riwayat berhasil dihapus."}), 200
