from flask import request, jsonify
from app.utils.jwt_utils import jwt_required, get_current_user_id
from app.services import history_service


@jwt_required
def history_page():
    """Render halaman riwayat diagnosis (SSR)."""
    from flask import render_template
    return render_template('pages/app/history.html')


@jwt_required
def get_history():
    """Ambil riwayat diagnosis milik user yang sedang login."""
    try:
        page     = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        per_page = min(per_page, 100)
    except ValueError:
        page, per_page = 1, 20

    start_date = request.args.get('start_date')
    end_date   = request.args.get('end_date')

    data = history_service.get_history_by_user(
        user_id=get_current_user_id(),
        page=page,
        per_page=per_page,
        start_date=start_date,
        end_date=end_date,
    )
    return jsonify({"success": True, "data": data}), 200


@jwt_required
def get_history_detail(history_id: int):
    """Ambil detail satu record riwayat."""
    record = history_service.get_history_detail(history_id, get_current_user_id())
    if not record:
        return jsonify({"success": False, "message": "Data tidak ditemukan."}), 404
    return jsonify({"success": True, "data": record.to_dict()}), 200


@jwt_required
def delete_history(history_id: int):
    """Hapus satu record riwayat."""
    deleted = history_service.delete_history(history_id, get_current_user_id())
    if not deleted:
        return jsonify({"success": False, "message": "Data tidak ditemukan."}), 404
    return jsonify({"success": True, "message": "Riwayat berhasil dihapus."}), 200
