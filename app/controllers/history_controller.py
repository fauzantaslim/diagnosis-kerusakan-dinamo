from flask import request, jsonify, render_template
from flask_login import current_user
from app.services import history_service

def history_page():
    """Controller untuk render halaman riwayat diagnosis."""
    return render_template('pages/app/history.html')

def get_history():
    """Controller untuk mengambil riwayat diagnosis milik user yang sedang login."""
    try:
        page     = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        per_page = min(per_page, 100)   # Batasi maksimal 100 per halaman
    except ValueError:
        page, per_page = 1, 20

    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    data = history_service.get_history_by_user(
        user_id=current_user.id,
        page=page,
        per_page=per_page,
        start_date=start_date,
        end_date=end_date,
    )
    return jsonify({"success": True, "data": data}), 200

def get_history_detail(history_id: int):
    """Controller untuk mengambil detail satu record riwayat berdasarkan ID."""
    record = history_service.get_history_detail(history_id, current_user.id)
    if not record:
        return jsonify({"success": False, "message": "Data tidak ditemukan."}), 404
    return jsonify({"success": True, "data": record.to_dict()}), 200

def delete_history(history_id: int):
    """Controller untuk menghapus satu record riwayat."""
    deleted = history_service.delete_history(history_id, current_user.id)
    if not deleted:
        return jsonify({"success": False, "message": "Data tidak ditemukan."}), 404
    return jsonify({"success": True, "message": "Riwayat berhasil dihapus."}), 200
