"""
Service untuk mengelola data riwayat diagnosis.
"""
from app.models import db
from app.models.history import History
from datetime import datetime


def get_history_by_user(user_id: int, page: int = 1, per_page: int = 20, start_date: str = None, end_date: str = None) -> dict:
    """
    Mengambil riwayat diagnosis milik user tertentu dengan pagination dan filter tanggal.

    Returns:
        dict dengan key: items, total, page, per_page, pages
    """
    query = History.query.filter_by(user_id=user_id)

    if start_date:
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            query = query.filter(History.tanggal >= start_dt)
        except ValueError:
            pass

    if end_date:
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
            query = query.filter(History.tanggal <= end_dt)
        except ValueError:
            pass

    pagination = (
        query
        .order_by(History.tanggal.desc())
        .paginate(page=page, per_page=per_page, error_out=False)
    )
    return {
        "items"   : [h.to_dict() for h in pagination.items],
        "total"   : pagination.total,
        "page"    : pagination.page,
        "per_page": pagination.per_page,
        "pages"   : pagination.pages,
    }


def get_history_detail(history_id: int, user_id: int):
    """
    Mengambil satu record riwayat berdasarkan ID.
    Hanya mengembalikan data milik user_id yang bersangkutan.

    Returns:
        History object atau None
    """
    return History.query.filter_by(id=history_id, user_id=user_id).first()


def save_history(user_id: int, input_data: dict, result: dict) -> History:
    """
    Menyimpan satu hasil diagnosis ke tabel history.

    Args:
        user_id   : ID user yang melakukan diagnosis
        input_data: dict berisi input form dari user
        result    : dict hasil dari ml_predict.predict()

    Returns:
        Objek History yang sudah tersimpan
    """
    def _bool(val) -> bool:
        return str(val).lower() in ("ya", "1", "true")

    record = History(
        user_id=user_id,
        # Spesifikasi
        daya=_safe_float(input_data.get("daya")),
        jumlah_pole=_safe_int(input_data.get("jumlah_pole")),
        kecepatan_putaran_rpm=_safe_float(input_data.get("kecepatan_putaran_rpm")),
        # Gejala
        suara_bising_abnormal=_bool(input_data.get("suara_bising_abnormal", "Tidak")),
        bau_hangus=_bool(input_data.get("bau_hangus", "Tidak")),
        indikasi_overheating=_bool(input_data.get("indikasi_overheating", "Tidak")),
        putaran_poros_seret=_bool(input_data.get("putaran_poros_seret", "Tidak")),
        getaran_berlebih=_bool(input_data.get("getaran_berlebih", "Tidak")),
        terminal_overheating=_bool(input_data.get("terminal_overheating", "Tidak")),
        kipas_pendingin_rusak=_bool(input_data.get("kipas_pendingin_rusak", "Tidak")),
        cooling_duct_tersumbat=_bool(input_data.get("cooling_duct_tersumbat", "Tidak")),
        resistansi_isolasi_tidak_seimbang=_bool(input_data.get("resistansi_isolasi_tidak_seimbang", "Tidak")),
        resistansi_winding_tidak_seimbang=_bool(input_data.get("resistansi_winding_tidak_seimbang", "Tidak")),
        arus_antar_fasa_tidak_seimbang=_bool(input_data.get("arus_antar_fasa_tidak_seimbang", "Tidak")),
        # Hasil
        diagnosis=result["diagnosis"],
        confidence=result["confidence"],
    )
    db.session.add(record)
    db.session.commit()
    return record


def delete_history(history_id: int, user_id: int) -> bool:
    """
    Menghapus satu record riwayat milik user.

    Returns:
        True jika berhasil, False jika tidak ditemukan
    """
    record = History.query.filter_by(id=history_id, user_id=user_id).first()
    if not record:
        return False
    db.session.delete(record)
    db.session.commit()
    return True


def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None
