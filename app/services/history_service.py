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
        jenis_mesin=input_data.get("jenis_mesin"),
        daya_hp_kw=_safe_float(input_data.get("daya_hp_kw")),
        jumlah_pole=_safe_int(input_data.get("jumlah_pole")),
        # Gejala
        suara_bising_abnormal=_bool(input_data.get("suara_bising_abnormal", "Tidak")),
        getaran_berlebih=_bool(input_data.get("getaran_berlebih", "Tidak")),
        motor_cepat_panas=_bool(input_data.get("motor_cepat_panas", "Tidak")),
        arus_melebihi_normal=_bool(input_data.get("arus_melebihi_normal", "Tidak")),
        tegangan_tidak_stabil=_bool(input_data.get("tegangan_tidak_stabil", "Tidak")),
        putaran_menurun=_bool(input_data.get("putaran_menurun", "Tidak")),
        sulit_start=_bool(input_data.get("sulit_start", "Tidak")),
        sering_trip_mcb=_bool(input_data.get("sering_trip_mcb", "Tidak")),
        trip_overload_relay=_bool(input_data.get("trip_overload_relay", "Tidak")),
        efisiensi_menurun=_bool(input_data.get("efisiensi_menurun", "Tidak")),
        bau_hangus=_bool(input_data.get("bau_hangus", "Tidak")),
        intermittent_stopping=_bool(input_data.get("intermittent_stopping", "Tidak")),
        warna_gulungan_berubah=_bool(input_data.get("warna_gulungan_berubah", "Tidak")),
        kipas_pendingin_rusak=_bool(input_data.get("kipas_pendingin_rusak", "Tidak")),
        terminal_terbakar=_bool(input_data.get("terminal_terbakar", "Tidak")),
        bearing_aus_pecah=_bool(input_data.get("bearing_aus_pecah", "Tidak")),
        housing_bearing_aus=_bool(input_data.get("housing_bearing_aus", "Tidak")),
        poros_shaft_aus=_bool(input_data.get("poros_shaft_aus", "Tidak")),
        kebocoran_pelumas=_bool(input_data.get("kebocoran_pelumas", "Tidak")),
        keretakan_dudukan=_bool(input_data.get("keretakan_dudukan", "Tidak")),
        sumbatan_sirip_pendingin=_bool(input_data.get("sumbatan_sirip_pendingin", "Tidak")),
        lubang_spi_aus=_bool(input_data.get("lubang_spi_aus", "Tidak")),
        # Pengukuran
        temperatur_c=_safe_float(input_data.get("temperatur_c")),
        arus_a=_safe_float(input_data.get("arus_a")),
        tegangan_v=_safe_float(input_data.get("tegangan_v")),
        resistansi_isolasi_mohm=_safe_float(input_data.get("resistansi_isolasi_mohm")),
        kecepatan_putaran_rpm=_safe_float(input_data.get("kecepatan_putaran_rpm")),
        ketidakseimbangan_arus_pct=_safe_float(input_data.get("ketidakseimbangan_arus_pct")),
        ketidakseimbangan_tegangan_pct=_safe_float(input_data.get("ketidakseimbangan_tegangan_pct")),
        faktor_daya=_safe_float(input_data.get("faktor_daya")),
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
