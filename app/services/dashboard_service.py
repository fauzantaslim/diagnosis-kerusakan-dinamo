from app.models.history import History

def get_recent_history(user_id, limit=5):
    return History.query.filter_by(user_id=user_id).order_by(History.tanggal.desc()).limit(limit).all()

