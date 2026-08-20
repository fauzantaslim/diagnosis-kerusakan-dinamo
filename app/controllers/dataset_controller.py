from collections import Counter
from flask import request, jsonify
from app.services import dataset_service
from app.services.dataset_service import load_split_data


def index():
    """GET /api/dataset — List semua dataset dengan pagination dan search."""
    try:
        page     = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        search   = request.args.get('search', '').strip()
        split    = request.args.get('split', '').strip().lower()

        result = dataset_service.get_all_datasets(page=page, per_page=per_page, search=search, split_filter=split)

        return jsonify({
            'success'   : True,
            'data'      : result['data'],
            'pagination': result['pagination'],
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


def store():
    """POST /api/dataset — Buat satu data baru."""
    data = request.get_json(silent=True) or request.form.to_dict()

    if not data:
        return jsonify({'success': False, 'message': 'Request body tidak boleh kosong.'}), 400

    try:
        dataset = dataset_service.create_dataset(data)
        return jsonify({
            'success': True,
            'message': 'Data berhasil ditambahkan.',
            'data'   : dataset.to_dict(),
        }), 201

    except ValueError as e:
        return jsonify({'success': False, 'message': str(e)}), 422

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


def show(dataset_id):
    """GET /api/dataset/<id> — Detail satu data."""
    dataset = dataset_service.get_dataset_by_id(dataset_id)

    if not dataset:
        return jsonify({'success': False, 'message': f'Data dengan id {dataset_id} tidak ditemukan.'}), 404

    return jsonify({'success': True, 'data': dataset.to_dict()}), 200


def update(dataset_id):
    """PUT /api/dataset/<id> — Update satu data."""
    data = request.get_json(silent=True) or request.form.to_dict()

    if not data:
        return jsonify({'success': False, 'message': 'Request body tidak boleh kosong.'}), 400

    try:
        dataset = dataset_service.update_dataset(dataset_id, data)

        if not dataset:
            return jsonify({'success': False, 'message': f'Data dengan id {dataset_id} tidak ditemukan.'}), 404

        return jsonify({
            'success': True,
            'message': 'Data berhasil diperbarui.',
            'data'   : dataset.to_dict(),
        }), 200

    except ValueError as e:
        return jsonify({'success': False, 'message': str(e)}), 422

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


def destroy(dataset_id):
    """DELETE /api/dataset/<id> — Hapus satu data."""
    deleted = dataset_service.delete_dataset(dataset_id)

    if not deleted:
        return jsonify({'success': False, 'message': f'Data dengan id {dataset_id} tidak ditemukan.'}), 404

    return jsonify({'success': True, 'message': 'Data berhasil dihapus.'}), 200


def import_data():
    """POST /api/dataset/import — Import data dari file CSV atau XLSX."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': "Field 'file' wajib ada pada request."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'success': False, 'message': 'Tidak ada file yang dipilih.'}), 400

    try:
        result = dataset_service.import_from_file(file)

        return jsonify({
            'success' : True,
            'message' : f"Import selesai. {result['imported']} data berhasil, {result['skipped']} dilewati.",
            'imported': result['imported'],
            'skipped' : result['skipped'],
            'errors'  : result['errors'],
        }), 200

    except (ValueError, RuntimeError) as e:
        return jsonify({'success': False, 'message': str(e)}), 422

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


def split_info():
    """GET /api/dataset/split-info — Info pembagian 80:20 training/testing."""
    try:
        X_train, X_test, y_train, y_test, X, y = load_split_data()

        return jsonify({
            'success'          : True,
            'n_total'          : len(X),
            'n_train'          : len(X_train),
            'n_test'           : len(X_test),
            'train_pct'        : round(len(X_train) / len(X) * 100, 1),
            'test_pct'         : round(len(X_test)  / len(X) * 100, 1),
            'class_dist_train' : dict(Counter(y_train)),
            'class_dist_test'  : dict(Counter(y_test)),
            'classes'          : sorted(set(y)),
        }), 200

    except ValueError as e:
        return jsonify({'success': False, 'message': str(e)}), 422

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
