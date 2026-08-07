from flask import jsonify
from app.utils.jwt_utils import jwt_required
from app.services import random_forest_service


@jwt_required
def get_tree(session_id: str, tree_id: int):
    """
    GET /api/rf/tree/<session_id>/<tree_id>
    Return data React Flow untuk satu decision tree.
    """
    try:
        data = random_forest_service.get_tree_react_flow(session_id, tree_id)
        return jsonify({"success": True, **data}), 200
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 404
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@jwt_required
def get_trees(session_id: str):
    """
    GET /api/rf/trees/<session_id>
    Return ringkasan vote semua tree dalam session.
    """
    try:
        data = random_forest_service.get_session_votes(session_id)
        return jsonify({"success": True, **data}), 200
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 404
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
