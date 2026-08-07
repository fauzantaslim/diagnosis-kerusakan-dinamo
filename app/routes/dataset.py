from flask import Blueprint
from app.controllers import dataset_controller

dataset_bp = Blueprint('dataset_bp', __name__)

# ------------------------------------------------------------------ #
#  Import harus di atas CRUD agar /import tidak ditangkap /<int:id>  #
# ------------------------------------------------------------------ #
dataset_bp.route('/import', methods=['POST'])(dataset_controller.import_data)

# CRUD
dataset_bp.route('',          methods=['GET']   )(dataset_controller.index)
dataset_bp.route('',          methods=['POST']  )(dataset_controller.store)
dataset_bp.route('/<int:dataset_id>', methods=['GET']   )(dataset_controller.show)
dataset_bp.route('/<int:dataset_id>', methods=['PUT']   )(dataset_controller.update)
dataset_bp.route('/<int:dataset_id>', methods=['DELETE'])(dataset_controller.destroy)
