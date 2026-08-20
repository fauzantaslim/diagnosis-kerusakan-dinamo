"""
Service layer untuk Random Forest manual.

Alur:
  1. Load data training dari tabel datasets
  2. Stratified split 80:20 → 200 train, 50 test
  3. Build RandomForest (100 trees, DFS, bootstrap dari 200 training)
  4. Predict dengan detail per-tree
  5. Simpan ke session cache (10 menit) untuk lazy-load tree detail
"""

import time
import uuid
from typing import Dict, Any, Tuple, List, Optional
from collections import Counter

from app.services.dataset_service import load_split_data
from app.ml.random_forest import (
    RandomForest, FEATURE_NAMES
)

# ================================================================ #
#  Session cache                                                   #
#  key: session_id → { forest, input_x, split_info, detail,       #
#                      expires_at }                                #
# ================================================================ #
_SESSION_CACHE: Dict[str, Dict] = {}
_SESSION_TTL: int = 3600  # 60 menit (1 jam)


def _cleanup() -> None:
    now     = time.time()
    expired = [k for k, v in _SESSION_CACHE.items() if v["expires_at"] < now]
    for k in expired:
        del _SESSION_CACHE[k]




# ================================================================ #
#  Prediksi utama                                                  #
# ================================================================ #

def predict_with_detail(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    1. Parse input → vektor binary
    2. Load data dari DB
    3. Stratified split 80:20 (200 train / 50 test)
    4. Build RandomForest dari training set
    5. Predict + detail
    6. Simpan ke session cache
    """
    input_x = _parse_input(input_data)

    # Ambil data dari DB dan split 80:20 via dataset_service (single source of truth)
    X_train, X_test, y_train, y_test, X, y = load_split_data()
    if len(X) < 10:
        raise ValueError("Dataset terlalu sedikit untuk training (minimal 10 data).")

    # Stratified 80:20 split sudah dilakukan oleh load_split_data()

    split_info = {
        "n_total"             : len(X),
        "n_train"             : len(X_train),
        "n_test"              : len(X_test),
        "train_pct"           : round(len(X_train) / len(X) * 100, 1),
        "test_pct"            : round(len(X_test)  / len(X) * 100, 1),
        "class_dist_train"    : dict(Counter(y_train)),
        "class_dist_test"     : dict(Counter(y_test)),
    }

    # Build forest dari training set
    rf = RandomForest(
        n_estimators=100,
        max_depth=7,
        min_samples_split=5,
        max_features="sqrt",
        random_state=42,
    )
    rf.fit(X_train, y_train)

    # Predict dengan detail
    result = rf.predict_with_detail(input_x, feature_names=FEATURE_NAMES)

    # Hitung SHAP (Exact Shapley Values) secara manual
    shap_result = rf.compute_shap_values(
        input_x,
        result["diagnosis"],
        feature_names=FEATURE_NAMES,
    )

    # Ganti feature_importances (MDI) dengan SHAP-based importances
    result["feature_importances"] = [
        {
            "feature"       : sv["feature"],
            "feature_index" : sv["feature_index"],
            "importance"    : sv["importance_pct"],
            "shap_value"    : sv["shap_value"],
        }
        for sv in shap_result["shap_values"]
    ]

    # Tambahkan shap_detail lengkap untuk detail perhitungan
    result["shap_detail"] = shap_result

    # Simpan ke session cache
    _cleanup()
    session_id = str(uuid.uuid4())
    _SESSION_CACHE[session_id] = {
        "forest"    : rf,
        "input_x"   : input_x,
        "split_info": split_info,
        "detail"    : result,
        "expires_at": time.time() + _SESSION_TTL,
    }

    result["session_id"]       = session_id
    result["split_info"]       = split_info
    result["n_training_samples"] = len(X_train)
    result["calculation_detail"]["n_training_samples"] = len(X_train)
    result["calculation_detail"]["n_total_dataset"]    = len(X)

    return result


# ================================================================ #
#  Tree detail (React Flow)                                        #
# ================================================================ #

def get_tree_react_flow(session_id: str, tree_id: int) -> Dict[str, Any]:
    """
    Ambil data React Flow untuk satu tree beserta bootstrap info.
    """
    _cleanup()
    session = _SESSION_CACHE.get(session_id)
    if not session:
        raise ValueError(
            "Session tidak ditemukan atau sudah expired (10 menit). "
            "Lakukan prediksi ulang."
        )

    rf: RandomForest = session["forest"]
    input_x: List[int] = session["input_x"]

    if tree_id < 0 or tree_id >= len(rf.trees):
        raise ValueError(
            f"tree_id {tree_id} tidak valid. Tersedia: 0 – {len(rf.trees) - 1}."
        )

    tree          = rf.trees[tree_id]
    boot_info     = rf.bootstrap_infos[tree_id]
    split_info    = session["split_info"]
    react_flow    = tree.to_react_flow(input_x=input_x, feature_names=FEATURE_NAMES)
    vote, path    = tree.predict_with_path(input_x)

    # Gabungkan info split + bootstrap untuk konteks lengkap
    sampling_detail = {
        **split_info,
        "bootstrap": {
            **boot_info,
            # Penjelasan: bootstrap diambil dari training set (n_train)
            "note": (
                f"Dari {split_info['n_train']} data training (80% dari {split_info['n_total']}), "
                f"diambil {boot_info['n_bootstrap']} sampel acak dengan pengembalian. "
                f"Didapat {boot_info['n_unique_samples']} sampel unik "
                f"({boot_info['unique_pct']}%) dan "
                f"{boot_info['n_oob_samples']} OOB ({boot_info['oob_pct']}%)."
            )
        }
    }

    return {
        "tree_id"        : tree_id,
        "session_id"     : session_id,
        "vote"           : vote,
        "path_length"    : len(path),
        "sampling_detail": sampling_detail,
        **react_flow,
    }


def get_session_votes(session_id: str) -> Dict[str, Any]:
    """Ringkasan vote semua tree dari session cache."""
    _cleanup()
    session = _SESSION_CACHE.get(session_id)
    if not session:
        raise ValueError(
            "Session tidak ditemukan atau sudah expired. Lakukan prediksi ulang."
        )

    detail: Dict = session["detail"]

    return {
        "session_id"    : session_id,
        "n_trees"       : detail["calculation_detail"]["n_trees"],
        "diagnosis"     : detail["diagnosis"],
        "votes"         : [
            {
                "tree_id"    : tv["tree_id"],
                "vote"       : tv["vote"],
                "path_length": tv["path_length"],
            }
            for tv in detail["tree_votes"]
        ],
        "majority_votes": detail["calculation_detail"]["majority_votes"],
        "split_info"    : session["split_info"],
        "shap_detail"   : detail.get("shap_detail"),
    }


# ================================================================ #
#  Helper: parse input                                             #
# ================================================================ #

def _parse_input(input_data: Dict[str, Any]) -> List[int]:
    """Konversi dict input ke vektor binary sesuai FEATURE_NAMES."""
    result = []
    for f in FEATURE_NAMES:
        val = input_data.get(f, 0)
        if isinstance(val, bool):
            result.append(1 if val else 0)
        elif isinstance(val, int):
            result.append(1 if val else 0)
        elif isinstance(val, str):
            result.append(1 if val.strip().upper() in ("YA", "1", "TRUE", "YES") else 0)
        else:
            result.append(0)
    return result
