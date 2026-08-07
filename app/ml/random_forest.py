"""
Implementasi Random Forest manual tanpa scikit-learn.

Alur:
  1. Stratified Split 80:20 (di service layer)
  2. Bootstrap Sampling dari training set (with replacement)
  3. Decision Tree per bootstrap sample:
     a. Random Feature Selection: m = floor(sqrt(11)) = 3
     b. Gini Impurity sebagai kriteria split
     c. DFS Build: rekursi kiri (TIDAK=0) dulu → kanan (YA=1)
     d. DFS Predict: traverse ke leaf
  4. Majority Voting dari semua tree

Konvensi branch (binary features, threshold=0.5):
  LEFT  → nilai fitur ≤ 0.5 = 0 (TIDAK)
  RIGHT → nilai fitur > 0.5 = 1 (YA)

Setiap node menyimpan detail kalkulasi Gini untuk dialog interaktif.
"""

import math
import random
from collections import Counter
from typing import Optional, List, Tuple, Dict, Any

FEATURE_NAMES: List[str] = [
    'suara_bising_abnormal',
    'bau_hangus',
    'indikasi_overheating',
    'putaran_poros_seret',
    'getaran_berlebih',
    'terminal_overheating',
    'kipas_pendingin_rusak',
    'cooling_duct_tersumbat',
    'resistansi_isolasi_tidak_seimbang',
    'resistansi_winding_tidak_seimbang',
    'arus_antar_fasa_tidak_seimbang',
]

_THRESHOLD: float = 0.5  # semua fitur binary (0/1)


# ================================================================ #
#  Node                                                            #
# ================================================================ #

class Node:
    """
    Satu node dalam Decision Tree.

    Atribut utama:
      evaluated_features — list detail evaluasi semua fitur kandidat (decision node)
      gini_detail        — step-by-step Gini calculation di node ini (semua tipe)
    """

    __slots__ = (
        'node_id', 'depth', 'n_samples', 'gini',
        'feature_index', 'feature_name',
        'gini_left', 'gini_right', 'n_left', 'n_right',
        'left', 'right',
        'is_leaf', 'value', 'class_counts',
        'evaluated_features',   # list[dict] — kandidat fitur & Gini-nya
        'gini_detail',          # dict — kalkulasi Gini step-by-step
        '_rx', '_ry',           # posisi React Flow
    )

    def __init__(
        self,
        *,
        node_id: int,
        depth: int,
        n_samples: int,
        gini: float,
        feature_index: Optional[int] = None,
        feature_name: Optional[str] = None,
        gini_left: float = 0.0,
        gini_right: float = 0.0,
        n_left: int = 0,
        n_right: int = 0,
        left: "Optional[Node]" = None,
        right: "Optional[Node]" = None,
        is_leaf: bool = False,
        value: Optional[str] = None,
        class_counts: Optional[Dict[str, int]] = None,
        evaluated_features: Optional[List[Dict]] = None,
        gini_detail: Optional[Dict] = None,
    ):
        self.node_id = node_id
        self.depth = depth
        self.n_samples = n_samples
        self.gini = gini
        self.feature_index = feature_index
        self.feature_name = feature_name
        self.gini_left = gini_left
        self.gini_right = gini_right
        self.n_left = n_left
        self.n_right = n_right
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.value = value
        self.class_counts = class_counts or {}
        self.evaluated_features = evaluated_features or []
        self.gini_detail = gini_detail or {}
        self._rx: float = 0.0
        self._ry: float = 0.0


# ================================================================ #
#  Decision Tree                                                   #
# ================================================================ #

class DecisionTree:
    """
    Decision Tree dengan:
    - Random feature selection (m = sqrt(n_features))
    - Gini impurity sebagai kriteria split
    - DFS build & predict: kiri (TIDAK=0) dulu, kanan (YA=1) kemudian
    - Menyimpan detail kalkulasi per node untuk dialog interaktif
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root: Optional[Node] = None
        self._node_counter: int = 0
        self._rng = random.Random(random_state)
        self.feature_importance_: Dict[int, float] = {}

    # ------------------------------------------------------------ #
    #  Helpers                                                      #
    # ------------------------------------------------------------ #

    def _next_id(self) -> int:
        self._node_counter += 1
        return self._node_counter

    def _gini(self, y: List[str]) -> float:
        """Gini impurity: 1 - Σ p_k²"""
        n = len(y)
        if n == 0:
            return 0.0
        counts = Counter(y)
        return 1.0 - sum((c / n) ** 2 for c in counts.values())

    def _compute_gini_detail(self, y: List[str]) -> Dict:
        """
        Hasilkan breakdown kalkulasi Gini step-by-step untuk satu node.
        Dipakai untuk dialog interaktif.

        Returns:
            {
              n_samples, class_breakdown: [{class, count, prob, prob_sq}],
              sum_sq, gini
            }
        """
        n = len(y)
        if n == 0:
            return {"n_samples": 0, "class_breakdown": [], "sum_sq": 0.0, "gini": 0.0}

        counts = Counter(y)
        breakdown = []
        sum_sq = 0.0

        for cls in sorted(counts.keys()):
            cnt = counts[cls]
            prob = cnt / n
            prob_sq = prob ** 2
            sum_sq += prob_sq
            breakdown.append({
                "class"  : cls,
                "count"  : cnt,
                "prob"   : round(prob, 4),
                "prob_sq": round(prob_sq, 4),
            })

        gini = 1.0 - sum_sq
        return {
            "n_samples"      : n,
            "class_breakdown": breakdown,
            "sum_sq"         : round(sum_sq, 4),
            "gini"           : round(gini, 4),
        }

    def _select_features(self, n_features: int) -> List[int]:
        """Random feature selection: pilih m fitur secara acak."""
        m = self.max_features or max(1, int(math.sqrt(n_features)))
        m = min(m, n_features)
        return self._rng.sample(range(n_features), m)

    def _majority_class(self, y: List[str]) -> str:
        return Counter(y).most_common(1)[0][0]

    def _evaluate_features(
        self,
        X: List[List[int]],
        y: List[str],
        feature_indices: List[int],
    ) -> Tuple[List[Dict], Optional[int], float, float, float, int, int]:
        """
        Evaluasi semua fitur kandidat dengan Gini impurity.
        Menyimpan detail kalkulasi tiap fitur untuk dialog interaktif.

        Returns:
            evaluations  — list detail setiap fitur
            best_feature — index fitur terbaik (atau None)
            best_gain    — Gini gain terbaik
            best_gl      — Gini kiri fitur terbaik
            best_gr      — Gini kanan fitur terbaik
            best_nl      — n sampel kiri fitur terbaik
            best_nr      — n sampel kanan fitur terbaik
        """
        n = len(y)
        parent_gini = self._gini(y)

        best_fi: Optional[int] = None
        best_gain: float = -1.0
        best_gl: float = 0.0
        best_gr: float = 0.0
        best_nl: int = 0
        best_nr: int = 0

        evaluations: List[Dict] = []

        for fi in feature_indices:
            fn = FEATURE_NAMES[fi] if fi < len(FEATURE_NAMES) else f"f{fi}"

            left_y  = [y[i] for i in range(n) if X[i][fi] <= _THRESHOLD]   # TIDAK=0
            right_y = [y[i] for i in range(n) if X[i][fi] > _THRESHOLD]    # YA=1
            nl, nr  = len(left_y), len(right_y)

            if nl == 0 or nr == 0:
                # Split tidak menghasilkan dua sisi → skip
                evaluations.append({
                    "feature_index"      : fi,
                    "feature_name"       : fn,
                    "skipped"            : True,
                    "skip_reason"        : "Semua sampel bernilai sama (tidak ada variasi fitur ini)",
                    "n_left"             : nl,
                    "n_right"            : nr,
                    "class_counts_left"  : dict(Counter(left_y)),
                    "class_counts_right" : dict(Counter(right_y)),
                    "gini_left"          : 0.0,
                    "gini_right"         : 0.0,
                    "weighted_gini"      : round(parent_gini, 4),
                    "gini_gain"          : 0.0,
                    "is_best"            : False,
                })
                continue

            g_left        = self._gini(left_y)
            g_right       = self._gini(right_y)
            weighted_gini = (nl / n) * g_left + (nr / n) * g_right
            gain          = parent_gini - weighted_gini

            evaluations.append({
                "feature_index"      : fi,
                "feature_name"       : fn,
                "skipped"            : False,
                "n_left"             : nl,     # TIDAK (0)
                "n_right"            : nr,     # YA (1)
                "class_counts_left"  : dict(Counter(left_y)),
                "class_counts_right" : dict(Counter(right_y)),
                "gini_left"          : round(g_left, 4),
                "gini_right"         : round(g_right, 4),
                "weighted_gini"      : round(weighted_gini, 4),
                "gini_gain"          : round(gain, 4),
                "is_best"            : False,
            })

            if gain > best_gain:
                best_gain = gain
                best_fi   = fi
                best_gl   = g_left
                best_gr   = g_right
                best_nl   = nl
                best_nr   = nr

        # Tandai fitur terbaik
        for ev in evaluations:
            if ev["feature_index"] == best_fi and not ev.get("skipped"):
                ev["is_best"] = True

        return evaluations, best_fi, best_gain, best_gl, best_gr, best_nl, best_nr

    # ------------------------------------------------------------ #
    #  Build Tree (DFS rekursif)                                   #
    # ------------------------------------------------------------ #

    def _build_tree(self, X: List[List[int]], y: List[str], depth: int = 0) -> Node:
        """
        DFS rekursif membangun decision tree.
        Kiri (TIDAK=0) dibangun dulu, kanan (YA=1) kemudian.
        Setiap node menyimpan detail Gini untuk dialog interaktif.
        """
        n = len(y)
        node_gini    = self._gini(y)
        class_counts = dict(Counter(y))
        node_id      = self._next_id()
        gini_detail  = self._compute_gini_detail(y)

        # ---- Terminal node → buat leaf ----
        if (
            depth >= self.max_depth
            or n < self.min_samples_split
            or node_gini == 0.0
        ):
            return Node(
                node_id=node_id, depth=depth, n_samples=n, gini=round(node_gini, 4),
                is_leaf=True, value=self._majority_class(y),
                class_counts=class_counts, gini_detail=gini_detail,
            )

        # ---- Random feature selection ----
        n_features      = len(X[0]) if X else 0
        feature_indices = self._select_features(n_features)

        # ---- Evaluasi semua fitur kandidat ----
        evaluations, best_fi, best_gain, g_left, g_right, nl, nr = \
            self._evaluate_features(X, y, feature_indices)

        # Tidak ada split menguntungkan → leaf
        if best_fi is None or best_gain <= 0.0:
            return Node(
                node_id=node_id, depth=depth, n_samples=n, gini=round(node_gini, 4),
                is_leaf=True, value=self._majority_class(y),
                class_counts=class_counts, gini_detail=gini_detail,
                evaluated_features=evaluations,
            )

        # ---- Pisahkan data ----
        left_X  = [X[i] for i in range(n) if X[i][best_fi] <= _THRESHOLD]
        left_y  = [y[i] for i in range(n) if X[i][best_fi] <= _THRESHOLD]
        right_X = [X[i] for i in range(n) if X[i][best_fi] > _THRESHOLD]
        right_y = [y[i] for i in range(n) if X[i][best_fi] > _THRESHOLD]

        # ---- DFS: kiri dulu (TIDAK=0), lalu kanan (YA=1) ----
        left_child  = self._build_tree(left_X,  left_y,  depth + 1)
        right_child = self._build_tree(right_X, right_y, depth + 1)

        # ---- MDI Feature Importance ----
        self.feature_importance_[best_fi] = (
            self.feature_importance_.get(best_fi, 0.0) + n * best_gain
        )

        feat_name = FEATURE_NAMES[best_fi] if best_fi < len(FEATURE_NAMES) else f"f{best_fi}"

        return Node(
            node_id=node_id, depth=depth, n_samples=n, gini=round(node_gini, 4),
            feature_index=best_fi, feature_name=feat_name,
            gini_left=round(g_left, 4), gini_right=round(g_right, 4),
            n_left=nl, n_right=nr,
            left=left_child, right=right_child,
            is_leaf=False, class_counts=class_counts,
            evaluated_features=evaluations, gini_detail=gini_detail,
        )

    def fit(self, X: List[List[int]], y: List[str]) -> "DecisionTree":
        self._node_counter    = 0
        self.feature_importance_ = {}
        self.root = self._build_tree(X, y, depth=0)
        return self

    # ------------------------------------------------------------ #
    #  Predict (DFS ke leaf)                                       #
    # ------------------------------------------------------------ #

    def _predict_one(self, node: Node, x: List[int]) -> str:
        if node.is_leaf:
            return node.value
        if x[node.feature_index] <= _THRESHOLD:
            return self._predict_one(node.left, x)   # TIDAK (0)
        else:
            return self._predict_one(node.right, x)  # YA (1)

    def predict(self, x: List[int]) -> str:
        return self._predict_one(self.root, x)

    def predict_with_path(self, x: List[int]) -> Tuple[str, List[int]]:
        """DFS predict + track path node_id yang dilalui."""
        path: List[int] = []

        def _traverse(node: Node) -> str:
            path.append(node.node_id)
            if node.is_leaf:
                return node.value
            if x[node.feature_index] <= _THRESHOLD:
                return _traverse(node.left)
            else:
                return _traverse(node.right)

        prediction = _traverse(self.root)
        return prediction, path

    # ------------------------------------------------------------ #
    #  React Flow serialization                                     #
    # ------------------------------------------------------------ #

    def _assign_positions(self) -> None:
        """
        DFS untuk assign posisi (x, y) ke setiap node.
        Leaf: x incremental (220px), y = depth * 150px.
        Parent: x = rata-rata children.
        """
        counter = [0]

        def _dfs_pos(node: Node, depth: int) -> None:
            node._ry = depth * 150.0
            if node.is_leaf:
                node._rx = counter[0] * 220.0
                counter[0] += 1
            else:
                if node.left:
                    _dfs_pos(node.left,  depth + 1)
                if node.right:
                    _dfs_pos(node.right, depth + 1)

                lx = node.left._rx  if node.left  else 0.0
                rx = node.right._rx if node.right else 0.0
                if node.left and node.right:
                    node._rx = (lx + rx) / 2
                elif node.left:
                    node._rx = lx
                elif node.right:
                    node._rx = rx
                else:
                    node._rx = 0.0

        if self.root:
            _dfs_pos(self.root, 0)

    def to_react_flow(
        self,
        input_x: Optional[List[int]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Serialisasi tree ke format React Flow.
        Setiap node data menyertakan detail Gini untuk dialog interaktif.

        Args:
            input_x: vektor input (11 × 0/1). Jika ada, on_path=True di jalur input.
        """
        fn = feature_names or FEATURE_NAMES

        # Path untuk input ini
        path_nodes: set = set()
        path_edges: set = set()
        if input_x is not None:
            _, path = self.predict_with_path(input_x)
            path_nodes = set(path)
            path_edges = {(path[i], path[i + 1]) for i in range(len(path) - 1)}

        self._assign_positions()

        nodes: List[Dict] = []
        edges: List[Dict] = []

        def _dfs_serialize(node: Node) -> None:
            """DFS serialize — kiri dulu, sesuai urutan build tree."""
            if node is None:
                return

            on_path = node.node_id in path_nodes

            base_data: Dict[str, Any] = {
                "id"          : str(node.node_id),
                "n_samples"   : node.n_samples,
                "gini"        : node.gini,
                "on_path"     : on_path,
                "depth"       : node.depth,
                "class_counts": node.class_counts,
                "gini_detail" : node.gini_detail,
            }

            if node.is_leaf:
                node_type = "leafNode"
                node_data = {
                    **base_data,
                    "type" : "leaf",
                    "label": node.value,
                }
            else:
                feat_n = fn[node.feature_index] if node.feature_index < len(fn) else f"f{node.feature_index}"
                node_type = "decisionNode"
                node_data = {
                    **base_data,
                    "type"               : "decision",
                    "feature_index"      : node.feature_index,
                    "feature_name"       : feat_n,
                    "gini_left"          : node.gini_left,
                    "gini_right"         : node.gini_right,
                    "n_left"             : node.n_left,
                    "n_right"            : node.n_right,
                    "evaluated_features" : node.evaluated_features,
                    "m_features_evaluated": len(node.evaluated_features),
                }

            nodes.append({
                "id"      : str(node.node_id),
                "type"    : node_type,
                "data"    : node_data,
                "position": {"x": round(node._rx), "y": round(node._ry)},
            })

            if not node.is_leaf:
                # Edge kiri: TIDAK (0)
                if node.left:
                    on_e = (node.node_id, node.left.node_id) in path_edges
                    edges.append({
                        "id"    : f"e{node.node_id}-{node.left.node_id}",
                        "source": str(node.node_id),
                        "target": str(node.left.node_id),
                        "label" : "TIDAK (0)",
                        "type"  : "left",
                        "data"  : {"on_path": on_e, "direction": "left", "value": 0},
                    })
                    _dfs_serialize(node.left)    # DFS kiri dulu

                # Edge kanan: YA (1)
                if node.right:
                    on_e = (node.node_id, node.right.node_id) in path_edges
                    edges.append({
                        "id"    : f"e{node.node_id}-{node.right.node_id}",
                        "source": str(node.node_id),
                        "target": str(node.right.node_id),
                        "label" : "YA (1)",
                        "type"  : "right",
                        "data"  : {"on_path": on_e, "direction": "right", "value": 1},
                    })
                    _dfs_serialize(node.right)   # DFS kanan setelah kiri

        _dfs_serialize(self.root)
        return {"nodes": nodes, "edges": edges}


# ================================================================ #
#  Random Forest                                                   #
# ================================================================ #

class RandomForest:
    """Random Forest manual — 50 trees, majority voting."""

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 10,
        min_samples_split: int = 2,
        max_features: str = "sqrt",
        random_state: int = 42,
    ):
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.min_samples_split= min_samples_split
        self.max_features     = max_features
        self.random_state     = random_state
        self.trees: List[DecisionTree] = []
        self.bootstrap_infos: List[Dict] = []   # info per tree
        self._rng = random.Random(random_state)

    def _resolve_m(self, n_features: int) -> int:
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(math.log2(n_features)))
        if isinstance(self.max_features, int):
            return self.max_features
        return n_features

    def _bootstrap(
        self, X: List[List[int]], y: List[str], seed: Optional[int]
    ) -> Tuple[List[List[int]], List[str], Dict]:
        """
        Bootstrap sampling dengan replacement dari training set.
        Mengembalikan (X_boot, y_boot, bootstrap_info).

        bootstrap_info:
          n_total_train, n_bootstrap, n_unique, n_oob,
          class_distribution_train, class_distribution_bootstrap
        """
        rng = random.Random(seed)
        n = len(X)
        indices = [rng.randint(0, n - 1) for _ in range(n)]

        X_boot = [X[i] for i in indices]
        y_boot = [y[i] for i in indices]

        unique_set   = set(indices)
        oob_indices  = [i for i in range(n) if i not in unique_set]

        # Ambil semua sampel bootstrap
        sample_rows = []
        for i in range(len(X_boot)):
            sample_rows.append({
                "original_index": indices[i],
                "features": X_boot[i],
                "label"   : y_boot[i],
            })

        info = {
            "n_total_train"               : n,
            "n_bootstrap"                 : n,
            "n_unique_samples"            : len(unique_set),
            "n_oob_samples"               : len(oob_indices),
            "unique_pct"                  : round(len(unique_set) / n * 100, 1),
            "oob_pct"                     : round(len(oob_indices) / n * 100, 1),
            "class_distribution_train"    : dict(Counter(y)),
            "class_distribution_bootstrap": dict(Counter(y_boot)),
            "sample_rows"                 : sample_rows,
        }
        return X_boot, y_boot, info

    def fit(self, X: List[List[int]], y: List[str]) -> "RandomForest":
        """
        Latih n_estimators trees.
        Setiap tree: bootstrap sample → DFS build decision tree.
        """
        self.trees           = []
        self.bootstrap_infos = []
        n_features = len(X[0]) if X else 0
        m = self._resolve_m(n_features)

        for i in range(self.n_estimators):
            seed = (self.random_state + i) if self.random_state is not None else None
            X_boot, y_boot, boot_info = self._bootstrap(X, y, seed)

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=m,
                random_state=seed,
            )
            tree.fit(X_boot, y_boot)

            self.trees.append(tree)
            self.bootstrap_infos.append(boot_info)

        return self

    def predict(self, x: List[int]) -> str:
        votes = [tree.predict(x) for tree in self.trees]
        return Counter(votes).most_common(1)[0][0]

    def predict_with_detail(
        self,
        x: List[int],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Prediksi lengkap dengan detail per-tree dan feature importances."""
        fn = feature_names or FEATURE_NAMES

        tree_votes = []
        for i, tree in enumerate(self.trees):
            vote, path = tree.predict_with_path(x)
            tree_votes.append({
                "tree_id"     : i,
                "vote"        : vote,
                "path_length" : len(path),
                "path_node_ids": path,
            })

        votes       = [tv["vote"] for tv in tree_votes]
        vote_counts = Counter(votes)
        total       = len(votes)
        diagnosis   = vote_counts.most_common(1)[0][0]
        confidence  = vote_counts[diagnosis] / total

        # MDI Feature Importance (rata-rata antar tree, dinormalisasi)
        fi_agg: Dict[int, float] = {}
        for tree in self.trees:
            total_tree = sum(tree.feature_importance_.values()) or 1.0
            for fi, val in tree.feature_importance_.items():
                fi_agg[fi] = fi_agg.get(fi, 0.0) + (val / total_tree)

        total_fi = sum(fi_agg.values()) or 1.0
        feature_importances = [
            {
                "feature"      : fn[fi] if fi < len(fn) else f"f{fi}",
                "feature_index": fi,
                "importance"   : round((val / total_fi) * 100, 2),
            }
            for fi, val in sorted(fi_agg.items(), key=lambda kv: -kv[1])
        ]

        m = self.trees[0].max_features if self.trees else 0

        return {
            "diagnosis"          : diagnosis,
            "confidence"         : round(confidence, 4),
            "confidence_pct"     : f"{confidence * 100:.2f}%",
            "probabilities"      : {
                label: round(cnt / total, 4)
                for label, cnt in sorted(vote_counts.items(), key=lambda kv: -kv[1])
            },
            "tree_votes"         : tree_votes,
            "feature_importances": feature_importances,
            "calculation_detail" : {
                "n_trees"             : total,
                "n_features_total"    : len(fn),
                "m_features_per_split": m,
                "majority_votes"      : dict(vote_counts),
            },
        }


# ================================================================ #
#  Stratified Split (80:20)                                        #
# ================================================================ #

def stratified_split(
    X: List[List[int]],
    y: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List, List, List, List]:
    """
    Stratified split 80:20 — jaga proporsi kelas di train & test.
    Tidak menggunakan sklearn.

    Returns: X_train, X_test, y_train, y_test
    """
    from collections import defaultdict
    rng = random.Random(random_state)

    class_indices: Dict[str, List[int]] = defaultdict(list)
    for i, label in enumerate(y):
        class_indices[label].append(i)

    train_idx: List[int] = []
    test_idx:  List[int] = []

    total_n = len(X)
    target_n_test = int(total_n * test_size)
    
    # Tahap 1: Hitung kuota awal test untuk tiap kelas (dibulatkan ke bawah)
    class_allocations = {}
    remainders = []
    
    for label, indices in class_indices.items():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        
        exact_test = len(shuffled) * test_size
        n_test = int(exact_test)
        
        class_allocations[label] = {
            'shuffled': shuffled,
            'n_test': n_test
        }
        remainders.append((exact_test - n_test, label))
        
    # Tahap 2: Tambahkan sisa kuota ke kelas dengan remainder terbesar (Largest Remainder Method)
    current_n_test = sum(alloc['n_test'] for alloc in class_allocations.values())
    shortfall = target_n_test - current_n_test
    
    remainders.sort(key=lambda x: x[0], reverse=True)
    for i in range(shortfall):
        if i < len(remainders):
            label = remainders[i][1]
            class_allocations[label]['n_test'] += 1
            
    # Tahap 3: Masukkan ke index akhir
    for label, alloc in class_allocations.items():
        shuffled = alloc['shuffled']
        n_test = alloc['n_test']
        test_idx.extend(shuffled[:n_test])
        train_idx.extend(shuffled[n_test:])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]
    y_test  = [y[i] for i in test_idx]

    return X_train, X_test, y_train, y_test
