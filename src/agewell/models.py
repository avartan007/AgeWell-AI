"""Statistical learning primitives and model builders for AgeWell-AI."""
from __future__ import annotations

from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd


def sigmoid_fn(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(y_score).rank(method="average").values
    sum_pos_ranks = float(np.sum(ranks[y_true == 1]))
    u_stat = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return float(u_stat / (n_pos * n_neg))


def average_precision_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return np.nan
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    pos_mask = y_sorted == 1
    return float(np.sum(precision[pos_mask]) / n_pos)


def brier_score_loss_np(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    return float(np.mean((y_true - y_prob) ** 2))


def log_loss_np(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-6) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


class SimpleImputerNP:
    def __init__(self, strategy: str = "median") -> None:
        self.strategy = strategy
        self.statistics_: Optional[np.ndarray] = None

    def fit(self, X: Any) -> "SimpleImputerNP":
        arr = np.asarray(X, dtype=float)
        if self.strategy == "median":
            self.statistics_ = np.nanmedian(arr, axis=0)
        elif self.strategy == "mean":
            self.statistics_ = np.nanmean(arr, axis=0)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")
        self.statistics_ = np.where(np.isnan(self.statistics_), 0.0, self.statistics_)
        return self

    def transform(self, X: Any) -> np.ndarray:
        if self.statistics_ is None:
            raise RuntimeError("SimpleImputerNP must be fitted first.")
        arr = np.asarray(X, dtype=float).copy()
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(self.statistics_, inds[1])
        return arr

    def fit_transform(self, X: Any) -> np.ndarray:
        return self.fit(X).transform(X)


class StandardScalerNP:
    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: Any) -> "StandardScalerNP":
        arr = np.asarray(X, dtype=float)
        self.mean_ = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        self.scale_ = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        return (arr - self.mean_) / self.scale_

    def fit_transform(self, X: Any) -> np.ndarray:
        return self.fit(X).transform(X)


class LogisticRegressionNP:
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 200,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self.random_state = random_state
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any) -> "LogisticRegressionNP":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n_samples, n_features = X_arr.shape

        if self.class_weight == "balanced":
            n_pos = max(np.sum(y_arr == 1), 1.0)
            n_neg = max(np.sum(y_arr == 0), 1.0)
            w_pos = n_samples / (2.0 * n_pos)
            w_neg = n_samples / (2.0 * n_neg)
            sample_w = np.where(y_arr == 1, w_pos, w_neg)
        else:
            sample_w = np.ones(n_samples, dtype=float)

        X_aug = np.column_stack([np.ones(n_samples), X_arr])
        theta = np.zeros(n_features + 1, dtype=float)
        reg_diag = np.ones(n_features + 1, dtype=float) / max(self.C, 1e-6)
        reg_diag[0] = 0.0

        for _ in range(min(self.max_iter, 50)):
            p = sigmoid_fn(X_aug @ theta)
            grad = X_aug.T @ (sample_w * (p - y_arr)) + reg_diag * theta
            W = sample_w * np.maximum(p * (1.0 - p), 1e-5)
            H = (X_aug.T * W) @ X_aug + np.diag(reg_diag + 1e-5)
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, grad, rcond=None)[0]
            theta -= step
            if np.max(np.abs(step)) < 1e-5:
                break

        self.intercept_ = np.array([theta[0]])
        self.coef_ = theta[1:].reshape(1, -1)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        logits = self.intercept_[0] + (X_arr @ self.coef_[0])
        p1 = sigmoid_fn(logits)
        return np.column_stack([1.0 - p1, p1])


class _FastCARTNode:
    __slots__ = ("feature", "threshold", "left", "right", "prob")

    def __init__(self, prob: float) -> None:
        self.feature: int = -1
        self.threshold: float = 0.0
        self.left: Optional["_FastCARTNode"] = None
        self.right: Optional["_FastCARTNode"] = None
        self.prob: float = prob


class _FastDecisionTree:
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_leaf: int = 18,
        max_features: int = 5,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.root_: Optional[_FastCARTNode] = None

    def _build(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, depth: int) -> _FastCARTNode:
        w_sum = float(np.sum(w))
        prob = float(np.sum(w * y) / max(w_sum, 1e-9))
        node = _FastCARTNode(prob=prob)

        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or prob <= 1e-4 or prob >= 1.0 - 1e-4:
            return node

        n_total, n_feat = X.shape
        feat_indices = self.rng.choice(n_feat, size=min(self.max_features, n_feat), replace=False)

        best_gain = 0.0
        best_feat = -1
        best_thresh = 0.0
        parent_impurity = prob * (1.0 - prob)

        for f in feat_indices:
            col = X[:, f]
            qs = np.quantile(col, [0.25, 0.5, 0.75])
            for thr in np.unique(qs):
                left_mask = col <= thr
                n_left = int(np.sum(left_mask))
                n_right = n_total - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                w_l = float(np.sum(w[left_mask]))
                w_r = w_sum - w_l
                if w_l <= 0 or w_r <= 0:
                    continue
                p_l = float(np.sum(w[left_mask] * y[left_mask]) / w_l)
                p_r = float(np.sum(w[~left_mask] * y[~left_mask]) / w_r)
                imp_l = p_l * (1.0 - p_l)
                imp_r = p_r * (1.0 - p_r)
                gain = parent_impurity - (w_l / w_sum) * imp_l - (w_r / w_sum) * imp_r
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f)
                    best_thresh = float(thr)

        if best_feat == -1 or best_gain < 1e-5:
            return node

        node.feature = best_feat
        node.threshold = best_thresh
        l_mask = X[:, best_feat] <= best_thresh
        node.left = self._build(X[l_mask], y[l_mask], w[l_mask], depth + 1)
        node.right = self._build(X[~l_mask], y[~l_mask], w[~l_mask], depth + 1)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> "_FastDecisionTree":
        self.root_ = self._build(X, y, w, depth=0)
        return self

    def _predict_node(self, node: _FastCARTNode, X: np.ndarray, idx: np.ndarray, out: np.ndarray) -> None:
        if node.feature == -1 or node.left is None or node.right is None or len(idx) == 0:
            out[idx] = node.prob
            return
        go_left = X[idx, node.feature] <= node.threshold
        self._predict_node(node.left, X, idx[go_left], out)
        self._predict_node(node.right, X, idx[~go_left], out)

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros(len(X), dtype=float)
        if self.root_ is not None:
            self._predict_node(self.root_, X, np.arange(len(X)), out)
        return out


class RandomForestClassifierNP:
    def __init__(
        self,
        n_estimators: int = 40,
        max_depth: int = 5,
        min_samples_leaf: int = 18,
        class_weight: Optional[str] = "balanced_subsample",
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = min(int(n_estimators), 40)
        self.max_depth = min(int(max_depth), 5)
        self.min_samples_leaf = int(min_samples_leaf)
        self.class_weight = class_weight
        self.random_state = random_state
        self.trees_: List[_FastDecisionTree] = []

    def fit(self, X: Any, y: Any) -> "RandomForestClassifierNP":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n_samples, n_features = X_arr.shape
        max_f = max(int(np.sqrt(n_features)) + 2, 3)

        if self.class_weight in ("balanced", "balanced_subsample"):
            n_pos = max(np.sum(y_arr == 1), 1.0)
            n_neg = max(np.sum(y_arr == 0), 1.0)
            base_w = np.where(y_arr == 1, (n_samples / (1.65 * n_pos)), (n_samples / (2.35 * n_neg)))
        else:
            base_w = np.ones(n_samples, dtype=float)

        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        for _ in range(self.n_estimators):
            boot_idx = rng.integers(0, n_samples, size=n_samples)
            tree = _FastDecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_f,
                rng=rng,
            )
            tree.fit(X_arr[boot_idx], y_arr[boot_idx], base_w[boot_idx])
            self.trees_.append(tree)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        preds = np.mean([t.predict(X_arr) for t in self.trees_], axis=0)
        p1 = np.clip(preds, 1e-4, 1.0 - 1e-4)
        return np.column_stack([1.0 - p1, p1])


class HistGradientBoostingClassifierNP:
    def __init__(
        self,
        max_iter: int = 35,
        max_depth: int = 4,
        learning_rate: float = 0.10,
        min_samples_leaf: int = 20,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.max_iter = min(int(max_iter), 35)
        self.max_depth = min(int(max_depth), 4)
        self.learning_rate = float(learning_rate)
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = random_state
        self.init_logit_: float = 0.0
        self.trees_: List[_FastDecisionTree] = []

    def fit(self, X: Any, y: Any) -> "HistGradientBoostingClassifierNP":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        p_mean = float(np.clip(np.mean(y_arr), 1e-4, 1.0 - 1e-4))
        self.init_logit_ = float(np.log(p_mean / (1.0 - p_mean)))

        f_val = np.full(len(y_arr), self.init_logit_, dtype=float)
        rng = np.random.default_rng(self.random_state)
        n_feat = X_arr.shape[1]
        unit_w = np.ones(len(y_arr), dtype=float)

        self.trees_ = []
        for _ in range(self.max_iter):
            p = sigmoid_fn(f_val)
            residual_target = np.clip(0.5 + 0.5 * (y_arr - p), 0.0, 1.0)
            tree = _FastDecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max(int(0.75 * n_feat), 3),
                rng=rng,
            )
            tree.fit(X_arr, residual_target, unit_w)
            step = (tree.predict(X_arr) - 0.5) * 4.0
            f_val += self.learning_rate * step
            self.trees_.append(tree)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        f_val = np.full(len(X_arr), self.init_logit_, dtype=float)
        for tree in self.trees_:
            f_val += self.learning_rate * ((tree.predict(X_arr) - 0.5) * 4.0)
        p1 = sigmoid_fn(f_val)
        return np.column_stack([1.0 - p1, p1])


class IsotonicRegressionNP:
    def __init__(self, y_min: float = 1e-4, y_max: float = 1.0 - 1e-4, **kwargs: Any) -> None:
        self.y_min = y_min
        self.y_max = y_max
        self.x_knots_: Optional[np.ndarray] = None
        self.y_knots_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "IsotonicRegressionNP":
        x_arr = np.asarray(X, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()
        order = np.argsort(x_arr)
        x_s = x_arr[order]
        y_s = y_arr[order]

        blocks: List[List[float]] = [[float(y_s[i]), 1.0, float(x_s[i]), float(x_s[i])] for i in range(len(x_s))]
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][0] > blocks[i + 1][0]:
                total_w = blocks[i][1] + blocks[i + 1][1]
                avg_v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / total_w
                blocks[i] = [avg_v, total_w, blocks[i][2], blocks[i + 1][3]]
                blocks.pop(i + 1)
                if i > 0:
                    i -= 1
            else:
                i += 1

        xs = []
        ys = []
        for val, _, xmin, xmax in blocks:
            v_clip = float(np.clip(val, self.y_min, self.y_max))
            xs.extend([xmin, xmax])
            ys.extend([v_clip, v_clip])
        self.x_knots_ = np.asarray(xs, dtype=float)
        self.y_knots_ = np.asarray(ys, dtype=float)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=float).ravel()
        return np.clip(np.interp(x_arr, self.x_knots_, self.y_knots_), self.y_min, self.y_max)


def permutation_importance_np(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    base_prob = model.predict_proba(X)[:, 1]
    base_auc = roc_auc_score_np(y, base_prob)

    n_cols = X.shape[1]
    drops = np.zeros((n_cols, n_repeats), dtype=float)
    X_base = X.values.copy()

    for col_idx in range(n_cols):
        for r in range(n_repeats):
            X_perm = X_base.copy()
            X_perm[:, col_idx] = rng.permutation(X_perm[:, col_idx])
            perm_prob = model.predict_proba(X_perm)[:, 1]
            perm_auc = roc_auc_score_np(y, perm_prob)
            drops[col_idx, r] = base_auc - perm_auc

    return np.mean(drops, axis=1), np.std(drops, axis=1)


def build_model(model_name: str, seed: int = 42, **kwargs: Any) -> Any:
    name = model_name.lower().strip()
    if name in ("logistic_regression", "logreg", "lr"):
        return LogisticRegressionNP(C=1.0, max_iter=150, class_weight="balanced", random_state=seed, **kwargs)
    if name in ("random_forest", "rf"):
        return RandomForestClassifierNP(
            n_estimators=40, max_depth=5, min_samples_leaf=18, class_weight="balanced_subsample", random_state=seed, **kwargs
        )
    if name in ("gradient_boosting", "hist_gradient_boosting", "gbdt"):
        return HistGradientBoostingClassifierNP(
            max_iter=35, max_depth=4, learning_rate=0.10, min_samples_leaf=20, random_state=seed, **kwargs
        )
    raise ValueError(f"Unsupported model_name: {model_name}")
