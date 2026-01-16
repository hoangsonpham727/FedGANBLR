from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.datasets import fetch_openml
try:
    from ucimlrepo import fetch_ucirepo
    UCIMLREPO_AVAILABLE = True
except ImportError:
    UCIMLREPO_AVAILABLE = False
    print("[Warning] ucimlrepo package not installed. Install with: pip install ucimlrepo")
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from pathlib import Path
from base_models.KDependenceBayesian import MLE_KDB, DiscBN_KDB
from typing import List

def _align_proba_cols(proba: np.ndarray, classes_pred: np.ndarray, C: int, eps: float = 1e-12):
    proba = np.asarray(proba, dtype=np.float64)
    n_samples = proba.shape[0]
    out = np.full((n_samples, C), eps, dtype=np.float64)

    # Map each column of proba to the correct class index in the full C-sized array.
    cols = np.asarray(classes_pred)
    for j, cls in enumerate(cols):
        try:
            cls_int = int(cls)
        except Exception:
            continue
        if 0 <= cls_int < C:
            out[:, cls_int] = proba[:, j]

    # Normalize rows (avoid division by zero)
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    out = out / row_sums
    return out


def _safe_log_loss(y_true: np.ndarray, proba: np.ndarray, C: int, eps: float = 1e-12) -> float:
    """
    Robust negative log-likelihood that never raises due to normalization issues.
    - Clips probabilities to [eps, 1-eps]
    - Renormalizes each row to sum to 1
    - Ignores samples whose labels fall outside [0, C-1]
    """
    # Debug: verify this function is actually being called
    import sys
    if not hasattr(_safe_log_loss, '_called'):
        print("[DEBUG] _safe_log_loss is being used (this message appears once)", file=sys.stderr)
        _safe_log_loss._called = True
    
    y_true = np.asarray(y_true, dtype=np.int64)
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2:
        raise ValueError(f"Expected 2D proba array, got shape {proba.shape}")

    # Ensure correct number of columns; if fewer, pad uniformly, if more, truncate
    n, k = proba.shape
    if k < C:
        pad = np.full((n, C - k), 1.0 / C, dtype=np.float64)
        proba = np.concatenate([proba, pad], axis=1)
    elif k > C:
        proba = proba[:, :C]

    # Clip and renormalize
    proba = np.clip(proba, eps, 1.0 - eps)
    row_sums = proba.sum(axis=1, keepdims=True)
    bad_rows = (row_sums <= 0) | ~np.isfinite(row_sums)
    if np.any(bad_rows):
        proba[bad_rows] = 1.0 / C
        row_sums[bad_rows] = 1.0
    proba = proba / row_sums

    # Only keep samples with in-range labels
    mask = (y_true >= 0) & (y_true < C)
    if not np.any(mask):
        return float("nan")
    y_used = y_true[mask]
    p_used = proba[mask, :]

    # Gather probabilities of the true class
    idx_rows = np.arange(y_used.shape[0], dtype=np.int64)
    p_true = p_used[idx_rows, y_used]
    p_true = np.clip(p_true, eps, 1.0)  # final safety
    return float(-np.mean(np.log(p_true)))

def _make_pipes():
    return {
        "lr":  Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore")), ("lr",  LogisticRegression(max_iter=500, multi_class="multinomial"))]),
        "mlp": Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore")), ("mlp", MLPClassifier(max_iter=300, random_state=0))]),
        "rf":  Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore")), ("rf",  RandomForestClassifier(n_estimators=200, random_state=0))]),
        "xgb": Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore")), ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", verbosity=0, random_state=0))]),
    }

def _ohe_pipeline(model, step_name: str = "clf"):
    """Return a Pipeline with a OneHotEncoder followed by the provided estimator."""
    return Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore")), (step_name, model)])

def _evaluate_synthetic_classifiers(X_syn: np.ndarray, y_syn: np.ndarray, X_test_int: np.ndarray, y_test_int: np.ndarray):
    """
    Train standard pipelines (one-hot + classifier) on synthetic (X_syn,y_syn)
    and evaluate on integer-encoded X_test_int,y_test_int.
    Returns dict: acc_lr, nll_lr, acc_mlp, nll_mlp, acc_rf, nll_rf
    """

    pipes = _make_pipes()
    out = {}
    C = int(y_test_int.max()) + 1
    # Compress synthetic labels to contiguous 0..K-1 for training (required by XGB)
    y_syn = np.asarray(y_syn, dtype=np.int64)
    classes_syn_sorted = np.unique(y_syn)
    y_syn_comp = np.searchsorted(classes_syn_sorted, y_syn)  # 0..K-1

    for key, pipe in pipes.items():
        pipe.fit(X_syn, y_syn_comp)
        proba = pipe.predict_proba(X_test_int)

        # Map estimator's classes_ (in compressed space) back to original label IDs, then align to 0..C-1
        est = pipe.named_steps.get(key, pipe.steps[-1][1])
        classes_pred_comp = getattr(est, "classes_", None)
        classes_pred_orig = None
        if classes_pred_comp is not None:
            classes_pred_comp = np.asarray(classes_pred_comp, dtype=np.int64)
            classes_pred_orig = classes_syn_sorted[classes_pred_comp]  # original label IDs in [0..C-1]

        if classes_pred_orig is not None and len(classes_pred_orig) != C:
            proba = _align_proba_cols(proba, classes_pred_orig, C)

        preds = proba.argmax(axis=1)
        out[f"acc_{key}"] = accuracy_score(y_test_int, preds)
        out[f"nll_{key}"] = _safe_log_loss(y_test_int, proba, C)
    return out

def discretize_train_test_no_leak(X_train_df, y_train_sr, X_test_df, y_test_sr,
                                  strategy="ef", ef_bins=10):
    """
    Discretize TRAIN -> fit mappings/bins on TRAIN only, then transform TEST.
    - Categorical: factorize on TRAIN; TEST unseen categories -> 'other' bin.
    - Numeric 'ef': KBinsDiscretizer(strategy='quantile') fit on TRAIN.
    - Numeric 'mdlp': Fayyad-Irani cuts computed on TRAIN labels; then digitize TRAIN/TEST.

    Returns:
        Xtr_int, Xte_int, ytr_int, yte_int, card_feat, classes
    """
    # y mapping from TRAIN categories only
    y_train_cat = y_train_sr.astype("category")
    classes = y_train_cat.cat.categories  # train-only classes
    y_map = {c: i for i, c in enumerate(classes)}
    ytr_int = y_train_sr.map(y_map).to_numpy().astype(np.int32)
    yte_mapped = y_test_sr.map(y_map)
    if yte_mapped.isnull().any():
        # Unseen label in TEST; fail fast to avoid silent leakage/mismatch
        missing = sorted(set(y_test_sr[yte_mapped.isnull()].unique().tolist()))
        raise ValueError(f"Test contains unseen class labels not present in train: {missing}")
    yte_int = yte_mapped.to_numpy().astype(np.int32)

    Xtr_blocks, Xte_blocks, card_feat = [], [], []

    # Identify categorical vs numeric
    cat_cols = [c for c in X_train_df.columns if X_train_df[c].dtype.kind in ("O","b","U","S")
                or str(X_train_df[c].dtype).startswith("category")]
    num_cols = [c for c in X_train_df.columns if c not in cat_cols]

    # Categorical factorization using TRAIN categories
    for c in cat_cols:
        tr_cat = X_train_df[c].astype("category")
        cats = list(tr_cat.cat.categories)
        r = len(cats)
        tr_codes = tr_cat.cat.codes.to_numpy().astype(np.int32)

        # Map TEST: unseen -> 'other' bin at index r
        te_cat = X_test_df[c].astype("category").cat.set_categories(cats)
        te_codes = te_cat.cat.codes.to_numpy().astype(np.int32)
        unseen_mask = (te_codes == -1)
        if unseen_mask.any():
            te_codes[unseen_mask] = r
            r_eff = r + 1
        else:
            r_eff = r

        Xtr_blocks.append(tr_codes.reshape(-1, 1))
        Xte_blocks.append(te_codes.reshape(-1, 1))
        card_feat.append(r_eff)

    # Numeric discretization
    if num_cols:
        if strategy == "ef":
            n_bins_eff = int(ef_bins) if ef_bins is not None else 10
            kbin = KBinsDiscretizer(n_bins=n_bins_eff, encode="ordinal", strategy="quantile")
            Xtr_disc = kbin.fit_transform(X_train_df[num_cols]).astype(np.int32)
            Xte_disc = kbin.transform(X_test_df[num_cols]).astype(np.int32)
            Xtr_disc = np.clip(Xtr_disc, 0, n_bins_eff - 1).astype(np.int32)
            Xte_disc = np.clip(Xte_disc, 0, n_bins_eff - 1).astype(np.int32)
            Xtr_blocks.append(Xtr_disc)
            Xte_blocks.append(Xte_disc)
            card_feat.extend([n_bins_eff] * Xtr_disc.shape[1])
        elif strategy == "mdlp":
            # Use MDLP cuts computed on TRAIN only
            Xtr_num = X_train_df[num_cols].to_numpy(dtype=float)
            Xte_num = X_test_df[num_cols].to_numpy(dtype=float)
            discs_tr, discs_te = [], []
            for j in range(Xtr_num.shape[1]):
                cuts = _mdlp_discretize_1d(Xtr_num[:, j], ytr_int)
                tr_b = np.digitize(Xtr_num[:, j], bins=cuts, right=False).astype(np.int32)
                te_b = np.digitize(Xte_num[:, j], bins=cuts, right=False).astype(np.int32)
                discs_tr.append(tr_b.reshape(-1, 1))
                discs_te.append(te_b.reshape(-1, 1))
                card_feat.append(len(cuts) + 1 if cuts.size > 0 else 1)
            Xtr_blocks.append(np.hstack(discs_tr) if discs_tr else np.empty((len(X_train_df), 0), dtype=np.int32))
            Xte_blocks.append(np.hstack(discs_te) if discs_te else np.empty((len(X_test_df), 0), dtype=np.int32))
        else:
            raise ValueError("Unknown strategy. Use 'ef' or 'mdlp'.")

    Xtr_int = np.hstack(Xtr_blocks) if Xtr_blocks else np.empty((len(X_train_df), 0), dtype=np.int32)
    Xte_int = np.hstack(Xte_blocks) if Xte_blocks else np.empty((len(X_test_df), 0), dtype=np.int32)
    return Xtr_int, Xte_int, ytr_int, yte_int, card_feat, classes

def dirichlet_split(X, y, num_clients: int, alpha: float, rng=np.random.default_rng(42)):
    X = np.asarray(X)
    y = np.asarray(y)
    classes = np.unique(y)
    idx_per_class = {c: np.where(y == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(idx_per_class[c])
    props = rng.dirichlet([alpha] * num_clients, size=len(classes))  # [C, num_clients]
    splits = [[] for _ in range(num_clients)]
    for ci, c in enumerate(classes):
        idxs = idx_per_class[c]
        if len(idxs) == 0: continue
        # turn proportions into counts
        cnt = (props[ci] / props[ci].sum() * len(idxs)).astype(int)
        # fix rounding
        while cnt.sum() < len(idxs):
            cnt[rng.integers(0, num_clients)] += 1
        off = 0
        for k in range(num_clients):
            if cnt[k] > 0:
                splits[k].extend(idxs[off:off+cnt[k]])
                off += cnt[k]
    out = []
    for k in range(num_clients):
        sel = np.array(splits[k], dtype=int)
        if sel.size == 0:
            # return empty views from X/y (not undefined Xtr_int/ytr_int)
            out.append((X[:0], y[:0]))
            # print the size of each client's data
            print(f"Client {k}: 0 samples")
        else:
            out.append((X[sel], y[sel]))
            print(f"Client {k}: {sel.size} samples")
        
    return out


# =============================================================================
# Utils: fetching
# =============================================================================
def fetch_openml_safely(name, target, data_id=None, alt_names=None, version=1):
    """
    Fetch dataset using ucimlrepo first (by data_id), then fallback to OpenML.
    
    Args:
        name: Primary dataset name to try (for OpenML fallback)
        target: Target column name
        data_id: Dataset ID (UCI ID for ucimlrepo, OpenML ID for fallback)
        alt_names: Alternative names to try (for OpenML fallback)
        version: Dataset version (default 1, for OpenML fallback)
    
    Returns:
        X, y: Feature DataFrame and target Series
    """
    last_err = None
    
    # Strategy 1: Try ucimlrepo by data_id (UCI ML Repository)
    if UCIMLREPO_AVAILABLE and data_id is not None:
        try:
            print(f"[Data Fetch] Trying ucimlrepo with ID={data_id}...")
            ds = fetch_ucirepo(id=data_id)
            
            # Extract features and targets
            X = ds.data.features
            y = ds.data.targets
            
            # Handle case where features might be None or empty
            if X is None:
                raise ValueError(f"No features found for ucimlrepo dataset ID {data_id}")
            
            # Handle target column - could be single column or multiple
            if y is None:
                # Some UCI datasets don't have separate targets - check if target column is in features
                if target in X.columns:
                    y = X[target]
                    X = X.drop(columns=[target])
                else:
                    raise ValueError(f"No target data found for ucimlrepo dataset ID {data_id}")
            elif hasattr(y, 'empty') and y.empty:
                # Check if target is in features instead
                if target in X.columns:
                    y = X[target]
                    X = X.drop(columns=[target])
                else:
                    raise ValueError(f"Target data is empty for ucimlrepo dataset ID {data_id}")
            
            # If targets is a DataFrame with multiple columns, try to find the target column
            if isinstance(y, pd.DataFrame):
                if target in y.columns:
                    y = y[target]
                elif len(y.columns) == 1:
                    y = y.iloc[:, 0]
                else:
                    # Use first column if target name not found
                    print(f"[Data Fetch] Warning: Target '{target}' not found in ucimlrepo targets. Using first column: {y.columns[0]}")
                    y = y.iloc[:, 0]
            
            # Ensure X is a DataFrame
            if not isinstance(X, pd.DataFrame):
                if hasattr(X, 'values'):
                    X = pd.DataFrame(X.values, columns=getattr(X, 'columns', None))
                else:
                    X = pd.DataFrame(X)
            
            # Ensure y is a Series
            if not isinstance(y, pd.Series):
                if hasattr(y, 'values'):
                    y = pd.Series(y.values.flatten() if hasattr(y.values, 'flatten') else y.values)
                elif hasattr(y, '__iter__') and not isinstance(y, str):
                    y = pd.Series(list(y))
                else:
                    y = pd.Series([y])
            
            # Reset indices to ensure alignment
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
            
            print(f"[Data Fetch] Successfully loaded from ucimlrepo: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            last_err = e
            print(f"[Data Fetch] ucimlrepo failed for ID={data_id}: {repr(e)}")
            print(f"[Data Fetch] Falling back to OpenML...")
    
    # Strategy 2: Fallback to OpenML - Try by name (primary + alternatives) with version
    names_to_try = [name] + (alt_names or []) if name else (alt_names or [])
    for nm in names_to_try:
        if nm is None:
            continue
        try:
            print(f"[Data Fetch] Trying OpenML with name='{nm}' (version={version})...")
            ds = fetch_openml(name=nm, version=version, as_frame=True, parser='auto')
            X = ds.frame.drop(columns=[target], errors="ignore")
            y = ds.frame[target] if target in ds.frame.columns else ds.target
            if y is None:
                raise ValueError(f"Target '{target}' not found for dataset {nm}.")
            print(f"[Data Fetch] Successfully loaded from OpenML: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
        except Exception as e:
            last_err = e
    
    # Strategy 3: Fallback to OpenML - Try by name without version
    for nm in names_to_try:
        if nm is None:
            continue
        try:
            print(f"[Data Fetch] Trying OpenML with name='{nm}' (no version)...")
            ds = fetch_openml(name=nm, as_frame=True, parser='auto')
            X = ds.frame.drop(columns=[target], errors="ignore")
            y = ds.frame[target] if target in ds.frame.columns else ds.target
            if y is None:
                raise ValueError(f"Target '{target}' not found for dataset {nm}.")
            print(f"[Data Fetch] Successfully loaded from OpenML: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
        except Exception as e:
            last_err = e
    
    # Strategy 4: Fallback to OpenML - Try by data_id with version
    if data_id is not None:
        try:
            print(f"[Data Fetch] Trying OpenML with data_id={data_id} (version={version})...")
            ds = fetch_openml(data_id=data_id, version=version, as_frame=True, parser='auto')
            X = ds.frame.drop(columns=[target], errors="ignore")
            y = ds.frame[target] if target in ds.frame.columns else ds.target
            if y is None:
                raise ValueError(f"Target '{target}' not found for dataset with data_id={data_id}.")
            print(f"[Data Fetch] Successfully loaded from OpenML: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
        except Exception as e:
            last_err = e
    
    # Strategy 5: Fallback to OpenML - Try by data_id without version
    if data_id is not None:
        try:
            print(f"[Data Fetch] Trying OpenML with data_id={data_id} (no version)...")
            ds = fetch_openml(data_id=data_id, as_frame=True, parser='auto')
            X = ds.frame.drop(columns=[target], errors="ignore")
            y = ds.frame[target] if target in ds.frame.columns else ds.target
            if y is None:
                raise ValueError(f"Target '{target}' not found for dataset with data_id={data_id}.")
            print(f"[Data Fetch] Successfully loaded from OpenML: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
        except Exception as e:
            last_err = e
    
    # All strategies failed
    identifier = f"name='{name}'" if name else f"data_id={data_id}"
    raise RuntimeError(f"Failed to fetch dataset {identifier}. Last error: {repr(last_err)}")

# =============================================================================
# Discretization helpers
#   - Categorical: factorize to ints on FULL dataset
#   - Numeric EF: KBins (quantile) on FULL dataset
#   - Numeric MDLP: Fayyad-Irani supervised MDL discretization per feature (implemented below)
# =============================================================================

def _entropy(counts):
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts[counts > 0] / n
    return float(-(p * np.log2(p)).sum())

def _class_counts(y):
    # expects integer-coded classes 0..C-1
    C = int(y.max()) + 1 if y.size > 0 else 0
    return np.bincount(y, minlength=C)

def _candidate_cutpoints(x, y):
    # x, y must be sorted by x
    candidates = []
    for i in range(1, len(x)):
        if y[i] != y[i-1] and x[i] > x[i-1]:
            candidates.append((x[i-1] + x[i]) / 2.0)
    return candidates

def _info_gain(y, y_left, y_right):
    n = len(y)
    n_l = len(y_left)
    n_r = len(y_right)
    H = _entropy(_class_counts(y))
    H_l = _entropy(_class_counts(y_left))
    H_r = _entropy(_class_counts(y_right))
    return H - (n_l/n)*H_l - (n_r/n)*H_r, H, H_l, H_r

def _mdlp_accept(y, y_left, y_right, gain, H, H_l, H_r):
    # Fayyad-Irani MDL criterion (1993)
    k  = int(y.max()) + 1 if y.size > 0 else 0              # classes before split
    k1 = int(y_left.max()) + 1 if y_left.size > 0 else 1
    k2 = int(y_right.max()) + 1 if y_right.size > 0 else 1
    delta = np.log2(max(1, 3**k - 2)) - (k*H - k1*H_l - k2*H_r)
    n = len(y)
    threshold = (np.log2(max(1, n - 1)) + delta) / max(1, n)
    return gain > threshold

def _mdlp_discretize_1d(x, y):
    # returns sorted cut points
    order = np.argsort(x, kind="mergesort")
    x_s = x[order]
    y_s = y[order]
    cuts = []

    def _recurse(xs, ys):
        cands = _candidate_cutpoints(xs, ys)
        if not cands:
            return
        best_gain = -1.0
        best_t = None
        best_split = None
        for t in cands:
            mask = xs <= t
            yl = ys[mask]
            yr = ys[~mask]
            gain, H, H_l, H_r = _info_gain(ys, yl, yr)
            if gain > best_gain:
                best_gain = gain
                best_t = t
                best_split = (yl, yr, H, H_l, H_r)
        yl, yr, H, H_l, H_r = best_split
        if _mdlp_accept(ys, yl, yr, best_gain, H, H_l, H_r):
            cuts.append(best_t)
            _recurse(xs[xs <= best_t], ys[xs <= best_t])
            _recurse(xs[xs >  best_t], ys[xs >  best_t])

    _recurse(x_s, y_s)
    cuts_sorted = np.array(sorted(set(cuts)), dtype=float)
    return cuts_sorted

def discretize_full_dataset(X_df, y_sr, strategy="ef", ef_bins=10):
    """
    Discretize FULL dataset before splitting.
    - All categorical/string columns → factorized 0..R-1
    - Numeric columns:
        * 'ef'   → quantile bins with n=ef_bins on FULL data
        * 'mdlp' → supervised MDL (Fayyad-Irani) using FULL y
    Returns:
        X_int (np.ndarray), y_int (np.ndarray), card_feat (list[int]), classes (Index)
    """
    # unify y
    y_cat = y_sr.astype("category")
    classes = y_cat.cat.categories
    y_int = y_cat.cat.codes.to_numpy().astype(np.int32)
    
    X = X_df.copy()
    # Identify categorical vs numeric
    cat_cols = [c for c in X.columns if X[c].dtype.kind in ("O","b","U","S") or str(X[c].dtype).startswith("category")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    feat_blocks = []
    card_feat = []

    # Categorical factorization (on FULL data)
    for c in cat_cols:
        cat = X[c].astype("category")
        codes = cat.cat.codes.to_numpy().astype(np.int32)
        r = len(cat.cat.categories)  # exact cardinality; no 'other' bin since FULL mapping
        feat_blocks.append(codes.reshape(-1,1))
        card_feat.append(r)

    # Numeric discretization
    if num_cols:
        if strategy == "ef":
            n_bins_eff = int(ef_bins) if ef_bins is not None else 10
            kbin = KBinsDiscretizer(n_bins=n_bins_eff, encode="ordinal", strategy="quantile")
            Xdisc = kbin.fit_transform(X[num_cols]).astype(np.int32)
            Xdisc = np.clip(Xdisc, 0, n_bins_eff-1).astype(np.int32)
            feat_blocks.append(Xdisc)
            card_feat.extend([n_bins_eff]*Xdisc.shape[1])
        elif strategy == "mdlp":
            # MDL supervised discretization per feature
            Xnum = X[num_cols].to_numpy(dtype=float)
            discs = []
            for j in range(Xnum.shape[1]):
                xj = Xnum[:, j]
                cuts = _mdlp_discretize_1d(xj, y_int)
                # bin using cuts; number of bins = len(cuts)+1
                bj = np.digitize(xj, bins=cuts, right=False).astype(np.int32)
                discs.append(bj.reshape(-1,1))
                card_feat.append(len(cuts)+1 if cuts.size>0 else 1)
            feat_blocks.append(np.hstack(discs) if discs else np.empty((len(X),0), dtype=np.int32))
        else:
            raise ValueError("Unknown strategy. Use 'ef' or 'mdlp'.")

    X_int = np.hstack(feat_blocks) if feat_blocks else np.empty((len(X),0), dtype=np.int32)
    return X_int, y_int, card_feat, classes
def onehot_all(X_int: np.ndarray, card_feat: List[int], dtype=np.float32) -> np.ndarray:
    """
    Convert integer-encoded feature matrix -> concatenated one-hot matrix.

    Parameters
    ----------
    X_int : np.ndarray
        Integer matrix shape [N, F] (features only). If 1D, treated as single column.
    card_feat : List[int]
        Per-feature cardinalities (length F). Each feature j will produce card_feat[j] columns.
        If None is passed, cardinalities will be inferred from the data (max value + 1 per column).
    dtype : data-type for output (default np.float32)

    Returns
    -------

    One-hot encoded matrix shape [N, sum(card_feat)].
    If X_int has values outside [0, card-1] they are ignored (row remains zero for that feature).
    """
    X = np.asarray(X_int)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    N, F = X.shape
    if F == 0:
        return np.zeros((N, 0), dtype=dtype)
    # Allow card_feat to be None — infer cardinalities from data
    if card_feat is None:
        inferred = []
        for j in range(F):
            col = X[:, j]
            # consider only finite, non-negative integers
            mask_valid = np.isfinite(col) & (col >= 0)
            if not mask_valid.any():
                inferred.append(0)
            else:
                # assume integer-encoded features; infer as max+1
                inferred.append(int(np.nanmax(col[mask_valid])) + 1)
        card_feat = inferred
    if len(card_feat) != F:
        raise ValueError(f"card_feat length ({len(card_feat)}) must match number of columns in X_int ({F})")
    parts = []
    for j, r in enumerate(card_feat):
        if r <= 0:
            # zero-card feature -> skip with zero-width block
            parts.append(np.zeros((N, 0), dtype=dtype))
            continue
        col = X[:, j].astype(np.int64, copy=False)
        mat = np.zeros((N, r), dtype=dtype)
        # valid positions only
        mask = (col >= 0) & (col < r)
        if mask.any():
            rows = np.nonzero(mask)[0]
            vals = col[mask]
            mat[rows, vals] = 1.0
        parts.append(mat)
    return np.concatenate(parts, axis=1) if parts else np.zeros((N, 0), dtype=dtype)

# Array builders and evaluation (split AFTER full discretization)
def build_full_table(y_train_int, y_test_int, Xtr_int, Xte_int):
    train_arr = np.column_stack([y_train_int, Xtr_int]).astype(np.int32)
    test_arr  = np.column_stack([y_test_int,  Xte_int]).astype(np.int32)
    return train_arr, test_arr

def _disc_to_py_thetas(gen: 'DiscBN_KDB') -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Extract py/thetas in probability space from a constrained DiscBN_KDB."""
    py = np.asarray(tf.nn.softmax(gen.logits_y)).ravel()
    thetas: dict[int, np.ndarray] = {}
    V = len(gen.card)
    for v in range(V):
        if v == gen.y_index:
            continue
        if gen.logits_v is not None: 
            thetas[v] = np.asarray(tf.nn.softmax(gen.logits_v[v], axis=-1))
    return py, thetas

def _mle_to_payload(mle: 'MLE_KDB', parents_excl_y: dict[int, list[int]]) -> dict:
    return dict(
        py=np.asarray(mle.py, dtype=np.float64),
        thetas={int(v): np.asarray(t, dtype=np.float64) for v, t in mle.theta.items()},
        card=list(mle.card), y_index=int(mle.y_index), parents={int(v): list(pa) for v, pa in parents_excl_y.items()}
    )

def _disc_to_payload(gen: 'DiscBN_KDB', parents_excl_y: dict[int, list[int]]) -> dict:
    py, thetas = _disc_to_py_thetas(gen)
    return dict(
        py=np.asarray(py, dtype=np.float64),
        thetas={int(v): np.asarray(t, dtype=np.float64) for v, t in thetas.items()},
        card=list(gen.card), y_index=int(gen.y_index), parents={int(v): list(pa) for v, pa in parents_excl_y.items()}
    )

def _gm_to_payload(gm: dict, card: list[int], parents_excl_y: dict[int, list[int]], y_index: int) -> dict:
    return dict(
        py=np.asarray(gm["py"], dtype=np.float64),
        thetas={int(v): np.asarray(t, dtype=np.float64) for v, t in gm["thetas"].items()},
        card=list(card), y_index=int(y_index), parents={int(v): list(pa) for v, pa in parents_excl_y.items()}
    )

def save_kdb_model_npz_json(out_npz: Path, out_json: Path, model: dict):
    """Save a KDB model payload to .npz (arrays) and .json (human-friendly)."""
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(int(v) for v in model["thetas"].keys())
    np.savez_compressed(
        out_npz,
        py=model["py"],
        card=np.asarray(model["card"], dtype=np.int32),
        y_index=np.asarray([model["y_index"]], dtype=np.int32),
        theta_keys=np.asarray(keys, dtype=np.int32),
        parents_json=np.asarray([json.dumps({str(k): v for k, v in model["parents"].items()})], dtype=object),
        **{f"theta_{int(v)}": np.asarray(model["thetas"][int(v)], dtype=np.float64) for v in keys}
    )
    with out_json.open("w") as f:
        json.dump(
            dict(
                py=model["py"].tolist(),
                thetas={str(int(v)): model["thetas"][int(v)].tolist() for v in keys},
                card=list(map(int, model["card"])),
                y_index=int(model["y_index"]),
                parents={str(int(k)): list(map(int, v)) for k, v in model["parents"].items()}
            ),
            f
        )

def save_fold_data_npz(out_npz: Path,
                       Xtr_int: np.ndarray, ytr_int: np.ndarray,
                       Xte_int: np.ndarray, yte_int: np.ndarray,
                       card_feat: list[int], num_classes: int):
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        Xtr_int=Xtr_int.astype(np.int32),
        ytr_int=ytr_int.astype(np.int32),
        Xte_int=Xte_int.astype(np.int32),
        yte_int=yte_int.astype(np.int32),
        card_feat=np.asarray(card_feat, dtype=np.int32),
        num_classes=np.asarray([num_classes], dtype=np.int32),
    )

def save_fold_data_csv(out_dir: Path,
                       Xtr_df: pd.DataFrame, ytr_sr: pd.Series,
                       Xte_df: pd.DataFrame, yte_sr: pd.Series,
                       target_name: str = "target",
                       include_index: bool = False,
                       save_meta: bool = True) -> None:
    
    """Save one train/test fold to CSV WITHOUT any discretization."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = Xtr_df.copy()
    train_df[target_name] = ytr_sr.values

    test_df = Xte_df.copy()
    test_df[target_name] = yte_sr.values

    # Save CSVs
    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    train_df.to_csv(train_path, index=include_index)
    test_df.to_csv(test_path, index=include_index)
    
