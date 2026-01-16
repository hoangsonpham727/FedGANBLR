import numpy as np
import pandas as pd
import flwr as fl
import time
from pathlib import Path
import keras
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from typing import Any, List, Optional
from utils import build_full_table, dirichlet_split, _evaluate_synthetic_classifiers, _gm_to_payload, _make_pipes,\
_align_proba_cols, _mle_to_payload, _disc_to_payload, fetch_openml_safely, discretize_train_test_no_leak,\
save_fold_data_csv, save_fold_data_npz, save_kdb_model_npz_json, _safe_log_loss
from base_models.KDependenceBayesian import _compute_strides
from base_models.Ganblr import SimpleGANBLR, GANBLR
from federated_models.FedGanblr import derive_global_meta, build_global_kdb_from_gm, KDBGANStrategy, GANBLRFederatedClient
from federated_models.FedMLE import run_one_fold_fed_mle_naive, _aggregate_mle_counts
import itertools
import gc
from scipy.stats import kstest, wasserstein_distance
from scipy.spatial.distance import cdist, jensenshannon
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from numpy.typing import ArrayLike


def total_variation_distance(p_dist: pd.Series, s_dist: pd.Series) -> float:
    """Total Variation Distance (TVD) between two categorical distributions."""
    all_categories = p_dist.index.union(s_dist.index)
    p_full = p_dist.reindex(all_categories, fill_value=0)
    s_full = s_dist.reindex(all_categories, fill_value=0)
    tvd_score = 0.5 * np.sum(np.abs(p_full - s_full))
    return float(tvd_score)


def calculate_fidelity(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[float, float, float]:
    """
    Column Fidelity, Row Fidelity, Aggregate Fidelity.
    Ported from `experiments.ipynb` so it can be used during CV.
    """
    col_fidelities: list[float] = []

    # Numeric: Kolmogorov-Smirnov Statistic (KSS)
    for col in numeric_cols:
        try:
            kss, _ = kstest(real_df[col], synthetic_df[col])
            omega_col = 1.0 - kss
            col_fidelities.append(float(omega_col))
        except Exception:
            # Best-effort; skip unstable columns
            continue

    # Categorical: TVD
    for col in categorical_cols:
        real_dist = real_df[col].value_counts(normalize=True)
        synth_dist = synthetic_df[col].value_counts(normalize=True)
        tvd_score = total_variation_distance(real_dist, synth_dist)
        omega_col = 1.0 - tvd_score
        col_fidelities.append(float(omega_col))

    column_fidelity = float(np.mean(col_fidelities)) if col_fidelities else 0.0

    # Row fidelity over all column pairs
    row_fidelities: list[float] = []
    all_cols = list(numeric_cols) + list(categorical_cols)
    for i in range(len(all_cols)):
        for j in range(i + 1, len(all_cols)):
            col_a, col_b = all_cols[i], all_cols[j]
            is_num_a = col_a in numeric_cols
            is_num_b = col_b in numeric_cols

            if is_num_a and is_num_b:
                real_corr = real_df[[col_a, col_b]].corr().iloc[0, 1]
                synth_corr = synthetic_df[[col_a, col_b]].corr().iloc[0, 1]
                pc_discrepancy = np.abs(np.asarray(real_corr) - np.asarray(synth_corr))
                omega_row = 1.0 - (0.5 * pc_discrepancy)
                row_fidelities.append(float(omega_row))
            elif (col_a in categorical_cols) and (col_b in categorical_cols):
                real_joint = real_df.groupby([col_a, col_b]).size().div(len(real_df))
                synth_joint = synthetic_df.groupby([col_a, col_b]).size().div(len(synthetic_df))
                tvd_score = total_variation_distance(real_joint, synth_joint)
                omega_row = 1.0 - tvd_score
                row_fidelities.append(float(omega_row))

    row_fidelity = float(np.mean(row_fidelities)) if row_fidelities else 0.0
    aggregate_fidelity = (column_fidelity + row_fidelity) / 2.0
    return float(column_fidelity), float(row_fidelity), float(aggregate_fidelity)


def calculate_coverage(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> float:
    """Coverage (γ) – diversity and range replication. Ported from notebook."""
    coverage_scores: list[float] = []

    for col in categorical_cols:
        real_categories = set(real_df[col].dropna().unique())
        synth_categories = set(synthetic_df[col].dropna().unique())
        if not real_categories:
            continue
        covered_count = len(real_categories.intersection(synth_categories))
        categorical_coverage = covered_count / len(real_categories)
        coverage_scores.append(float(categorical_coverage))

    for col in numeric_cols:
        real_min, real_max = real_df[col].min(), real_df[col].max()
        synth_min, synth_max = synthetic_df[col].min(), synthetic_df[col].max()
        real_range = real_max - real_min
        if real_range == 0:
            coverage_scores.append(1.0)
            continue
        tau = np.maximum(0.0, (synth_min - real_min) / real_range)
        chi = np.maximum(0.0, (real_max - synth_max) / real_range)
        numerical_coverage = 1.0 - (tau + chi)
        coverage_scores.append(float(numerical_coverage))

    coverage_score = np.mean(coverage_scores) if coverage_scores else 0.0
    return float(coverage_score)


def _safe_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def calculate_dcr_privacy(real_df: pd.DataFrame,
                          synthetic_df: pd.DataFrame,
                          numeric_cols: List[str],
                          k: int = 1,
                          exclude_exact_matches: bool = True,
                          zero_eps: float = 1e-12,
                          sample_size: Optional[int] = None,
                          synth_batch_size: int = 50000,
                          real_batch_size: int = 50000) -> float:
    """
    Distance to Closest Records(DCR) quantifies the degree to which 
    the synthetic data inhibits the identification of original data
    entries.
    
    Sampling-based: For large datasets, samples data to reduce runtime while
    maintaining statistical validity.
    
    Memory-optimized: Uses double batching to process the sampled dataset without OOM.
    - Outer loop: batch over synthetic data
    - Inner loop: batch over real data (for very large real datasets)
    
    Parameters
    ----------
    real_df : pd.DataFrame
        Real training data
    synthetic_df : pd.DataFrame  
        Synthetic data
    numeric_cols : List[str]
        List of numeric column names
    k : int
        k-th nearest neighbor to use (default 1)
    exclude_exact_matches : bool
        Whether to exclude exact matches
    zero_eps : float
        Threshold for considering a match as exact
    sample_size : Optional[int]
        If provided, randomly sample up to this many records from both real and synthetic.
        If None, uses full dataset (default behavior).
    synth_batch_size : int
        Process synthetic data in batches of this size
    real_batch_size : int
        Process real data in batches of this size (for finding min distances)
    """
    real = real_df.copy()
    synth = synthetic_df.copy()
    synth = synth[real.columns.tolist()]
    
    # Sample data if sample_size is provided
    rng = np.random.default_rng(42)
    if sample_size is not None:
        if len(real) > sample_size:
            sample_idx = rng.choice(len(real), size=sample_size, replace=False)
            real = real.iloc[sample_idx].reset_index(drop=True)
        if len(synth) > sample_size:
            sample_idx = rng.choice(len(synth), size=sample_size, replace=False)
            synth = synth.iloc[sample_idx].reset_index(drop=True)
    
    num = [c for c in numeric_cols if c in real.columns]
    cat = [c for c in real.columns if c not in num]
    if real.shape[1] == 0 or len(real) == 0 or len(synth) == 0:
        return float("nan")
    if cat:
        ohe = _safe_ohe()
        ohe.fit(real[cat].astype("category"))
        Rcat = ohe.transform(real[cat].astype("category"))
        Scat = ohe.transform(synth[cat].astype("category"))
    else:
        Rcat = np.empty((len(real), 0))
        Scat = np.empty((len(synth), 0))
    if num:
        Rnum = real[num].to_numpy(dtype=float)
        Snum = synth[num].to_numpy(dtype=float)
    else:
        Rnum = np.empty((len(real), 0))
        Snum = np.empty((len(synth), 0))

    # --- Ensure dense numpy arrays (convert sparse OHE outputs if any) ---
    def _ensure_dense(arr):
        # convert sparse matrix (has .toarray) -> dense ndarray, otherwise np.asarray
        try:
            if hasattr(arr, "toarray"):
                return np.asarray(arr.toarray(), dtype=float)
        except Exception:
            pass
        return np.asarray(arr, dtype=float)

    Rnum = _ensure_dense(Rnum)
    Snum = _ensure_dense(Snum)
    Rcat = _ensure_dense(Rcat)
    Scat = _ensure_dense(Scat)

    R = np.hstack([Rnum, Rcat]) if (Rnum.size or Rcat.size) else np.zeros((len(real), 0))
    S = np.hstack([Snum, Scat]) if (Snum.size or Scat.size) else np.zeros((len(synth), 0))
    
    if R.shape[1] == 0:
        return float("nan")
    scaler = MinMaxScaler()
    R_s = scaler.fit_transform(R)
    S_s = scaler.transform(S)
    
    # Free intermediate arrays
    del R, S, Rnum, Snum, Rcat, Scat
    
    # Double batching to handle very large datasets:
    # - Outer loop batches over synthetic data
    # - Inner loop batches over real data to find k-NN
    k_idx = max(0, k - 1)
    nn_distances = []
    
    n_synth = S_s.shape[0]
    n_real = R_s.shape[0]
    
    for s_start in range(0, n_synth, synth_batch_size):
        s_end = min(s_start + synth_batch_size, n_synth)
        S_batch = S_s[s_start:s_end]
        batch_size_actual = S_batch.shape[0]
        
        # For each synthetic point, track the k smallest distances across all real batches
        # We need to keep track of candidates for k-NN
        if k == 1:
            # Optimization for k=1: just track minimum
            min_dists = np.full(batch_size_actual, np.inf)
        else:
            # For k>1, track top-k candidates per synthetic point
            topk_dists = [[] for _ in range(batch_size_actual)]
        
        # Inner loop: batch over real data
        for r_start in range(0, n_real, real_batch_size):
            r_end = min(r_start + real_batch_size, n_real)
            R_batch = R_s[r_start:r_end]
            
            # Compute distances for this (synth_batch, real_batch) pair
            dist_block = cdist(S_batch, R_batch, metric="euclidean")
            
            if exclude_exact_matches:
                dist_block[dist_block <= zero_eps] = np.inf
            
            if k == 1:
                # Update minimum distances
                block_mins = np.min(dist_block, axis=1)
                min_dists = np.minimum(min_dists, block_mins)
            else:
                # Collect candidates for k-NN
                for i in range(batch_size_actual):
                    row = dist_block[i]
                    finite = row[np.isfinite(row)]
                    if finite.size > 0:
                        # Keep only top-k smallest from this block
                        if finite.size <= k:
                            topk_dists[i].extend(finite.tolist())
                        else:
                            topk_dists[i].extend(np.partition(finite, k)[:k].tolist())
            
            # Free memory
            del dist_block
        
        # Finalize distances for this synthetic batch
        if k == 1:
            for d in min_dists:
                nn_distances.append(0.0 if np.isinf(d) else d)
        else:
            for i in range(batch_size_actual):
                candidates = np.array(topk_dists[i])
                if candidates.size == 0:
                    nn_distances.append(0.0)
                elif candidates.size <= k_idx:
                    nn_distances.append(np.max(candidates))
                else:
                    nn_distances.append(np.partition(candidates, k_idx)[k_idx])
    
    return float(np.nanmedian(nn_distances))


def jensen_shannon_divergence(p: ArrayLike, q: ArrayLike) -> float:
    """Jensen–Shannon divergence between two discrete distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(jensenshannon(p, q) ** 2)


def wasserstein_distance_categorical(real_counts: ArrayLike, syn_counts: ArrayLike) -> float:
    """Wasserstein distance for categorical distributions, scaled to [0, 1].
    
    The raw Wasserstein distance is normalized by dividing by the maximum
    possible distance (n_categories - 1), which occurs when all mass is
    concentrated at opposite ends of the category range.
    """
    real_counts = np.asarray(real_counts, dtype=np.float64)
    syn_counts = np.asarray(syn_counts, dtype=np.float64)
    p = real_counts / (real_counts.sum() + 1e-12)
    q = syn_counts / (syn_counts.sum() + 1e-12)
    categories = np.arange(len(p))
    raw_wd = wasserstein_distance(categories, categories, p, q)
    # Normalize to [0, 1]: max WD is (n_categories - 1) when distributions are at opposite ends
    max_wd = len(categories) - 1
    if max_wd <= 0:
        return 0.0
    return float(raw_wd / max_wd)


def compute_distributional_metrics(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> dict[str, Any]:
    """Compute per-column JSD/WD and their means."""
    jsd_values: list[float] = []
    wd_values: list[float] = []

    for col in real_df.columns:
        real_counts = real_df[col].value_counts().sort_index()
        syn_counts = syn_df[col].value_counts().sort_index()
        all_cats = real_counts.index.union(syn_counts.index)
        real_aligned = np.asarray(real_counts.reindex(all_cats, fill_value=0).values, dtype=np.float32)
        syn_aligned = np.asarray(syn_counts.reindex(all_cats, fill_value=0).values, dtype=np.float32)
        jsd = jensen_shannon_divergence(real_aligned, syn_aligned)
        wd = wasserstein_distance_categorical(real_aligned, syn_aligned)
        jsd_values.append(jsd)
        wd_values.append(wd)

    return dict(
        jsd_mean=float(np.mean(jsd_values)) if jsd_values else float("nan"),
        wd_mean=float(np.mean(wd_values)) if wd_values else float("nan"),
        jsd_values=jsd_values,
        wd_values=wd_values,
    )


def _build_real_df(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Helper: build a discretized 'real' DataFrame from X/y (all categorical)."""
    cols = ["y"] + [f"f{j}" for j in range(X.shape[1])]
    arr = np.column_stack([y, X]).astype(np.int32)
    return pd.DataFrame(arr, columns=cols)


def _build_syn_df(syn_full: np.ndarray) -> pd.DataFrame:
    y_syn = syn_full[:, 0]
    X_syn = syn_full[:, 1:]
    return _build_real_df(X_syn, y_syn)


def _build_kdb_from_payload(payload: dict) -> Any:
    """
    Reconstruct a KDB generator from a stored payload.
    Payload structure matches that used in experiment notebook helpers.
    """
    gm = dict(py=payload["py"], thetas=payload["thetas"])
    return build_global_kdb_from_gm(
        gm,
        card=payload["card"],
        parents=payload["parents"],
        y_index=payload["y_index"],
    )


def _sample_full_from_payload(payload: dict, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    gen = _build_kdb_from_payload(payload)
    return gen.sample(n=n, rng=(rng or np.random.default_rng(0)), order=None, return_y=True)

def run_one_fold_fed_ganblr(
    Xtr_int, ytr_int, Xte_int, yte_int, card_feat, num_classes,
    k_global=2, num_clients=5, num_rounds=5, dir_alpha=0.1,
    gamma=0.6, local_epochs=3, batch_size=1024, disc_epochs=1,
    cpt_mix=0.25, beta = 0.5 ,alpha_dir=1e-3, cap_train=None, clf="lr", verbose=False,
    eval_syn_frac: float = 0.5,          
    ray_local_mode: bool = False
):
    """
    Run one federated simulation on this fold and return TSTR metrics.
    eval_syn_frac: fraction of training size to sample for synthetic evaluation (<=1.0).
    ray_local_mode: use Ray local_mode to reduce process/memory overhead.
    """
    if cap_train is not None and len(Xtr_int) > cap_train:
        sel = np.random.default_rng(1).choice(len(Xtr_int), size=cap_train, replace=False)
        Xtr_int = Xtr_int[sel]; ytr_int = ytr_int[sel]

    card_all = [num_classes] + list(card_feat)
    y_index = 0
    train_arr, test_arr = build_full_table(ytr_int, yte_int, Xtr_int, Xte_int)
    card_glob, parents_glob, y_index_glob = derive_global_meta(train_arr, card_all, y_index, k=k_global)

    clients_data = dirichlet_split(Xtr_int, ytr_int, num_clients=num_clients, alpha=dir_alpha)

    strategy = KDBGANStrategy(
        k=k_global,
        gamma=gamma,
        local_epochs=local_epochs,
        batch_size=batch_size,
        disc_epochs=disc_epochs,
        cpt_mix=cpt_mix,
        beta_pow=beta,
        alpha_dir=alpha_dir,
        adversarial=False
    )
    strategy.set_global_meta(card=card_glob, parents=parents_glob, y_index=y_index_glob)

    def client_fn(cid: str):
        i = int(cid)
        Xc, yc = clients_data[i]
        return GANBLRFederatedClient(cid, Xc, yc)

    # ---- Controlled Ray init (reuse if already initialized) ----
    import ray
    if not ray.is_initialized():
        try:
            ray.init(local_mode=ray_local_mode, ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")
        except Exception:
            pass

    t0 = time.time()
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(clients_data),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    sim_time = time.time() - t0

    gm = strategy.global_model
    if gm is None:
        # Cleanup minimal
        try: del hist
        except: pass
        return dict(acc_lr=np.nan, nll_lr=np.nan, acc_mlp=np.nan, nll_mlp=np.nan,
                    acc_rf=np.nan, nll_rf=np.nan, train_time_sec=sim_time, weights=None, s_y=None)

    # ---- Shrink global model to float32 before sampling ----
    gm["py"] = np.asarray(gm["py"], dtype=np.float32)
    gm["thetas"] = {v: np.asarray(t, dtype=np.float32) for v, t in gm["thetas"].items()}

    gen_global = build_global_kdb_from_gm(gm, card_glob, parents_glob, y_index_glob)

    # Reduced synthetic sample size
    n_full = len(Xtr_int)
    n_syn = max(1, int(n_full * float(eval_syn_frac)))
    syn_full = gen_global.sample(n=n_syn, rng=np.random.default_rng(0), order=None, return_y=True)
    X_syn, y_syn = syn_full[:, 1:], syn_full[:, 0]

    ev = _evaluate_synthetic_classifiers(X_syn, y_syn, Xte_int, yte_int)

    gm_weights = gm.get("weights", None)
    weights_list = gm_weights.tolist() if isinstance(gm_weights, np.ndarray) else (gm_weights if isinstance(gm_weights, list) else None)
    gm_s_y = gm.get("s_y", None)
    s_y_list = gm_s_y.tolist() if isinstance(gm_s_y, np.ndarray) else (gm_s_y if isinstance(gm_s_y, list) else None)

    out = {"train_time_sec": sim_time, "weights": weights_list, "s_y": s_y_list}
    out.update(ev)

    parents_excl_y = {int(v): list(pa) for v, pa in parents_glob.items()}
    try:
        out["fed_global_model"] = _gm_to_payload(gm, card_glob, parents_excl_y, y_index_glob)
    except Exception:
        out["fed_global_model"] = gm

    # Explicit cleanup
    try: del hist
    except: pass
    # Optionally shutdown Ray after each fold (leave running if reusing):
    if ray_local_mode:
        try:
            ray.shutdown()
        except Exception:
            pass
    return out


def _soft_clear_tf_and_ray():
    """Best-effort cleanup for TF/Keras and Ray to release memory between runs."""
    # Clear TensorFlow/Keras graphs/allocations
    try:
        
        try:
            keras.backend.clear_session()
        except Exception:
            pass
    except Exception:
        pass
    # Force Python GC
    try:
        gc.collect()
    except Exception:
        pass
    # If Flower started Ray under the hood, shut it down
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass

def _free_locals(*objs):
    """Clear/Del provided objects and trigger GC."""
    for o in objs:
        try:
            if isinstance(o, dict):
                o.clear()
        except Exception:
            pass
    try:
        del objs
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass

def _eval_real_baseline(Xtr_int, ytr_int, Xte_int, yte_int, num_classes):
    out: dict[str, Any] = {}
    C = int(yte_int.max()) + 1
    for key, pipe in _make_pipes().items():
        pipe.fit(Xtr_int, ytr_int)
        proba = pipe.predict_proba(Xte_int)
        est = pipe.named_steps.get(key, pipe.steps[-1][1])
        classes_pred = getattr(est, "classes_", None)
        if classes_pred is not None and len(classes_pred) != C:
            proba = _align_proba_cols(proba, classes_pred, C)

        preds = proba.argmax(axis=1)
        out[f"acc_real_{key}"] = accuracy_score(yte_int, preds)
        out[f"nll_real_{key}"] = _safe_log_loss(yte_int, proba, C)
    return out
    
def _eval_mle_tstr(Xtr_int, ytr_int, Xte_int, yte_int, k_global=2, warmup=1, return_model: bool = False):
    """
    Train central generator & return synthetic-evaluation results.
    Uses SimpleGANBLR (MLE model). If return_model, also returns an exportable KDB payload.
    """
    mle_gen = SimpleGANBLR(alpha=1.0)
    t0 = time.time()
    mle_gen.fit(Xtr_int, ytr_int, k=k_global, verbose=0, warmup_epochs=warmup)
    t1 = time.time()
    X_syn, y_syn = mle_gen.sample(size=len(Xtr_int), return_decoded=False)
    out: dict[str, Any] = {"time_mle_sec": t1 - t0}
    ev = _evaluate_synthetic_classifiers(X_syn, y_syn, Xte_int, yte_int)
    for k, v in ev.items():
        out[f"mle_{k}"] = v

    if return_model and mle_gen.generator is not None:
        # Build exportable payload from the MLE generator and SimpleGANBLR.parents (exclude Y)
        parents_excl_y = {int(v): list(pa) for v, pa in mle_gen.parents.items()}
        payload = _mle_to_payload(mle_gen.generator, parents_excl_y)
        out["mle_model"] = payload
    return out
    

def _eval_central_tstr(Xtr_int, ytr_int, Xte_int, yte_int, k_global=2, epochs=100, batch_size=1024, warmup=1, return_model: bool = False):
    try:
        ganblr = GANBLR(alpha=1.0)
        print(f"[Central Training] Starting: epochs={epochs}, warmup={warmup}, batch_size={batch_size}")
        t0 = time.time()
        ganblr.fit(Xtr_int, ytr_int, k=k_global, epochs=epochs, batch_size=batch_size, warmup_epochs=warmup, verbose=1, adversarial=False)
        t1 = time.time()
        X_syn, y_syn = ganblr.sample(size=len(Xtr_int), return_decoded=False)
        out: dict[str, Any] = {"time_central_sec": t1 - t0}
        ev = _evaluate_synthetic_classifiers(X_syn, y_syn, Xte_int, yte_int)
        # Checkpoint: Verify evaluation results
        if not isinstance(ev, dict) or not ev:
            print(f"[CHECKPOINT] WARNING: _evaluate_synthetic_classifiers returned empty/invalid: {type(ev)}, keys: {list(ev.keys()) if isinstance(ev, dict) else 'N/A'}")
        for k, v in ev.items():
            out[f"central_{k}"] = v

        if return_model and ganblr.generator is not None:
            parents_excl_y = {int(v): list(pa) for v, pa in ganblr.parents.items()}
            payload = _disc_to_payload(ganblr.generator, parents_excl_y)
            out["central_model"] = payload
        return out
    except Exception as e:
        print(f"[CHECKPOINT] ERROR in _eval_central_tstr: {repr(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    
def compare_real_central_fed_cv_all_datasets(
    datasets_spec,
    n_splits=3, n_repeats=1, random_state=2024,
    # discretization
    disc_strategy_use=None, ef_bins_default=12,
    # central GANBLR params
    k_global=2, epochs_ganblr=10, batch_ganblr=256, warmup_ganblr=1,
    # federated params
    num_clients=5, num_rounds=5, dir_alpha=0.3,
    gamma=0.6, local_epochs=3, batch_size=1024, disc_epochs=1,
    cpt_mix=0.25, alpha_dir=1e-3, cap_train=60000,
    # memory management
    max_memory_gb=8.0, force_gc_every_n_folds=3,
    # output paths
    save_root="fold_artifacts",
    out_folds_path: Path | str = Path("cv_compare_perfold_nal.csv"),
    out_sum_path: Path | str = Path("cv_compare_summary_nal.csv"),
):
    """
    Enhanced version with robust memory management and error handling.
    Uses existing _soft_clear_tf_and_ray() and _free_locals() functions.
    """
    import psutil
    
    out_folds = Path(out_folds_path)
    out_sum = Path(out_sum_path)
    # Additional outputs for fidelity/coverage/privacy + JSD/WD computed inline
    out_metrics = Path("fold_artifacts/metrics_fidelity_coverage_privacy_jsdwd.csv")
    out_metrics_sum = Path("fold_artifacts/metrics_fidelity_coverage_privacy_jsdwd_summary.csv")
    
    # Prepare stable header for incremental CSV writing (accuracy/NLL + timing)
    classifiers = ["lr", "mlp", "rf", "xgb"]
    base_cols = [
        "dataset", "fold", "k", "n_train", "n_test",
        "time_central_sec", "time_fed_sec", "time_fedmle_sec",
        "weights", "s_y"
    ]
    metric_cols = []
    for clf in classifiers:
        metric_cols += [f"acc_real_{clf}", f"nll_real_{clf}"]
    for clf in classifiers:
        metric_cols += [f"acc_central_{clf}", f"nll_central_{clf}"]
    for clf in classifiers:
        metric_cols += [f"acc_fed_{clf}", f"nll_fed_{clf}"]
    for clf in classifiers:
        metric_cols += [f"acc_fedmle_{clf}", f"nll_fedmle_{clf}"]

    header_cols = base_cols + metric_cols

    # Initialize main per-fold output file with header
    out_folds.parent.mkdir(parents=True, exist_ok=True)
    if (not out_folds.exists()) or (out_folds.stat().st_size == 0):
        pd.DataFrame(columns=header_cols).to_csv(out_folds, index=False)

    print(f"Incremental per-fold CSV: {out_folds.resolve()}")
    
    # Container for inline fidelity/coverage/privacy + JSD/WD metrics
    metrics_rows: list[dict[str, Any]] = []

    # Memory monitoring
    process = psutil.Process()
    fold_counter = 0

    def check_memory():
        """Check if memory usage exceeds threshold."""
        memory_gb = process.memory_info().rss / (1024**3)
        if memory_gb > max_memory_gb:
            print(f"Warning: Memory usage {memory_gb:.1f}GB exceeds threshold {max_memory_gb}GB")
            return True
        return False

    try:
        # Process each dataset
        for dataset_idx, spec in enumerate(datasets_spec):
            # Use the human-readable dataset name (string) for logging/paths
            # and keep the alternative OpenML names as a separate list.
            name = spec["name"]
            target = spec["target"]
            data_id = spec["data_id"]
            ef_hint = spec.get("ef_bins", None)
            alt = spec.get("alt", [])
            
            print(f"\n=== CV Compare: {name} ({n_repeats}x{n_splits}-fold, k={k_global}, clients={num_clients}, rounds={num_rounds}) ===")
            
            # Memory check before processing dataset
            if check_memory():
                _soft_clear_tf_and_ray()  # Use your existing function
            
            try:
                # Fetch and prepare data once per dataset
                X_df, y_sr = fetch_openml_safely(name=name, data_id=data_id, target=target, alt_names=alt, version=1)
                df_combined = pd.concat([X_df, y_sr.rename("target")], axis=1).dropna()
                X_full = df_combined.drop(columns=["target"])
                y_full = df_combined["target"]
                # Default strategy is "ef" (equal-frequency discretization)
                strategy = "ef" if disc_strategy_use is None else disc_strategy_use
                ef_bins_use = ef_bins_default if ef_hint is None else ef_hint

                # Create CV splits
                rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
                
                for fold_idx, (tr_idx, te_idx) in enumerate(rskf.split(X_full, y_full), 1):
                    fold_counter += 1
                    
                    # Memory management
                    if fold_counter % force_gc_every_n_folds == 0:
                        _soft_clear_tf_and_ray()  # Use your existing function
                    
                    try:
                        Xtr_df, Xte_df = X_full.iloc[tr_idx].copy(), X_full.iloc[te_idx].copy()
                        ytr_sr, yte_sr = y_full.iloc[tr_idx].copy(), y_full.iloc[te_idx].copy()

                        t_fold_start = time.time()

                        # Save raw fold data if requested
                        if save_root is not None:
                            save_fold_data_csv(
                                out_dir=Path("raw_folds") / name / f"fold_{fold_idx:02d}",
                                Xtr_df=Xtr_df, ytr_sr=ytr_sr,
                                Xte_df=Xte_df, yte_sr=yte_sr,
                                target_name="target",
                                include_index=False,
                                save_meta=False
                            )

                        # Discretize with no leakage
                        Xtr_int, Xte_int, ytr_int, yte_int, card_feat, classes = discretize_train_test_no_leak(
                            Xtr_df, ytr_sr, Xte_df, yte_sr, strategy=strategy, ef_bins=ef_bins_use
                        )
                        num_classes = len(classes)

                        # Apply training cap
                        if cap_train is not None and len(Xtr_int) > cap_train:
                            sel = np.random.default_rng(1).choice(len(Xtr_int), size=cap_train, replace=False)
                            Xtr_int, ytr_int = Xtr_int[sel], ytr_int[sel]

                        # Initialize results containers
                        res_real = res_central = res_fed = res_fed_mle = {}
                        
                        try:
                            # A) Real baseline evaluation
                            res_real = _eval_real_baseline(Xtr_int, ytr_int, Xte_int, yte_int, num_classes)
                            # Checkpoint: Verify res_real structure
                            if not isinstance(res_real, dict) or not res_real:
                                print(f"[CHECKPOINT] WARNING: res_real is empty or invalid: {type(res_real)}, keys: {list(res_real.keys()) if isinstance(res_real, dict) else 'N/A'}")
                            
                            # Clear intermediate objects
                            _free_locals(Xtr_df, Xte_df, ytr_sr, yte_sr)  # Use your existing function

                            # B) Central GANBLR evaluation  
                            res_central = _eval_central_tstr(
                                Xtr_int, ytr_int, Xte_int, yte_int, k_global,
                                epochs=epochs_ganblr, batch_size=batch_ganblr, warmup=warmup_ganblr, return_model=True
                            )
                            # Checkpoint: Verify res_central structure
                            if not isinstance(res_central, dict):
                                print(f"[CHECKPOINT] WARNING: res_central is not a dict: {type(res_central)}")
                            elif not any(k.startswith('central_acc_') or k.startswith('central_nll_') for k in res_central.keys()):
                                print(f"[CHECKPOINT] WARNING: res_central missing expected keys. Actual keys: {list(res_central.keys())[:10]}")
                            
                            # Clear TF session after central training
                            _soft_clear_tf_and_ray()  # Use your existing function

                            # C) Federated GANBLR evaluation
                            res_fed = run_one_fold_fed_ganblr(
                                Xtr_int, ytr_int, Xte_int, yte_int, card_feat, num_classes,
                                k_global=k_global, num_clients=num_clients, num_rounds=num_rounds, dir_alpha=dir_alpha,
                                gamma=gamma, local_epochs=local_epochs, batch_size=batch_size, disc_epochs=disc_epochs,
                                cpt_mix=cpt_mix, alpha_dir=alpha_dir, cap_train=None,
                                ray_local_mode=False  # Use local mode to reduce overhead
                            )
                            # Checkpoint: Verify res_fed structure
                            if not isinstance(res_fed, dict):
                                print(f"[CHECKPOINT] WARNING: res_fed is not a dict: {type(res_fed)}")
                            elif not any(k.startswith('acc_') or k.startswith('nll_') for k in res_fed.keys()):
                                print(f"[CHECKPOINT] WARNING: res_fed missing expected keys. Actual keys: {list(res_fed.keys())[:10]}")
                            
                            # Clear after federated run
                            _soft_clear_tf_and_ray()  # Use your existing function

                            # D) Federated MLE evaluation  
                            res_fed_mle = run_one_fold_fed_mle_naive(
                                Xtr_int, ytr_int, Xte_int, yte_int, card_feat, num_classes,
                                k_global=k_global, num_clients=num_clients, dir_alpha=dir_alpha, cap_train=None, verbose=False
                            )
                            # Checkpoint: Verify res_fed_mle structure
                            if not isinstance(res_fed_mle, dict):
                                print(f"[CHECKPOINT] WARNING: res_fed_mle is not a dict: {type(res_fed_mle)}")

                        except Exception as e_eval:
                            import traceback
                            print(f" !! Error during model evaluation for {name} fold {fold_idx}: {repr(e_eval)}")
                            print(f" !! Full traceback:")
                            traceback.print_exc()
                            # Fill with NaN results
                            res_real = {f"acc_real_{clf}": np.nan for clf in classifiers} | {f"nll_real_{clf}": np.nan for clf in classifiers}
                            res_central = {f"central_acc_{clf}": np.nan for clf in classifiers} | {f"central_nll_{clf}": np.nan for clf in classifiers} | {"time_central_sec": np.nan}
                            res_fed = {f"acc_{clf}": np.nan for clf in classifiers} | {f"nll_{clf}": np.nan for clf in classifiers} | {"train_time_sec": np.nan, "weights": None, "s_y": None}
                            res_fed_mle = {f"acc_{clf}": np.nan for clf in classifiers} | {f"nll_{clf}": np.nan for clf in classifiers} | {"train_time_sec": np.nan}

                        # Save fold artifacts if requested
                        if save_root is not None:
                            try:
                                root = Path(save_root) / name / f"fold_{fold_idx:02d}"
                                save_fold_data_npz(root / "data.npz", Xtr_int, ytr_int, Xte_int, yte_int, card_feat, num_classes)
                                
                                # Save models safely
                                for model_key, filename in [
                                    ("central_model", "central_ganblr_model"),
                                    ("fed_global_model", "fed_global_model"),
                                    ("fed_global_model", "fedmle_model")  # fedmle uses same key
                                ]:
                                    model_dict = (res_central if "central" in filename else 
                                                res_fed if "fed_global" in filename else 
                                                res_fed_mle).get(model_key)
                                    if isinstance(model_dict, dict):
                                        save_kdb_model_npz_json(
                                            root / f"{filename}.npz", 
                                            root / f"{filename}.json", 
                                            model_dict
                                        )
                            except Exception as e_save:
                                print(f" !! Error saving artifacts for {name} fold {fold_idx}: {repr(e_save)}")

                        # Build and save result row (classification metrics)
                        row = dict(
                            dataset=name, fold=fold_idx, k=k_global,
                            n_train=len(Xtr_int), n_test=len(Xte_int),
                            time_central_sec=res_central.get("time_central_sec", np.nan),
                            time_fed_sec=res_fed.get("train_time_sec", np.nan),
                            time_fedmle_sec=res_fed_mle.get("train_time_sec", np.nan),
                            weights=res_fed.get("weights", None),
                            s_y=res_fed.get("s_y", None)
                        )
                        
                        # Add metric results
                        row.update(res_real)
                        for key in classifiers:
                            row[f"acc_central_{key}"] = res_central.get(f"central_acc_{key}", np.nan)
                            row[f"nll_central_{key}"] = res_central.get(f"central_nll_{key}", np.nan)
                            row[f"acc_fed_{key}"] = res_fed.get(f"acc_{key}", np.nan)
                            row[f"nll_fed_{key}"] = res_fed.get(f"nll_{key}", np.nan)
                            row[f"acc_fedmle_{key}"] = res_fed_mle.get(f"acc_{key}", np.nan)
                            row[f"nll_fedmle_{key}"] = res_fed_mle.get(f"nll_{key}", np.nan)

                        # Checkpoint: Verify row has all required columns before writing
                        missing_cols = set(header_cols) - set(row.keys())
                        if missing_cols:
                            print(f"[CHECKPOINT] ERROR: Missing {len(missing_cols)} columns in row. Missing: {list(missing_cols)[:5]}...")
                            for col in missing_cols:
                                row[col] = np.nan
                        
                        # Checkpoint: Count non-null values
                        non_null_count = sum(1 for v in row.values() if v is not np.nan and v is not None and v != '')
                        if non_null_count < 10:  # Should have at least dataset, fold, k, n_train, n_test + some metrics
                            print(f"[CHECKPOINT] WARNING: Row has only {non_null_count} non-null values. Sample keys with values: {[(k, v) for k, v in list(row.items())[:10] if v is not np.nan and v is not None]}")

                        # Write row incrementally
                        try:
                            pd.DataFrame([row], columns=header_cols).to_csv(out_folds, mode="a", header=False, index=False)
                        except Exception as e_write:
                            print(f"[CHECKPOINT] ERROR writing CSV: {repr(e_write)}")
                            print(f"[CHECKPOINT] Row keys: {list(row.keys())[:20]}, Header cols: {list(header_cols)[:20]}")
                            raise
                        
                        # -- Inline fidelity/coverage/privacy and JSD/WD metrics per method --
                        try:
                            real_df = _build_real_df(Xtr_int, ytr_int)
                            real_feat_df = real_df.drop(columns=["y"], errors="ignore")
                            # In discretized space we treat all columns as categorical
                            numeric_cols_metrics: list[str] = []
                            categorical_cols_metrics: list[str] = list(real_df.columns)

                            method_payloads: dict[str, dict] = {
                                "central_ganblr": res_central.get("central_model", None),
                                "fed": res_fed.get("fed_global_model", None),
                            }

                            for method_name, payload in method_payloads.items():
                                if not isinstance(payload, dict):
                                    continue
                                syn_full = _sample_full_from_payload(
                                    payload, n=len(Xtr_int), rng=np.random.default_rng(0)
                                )
                                syn_df = _build_syn_df(syn_full)
                                syn_feat_df = syn_df.drop(columns=["y"], errors="ignore")

                                col_fid, row_fid, agg_fid = calculate_fidelity(
                                    real_df, syn_df, numeric_cols_metrics, categorical_cols_metrics
                                )
                                coverage = calculate_coverage(
                                    real_df, syn_df, numeric_cols_metrics, categorical_cols_metrics
                                )
                                # Use sampling for DCR to reduce runtime on large datasets
                                dcr_sample_size = min(5000, len(real_df)) if len(real_df) > 5000 else None
                                dcr = calculate_dcr_privacy(
                                    real_df, syn_df, numeric_cols_metrics, sample_size=dcr_sample_size
                                )
                                dist_metrics = compute_distributional_metrics(
                                    real_feat_df, syn_feat_df
                                )

                                metrics_rows.append(
                                    dict(
                                        dataset=name,
                                        fold=fold_idx,
                                        method=method_name,
                                        fidelity_col=col_fid,
                                        fidelity_row=row_fid,
                                        fidelity_agg=agg_fid,
                                        coverage=coverage,
                                        privacy_dcr=dcr,
                                        jsd_mean=dist_metrics["jsd_mean"],
                                        wd_mean=dist_metrics["wd_mean"],
                                    )
                                )
                        except Exception as e_metrics:
                            print(
                                f"[CHECKPOINT] WARNING: inline metrics failed for {name} fold {fold_idx}: {repr(e_metrics)}"
                            )

                        t_fold_end = time.time()
                        memory_gb = process.memory_info().rss / (1024**3)
                        print(f"Completed {name} fold {fold_idx} (wall_time={t_fold_end-t_fold_start:.1f}s, memory={memory_gb:.1f}GB)")

                    except Exception as e_fold:
                        print(f" !! Skipping fold {fold_idx} for {name} due to error: {repr(e_fold)}")
                        
                    finally:
                        # Explicit cleanup for each fold using your existing functions
                        local_vars = ['Xtr_int', 'Xte_int', 'ytr_int', 'yte_int', 'card_feat', 'classes',
                                     'res_real', 'res_central', 'res_fed', 'res_fed_mle', 'row']
                        
                        # Use your existing cleanup functions
                        _free_locals(*[locals().get(var) for var in local_vars])
                        
                        # Periodic cleanup
                        if fold_counter % 2 == 0:  # Every 2 folds
                            _soft_clear_tf_and_ray()

            except Exception as e_dataset:
                print(f"  !! Skipping {name} due to error: {repr(e_dataset)}")
                
            finally:
                # Clean up dataset-level variables using your existing function
                # Only pass objects that exist in the current local scope to avoid UnboundLocalError
                _free_locals(*[locals().get(n) for n in ("X_df", "y_sr", "df_combined", "X_full", "y_full", "rskf") if locals().get(n) is not None])
                _soft_clear_tf_and_ray()

    finally:
        # Final cleanup using your existing function
        _soft_clear_tf_and_ray()

    # Read incremental CSV and compute summary
    if not out_folds.exists():
        print(f"No per-fold CSV found at {out_folds}; returning empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        df = pd.read_csv(out_folds)
        if df.empty:
            return df, pd.DataFrame()

        # Compute summary statistics for classification metrics
        numeric_cols = [c for c in df.columns if c not in ("dataset", "weights", "s_y")]
        df_sum = df.groupby("dataset")[numeric_cols].mean().reset_index()

        # Save final results
        out_sum.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_folds, index=False)
        df_sum.to_csv(out_sum, index=False)

        print(f"\nWrote per-fold results: {out_folds}")
        print(f"Wrote summary: {out_sum}")

        # Also write per-fold fidelity/coverage/privacy + JSD/WD metrics if any
        if metrics_rows:
            df_metrics = pd.DataFrame(metrics_rows).sort_values(
                ["dataset", "fold", "method"]
            ).reset_index(drop=True)
            out_metrics.parent.mkdir(parents=True, exist_ok=True)
            df_metrics.to_csv(out_metrics, index=False)

            # Summary (mean/std) per dataset/method
            metric_cols = [
                "fidelity_col",
                "fidelity_row",
                "fidelity_agg",
                "coverage",
                "privacy_dcr",
                "jsd_mean",
                "wd_mean",
            ]
            grp = df_metrics.groupby(["dataset", "method"])
            mean_df = grp[metric_cols].mean().rename(
                columns={m: f"{m}_mean" for m in metric_cols}
            )
            std_df = grp[metric_cols].std(ddof=1).fillna(0.0).rename(
                columns={m: f"{m}_std" for m in metric_cols}
            )
            n_df = grp.size().to_frame("n_folds")
            df_metrics_sum = mean_df.join(std_df).join(n_df).reset_index()
            df_metrics_sum.to_csv(out_metrics_sum, index=False)

            print(f"Wrote fidelity/coverage/privacy+JSD/WD per-fold: {out_metrics}")
            print(f"Wrote fidelity/coverage/privacy+JSD/WD summary: {out_metrics_sum}")

        return df, df_sum

    except Exception as e_final:
        print(f"Error in final processing: {repr(e_final)}")
        
    


def run_one_fold_fed_mle_counts(
    Xtr_int, ytr_int, Xte_int, yte_int, card_feat, num_classes,
    k_global=2, num_clients=5, dir_alpha=0.1, cap_train=None, verbose=False
):
    """
    Federated MLE with count aggregation:
      - Split TRAIN into clients (Dirichlet)
      - Each client fits local MLE_KDB and extracts raw counts (before normalization)
      - Server aggregates counts directly, then normalizes to get global probabilities
      - Sample synthetic from aggregated model and run TSTR
    """
    t0 = time.time()

    # Optional cap
    if cap_train is not None and len(Xtr_int) > cap_train:
        sel = np.random.default_rng(1).choice(len(Xtr_int), size=cap_train, replace=False)
        Xtr_int = Xtr_int[sel]; ytr_int = ytr_int[sel]

    # Global schema from TRAIN
    card_all = [num_classes] + list(card_feat)
    y_index = 0
    train_arr, test_arr = build_full_table(ytr_int, yte_int, Xtr_int, Xte_int)
    card_glob, parents_glob, y_index_glob = derive_global_meta(train_arr, card_all, y_index, k=k_global)

    # Build non-IID splits
    clients_data = dirichlet_split(Xtr_int, ytr_int, num_clients=num_clients, alpha=dir_alpha)

    # Fit local MLEs and extract counts
    local_models = []
    for i, (Xc, yc) in enumerate(clients_data):
        n_i = int(len(Xc))
        if n_i == 0:
            continue
            
        data_i = np.column_stack([yc, Xc]).astype(np.int32)
        
        # Extract raw counts before normalization
        py_counts = np.bincount(yc, minlength=num_classes).astype(np.float64)
        
        # Extract CPT counts for each feature
        theta_counts = {}
        V = len(card_glob)
        
        for v in range(V):
            if v == y_index_glob:
                continue
                
            pa = [y_index_glob] + parents_glob.get(v, [])
            pa_cards = [card_glob[p] for p in pa]
            strides = _compute_strides(pa_cards)
            num_cfg = int(np.prod(pa_cards)) if len(pa_cards) > 0 else 1
            rv = card_glob[v]
            
            # Raw counts without Laplace smoothing
            counts = np.zeros((num_cfg, rv), dtype=np.float64)
            
            for row_idx in range(data_i.shape[0]):
                vals = [data_i[row_idx, p] for p in pa]
                # Compute flattened index
                idx = 0
                for s, val in zip(strides, vals):
                    idx += int(s) * int(val)
                counts[idx, data_i[row_idx, v]] += 1
            
            theta_counts[v] = counts
        
        local_models.append(dict(
            py_counts=py_counts, 
            theta_counts=theta_counts, 
            n=n_i
        ))
        
        if verbose:
            print(f"[fed-mle-counts] client {i}: n={n_i}")

    if not local_models:
        return dict(acc_lr=np.nan, nll_lr=np.nan, acc_mlp=np.nan, nll_mlp=np.nan, 
                   acc_rf=np.nan, nll_rf=np.nan, acc_xgb=np.nan, nll_xgb=np.nan,
                   train_time_sec=0.0, weights=None)

    # Aggregate counts
    gm = _aggregate_mle_counts(local_models)

    # Build generator and TSTR
    gen_global = build_global_kdb_from_gm(gm, card_glob, parents_glob, y_index_glob)
    n_syn = len(Xtr_int)
    syn_full = gen_global.sample(n=n_syn, rng=np.random.default_rng(0), order=None, return_y=True)
    X_syn, y_syn = syn_full[:, 1:], syn_full[:, 0]
    ev = _evaluate_synthetic_classifiers(X_syn, y_syn, Xte_int, yte_int)

    t1 = time.time()
    out = dict(train_time_sec=t1 - t0, weights=gm.get("weights", None))
    out.update(ev)
    return out