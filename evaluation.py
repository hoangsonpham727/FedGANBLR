import numpy as np
import pandas as pd
import flwr as fl
import time
from pathlib import Path
import json
import keras
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from typing import Any, List, Tuple
from numpy.typing import ArrayLike
from scipy.stats import kstest
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from utils import build_full_table, dirichlet_split, _evaluate_synthetic_classifiers, _gm_to_payload, _make_pipes,\
_align_proba_cols, _mle_to_payload, _disc_to_payload, fetch_openml_safely, discretize_train_test_no_leak,\
save_fold_data_csv, save_fold_data_npz, save_kdb_model_npz_json, _safe_log_loss, preprocess_covertype_binary_columns
from base_models.KDependenceBayesian import _compute_strides
from base_models.Ganblr import SimpleGANBLR, GANBLR
from federated_models.FedGanblr import derive_global_meta, build_global_kdb_from_gm, KDBGANStrategy, GANBLRFederatedClient
from federated_models.FedMLE import run_one_fold_fed_mle_naive, _aggregate_mle_counts
import gc


def total_variation_distance(p_dist: pd.Series, s_dist: pd.Series) -> float:
    """Calculates the Total Variation Distance (TVD) between two categorical distributions."""
    # Ensure both distributions cover the union of all categories
    all_categories = p_dist.index.union(s_dist.index)
    p_full = p_dist.reindex(all_categories, fill_value=0)
    s_full = s_dist.reindex(all_categories, fill_value=0)
    # TVD = 1/2 * Sum(|p_c - s_c|)
    tvd_score = 0.5 * np.sum(np.abs(p_full - s_full))
    return float(tvd_score)


def calculate_fidelity(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[float, float, float]:
    """
    Calculates Column Fidelity, Row Fidelity, and Aggregate Fidelity.
    Fidelity (Ω) assesses how closely synthetic data emulates real data.
    """
    # --- 1.1 Column Fidelity (Ω_col) ---
    col_fidelities: list[float] = []

    # Numeric Columns: use Kolmogorov-Smirnov Statistic (KSS)
    for col in numeric_cols:
        # KSS = 1 - KS(x^d, s^d)
        try:
            kss, _ = kstest(real_df[col], synthetic_df[col])
            omega_col = 1.0 - float(kss)
            col_fidelities.append(float(omega_col))
        except Exception as e:
            print(f"Warning: KSS failed for numeric col {col}. Skipping. Error: {e}")

    # Categorical Columns: use Total Variation Distance (TVD)
    for col in categorical_cols:
        real_dist = real_df[col].value_counts(normalize=True)
        synth_dist = synthetic_df[col].value_counts(normalize=True)
        tvd_score = total_variation_distance(real_dist, synth_dist)
        # Ω_col = 1 - TVD(x^d, s^d) (note TVD already includes the 1/2)
        omega_col = 1.0 - float(tvd_score)
        col_fidelities.append(float(omega_col))

    column_fidelity = float(np.mean(np.asarray(col_fidelities, dtype=float))) if col_fidelities else 0.0

    # --- 1.2 Row Fidelity (Ω_row) ---
    row_fidelities: list[float] = []
    all_cols = list(numeric_cols) + list(categorical_cols)
    for i in range(len(all_cols)):
        for j in range(i + 1, len(all_cols)):
            col_a, col_b = all_cols[i], all_cols[j]
            is_num_a = col_a in numeric_cols
            is_num_b = col_b in numeric_cols

            if is_num_a and is_num_b:
                # Numeric Pair: Pearson Correlation
                real_corr = real_df[[col_a, col_b]].corr().iloc[0, 1]
                synth_corr = synthetic_df[[col_a, col_b]].corr().iloc[0, 1]
                pc_discrepancy = np.abs(np.asarray(real_corr) - np.asarray(synth_corr))
                omega_row = 1.0 - (0.5 * float(pc_discrepancy))
                row_fidelities.append(float(omega_row))

            elif (col_a in categorical_cols) and (col_b in categorical_cols):
                # Categorical Pair: TVD over joint distributions
                real_joint = real_df.groupby([col_a, col_b]).size().div(len(real_df))
                synth_joint = synthetic_df.groupby([col_a, col_b]).size().div(len(synthetic_df))
                tvd_score = total_variation_distance(real_joint, synth_joint)
                omega_row = 1.0 - float(tvd_score)
                row_fidelities.append(float(omega_row))

            # Mixed numeric/categorical case skipped (matches notebook behavior)

    row_fidelity = float(np.mean(np.asarray(row_fidelities, dtype=float))) if row_fidelities else 0.0

    # --- 1.3 Aggregate Fidelity (Ω) ---
    aggregate_fidelity = (float(column_fidelity) + float(row_fidelity)) / 2.0
    return float(column_fidelity), float(row_fidelity), float(aggregate_fidelity)


def calculate_coverage(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> float:
    """
    Calculates Coverage (γ), assessing the replication of diversity (categories or range).
    """
    coverage_scores: list[float] = []

    # Categorical: proportion of real categories found in synthetic
    for col in categorical_cols:
        real_categories = set(real_df[col].dropna().unique())
        synth_categories = set(synthetic_df[col].dropna().unique())
        if not real_categories:
            continue
        covered_count = len(real_categories.intersection(synth_categories))
        coverage_scores.append(float(covered_count / len(real_categories)))

    # Numerical: range alignment
    for col in numeric_cols:
        real_min, real_max = real_df[col].min(), real_df[col].max()
        synth_min, synth_max = synthetic_df[col].min(), synthetic_df[col].max()
        real_range = real_max - real_min
        if real_range == 0:
            coverage_scores.append(1.0)
            continue
        tau = np.maximum(0, (synth_min - real_min) / real_range)
        chi = np.maximum(0, (real_max - synth_max) / real_range)
        coverage_scores.append(float(1.0 - (tau + chi)))

    return float(np.mean(coverage_scores)) if coverage_scores else 0.0


def _safe_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def calculate_dcr_privacy(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    numeric_cols: list[str],
    k: int = 1,
    exclude_exact_matches: bool = True,
    zero_eps: float = 1e-12,
) -> float:
    """
    Distance-to-Closest-Record (DCR) with robust defaults:
    - Uses OHE for categoricals + MinMax scaling on real-fit features.
    - If exclude_exact_matches=True, ignores 0-distance exact matches and
      uses distance to the first strictly-positive neighbor (k-th if k>1).
    - Returns NaN when undefined (no features or no comparable samples).
    """
    real = real_df.copy()
    synth = synthetic_df.copy()
    for lab in ("y", "label", "target"):
        if lab in real.columns:
            real = real.drop(columns=[lab])
        if lab in synth.columns:
            synth = synth.drop(columns=[lab])

    if real.shape[1] == 0:
        return float("nan")

    # Align columns
    synth = synth[real.columns.tolist()]

    numeric_cols = [c for c in numeric_cols if c in real.columns]
    categorical_cols = [c for c in real.columns if c not in numeric_cols]

    # OHE
    if categorical_cols:
        ohe = _safe_ohe()
        ohe.fit(real[categorical_cols].astype("category"))
        Xr_cat = ohe.transform(real[categorical_cols].astype("category"))
        Xs_cat = ohe.transform(synth[categorical_cols].astype("category"))
    else:
        Xr_cat = np.empty((len(real), 0), dtype=float)
        Xs_cat = np.empty((len(synth), 0), dtype=float)

    # Numeric
    if numeric_cols:
        Xr_num = real[numeric_cols].to_numpy(dtype=float)
        Xs_num = synth[numeric_cols].to_numpy(dtype=float)
    else:
        Xr_num = np.empty((len(real), 0), dtype=float)
        Xs_num = np.empty((len(synth), 0), dtype=float)

    Xr = np.hstack([np.asarray(Xr_num, dtype=float), np.asarray(Xr_cat, dtype=float)])
    Xs = np.hstack([np.asarray(Xs_num, dtype=float), np.asarray(Xs_cat, dtype=float)])
    if Xr.shape[1] == 0 or len(Xr) == 0 or len(Xs) == 0:
        return float("nan")

    scaler = MinMaxScaler()
    Xr_s = scaler.fit_transform(Xr)
    Xs_s = scaler.transform(Xs)

    dist = cdist(Xs_s, Xr_s, metric="euclidean")
    if dist.size == 0:
        return float("nan")

    dist_masked = dist
    if exclude_exact_matches:
        dist_masked = dist.copy()
        dist_masked[dist_masked <= max(zero_eps, 0.0)] = np.inf

    k_idx = max(1, int(k)) - 1
    nn = np.full(dist_masked.shape[0], np.nan, dtype=float)
    for i in range(dist_masked.shape[0]):
        finite = dist_masked[i][np.isfinite(dist_masked[i])]
        if finite.size == 0:
            nn[i] = 0.0
        else:
            if k_idx < finite.size:
                nn[i] = np.partition(finite, k_idx)[k_idx]
            else:
                nn[i] = np.max(finite)

    return float(np.nanmedian(nn))


def jensen_shannon_divergence(p: ArrayLike, q: ArrayLike) -> float:
    """Calculate Jensen-Shannon divergence between two probability distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    # scipy returns sqrt(JS); square to get JS divergence
    return float(jensenshannon(p, q) ** 2)


def wasserstein_distance_categorical(real_counts: ArrayLike, syn_counts: ArrayLike) -> float:
    """Calculate Wasserstein distance for categorical data."""
    real_counts = np.asarray(real_counts, dtype=np.float64)
    syn_counts = np.asarray(syn_counts, dtype=np.float64)
    p = real_counts / (real_counts.sum() + 1e-12)
    q = syn_counts / (syn_counts.sum() + 1e-12)
    categories = np.arange(len(p))
    return float(wasserstein_distance(categories, categories, p, q))


def compute_distributional_metrics(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> dict:
    """Compute JSD and WD metrics between real and synthetic data (mean across columns)."""
    jsd_values: list[float] = []
    wd_values: list[float] = []
    for col in real_df.columns:
        real_counts = real_df[col].value_counts().sort_index()
        syn_counts = syn_df[col].value_counts().sort_index()
        all_cats = real_counts.index.union(syn_counts.index)
        real_aligned = real_counts.reindex(all_cats, fill_value=0).values
        syn_aligned = syn_counts.reindex(all_cats, fill_value=0).values
        jsd_values.append(jensen_shannon_divergence(real_aligned, syn_aligned))
        wd_values.append(wasserstein_distance_categorical(real_aligned, syn_aligned))
    return dict(
        jsd_mean=float(np.mean(jsd_values)) if jsd_values else float("nan"),
        wd_mean=float(np.mean(wd_values)) if wd_values else float("nan"),
        jsd_values=jsd_values,
        wd_values=wd_values,
    )


def _build_real_df(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Build a DataFrame from discretized arrays; all columns treated categorical."""
    cols = ["y"] + [f"f{j}" for j in range(X.shape[1])]
    arr = np.column_stack([y, X]).astype(np.int32)
    return pd.DataFrame(arr, columns=cols)


def _build_syn_df(syn_full: np.ndarray) -> pd.DataFrame:
    y_syn = syn_full[:, 0]
    X_syn = syn_full[:, 1:]
    return _build_real_df(X_syn, y_syn)


def _build_kdb_from_payload(payload: dict):
    """Build a generator from an exportable payload (card/parents/y_index + py/thetas)."""
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be dict, got {type(payload)}")
    required = {"py", "thetas", "card", "parents", "y_index"}
    if not required.issubset(set(payload.keys())):
        raise KeyError(f"payload missing keys: {sorted(required - set(payload.keys()))}")
    gen = build_global_kdb_from_gm(
        gm=dict(py=payload["py"], thetas=payload["thetas"]),
        card=payload["card"],
        parents=payload["parents"],  # parents exclude Y in payload
        y_index=payload["y_index"],
    )
    return gen


def _sample_full_from_payload(payload: dict, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    gen = _build_kdb_from_payload(payload)
    return gen.sample(n=n, rng=(rng or np.random.default_rng(0)), order=None, return_y=True)


def _jsd_wd_features(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> tuple[float, float]:
    real_feat = real_df.drop(columns=["y"], errors="ignore")
    syn_feat = syn_df.drop(columns=["y"], errors="ignore")
    m = compute_distributional_metrics(real_feat, syn_feat)
    return float(m["jsd_mean"]), float(m["wd_mean"])


def _eval_fidelity_coverage_privacy(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> dict[str, float]:
    # In discretized space, treat everything (including y) as categorical.
    numeric_cols: list[str] = []
    categorical_cols = real_df.columns.tolist()
    col_fid, row_fid, agg_fid = calculate_fidelity(real_df, syn_df, numeric_cols, categorical_cols)
    coverage = calculate_coverage(real_df, syn_df, numeric_cols, categorical_cols)
    dcr = calculate_dcr_privacy(real_df, syn_df, numeric_cols)
    return dict(fidelity_col=col_fid, fidelity_row=row_fid, fidelity_agg=agg_fid, coverage=coverage, privacy_dcr=dcr)


def run_one_fold_fed_ganblr(
    Xtr_int, ytr_int, Xte_int, yte_int, card_feat, num_classes,
    k_global=2, num_clients=5, num_rounds=5, dir_alpha=0.1,
    gamma=0.6, local_epochs=3, batch_size=1024, disc_epochs=1,
    cpt_mix=0.25, alpha_dir=1e-3, beta_pow=0.5, cap_train=None, clf="lr", verbose=False,
    eval_syn_frac: float = 0.5,          
    ray_local_mode: bool = False,
    diagnostics_dir: str | Path | None = None
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

    diag_dir = Path(diagnostics_dir) if diagnostics_dir is not None else None
    if diag_dir:
        diag_dir.mkdir(parents=True, exist_ok=True)
        try:
            train_counts = np.bincount(ytr_int, minlength=num_classes).tolist()
            test_counts = np.bincount(yte_int, minlength=num_classes).tolist()
            client_stats = []
            for cid, (_, yc) in enumerate(clients_data):
                counts = np.bincount(yc, minlength=num_classes)
                total = int(counts.sum())
                frac = (counts / max(1, total)).tolist()
                client_stats.append({
                    "client": cid,
                    "n": total,
                    "label_counts": counts.astype(int).tolist(),
                    "label_frac": frac,
                })
            meta = {
                "num_clients": num_clients,
                "num_rounds": num_rounds,
                "dir_alpha": dir_alpha,
                "gamma": gamma,
                "local_epochs": local_epochs,
                "batch_size": batch_size,
                "disc_epochs": disc_epochs,
                "cpt_mix": cpt_mix,
                "alpha_dir": alpha_dir,
                "eval_syn_frac": eval_syn_frac,
                "train_class_counts": train_counts,
                "test_class_counts": test_counts,
                "client_splits": client_stats,
            }
            (diag_dir / "client_split_stats.json").write_text(json.dumps(meta, indent=2))
        except Exception as e:
            print(f"[diag] Failed to write client split stats: {e}")

    strategy = KDBGANStrategy(
        k=k_global,
        gamma=gamma,
        local_epochs=local_epochs,
        batch_size=batch_size,
        disc_epochs=disc_epochs,
        cpt_mix=cpt_mix,
        alpha_dir=alpha_dir,
        beta_pow=beta_pow,
        adversarial=True,
        nll_csv_path=None,  # disable NLL CSV writing
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

    if diag_dir:
        round_rows = []
        for rnd, snap in sorted(strategy.model_snapshots.items()):
            row = {"round": int(rnd)}
            for key in ("py", "s_y", "weights"):
                arr = snap.get(key, None)
                if arr is not None:
                    row[f"{key}_json"] = json.dumps(np.asarray(arr).tolist())
            client_kls = snap.get("client_kls", None)
            if client_kls is not None:
                arr = np.asarray(client_kls, dtype=np.float64)
                row["kl_mean"] = float(np.nanmean(arr)) if arr.size else np.nan
                row["kl_max"] = float(np.nanmax(arr)) if arr.size else np.nan
            client_ns = snap.get("client_ns", None)
            if client_ns is not None:
                row["client_ns_json"] = json.dumps(np.asarray(client_ns).astype(int).tolist())
            if "nll_clients" in snap:
                row["nll_clients"] = int(snap.get("nll_clients", 0))
            round_rows.append(row)
        try:
            pd.DataFrame(round_rows).to_csv(diag_dir / "global_round_stats.csv", index=False)
        except Exception as e:
            print(f"[diag] Failed to write global_round_stats: {e}")

        try:
            fold_config = {
                "k_global": k_global,
                "num_clients": num_clients,
                "num_rounds": num_rounds,
                "dir_alpha": dir_alpha,
                "gamma": gamma,
                "local_epochs": local_epochs,
                "batch_size": batch_size,
                "disc_epochs": disc_epochs,
                "cpt_mix": cpt_mix,
                "alpha_dir": alpha_dir,
                "eval_syn_frac": eval_syn_frac,
            }
            (diag_dir / "fold_config.json").write_text(json.dumps(fold_config, indent=2))
        except Exception as e:
            print(f"[diag] Failed to write fold_config: {e}")

        try:
            parents_excl_y = {int(v): list(pa) for v, pa in parents_glob.items()}
            payload = _gm_to_payload(gm, card_glob, parents_excl_y, y_index_glob)
            (diag_dir / "final_global_model.json").write_text(json.dumps(payload))
        except Exception as e:
            print(f"[diag] Failed to write final_global_model: {e}")
        out["diagnostics_dir"] = str(diag_dir)
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
        t0 = time.time()
        ganblr.fit(Xtr_int, ytr_int, k=k_global, epochs=epochs, batch_size=batch_size, warmup_epochs=warmup, verbose=0, adversarial=True)
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
    n_splits=2, n_repeats=2, random_state=2024,
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
    out_folds_path: Path | str = Path("fedganblr_results.csv"),
    out_sum_path: Path | str = Path("fedganblr_summary.csv"),
    diagnostics_root: Path | str | None = Path("diagnostics"),
):
    """
    Function to run the TSTR experiments on all models and calculate other metrics for FedGANBLR only

    """
    import psutil
    
    out_folds = Path(out_folds_path)
    out_sum = Path(out_sum_path)
    
    # Prepare stable header for incremental CSV writing
    classifiers = ["lr", "mlp", "rf", "xgb"]
    base_cols = [
        "dataset", "fold", "k", "n_train", "n_test",
        "time_central_sec", "time_fed_sec", "time_fedmle_sec",
        "weights", "s_y",
        # Distributional / privacy metrics (computed on discretized TRAIN vs sampled synthetic)
        "fidelity_col", "fidelity_row", "fidelity_agg",
        "coverage", "privacy_dcr",
        "jsd", "wd",
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

    # Initialize output file with header; if an existing file has mismatched columns, back it up and start fresh.
    out_folds.parent.mkdir(parents=True, exist_ok=True)
    need_header = True
    if out_folds.exists() and out_folds.stat().st_size > 0:
        try:
            existing_cols = pd.read_csv(out_folds, nrows=0).columns.tolist()
            if existing_cols == header_cols:
                need_header = False
            else:
                backup = out_folds.with_suffix(out_folds.suffix + ".bak")
                out_folds.rename(backup)
                print(f"[init] Existing {out_folds} had mismatched header; backed up to {backup} and writing fresh header.")
        except Exception as e:
            print(f"[init] Could not inspect existing {out_folds}: {e}. Rewriting header.")
    if need_header:
        pd.DataFrame(columns=header_cols).to_csv(out_folds, index=False)

    print(f"Incremental per-fold CSV: {out_folds.resolve()}")
    
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

                        # Apply dataset-specific preprocessing (before discretization)
                        if name.lower() == "covertype":
                            Xtr_df = preprocess_covertype_binary_columns(Xtr_df)
                            Xte_df = preprocess_covertype_binary_columns(Xte_df)

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
                        diag_dir = Path(diagnostics_root) / name / f"fold_{fold_idx:02d}" if diagnostics_root is not None else None
                        
                        try:
                            # A) Real baseline evaluation
                            res_real = _eval_real_baseline(Xtr_int, ytr_int, Xte_int, yte_int, num_classes)
                            # Checkpoint: Verify res_real structure
                            if not isinstance(res_real, dict) or not res_real:
                                print(f"[CHECKPOINT] WARNING: res_real is empty or invalid: {type(res_real)}, keys: {list(res_real.keys()) if isinstance(res_real, dict) else 'N/A'}")
                            
                            # # Clear intermediate objects
                            _free_locals(Xtr_df, Xte_df, ytr_sr, yte_sr)  # Use your existing function

                            #B) Central GANBLR evaluation  
                            res_central = _eval_central_tstr(
                                Xtr_int, ytr_int, Xte_int, yte_int, k_global,
                                epochs=epochs_ganblr, batch_size=batch_ganblr, warmup=warmup_ganblr, return_model=True
                            )
                            #Checkpoint: Verify res_central structure
                            if not isinstance(res_central, dict):
                                print(f"[CHECKPOINT] WARNING: res_central is not a dict: {type(res_central)}")
                            elif not any(k.startswith('central_acc_') or k.startswith('central_nll_') for k in res_central.keys()):
                                print(f"[CHECKPOINT] WARNING: res_central missing expected keys. Actual keys: {list(res_central.keys())[:10]}")
                            
                            #Clear TF session after central training
                            _soft_clear_tf_and_ray()  # Use your existing function

                            # C) Federated GANBLR evaluation
                            res_fed = run_one_fold_fed_ganblr(
                                Xtr_int, ytr_int, Xte_int, yte_int, card_feat, num_classes,
                                k_global=k_global, num_clients=num_clients, num_rounds=num_rounds, dir_alpha=dir_alpha,
                                gamma=gamma, local_epochs=local_epochs, batch_size=batch_size, disc_epochs=disc_epochs,
                                cpt_mix=cpt_mix, alpha_dir=alpha_dir, cap_train=None,
                                ray_local_mode=False,  # Use local mode to reduce overhead
                                diagnostics_dir=diag_dir
                            )
                            # Checkpoint: Verify res_fed structure
                            if not isinstance(res_fed, dict):
                                print(f"[CHECKPOINT] WARNING: res_fed is not a dict: {type(res_fed)}")
                            elif not any(k.startswith('acc_') or k.startswith('nll_') for k in res_fed.keys()):
                                print(f"[CHECKPOINT] WARNING: res_fed missing expected keys. Actual keys: {list(res_fed.keys())[:10]}")
                            
                            # Clear after federated run
                            _soft_clear_tf_and_ray()  # Use your existing function

                            # D) Federated MLE evaluation  
                            # res_fed_mle = run_one_fold_fed_mle_naive(
                            #     Xtr_int, ytr_int, Xte_int, yte_int, card_feat, num_classes,
                            #     k_global=k_global, num_clients=num_clients, dir_alpha=dir_alpha, cap_train=None, verbose=False
                            # )
                            # Checkpoint: Verify res_fed_mle structure
                            # if not isinstance(res_fed_mle, dict):
                            #     print(f"[CHECKPOINT] WARNING: res_fed_mle is not a dict: {type(res_fed_mle)}")

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

                        # --- Fidelity / Coverage / Privacy + JSD / WD (fed only, from final global model) ---
                        fid_cov_priv = dict(
                            fidelity_col=np.nan, fidelity_row=np.nan, fidelity_agg=np.nan,
                            coverage=np.nan, privacy_dcr=np.nan,
                        )
                        jsd = np.nan
                        wd = np.nan
                        try:
                            if isinstance(res_fed, dict):
                                payload = res_fed.get("fed_global_model", None)
                                if isinstance(payload, dict):
                                    real_df = _build_real_df(Xtr_int, ytr_int)
                                    syn_full = _sample_full_from_payload(payload, n=len(Xtr_int), rng=np.random.default_rng(0))
                                    syn_df = _build_syn_df(syn_full)
                                    fid_cov_priv = _eval_fidelity_coverage_privacy(real_df, syn_df)
                                    jsd, wd = _jsd_wd_features(real_df, syn_df)
                        except Exception as e_metrics:
                            print(f"[metrics] Failed to compute fidelity/coverage/privacy/jsd/wd for {name} fold {fold_idx}: {repr(e_metrics)}")

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

                        # Build and save result row
                        row = dict(
                            dataset=name, fold=fold_idx, k=k_global,
                            n_train=len(Xtr_int), n_test=len(Xte_int),
                            time_central_sec=res_central.get("time_central_sec", np.nan),
                            time_fed_sec=res_fed.get("train_time_sec", np.nan),
                            time_fedmle_sec=res_fed_mle.get("train_time_sec", np.nan),
                            weights=res_fed.get("weights", None),
                            s_y=res_fed.get("s_y", None),
                            fidelity_col=fid_cov_priv["fidelity_col"],
                            fidelity_row=fid_cov_priv["fidelity_row"],
                            fidelity_agg=fid_cov_priv["fidelity_agg"],
                            coverage=fid_cov_priv["coverage"],
                            privacy_dcr=fid_cov_priv["privacy_dcr"],
                            jsd=jsd,
                            wd=wd,
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

        # Compute summary statistics
        numeric_cols = [c for c in df.columns if c not in ("dataset", "weights", "s_y")]
        df_sum = df.groupby("dataset")[numeric_cols].mean().reset_index()

        # Save final results
        out_sum.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_folds, index=False)
        df_sum.to_csv(out_sum, index=False)
        
        print(f"\nWrote per-fold results: {out_folds}")
        print(f"Wrote summary: {out_sum}")
        
        return df, df_sum

    except Exception as e_final:
        print(f"Error in final processing: {repr(e_final)}")
        return pd.DataFrame(), pd.DataFrame()
    


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