import numpy as np
import time
from utils import dirichlet_split, _evaluate_synthetic_classifiers, build_full_table
from federated_models.FedGanblr import build_global_kdb_from_gm, derive_global_meta, _local_cpt_counts
from base_models.Ganblr import MLE_KDB
def _aggregate_mle_avg_only(local_models: list[dict]) -> dict:
    """
    Average-only aggregation (uniform over clients).
    local_models: list of dicts with keys: 'py' (np.ndarray), 'thetas' (dict[int]->np.ndarray), 'n' (int)
    Returns gm dict: {'py': np.ndarray, 'thetas': dict[int]->np.ndarray, 'weights': list[float]}
    """
    if not local_models:
        return dict(py=None, thetas={}, weights=[])
    K = len(local_models)
    weights = np.ones(K, dtype=np.float64) / K

    # py
    C = local_models[0]["py"].shape[0]
    py_stack = np.stack([m["py"] for m in local_models], axis=0)  # [K,C]
    py = np.sum(weights[:, None] * py_stack, axis=0)
    py = np.clip(py, 1e-12, None); py = py / py.sum()

    # thetas
    # ensure same keys and shapes across clients
    thetas_global: dict[int, np.ndarray] = {}
    keys = sorted(local_models[0]["thetas"].keys())
    for v in keys:
        t_stack = np.stack([m["thetas"][v] for m in local_models], axis=0)  # [K, num_cfg, r_v]
        tg = np.sum(weights[:, None, None] * t_stack, axis=0)
        tg = np.clip(tg, 1e-12, None)
        tg = tg / tg.sum(axis=1, keepdims=True)
        thetas_global[v] = tg

    return dict(py=py, thetas=thetas_global, weights=weights.tolist())


def run_one_fold_fed_mle_naive(
    Xtr_int, ytr_int, Xte_int, yte_int, card_feat, num_classes,
    k_global=2, num_clients=5, dir_alpha=0.1, cap_train=None, verbose=False
):
    """
    Naive federated MLE:
      - Split TRAIN into clients (Dirichlet)
      - Each client fits local MLE_KDB with a COMMON global structure
      - Server aggregates by simple uniform average of probabilities (py, CPT rows)
      - Sample synthetic from aggregated model and run TSTR (LR/MLP/RF)
    Returns acc_*/nll_* dict plus timing and weights.
    """
    t0 = time.time()

    # Optional cap
    if cap_train is not None and len(Xtr_int) > cap_train:
        sel = np.random.default_rng(1).choice(len(Xtr_int), size=cap_train, replace=False)
        Xtr_int = Xtr_int[sel]; ytr_int = ytr_int[sel]

    # Global schema from TRAIN (common parents across clients)
    card_all = [num_classes] + list(card_feat)
    y_index = 0
    train_arr, test_arr = build_full_table(ytr_int, yte_int, Xtr_int, Xte_int)
    card_glob, parents_glob, y_index_glob = derive_global_meta(train_arr, card_all, y_index, k=k_global)

    # Build non-IID splits
    clients_data = dirichlet_split(Xtr_int, ytr_int, num_clients=num_clients, alpha=dir_alpha)

    # Fit local MLEs
    local_models = []
    for i, (Xc, yc) in enumerate(clients_data):
        n_i = int(len(Xc))
        if n_i == 0:
            continue
        data_i = np.column_stack([yc, Xc]).astype(np.int32)
        mle_i = MLE_KDB(card_glob, y_index_glob, parents_glob, alpha=1.0)
        mle_i.fit(data_i)
        local_models.append(dict(py=mle_i.py.copy(), thetas={v: t.copy() for v, t in mle_i.theta.items()}, n=n_i))
        if verbose:
            print(f"[naive-fed-mle] client {i}: n={n_i}")

    if not local_models:
        return dict(acc_lr=np.nan, nll_lr=np.nan, acc_mlp=np.nan, nll_mlp=np.nan, acc_rf=np.nan, nll_rf=np.nan,
                    train_time_sec=0.0, weights=None)

    # Aggregate (uniform average)
    gm = _aggregate_mle_avg_only(local_models)

    # Build generator and TSTR
    gen_global = build_global_kdb_from_gm(gm, card_glob, parents_glob, y_index_glob)
    n_syn = len(Xtr_int)
    syn_full = gen_global.sample(n=n_syn, rng=np.random.default_rng(0), order=None, return_y=True)
    X_syn, y_syn = syn_full[:, 1:], syn_full[:, 0]
    ev = _evaluate_synthetic_classifiers(X_syn, y_syn, Xte_int, yte_int)

    t1 = time.time()
    out = dict(train_time_sec=t1 - t0, weights=gm.get("weights", None))
    out.update(ev)  # acc_lr, nll_lr, acc_mlp, nll_mlp, acc_rf, nll_rf
    return out

def _aggregate_mle_counts(local_models: list[dict]) -> dict:
    """
    Count-based aggregation for federated MLE.
    local_models: list of dicts with keys: 'py_counts', 'theta_counts' (dict[int]->np.ndarray), 'n' (int)
    Returns gm dict: {'py': np.ndarray, 'thetas': dict[int]->np.ndarray, 'weights': list[float]}
    """
    if not local_models:
        return dict(py=None, thetas={}, weights=[])
    
    K = len(local_models)
    weights = np.ones(K, dtype=np.float64) / K

    # Aggregate py counts
    py_counts_sum = np.zeros_like(local_models[0]["py_counts"], dtype=np.float64)
    for m in local_models:
        py_counts_sum += m["py_counts"]
    
    # Convert to probabilities with Laplace smoothing
    py_global = (py_counts_sum + 1.0) / (py_counts_sum.sum() + py_counts_sum.shape[0])

    # Aggregate theta counts per feature
    thetas_global = {}
    keys = sorted(local_models[0]["theta_counts"].keys())
    for v in keys:
        # Sum counts across all clients
        theta_counts_sum = np.zeros_like(local_models[0]["theta_counts"][v], dtype=np.float64)
        for m in local_models:
            theta_counts_sum += m["theta_counts"][v]
        
        # Convert to probabilities with Laplace smoothing
        tg = theta_counts_sum + 1.0  # Add Laplace smoothing
        tg = tg / tg.sum(axis=1, keepdims=True)  # Normalize each row
        thetas_global[v] = tg

    return dict(py=py_global, thetas=thetas_global, weights=weights.tolist())


def run_one_fold_fed_mle(
    Xtr_int, ytr_int, Xte_int, yte_int, card_feat, num_classes,
    k_global=2, num_clients=5, dir_alpha=0.2,
    aggregation="avg",
    eval_syn_frac=1.0, cap_train=None, split_seed=42,
    verbose=False, diagnostics_dir=None, ray_local_mode=False,
    **_ignored,
):
    """
    Framework-compatible simple federated MLE (one-shot, no training rounds).

      1. Split TRAIN into `num_clients` non-IID Dirichlet shards.
      2. Each client fits a local MLE_KDB on a COMMON global KDB structure (k=k_global).
      3. Server aggregates:
           - aggregation="avg":    uniform average of the per-client probability
                                   tables (py, CPT rows). "Average the values at
                                   the server."
           - aggregation="counts": pool raw per-client counts, then Laplace-normalise
                                   once (the count-exact federated MLE).
      4. Sample n_syn = eval_syn_frac * n_train synthetic rows and run TSTR
         (LR / MLP / RF / XGB), scoring accuracy + NLL on the real test set.

    Accepts (and ignores) the extra ablation-framework kwargs that don't apply to a
    one-shot MLE (num_rounds, local_epochs, batch_size, disc_epochs, gamma, kl_lambda,
    beta_pow, cpt_mix, use_theta_weights, alpha_mix, tau_floor) via **_ignored.
    """
    t0 = time.time()

    if cap_train is not None and len(Xtr_int) > cap_train:
        sel = np.random.default_rng(1).choice(len(Xtr_int), size=cap_train, replace=False)
        Xtr_int = Xtr_int[sel]; ytr_int = ytr_int[sel]

    # Common global schema/structure from TRAIN
    card_all = [int(num_classes)] + list(card_feat)
    y_index = 0
    train_arr, _ = build_full_table(ytr_int, yte_int, Xtr_int, Xte_int)
    card_glob, parents_glob, y_index_glob = derive_global_meta(train_arr, card_all, y_index,
                                                               k=int(k_global))
    C = int(card_glob[y_index_glob])
    # parents WITH Y first (for count extraction)
    parents_full = {v: [y_index_glob] + list(parents_glob.get(v, []))
                    for v in range(len(card_glob)) if v != y_index_glob}

    # Deterministic client partition (paired across configs for a given split_seed)
    clients_data = dirichlet_split(Xtr_int, ytr_int, num_clients=num_clients, alpha=dir_alpha,
                                   rng=np.random.default_rng(int(split_seed)))

    use_counts = str(aggregation).lower() == "counts"

    local_models = []
    for i, (Xc, yc) in enumerate(clients_data):
        n_i = int(len(Xc))
        if n_i == 0:
            continue
        data_i = np.column_stack([yc, Xc]).astype(np.int32)
        if use_counts:
            py_counts = np.bincount(np.asarray(yc, dtype=np.int64), minlength=C).astype(np.float64)
            theta_counts = _local_cpt_counts(data_i, card_glob, y_index_glob, parents_full)
            local_models.append(dict(py_counts=py_counts, theta_counts=theta_counts, n=n_i))
        else:
            mle_i = MLE_KDB(card_glob, y_index_glob, parents_glob, alpha=1.0)
            mle_i.fit(data_i)
            local_models.append(dict(py=mle_i.py.copy(),
                                     thetas={v: t.copy() for v, t in mle_i.theta.items()},
                                     n=n_i))
        if verbose:
            print(f"[fed-mle:{aggregation}] client {i}: n={n_i}")

    if not local_models:
        return dict(acc_lr=np.nan, nll_lr=np.nan, acc_mlp=np.nan, nll_mlp=np.nan,
                    acc_rf=np.nan, nll_rf=np.nan, acc_xgb=np.nan, nll_xgb=np.nan,
                    train_time_sec=time.time() - t0, weights=None)

    gm = _aggregate_mle_counts(local_models) if use_counts else _aggregate_mle_avg_only(local_models)

    gen_global = build_global_kdb_from_gm(gm, card_glob, parents_glob, y_index_glob)
    n_full = len(Xtr_int)
    n_syn = max(1, int(n_full * float(eval_syn_frac)))
    syn_full = gen_global.sample(n=n_syn, rng=np.random.default_rng(0), order=None, return_y=True)
    X_syn, y_syn = syn_full[:, 1:], syn_full[:, 0]
    ev = _evaluate_synthetic_classifiers(X_syn, y_syn, Xte_int, yte_int)

    out = dict(train_time_sec=time.time() - t0, weights=gm.get("weights", None))
    out.update(ev)
    return out