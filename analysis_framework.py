"""
Statistical Analysis Framework for FedGANBLR
=============================================
Analyses how KL divergence, importance weighting, and weighted aggregation
affect model learning stability and soundness.

Usage:
    # Convergence analysis on existing diagnostics
    python analysis_framework.py convergence --diagnostics-dir diagnostics/adult/fold_01

    # Ablation study (runs training with toggled components)
    python analysis_framework.py ablation --datasets adult nursery --num-rounds 10

    # Full analysis (ablation + convergence on generated diagnostics)
    python analysis_framework.py full --datasets adult --num-rounds 10
"""

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import RepeatedStratifiedKFold

# --- Project imports (reuse existing functions, never reimplement) ---
from evaluation import run_one_fold_fed_ganblr, run_one_fold_ganblr, _soft_clear_tf_and_ray
from evaluation_fedstruct import run_one_fold_fed_ganblr_fedstruct
from federated_models.FedMLE import run_one_fold_fed_mle
from utils import (
    fetch_openml_safely,
    discretize_train_test_no_leak,
    preprocess_covertype_binary_columns,
    _evaluate_synthetic_classifiers,
)
from federated_models.FedGanblr import _kl, build_global_kdb_from_gm
from base_models.KDependenceBayesian import MLE_KDB

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("analysis_output")

# ---------------------------------------------------------------------------
# Ablation configs for FedGANBLR
# ---------------------------------------------------------------------------
# Two KL mechanisms must be kept distinct:
#   gamma     -> SERVER-side KL-weighted aggregation: client_i weighted by
#                n_i * exp(-gamma * KL_i). gamma=0 => plain n-weighted FedAvg.
#   kl_lambda -> CLIENT-side KL proximal regulariser in the local training loss
#                (pull each client's CPTs toward the global CPTs).
# Other components:
#   beta_pow          -> importance-weighting exponent on the per-row KL weights
#                        W = mix^(-beta_pow); beta_pow=0 => uniform W (mechanism OFF).
#   use_theta_weights -> True: build W from a mixture of learned PARAMETERS
#                        (theta_local vs theta_global); False: from raw empirical
#                        COUNTS. Only has any effect when beta_pow > 0.
#   alpha_mix         -> mixture weight (alpha_mix*global + (1-alpha_mix)*local) for W.
#   cpt_mix           -> post-training CPT interpolation toward the global model.
#
# The configs are grouped so each group answers one analysis question. Default
# component values when a key is omitted come from the runner signature.
#
# NEW BASELINE = the recommended light config from the previous ablation
# (alpha_dir removed entirely). Every other config is exactly one step away from
# it, so the run answers: "is this baseline the best, or does some neighbour win?"
#   gamma=0.5  (server KL agg, slightly stronger) | kl_lambda=0.5 (client KL prox)
#   beta_pow=0.5 + use_theta_weights=True (parameter-based importance weighting)
#   cpt_mix=0.0 (post-training CPT mixing OFF by default)
_BASE = dict(gamma=0.5, beta_pow=0.5, cpt_mix=0.0, kl_lambda=0.5,
             use_theta_weights=True, alpha_mix=0.5)

ABLATION_CONFIGS = {
    # --- A. The recommended baseline ---
    "baseline":              {**_BASE},

    # --- B. Leave-one-out: turn OFF each ACTIVE component of the baseline ---
    #     If accuracy drops, the component is pulling its weight.
    "no_gamma":              {**_BASE, "gamma": 0.0},        # server-side KL agg OFF
    "no_iw":                 {**_BASE, "beta_pow": 0.0},     # importance weighting OFF
    "no_kl":                 {**_BASE, "kl_lambda": 0.0},    # client-side KL prox OFF

    # --- C. Decomposition: FedAvg floor + each KL anchor alone ---
    "all_off":               {**_BASE, "gamma": 0.0, "beta_pow": 0.0, "kl_lambda": 0.0},
    "gamma_only":            {**_BASE, "beta_pow": 0.0, "kl_lambda": 0.0},  # server KL alone
    "kl_lambda_only":        {**_BASE, "gamma": 0.0, "beta_pow": 0.0},      # client KL alone

    # --- D. gamma sensitivity (vary only gamma; baseline=0.5) ---
    "gamma_0p25":            {**_BASE, "gamma": 0.25},
    "gamma_1p0":             {**_BASE, "gamma": 1.0},

    # --- E. kl_lambda sensitivity (vary only kl_lambda; baseline=0.5) ---
    "klam_0p1":              {**_BASE, "kl_lambda": 0.1},
    "klam_1p0":              {**_BASE, "kl_lambda": 1.0},

    # --- F. cpt_mix ADD-BACK (baseline=0.0; does adding it help, esp. car/adult?) ---
    "cpt_0p1":               {**_BASE, "cpt_mix": 0.1},
    "cpt_0p25":              {**_BASE, "cpt_mix": 0.25},
    "cpt_0p6":               {**_BASE, "cpt_mix": 0.6},

    # --- G. Importance weighting: PARAMETERS (baseline) vs raw COUNTS ---
    #     baseline already IS the parameters arm; this is the counts counterpart.
    "iw_counts":             {**_BASE, "use_theta_weights": False},
}

# Ablation configs for FedStruct — identical component sweep. FedStruct adds a
# structure-learning round (round 1) on top of the parameter rounds; the same
# components apply to the parameter rounds and the structure round is always active.
# Aliased to the single source of truth above to avoid the two sets drifting apart.
ABLATION_CONFIGS_FEDSTRUCT = ABLATION_CONFIGS

# ---------------------------------------------------------------------------
# Ablation configs for the simple federated MLE (one-shot, server-side averaging)
# ---------------------------------------------------------------------------
# This model has no training-round regularisers. Its only meaningful axes are:
#   aggregation -> "avg"   : uniform average of per-client probability tables
#                            ("average the values at the server"), or
#                  "counts": pool raw counts then Laplace-normalise once.
#   k_global    -> KDB structure complexity (0 = naive Bayes, no feature parents).
ABLATION_CONFIGS_FEDMLE = {
    "baseline":    dict(aggregation="avg",    k_global=2),  # server averages prob tables, k=2
    "counts_agg":  dict(aggregation="counts", k_global=2),  # pool counts then normalise
    "naive_bayes": dict(aggregation="avg",    k_global=0),  # k=0 (Y-only parents)
    "k1":          dict(aggregation="avg",    k_global=1),
    "k3":          dict(aggregation="avg",    k_global=3),
}

# Map model name → (runner function, ablation configs, shared params)
# Resolved at runtime after DEFAULT_SHARED_PARAMS_FEDSTRUCT is defined below.

DATASET_SPECS = [
    dict(name="nursery",              data_id=76,  target="class", ef_bins=None),
    dict(name="chess",                data_id=23,  target="class", ef_bins=None),
    dict(name="car",                  data_id=19,  target="class", ef_bins=None),
    dict(name="adult",                data_id=2,   target="class", ef_bins=None),
]

CLASSIFIERS = ["lr", "mlp", "rf", "xgb"]

DEFAULT_SHARED_PARAMS = dict(
    k_global=2,
    num_clients=5,
    num_rounds=10,
    dir_alpha=0.2,
    local_epochs=3,
    batch_size=512,
    disc_epochs=1,
    eval_syn_frac=1.0,   # generate as many synthetic rows as n_train (full TSTR)
    cap_train=None,
    # kl_lambda is intentionally NOT here — each ablation config sets it explicitly
    # so the baseline vs no_kl comparison is clean.
)

# FedStruct uses the same parameters; num_rounds here = parameter-training rounds
# (an extra structure-learning round is added automatically inside the runner).
DEFAULT_SHARED_PARAMS_FEDSTRUCT = dict(
    k_global=2,
    num_clients=5,
    num_rounds=10,
    dir_alpha=0.2,
    local_epochs=3,
    batch_size=512,
    disc_epochs=1,
    eval_syn_frac=1.0,   # generate as many synthetic rows as n_train (full TSTR)
    cap_train=None,
)

# ---------------------------------------------------------------------------
# Bare (centralized) GANBLR — reference, not federated.
# Only meaningful axis here is whether the adversarial round is on.
# ---------------------------------------------------------------------------
ABLATION_CONFIGS_GANBLR = {
    "baseline":       dict(adversarial=True),    # GANBLR WITH adversarial rounds
    "no_adversarial": dict(adversarial=False),   # GANBLR-nAL (generator-only CLL)
}

DEFAULT_SHARED_PARAMS_GANBLR = dict(
    k_global=2,
    epochs=10,            # adversarial epochs (overridden by --num-rounds if given)
    batch_size=512,
    disc_epochs=1,
    warmup_epochs=1,
    eval_syn_frac=1.0,    # generate n_train synthetic rows
    cap_train=None,
)

# Simple federated MLE is one-shot: no training rounds / adversarial knobs.
DEFAULT_SHARED_PARAMS_FEDMLE = dict(
    k_global=2,
    num_clients=5,
    dir_alpha=0.2,
    eval_syn_frac=1.0,   # generate n_train synthetic rows
    cap_train=None,
)

# Registry: model name → (runner, ablation configs dict, default shared params)
MODEL_REGISTRY = {
    "fedganblr":  (run_one_fold_fed_ganblr,          ABLATION_CONFIGS,          DEFAULT_SHARED_PARAMS),
    "fedstruct":  (run_one_fold_fed_ganblr_fedstruct, ABLATION_CONFIGS_FEDSTRUCT, DEFAULT_SHARED_PARAMS_FEDSTRUCT),
    "fedmle":     (run_one_fold_fed_mle,              ABLATION_CONFIGS_FEDMLE,    DEFAULT_SHARED_PARAMS_FEDMLE),
    "ganblr":     (run_one_fold_ganblr,              ABLATION_CONFIGS_GANBLR,    DEFAULT_SHARED_PARAMS_GANBLR),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gini(weights: np.ndarray) -> float:
    """Gini coefficient of a weight vector (0=equal, 1=maximally unequal)."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.sum() == 0 or len(w) <= 1:
        return 0.0
    sorted_w = np.sort(w)
    n = len(sorted_w)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * sorted_w) / (n * np.sum(sorted_w))) - (n + 1.0) / n)


def _entropy(weights: np.ndarray) -> float:
    """Shannon entropy of a weight distribution."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    s = w.sum()
    if s <= 0:
        return 0.0
    w = w / s
    return float(-np.sum(w * np.log(w + 1e-12)))


def _effective_n(weights: np.ndarray) -> float:
    """Effective number of clients = 1 / sum(w_i^2)."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    s = w.sum()
    if s <= 0:
        return 0.0
    w = w / s
    return float(1.0 / np.sum(w ** 2))


def _setup_style():
    """Apply consistent matplotlib style."""
    plt.rcParams.update({
        "figure.figsize": (12, 8),
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 150,
    })


# ---------------------------------------------------------------------------
# Part 1: Data Loaders
# ---------------------------------------------------------------------------

def load_round_stats(diagnostics_dir: Path) -> pd.DataFrame:
    """Load global_round_stats.csv and parse JSON columns into arrays."""
    csv_path = diagnostics_dir / "global_round_stats.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Round stats not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Parse JSON columns
    for col, out in [
        ("py_json", "py_arr"),
        ("s_y_json", "s_y_arr"),
        ("weights_json", "weights_arr"),
        ("client_ns_json", "client_ns_arr"),
    ]:
        if col in df.columns:
            df[out] = df[col].apply(
                lambda x: json.loads(x) if pd.notna(x) and x else None
            )
    # Parse client_kls if stored (it may be embedded in weights_json or separate)
    return df


def load_nll_convergence(diagnostics_dir: Path) -> pd.DataFrame | None:
    """Load nll_convergence.csv if it exists. Returns None otherwise."""
    csv_path = diagnostics_dir / "nll_convergence.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def load_fold_config(diagnostics_dir: Path) -> dict:
    """Load fold_config.json."""
    p = diagnostics_dir / "fold_config.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def load_client_split_stats(diagnostics_dir: Path) -> dict:
    """Load client_split_stats.json."""
    p = diagnostics_dir / "client_split_stats.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Part 1: Convergence & Stability Plot Functions
# ---------------------------------------------------------------------------

def plot_kl_convergence(df_rounds: pd.DataFrame, out_dir: Path) -> Path:
    """
    Plot KL mean/max per round with convergence rate.
    Produces: kl_convergence.png
    """
    _setup_style()
    rounds = df_rounds["round"].values
    kl_mean = df_rounds["kl_mean"].values.astype(float)
    kl_max = df_rounds["kl_max"].values.astype(float)

    delta_kl = np.diff(kl_mean)
    convergence_rate = float(np.mean(delta_kl[-min(5, len(delta_kl)):]))  if len(delta_kl) > 0 else 0.0
    is_monotonic = bool(np.all(delta_kl <= 1e-8)) if len(delta_kl) > 0 else True

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(rounds, kl_mean, "b-o", markersize=4, label="KL mean", linewidth=2)
    ax1.plot(rounds, kl_max, "r--s", markersize=3, label="KL max", linewidth=1.5)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("KL Divergence")
    ax1.set_title(f"KL Convergence (rate={convergence_rate:.4f}/round, monotonic={is_monotonic})")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    if len(delta_kl) > 0:
        ax2 = ax1.twinx()
        ax2.bar(rounds[1:], delta_kl, alpha=0.25, color="gray", label="ΔKL")
        ax2.set_ylabel("ΔKL (per round)", color="gray")
        ax2.axhline(0, color="gray", linestyle=":", linewidth=0.5)
        ax2.legend(loc="upper right")

    out_path = out_dir / "kl_convergence.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"  KL Convergence: final_kl_mean={kl_mean[-1]:.4f}, "
          f"convergence_rate={convergence_rate:.4f}/round, monotonic={is_monotonic}")
    return out_path


def plot_weight_distribution(df_rounds: pd.DataFrame, out_dir: Path) -> Path:
    """
    Plot per-client aggregation weights over rounds + Gini/entropy.
    Produces: weight_distribution.png
    """
    _setup_style()
    rounds = df_rounds["round"].values

    # Build weight matrix [rounds x clients]
    weight_lists = df_rounds["weights_arr"].tolist()
    valid = [w for w in weight_lists if w is not None]
    if not valid:
        warnings.warn("No weight data available for weight distribution plot.")
        return out_dir / "weight_distribution.png"

    n_clients = len(valid[0])
    W = np.zeros((len(rounds), n_clients))
    for i, wl in enumerate(weight_lists):
        if wl is not None:
            arr = np.asarray(wl, dtype=np.float64)
            W[i, :len(arr)] = arr[:n_clients]

    # Compute Gini and entropy per round
    gini_vals = [_gini(W[i]) for i in range(len(rounds))]
    entropy_vals = [_entropy(W[i]) for i in range(len(rounds))]
    max_entropy = np.log(n_clients)  # for normalization

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-client weight evolution
    for c in range(n_clients):
        ax1.plot(rounds, W[:, c], "-o", markersize=3, label=f"Client {c}", linewidth=1.5)
    ax1.axhline(1.0 / n_clients, color="black", linestyle=":", linewidth=1, label="Uniform")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Aggregation Weight")
    ax1.set_title("Client Weight Evolution")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Right: Gini + normalized entropy
    ax2.plot(rounds, gini_vals, "b-o", markersize=4, label="Gini", linewidth=2)
    ax2_twin = ax2.twinx()
    norm_ent = [e / max_entropy if max_entropy > 0 else 0 for e in entropy_vals]
    ax2_twin.plot(rounds, norm_ent, "orange", marker="s", markersize=3,
                  label="Norm. Entropy", linewidth=1.5)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Gini Coefficient", color="blue")
    ax2_twin.set_ylabel("Normalized Entropy", color="orange")
    ax2.set_title("Weight Concentration Metrics")
    ax2.legend(loc="upper left")
    ax2_twin.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "weight_distribution.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    max_swing = float(np.max(np.abs(np.diff(W, axis=0)))) if W.shape[0] > 1 else 0.0
    print(f"  Weight stability: final_gini={gini_vals[-1]:.4f}, "
          f"mean_entropy={np.mean(entropy_vals):.4f}, max_weight_swing={max_swing:.4f}")
    return out_path


def plot_nll_convergence(df_nll: pd.DataFrame | None, out_dir: Path) -> Path | None:
    """
    Plot per-client NLL over rounds with divergence detection.
    Produces: nll_convergence.png
    """
    if df_nll is None or df_nll.empty:
        print("  NLL convergence: skipped (no nll_convergence.csv found)")
        return None

    _setup_style()

    # Pivot: rows=round, columns=client_id, values=nll
    pivot = df_nll.pivot_table(index="round", columns="client_id", values="nll", aggfunc="mean")
    rounds = pivot.index.values
    clients = pivot.columns.tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    for cid in clients:
        nll_vals = pivot[cid].values
        ax.plot(rounds, nll_vals, "-o", markersize=3, label=f"Client {cid}", linewidth=1.5)

        # Detect divergence: 3+ consecutive NLL increases
        increases = np.diff(nll_vals) > 0
        for i in range(len(increases) - 2):
            if increases[i] and increases[i + 1] and increases[i + 2]:
                ax.axvspan(rounds[i + 1], rounds[i + 3], alpha=0.15, color="red")

    ax.set_xlabel("Round")
    ax.set_ylabel("Negative Log-Likelihood")
    ax.set_title("Per-Client NLL Convergence")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    out_path = out_dir / "nll_convergence.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    final_nlls = pivot.iloc[-1].values
    print(f"  NLL convergence: final_mean={np.nanmean(final_nlls):.4f}, "
          f"final_std={np.nanstd(final_nlls):.4f}")
    return out_path


def plot_class_prior_drift(df_rounds: pd.DataFrame, out_dir: Path) -> Path:
    """
    Track how class prior py changes across rounds.
    Uses the existing _kl() from FedGanblr.py.
    Produces: class_prior_drift.png
    """
    _setup_style()
    rounds = df_rounds["round"].values
    py_lists = df_rounds["py_arr"].tolist()

    valid_pys = [(r, np.asarray(p, dtype=np.float64)) for r, p in zip(rounds, py_lists) if p is not None]
    if len(valid_pys) < 2:
        print("  Class prior drift: skipped (insufficient data)")
        return out_dir / "class_prior_drift.png"

    py_rounds, py_arrays = zip(*valid_pys)
    py_rounds = np.array(py_rounds)
    n_classes = len(py_arrays[0])

    # Compute KL between consecutive rounds
    kl_drifts = []
    for i in range(1, len(py_arrays)):
        kl_val = _kl(py_arrays[i], py_arrays[i - 1])
        kl_drifts.append(kl_val)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])

    # Top: per-class prior over rounds
    for c in range(n_classes):
        vals = [p[c] for p in py_arrays]
        ax1.plot(py_rounds, vals, "-o", markersize=3, label=f"Class {c}", linewidth=1.5)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("P(Y=c)")
    ax1.set_title("Class Prior Evolution")
    ax1.legend(fontsize=8, ncol=min(n_classes, 5))
    ax1.grid(True, alpha=0.3)

    # Bottom: KL drift
    ax2.bar(py_rounds[1:], kl_drifts, color="steelblue", alpha=0.7)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("KL(py_t || py_{t-1})")
    ax2.set_title("Class Prior Drift per Round")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "class_prior_drift.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    total_drift = sum(kl_drifts)
    max_drift = max(kl_drifts) if kl_drifts else 0.0
    print(f"  Class prior drift: total_KL_drift={total_drift:.6f}, "
          f"max_single_round_drift={max_drift:.6f}")
    return out_path


def plot_effective_clients(df_rounds: pd.DataFrame, out_dir: Path) -> Path:
    """
    Compute K_eff = 1 / sum(w_i^2) per round.
    Produces: effective_clients.png
    """
    _setup_style()
    rounds = df_rounds["round"].values
    weight_lists = df_rounds["weights_arr"].tolist()

    k_effs = []
    n_actual = 0
    for wl in weight_lists:
        if wl is not None:
            w = np.asarray(wl, dtype=np.float64)
            n_actual = len(w)
            k_effs.append(_effective_n(w))
        else:
            k_effs.append(0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, k_effs, "b-o", markersize=5, linewidth=2, label="K_eff")
    ax.axhline(n_actual, color="red", linestyle="--", linewidth=1.5, label=f"Total clients ({n_actual})")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="Single client dominance")
    ax.set_xlabel("Round")
    ax.set_ylabel("Effective Number of Clients")
    ax.set_title("Effective Client Participation (K_eff = 1 / Σw²)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    out_path = out_dir / "effective_clients.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"  Effective clients: final={k_effs[-1]:.2f}/{n_actual}, "
          f"mean={np.mean(k_effs):.2f}, min={np.min(k_effs):.2f}")
    return out_path


def plot_gamma_sensitivity(df_rounds: pd.DataFrame, out_dir: Path,
                           fold_config: dict | None = None) -> Path:
    """
    Analytically recompute weights for different gamma values using stored
    client_kls and client_ns. No retraining needed.
    Produces: gamma_sensitivity.png
    """
    _setup_style()
    gammas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    actual_gamma = fold_config.get("gamma", 0.25) if fold_config else 0.25

    # Use the last round's data for analysis
    last_row = df_rounds.iloc[-1]
    client_ns_raw = last_row.get("client_ns_arr", None)
    # We need client KLs. They're not directly in the CSV as an array column,
    # but kl_mean/kl_max are. For analytical sensitivity we need per-client KLs.
    # We'll use all rounds and approximate from weights + ns if needed.

    # Try to reconstruct per-client KLs from weights: w_i = n_i * exp(-gamma * kl_i)
    # => kl_i = -log(w_i / n_i) / gamma  (if gamma > 0)
    weights_raw = last_row.get("weights_arr", None)

    if client_ns_raw is None or weights_raw is None:
        print("  Gamma sensitivity: skipped (missing client_ns or weights data)")
        return out_dir / "gamma_sensitivity.png"

    ns = np.asarray(client_ns_raw, dtype=np.float64)
    ws = np.asarray(weights_raw, dtype=np.float64)
    K = len(ns)

    # Recover per-client KLs: w_i (unnormalized) = n_i * exp(-gamma * kl_i)
    # w_i / sum(w) = ws[i] => w_i_unnorm = ws[i] * C  for some constant C
    # We can recover kl_i from: ws[i] = n_i * exp(-gamma * kl_i) / Z
    # => log(ws[i]) = log(n_i) - gamma * kl_i - log(Z)
    # => kl_i = (log(n_i) - log(ws[i]) - log(Z)) / gamma
    # We don't know Z, but we can use the kl_mean for calibration.
    if actual_gamma > 1e-8:
        # Use ratio approach: ws[i]/ws[j] = (n_i/n_j) * exp(-gamma*(kl_i - kl_j))
        # Set kl_0 = 0 (reference), then kl_i = log(n_i * ws[0] / (n_0 * ws[i])) / gamma
        ref_idx = 0
        kls = np.zeros(K)
        for i in range(K):
            ratio = (ns[i] * ws[ref_idx]) / (ns[ref_idx] * ws[i] + 1e-15)
            kls[i] = np.log(max(ratio, 1e-15)) / actual_gamma
        # Shift so min KL is 0 (KL is non-negative)
        kls -= kls.min()
    else:
        # gamma=0 means uniform weighting by n, can't recover KLs
        # Use kl_mean as fallback: assume all clients have kl_mean
        kl_mean = float(last_row.get("kl_mean", 0.1))
        kls = np.full(K, kl_mean)
        # Add small perturbation to show sensitivity
        rng = np.random.default_rng(42)
        kls += rng.uniform(0, kl_mean * 0.5, K)

    # Now recompute weights for each gamma
    k_effs = []
    ginis = []
    for g in gammas:
        w_raw = ns * np.exp(-g * kls)
        w_sum = w_raw.sum()
        if w_sum > 0:
            w_norm = w_raw / w_sum
        else:
            w_norm = np.ones(K) / K
        k_effs.append(_effective_n(w_norm))
        ginis.append(_gini(w_norm))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(gammas, k_effs, "b-o", markersize=6, linewidth=2)
    ax1.axvline(actual_gamma, color="red", linestyle="--", linewidth=1.5,
                label=f"Actual γ={actual_gamma}")
    ax1.axhline(K, color="gray", linestyle=":", linewidth=1, label=f"Max ({K})")
    ax1.set_xlabel("γ (gamma)")
    ax1.set_ylabel("K_eff")
    ax1.set_title("Effective Clients vs γ")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(gammas, ginis, "orange", marker="s", markersize=6, linewidth=2)
    ax2.axvline(actual_gamma, color="red", linestyle="--", linewidth=1.5,
                label=f"Actual γ={actual_gamma}")
    ax2.set_xlabel("γ (gamma)")
    ax2.set_ylabel("Gini Coefficient")
    ax2.set_title("Weight Inequality vs γ")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "gamma_sensitivity.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"  Gamma sensitivity: at γ={actual_gamma}, K_eff={k_effs[gammas.index(actual_gamma) if actual_gamma in gammas else 2]:.2f}")
    return out_path


def plot_component_summary(df_rounds: pd.DataFrame, out_dir: Path) -> Path:
    """
    Combined 2x2 dashboard: KL convergence, weight distribution, K_eff, py drift.
    Produces: component_summary.png
    """
    _setup_style()
    rounds = df_rounds["round"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) KL convergence
    ax = axes[0, 0]
    kl_mean = df_rounds["kl_mean"].values.astype(float)
    kl_max = df_rounds["kl_max"].values.astype(float)
    ax.plot(rounds, kl_mean, "b-o", markersize=3, label="KL mean", linewidth=2)
    ax.plot(rounds, kl_max, "r--s", markersize=2, label="KL max", linewidth=1.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) Weight distribution
    ax = axes[0, 1]
    weight_lists = df_rounds["weights_arr"].tolist()
    valid = [w for w in weight_lists if w is not None]
    if valid:
        n_clients = len(valid[0])
        W = np.zeros((len(rounds), n_clients))
        for i, wl in enumerate(weight_lists):
            if wl is not None:
                arr = np.asarray(wl, dtype=np.float64)
                W[i, :len(arr)] = arr[:n_clients]
        for c in range(n_clients):
            ax.plot(rounds, W[:, c], "-", linewidth=1.2, label=f"C{c}")
        ax.axhline(1.0 / n_clients, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Round")
    ax.set_ylabel("Weight")
    ax.set_title("Client Weights")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (1,0) Effective clients
    ax = axes[1, 0]
    k_effs = []
    n_actual = 0
    for wl in weight_lists:
        if wl is not None:
            w = np.asarray(wl, dtype=np.float64)
            n_actual = len(w)
            k_effs.append(_effective_n(w))
        else:
            k_effs.append(0.0)
    ax.plot(rounds, k_effs, "b-o", markersize=4, linewidth=2)
    ax.axhline(n_actual, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("K_eff")
    ax.set_title("Effective Clients")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # (1,1) Class prior drift
    ax = axes[1, 1]
    py_lists = df_rounds["py_arr"].tolist()
    valid_pys = [(r, np.asarray(p, dtype=np.float64)) for r, p in zip(rounds, py_lists) if p is not None]
    if len(valid_pys) >= 2:
        py_rounds, py_arrays = zip(*valid_pys)
        kl_drifts = []
        for i in range(1, len(py_arrays)):
            kl_drifts.append(_kl(py_arrays[i], py_arrays[i - 1]))
        ax.bar(py_rounds[1:], kl_drifts, color="steelblue", alpha=0.7)
    ax.set_xlabel("Round")
    ax.set_ylabel("KL(py_t || py_{t-1})")
    ax.set_title("Class Prior Drift")
    ax.grid(True, alpha=0.3)

    fig.suptitle("FedGANBLR Component Analysis Dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = out_dir / "component_summary.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Part 1: Orchestrator
# ---------------------------------------------------------------------------

def run_convergence_analysis(diagnostics_dir: Path, out_dir: Path) -> dict:
    """Run all convergence and stability analyses on existing diagnostics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    print(f"\n{'='*60}")
    print(f"Convergence Analysis: {diagnostics_dir}")
    print(f"{'='*60}")

    try:
        df_rounds = load_round_stats(diagnostics_dir)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return {"error": str(e)}

    fold_config = load_fold_config(diagnostics_dir)
    client_stats = load_client_split_stats(diagnostics_dir)
    df_nll = load_nll_convergence(diagnostics_dir)

    print(f"  Loaded {len(df_rounds)} rounds of data")
    if fold_config:
        print(f"  Config: gamma={fold_config.get('gamma')}, "
              f"cpt_mix={fold_config.get('cpt_mix')}, "
              f"kl_lambda={fold_config.get('kl_lambda')}")

    # Run all plot functions
    plot_kl_convergence(df_rounds, out_dir)
    plot_weight_distribution(df_rounds, out_dir)
    plot_nll_convergence(df_nll, out_dir)
    plot_class_prior_drift(df_rounds, out_dir)
    plot_effective_clients(df_rounds, out_dir)
    plot_gamma_sensitivity(df_rounds, out_dir, fold_config)
    plot_component_summary(df_rounds, out_dir)

    print(f"\n  All plots saved to: {out_dir}")
    return summary


# ---------------------------------------------------------------------------
# Part 2: Ablation Study
# ---------------------------------------------------------------------------

def prepare_dataset_folds(dataset_spec: dict, num_folds: int = 1, ef_bins: int = 12,
                          random_state: int = 2025) -> list:
    """
    Load a dataset once and yield up to `num_folds` stratified train/test splits.

    Returns a list of tuples:
        (fold_idx, Xtr_int, ytr_int, Xte_int, yte_int, card_feat, num_classes).

    num_folds=1 reproduces the original single-split behaviour (n_splits=2, first
    fold, ~50% test). num_folds>1 uses n_splits=num_folds (test ~ 1/num_folds each),
    so every sample is held out exactly once across the folds.
    """
    name = dataset_spec["name"]
    num_folds = max(1, int(num_folds))
    print(f"\n  Preparing dataset: {name} ({num_folds} fold(s))")

    X, y = fetch_openml_safely(name=name, data_id=dataset_spec.get("data_id"),
                                target=dataset_spec["target"])
    y = y.astype("category").cat.codes

    # n_splits must be >= 2; for a single fold we keep the historical 50/50 split.
    n_splits = max(2, num_folds)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=random_state)
    ef_bins_use = dataset_spec.get("ef_bins") or ef_bins

    folds = []
    for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
        if fold_idx >= num_folds:
            break
        Xtr_df, Xte_df = X.iloc[tr_idx], X.iloc[te_idx]
        ytr_sr, yte_sr = y.iloc[tr_idx], y.iloc[te_idx]

        if name.lower() == "covertype":
            Xtr_df = preprocess_covertype_binary_columns(Xtr_df)
            Xte_df = preprocess_covertype_binary_columns(Xte_df)

        Xtr_int, Xte_int, ytr_int, yte_int, card_feat, classes = discretize_train_test_no_leak(
            Xtr_df, ytr_sr, Xte_df, yte_sr, strategy="ef", ef_bins=ef_bins_use
        )
        num_classes = len(classes)
        print(f"    fold {fold_idx}: n_train={len(Xtr_int)}, n_test={len(Xte_int)}, "
              f"n_features={Xtr_int.shape[1]}, n_classes={num_classes}")
        folds.append((fold_idx, Xtr_int, ytr_int, Xte_int, yte_int, card_feat, num_classes))

    return folds


def prepare_dataset(dataset_spec: dict, ef_bins: int = 12,
                    random_state: int = 2025) -> tuple:
    """
    Single train/test split (fold 0). Thin wrapper over prepare_dataset_folds for
    callers that only need one split (e.g. reconstruct_from_diagnostics).
    Returns (Xtr_int, ytr_int, Xte_int, yte_int, card_feat, num_classes).
    """
    folds = prepare_dataset_folds(dataset_spec, num_folds=1, ef_bins=ef_bins,
                                  random_state=random_state)
    _, Xtr_int, ytr_int, Xte_int, yte_int, card_feat, num_classes = folds[0]
    return Xtr_int, ytr_int, Xte_int, yte_int, card_feat, num_classes


def run_single_ablation(config_name: str, config_params: dict,
                        Xtr_int, ytr_int, Xte_int, yte_int,
                        card_feat, num_classes,
                        shared_params: dict,
                        diagnostics_root: Path | None = None,
                        model: str = "fedganblr") -> dict:
    """Run a single ablation configuration and return results."""
    runner, _, _ = MODEL_REGISTRY.get(model, MODEL_REGISTRY["fedganblr"])

    merged = {**shared_params, **config_params}
    diag_dir = None
    if diagnostics_root is not None:
        diag_dir = diagnostics_root / config_name
        diag_dir.mkdir(parents=True, exist_ok=True)

    # Show the knobs that actually differ for this config (model-agnostic)
    cfg_summary = ", ".join(f"{k}={v}" for k, v in config_params.items())
    print(f"    [{model}] config '{config_name}': {cfg_summary}")

    t0 = time.time()
    try:
        res = runner(
            Xtr_int=Xtr_int, ytr_int=ytr_int,
            Xte_int=Xte_int, yte_int=yte_int,
            card_feat=card_feat, num_classes=num_classes,
            ray_local_mode=True,
            diagnostics_dir=diag_dir,
            **merged,
        )
    except Exception as e:
        warnings.warn(f"    Config '{config_name}' failed: {e}")
        res = {f"acc_{c}": np.nan for c in CLASSIFIERS}
        res.update({f"nll_{c}": np.nan for c in CLASSIFIERS})
        res["train_time_sec"] = time.time() - t0

    try:
        _soft_clear_tf_and_ray()
    except Exception:
        pass

    out = {"config_name": config_name}
    for c in CLASSIFIERS:
        out[f"acc_{c}"] = res.get(f"acc_{c}", np.nan)
        out[f"nll_{c}"] = res.get(f"nll_{c}", np.nan)
    out["train_time_sec"] = res.get("train_time_sec", np.nan)

    acc_vals = [out[f"acc_{c}"] for c in CLASSIFIERS if not np.isnan(out.get(f"acc_{c}", np.nan))]
    mean_acc = np.mean(acc_vals) if acc_vals else np.nan
    print(f"      -> mean_acc={mean_acc:.4f}, time={out['train_time_sec']:.1f}s")
    return out


def run_ablation_study(dataset_specs: list, out_dir: Path,
                       configs: dict | None = None,
                       shared_params: dict | None = None,
                       model: str = "fedganblr",
                       num_folds: int = 1) -> pd.DataFrame:
    """
    Run ablation study across datasets and configurations.

    num_folds > 1 repeats every (dataset, config) across that many stratified
    train/test splits, so downstream aggregation can report mean ± std and tell
    real component effects from fold noise. Rich diagnostics (round stats, saved
    model) are written only for fold 0 to keep the on-disk layout stable; metrics
    are collected for every fold.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    _, default_configs, default_shared = MODEL_REGISTRY.get(model, MODEL_REGISTRY["fedganblr"])
    if configs is None:
        configs = default_configs
    if shared_params is None:
        shared_params = dict(default_shared)

    num_folds = max(1, int(num_folds))
    print(f"\n  Model: {model}  |  Folds: {num_folds}  |  Configs: {list(configs.keys())}")

    all_results = []

    for spec in dataset_specs:
        name = spec["name"]
        print(f"\n{'='*60}")
        print(f"Ablation Study [{model}]: {name}")
        print(f"{'='*60}")

        try:
            folds = prepare_dataset_folds(spec, num_folds=num_folds)
        except Exception as e:
            print(f"  Skipping {name}: {e}")
            continue

        diag_root = out_dir / "diagnostics" / name

        for fold_idx, Xtr_int, ytr_int, Xte_int, yte_int, card_feat, num_classes in folds:
            if num_folds > 1:
                print(f"\n  --- {name}: fold {fold_idx + 1}/{num_folds} ---")
            # Same client partition for every config in this fold (paired comparison);
            # different folds get a distinct, reproducible split.
            fold_shared = {**shared_params, "split_seed": 1000 + fold_idx}
            for config_name, config_params in configs.items():
                result = run_single_ablation(
                    config_name=config_name,
                    config_params=config_params,
                    Xtr_int=Xtr_int, ytr_int=ytr_int,
                    Xte_int=Xte_int, yte_int=yte_int,
                    card_feat=card_feat, num_classes=num_classes,
                    shared_params=fold_shared,
                    # Rich diagnostics only for fold 0 (keeps reconstruct/convergence layout)
                    diagnostics_root=(diag_root if fold_idx == 0 else None),
                    model=model,
                )
                result["dataset"] = name
                result["fold"] = fold_idx
                all_results.append(result)

    df = pd.DataFrame(all_results)
    csv_path = out_dir / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")
    return df


def _collapse_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Average metrics over folds so there is one row per (dataset, config_name).

    No-op when there is no 'fold' column / already one row per (dataset, config).
    Used to feed the per-dataset charts and delta tables, which assume a single
    row per (dataset, config).
    """
    if "config_name" not in df.columns:
        return df
    keys = [k for k in ("dataset", "config_name") if k in df.columns]
    metric_cols = [c for c in df.columns
                   if c.startswith(("acc_", "nll_")) or c == "train_time_sec"]
    if not metric_cols:
        return df
    return df.groupby(keys, sort=False)[metric_cols].mean().reset_index()


def compute_ablation_deltas(df_ablation: pd.DataFrame) -> pd.DataFrame:
    """Compute differences from baseline for each configuration."""
    df_ablation = _collapse_folds(df_ablation)  # average folds first if present
    rows = []
    metric_cols = [c for c in df_ablation.columns if c.startswith(("acc_", "nll_"))]

    for dataset in df_ablation["dataset"].unique():
        df_ds = df_ablation[df_ablation["dataset"] == dataset]
        baseline_row = df_ds[df_ds["config_name"] == "baseline"]
        if baseline_row.empty:
            continue
        baseline = baseline_row.iloc[0]

        for _, row in df_ds.iterrows():
            if row["config_name"] == "baseline":
                continue
            for metric in metric_cols:
                base_val = float(baseline[metric])
                curr_val = float(row[metric])
                delta = curr_val - base_val
                pct = (delta / abs(base_val) * 100) if abs(base_val) > 1e-10 else 0.0
                rows.append({
                    "dataset": dataset,
                    "config_name": row["config_name"],
                    "metric": metric,
                    "baseline_val": base_val,
                    "value": curr_val,
                    "delta": delta,
                    "pct_change": pct,
                })

    df_deltas = pd.DataFrame(rows)

    # Print summary table
    if not df_deltas.empty:
        print(f"\n{'='*60}")
        print("Ablation Deltas from Baseline")
        print(f"{'='*60}")
        for dataset in df_deltas["dataset"].unique():
            print(f"\n  Dataset: {dataset}")
            df_ds = df_deltas[df_deltas["dataset"] == dataset]
            pivot = df_ds.pivot_table(index="config_name", columns="metric",
                                      values="delta", aggfunc="first")
            # Show only accuracy columns for readability
            acc_cols = [c for c in pivot.columns if c.startswith("acc_")]
            if acc_cols:
                print(pivot[acc_cols].to_string(float_format="{:+.4f}".format))

    return df_deltas


def plot_ablation_bar_chart(df_ablation: pd.DataFrame, out_dir: Path,
                            metric_type: str = "acc") -> Path:
    """
    Grouped bar chart comparing configurations.
    metric_type: 'acc' or 'nll'
    """
    _setup_style()
    df_ablation = _collapse_folds(df_ablation)  # average folds -> one bar per config
    datasets = df_ablation["dataset"].unique()
    n_datasets = len(datasets)
    metric_cols = [f"{metric_type}_{c}" for c in CLASSIFIERS]

    fig, axes = plt.subplots(1, max(n_datasets, 1), figsize=(7 * n_datasets, 6),
                              squeeze=False)

    for idx, dataset in enumerate(datasets):
        ax = axes[0, idx]
        df_ds = df_ablation[df_ablation["dataset"] == dataset].copy()
        configs = df_ds["config_name"].values
        n_configs = len(configs)
        n_metrics = len(metric_cols)

        x = np.arange(n_configs)
        width = 0.8 / n_metrics
        colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))

        for j, col in enumerate(metric_cols):
            vals = df_ds[col].values.astype(float)
            offset = (j - n_metrics / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=col.split("_")[1].upper(),
                         color=colors[j], edgecolor="gray", linewidth=0.5)
            # Add value labels
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                           f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(metric_type.upper())
        ax.set_title(f"{dataset}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(f"Ablation Study: {metric_type.upper()} by Configuration",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / f"ablation_{metric_type}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ablation_delta_heatmap(df_deltas: pd.DataFrame, out_dir: Path) -> Path:
    """
    Heatmap of accuracy deltas from baseline.
    y-axis: configs, x-axis: metrics, color: delta.
    """
    _setup_style()
    if df_deltas.empty:
        print("  Heatmap: skipped (no delta data)")
        return out_dir / "ablation_heatmap.png"

    datasets = df_deltas["dataset"].unique()
    n_datasets = len(datasets)

    fig, axes = plt.subplots(1, max(n_datasets, 1), figsize=(8 * n_datasets, 5),
                              squeeze=False)

    for idx, dataset in enumerate(datasets):
        ax = axes[0, idx]
        df_ds = df_deltas[df_deltas["dataset"] == dataset]
        acc_metrics = [m for m in df_ds["metric"].unique() if m.startswith("acc_")]

        if not acc_metrics:
            continue

        pivot = df_ds[df_ds["metric"].isin(acc_metrics)].pivot_table(
            index="config_name", columns="metric", values="delta", aggfunc="first"
        )

        if pivot.empty:
            continue

        # Plot heatmap
        data = pivot.values
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto",
                        vmin=-np.nanmax(np.abs(data)), vmax=np.nanmax(np.abs(data)))

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([c.replace("acc_", "").upper() for c in pivot.columns],
                           fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)

        # Annotate cells
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > np.nanmax(np.abs(data)) * 0.6 else "black"
                    ax.text(j, i, f"{val:+.4f}", ha="center", va="center",
                           fontsize=8, color=color)

        ax.set_title(f"{dataset}: Accuracy Δ from Baseline")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    out_path = out_dir / "ablation_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Aggregate across datasets: mean metric per (config, classifier)
# ---------------------------------------------------------------------------

def compute_dataset_average(df_ablation: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Average each metric across datasets for every (config, classifier).

    Produces one row per config holding the mean over all datasets of each
    acc_/nll_ column, plus an across-classifier mean (acc_mean / nll_mean).
    Rows are ranked by mean accuracy (best first). Writes
    'ablation_mean_across_datasets.csv' (means + per-metric std) and prints a
    ranked summary so the best overall config is obvious.
    """
    metric_cols = [c for c in df_ablation.columns if c.startswith(("acc_", "nll_"))]
    if not metric_cols or "config_name" not in df_ablation.columns:
        print("  Dataset average: skipped (no metric columns or config_name)")
        return pd.DataFrame()

    has_ds = "dataset" in df_ablation.columns
    n_datasets = int(df_ablation["dataset"].nunique()) if has_ds else 1

    # --- Step 1: collapse folds -> one mean per (dataset, config) ---
    unit_keys = ["dataset", "config_name"] if has_ds else ["config_name"]
    per_unit = df_ablation.groupby(unit_keys, sort=False)[metric_cols].mean().reset_index()

    # Typical within-(dataset) fold noise: std over folds, averaged across datasets.
    fold_counts = df_ablation.groupby(unit_keys, sort=False).size()
    n_folds = int(fold_counts.max()) if len(fold_counts) else 1
    if n_folds > 1:
        fold_std = (df_ablation.groupby(unit_keys, sort=False)[metric_cols]
                    .std(ddof=1)
                    .groupby(level="config_name" if has_ds else None, sort=False).mean()
                    if has_ds else
                    df_ablation.groupby("config_name", sort=False)[metric_cols].std(ddof=1))
    else:
        fold_std = per_unit.groupby("config_name", sort=False)[metric_cols].mean() * 0.0

    # --- Step 2: average across datasets (each dataset weighted equally) ---
    grouped = per_unit.groupby("config_name", sort=False)
    mean_df = grouped[metric_cols].mean()
    ds_std = grouped[metric_cols].std(ddof=0)  # between-dataset spread

    acc_cols = [c for c in metric_cols if c.startswith("acc_")]
    nll_cols = [c for c in metric_cols if c.startswith("nll_")]
    if acc_cols:
        mean_df["acc_mean"] = mean_df[acc_cols].mean(axis=1)
    if nll_cols:
        mean_df["nll_mean"] = mean_df[nll_cols].mean(axis=1)

    # Rank by overall mean accuracy (desc); fall back to first column
    sort_key = "acc_mean" if "acc_mean" in mean_df.columns else mean_df.columns[0]
    mean_df = mean_df.sort_values(sort_key, ascending=False)

    # Per-config scalar fold noise on the headline acc_mean metric (avg over classifiers)
    acc_fold_noise = (fold_std[acc_cols].mean(axis=1).reindex(mean_df.index)
                      if acc_cols and not fold_std.empty else
                      pd.Series(0.0, index=mean_df.index))

    # --- CSV: means + between-dataset std + within-dataset fold std ---
    out = mean_df.copy()
    for c in metric_cols:
        out[f"{c}_ds_std"] = ds_std[c]
        if c in fold_std.columns:
            out[f"{c}_fold_std"] = fold_std[c].reindex(mean_df.index)
    out.insert(0, "n_folds", n_folds)
    out.insert(0, "n_datasets", n_datasets)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ablation_mean_across_datasets.csv"
    out.to_csv(csv_path)
    print(f"\n  Mean-across-datasets table saved to: {csv_path}")

    if acc_cols:
        show = mean_df[acc_cols + (["acc_mean"] if "acc_mean" in mean_df.columns else [])].copy()
        if n_folds > 1:
            show["±fold"] = acc_fold_noise
        print(f"\n{'='*64}")
        print(f"Mean accuracy across {n_datasets} datasets, {n_folds} fold(s) "
              f"(ranked, best first)")
        print(f"{'='*64}")
        print(show.to_string(float_format="{:.4f}".format))

        best = mean_df[sort_key].idxmax()
        best_val = float(mean_df.loc[best, sort_key])
        print(f"\n  Best config by mean accuracy: '{best}' ({best_val:.4f})")
        if "baseline" in mean_df.index:
            print(f"  Baseline mean accuracy:        {float(mean_df.loc['baseline', sort_key]):.4f} "
                  f"(rank {list(mean_df.index).index('baseline') + 1}/{len(mean_df)})")

        # --- Significance verdict: is the top gap larger than fold noise? ---
        if n_folds > 1 and len(mean_df) > 1:
            runner_up = mean_df[sort_key].iloc[1]
            gap = best_val - float(runner_up)
            noise = float(acc_fold_noise.loc[best]) if best in acc_fold_noise.index else 0.0
            verdict = ("LIKELY REAL" if gap > noise else "WITHIN FOLD NOISE")
            print(f"\n  Top-1 vs Top-2 gap = {gap:.4f}; fold noise(±) ≈ {noise:.4f} "
                  f"-> {verdict}")
            if gap <= noise:
                print("  => The winner is not clearly separable from the runner-up; "
                      "treat the top cluster as tied.")
    else:
        print("  (no accuracy columns to rank)")

    return mean_df


def plot_dataset_average(mean_df: pd.DataFrame, out_dir: Path,
                         metric_type: str = "acc") -> Path:
    """Grouped bar chart of the across-datasets mean per config & classifier."""
    _setup_style()
    out_path = out_dir / f"ablation_mean_{metric_type}.png"
    if mean_df is None or mean_df.empty:
        print(f"  Dataset-average bar ({metric_type}): skipped (no data)")
        return out_path

    cols = [f"{metric_type}_{c}" for c in CLASSIFIERS if f"{metric_type}_{c}" in mean_df.columns]
    if not cols:
        return out_path

    configs = mean_df.index.tolist()
    x = np.arange(len(configs))
    n_metrics = len(cols)
    width = 0.8 / n_metrics
    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(configs) + 4), 6))
    for j, col in enumerate(cols):
        vals = mean_df[col].values.astype(float)
        offset = (j - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=col.split("_")[1].upper(),
               color=colors[j], edgecolor="gray", linewidth=0.5)

    mean_col = f"{metric_type}_mean"
    if mean_col in mean_df.columns:
        ax.plot(x, mean_df[mean_col].values.astype(float), "k--o",
                markersize=4, linewidth=1.5, label=f"{metric_type.upper()} mean")

    ax.set_xticks(x)
    labels = [("★ " + c if c == "baseline" else c) for c in configs]
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(f"{metric_type.upper()} (mean across datasets)")
    ax.set_title(f"Ablation: mean {metric_type.upper()} across datasets "
                 f"(per config & classifier, ranked)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_dataset_average_heatmap(mean_df: pd.DataFrame, out_dir: Path,
                                 metric_type: str = "acc") -> Path:
    """Heatmap of the across-datasets mean: rows=config (ranked), cols=classifier."""
    _setup_style()
    out_path = out_dir / f"ablation_mean_{metric_type}_heatmap.png"
    if mean_df is None or mean_df.empty:
        print(f"  Dataset-average heatmap ({metric_type}): skipped (no data)")
        return out_path

    cols = [f"{metric_type}_{c}" for c in CLASSIFIERS if f"{metric_type}_{c}" in mean_df.columns]
    mean_col = f"{metric_type}_mean"
    show_cols = cols + ([mean_col] if mean_col in mean_df.columns else [])
    if not show_cols:
        return out_path

    data = mean_df[show_cols].values.astype(float)
    cmap = "YlGn" if metric_type == "acc" else "YlOrRd_r"  # higher acc greener, lower nll better

    fig, ax = plt.subplots(figsize=(1.5 * len(show_cols) + 3, 0.5 * len(mean_df) + 2))
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(show_cols)))
    ax.set_xticklabels([c.replace(f"{metric_type}_", "").upper() for c in show_cols], fontsize=9)
    ax.set_yticks(range(len(mean_df)))
    ax.set_yticklabels(mean_df.index.tolist(), fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Mean {metric_type.upper()} across datasets (config × classifier)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def emit_dataset_average(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Convenience: compute the mean-across-datasets table and its charts."""
    mean_df = compute_dataset_average(df, out_dir)
    plot_dataset_average(mean_df, out_dir, metric_type="acc")
    plot_dataset_average(mean_df, out_dir, metric_type="nll")
    plot_dataset_average_heatmap(mean_df, out_dir, metric_type="acc")
    plot_dataset_average_heatmap(mean_df, out_dir, metric_type="nll")
    return mean_df


# ---------------------------------------------------------------------------
# Reconstruct ablation_results.csv from diagnostics directories
# ---------------------------------------------------------------------------

def reconstruct_from_diagnostics(diagnostics_root: Path, out_dir: Path,
                                  eval_syn_frac: float = 1.0) -> pd.DataFrame:
    """
    Walk diagnostics_root/{dataset}/{config}/final_global_model.json,
    regenerate synthetic data from each saved model, evaluate TSTR metrics,
    and write ablation_results.csv — without rerunning any training.

    Expected directory layout (produced by run_ablation_study):
        diagnostics_root/
          {dataset_name}/
            {config_name}/
              final_global_model.json
              fold_config.json          (optional, for metadata)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover all (dataset, config) pairs present on disk
    pairs = []
    for ds_dir in sorted(diagnostics_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        for cfg_dir in sorted(ds_dir.iterdir()):
            if not cfg_dir.is_dir():
                continue
            model_path = cfg_dir / "final_global_model.json"
            if model_path.exists():
                pairs.append((ds_dir.name, cfg_dir.name, cfg_dir))

    if not pairs:
        print(f"No final_global_model.json files found under {diagnostics_root}")
        return pd.DataFrame()

    print(f"\nFound {len(pairs)} (dataset, config) pairs to reconstruct:")
    for ds, cfg, _ in pairs:
        print(f"  {ds} / {cfg}")

    # Cache fetched datasets so each dataset is only downloaded once
    dataset_cache: dict[str, tuple] = {}

    all_results = []
    for dataset_name, config_name, cfg_dir in pairs:
        print(f"\n{'='*60}")
        print(f"Reconstructing: {dataset_name} / {config_name}")
        print(f"{'='*60}")

        # --- Load final global model ---
        try:
            raw = json.loads((cfg_dir / "final_global_model.json").read_text())
            py      = np.asarray(raw["py"], dtype=np.float64)
            thetas  = {int(v): np.asarray(t, dtype=np.float64)
                       for v, t in raw["thetas"].items()}
            card    = list(map(int, raw["card"]))
            y_index = int(raw["y_index"])
            parents = {int(k): list(map(int, v))
                       for k, v in raw["parents"].items()}
        except Exception as e:
            print(f"  Skipping — could not load model: {e}")
            continue

        # --- Fetch & discretize real data (cached per dataset) ---
        if dataset_name not in dataset_cache:
            spec = next((s for s in DATASET_SPECS if s["name"] == dataset_name), None)
            if spec is None:
                print(f"  Skipping — '{dataset_name}' not in DATASET_SPECS")
                continue
            try:
                data_tuple = prepare_dataset(spec)
                dataset_cache[dataset_name] = data_tuple
            except Exception as e:
                print(f"  Skipping — could not prepare dataset: {e}")
                continue

        Xtr_int, ytr_int, Xte_int, yte_int, card_feat, num_classes = dataset_cache[dataset_name]

        # --- Rebuild MLE_KDB from saved parameters ---
        try:
            gm = {"py": py, "thetas": thetas}
            gen = build_global_kdb_from_gm(gm, card, parents, y_index)
        except Exception as e:
            print(f"  Skipping — could not rebuild model: {e}")
            continue

        # --- Generate synthetic data ---
        try:
            n_syn = max(1, int(len(Xtr_int) * eval_syn_frac))
            syn_full = gen.sample(n=n_syn, rng=np.random.default_rng(0),
                                  order=None, return_y=True)
            X_syn = syn_full[:, [i for i in range(syn_full.shape[1]) if i != y_index]]
            y_syn = syn_full[:, y_index]
            print(f"  Generated {n_syn} synthetic samples")
        except Exception as e:
            print(f"  Skipping — synthetic generation failed: {e}")
            continue

        # --- Evaluate TSTR ---
        try:
            ev = _evaluate_synthetic_classifiers(X_syn, y_syn, Xte_int, yte_int)
        except Exception as e:
            print(f"  Warning — evaluation failed: {e}")
            ev = {f"acc_{c}": np.nan for c in CLASSIFIERS}
            ev.update({f"nll_{c}": np.nan for c in CLASSIFIERS})

        row = {"dataset": dataset_name, "config_name": config_name}
        row.update(ev)
        row["train_time_sec"] = np.nan   # not available post-hoc

        acc_vals = [ev.get(f"acc_{c}", np.nan) for c in CLASSIFIERS]
        mean_acc = np.nanmean([v for v in acc_vals if not np.isnan(v)]) if acc_vals else np.nan
        print(f"  mean_acc={mean_acc:.4f}  "
              + "  ".join(f"acc_{c}={ev.get(f'acc_{c}', np.nan):.4f}" for c in CLASSIFIERS))
        all_results.append(row)

    df = pd.DataFrame(all_results)
    csv_path = out_dir / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved {len(df)} rows → {csv_path}")
    return df


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Statistical Analysis Framework for FedGANBLR: "
                    "convergence/stability analysis and ablation studies."
    )
    subparsers = parser.add_subparsers(dest="command", help="Analysis mode")

    # --- convergence subcommand ---
    p_conv = subparsers.add_parser(
        "convergence",
        help="Run convergence/stability analysis on existing diagnostics"
    )
    p_conv.add_argument(
        "--diagnostics-dir", type=str, required=True,
        help="Path to diagnostics directory (e.g., diagnostics/adult/fold_01)"
    )
    p_conv.add_argument(
        "--output-dir", type=str,
        default=str(DEFAULT_OUTPUT_DIR / "convergence")
    )

    all_dataset_names = [s["name"] for s in DATASET_SPECS]
    model_choices = list(MODEL_REGISTRY.keys())

    def _add_model_arg(p):
        p.add_argument(
            "--model", choices=model_choices, default="fedganblr",
            help=f"Federated model to use: {model_choices} (default: fedganblr)"
        )

    def _add_training_args(p):
        p.add_argument("--datasets", nargs="+", default=all_dataset_names,
                       help=f"Dataset names (default: all)")
        p.add_argument("--configs", nargs="+", default=None,
                       help="Subset of ablation config names to run (default: all). "
                            "See ABLATION_CONFIGS for available names.")
        p.add_argument("--num-rounds", type=int, default=10)
        p.add_argument("--num-clients", type=int, default=5)
        p.add_argument("--dir-alpha", type=float, default=0.2)
        p.add_argument("--num-folds", type=int, default=1,
                       help="Stratified folds per (dataset, config) for mean ± std "
                            "and a fold-noise significance check (default: 1)")
        _add_model_arg(p)

    # --- ablation subcommand ---
    p_abl = subparsers.add_parser("ablation", help="Run ablation study")
    _add_training_args(p_abl)
    p_abl.add_argument("--output-dir", type=str,
                       default=str(DEFAULT_OUTPUT_DIR / "ablation"))

    # --- reconstruct subcommand ---
    p_rec = subparsers.add_parser(
        "reconstruct",
        help="Rebuild ablation_results.csv from saved diagnostics, then generate charts"
    )
    p_rec.add_argument("--diagnostics-dir", type=str, required=True,
                       help="Root diagnostics directory")
    p_rec.add_argument("--output-dir", type=str,
                       default=str(DEFAULT_OUTPUT_DIR / "ablation"))
    p_rec.add_argument("--eval-syn-frac", type=float, default=1.0)

    # --- charts subcommand ---
    p_charts = subparsers.add_parser(
        "charts",
        help="Generate charts from an existing ablation_results.csv (no retraining)"
    )
    p_charts.add_argument("--csv", type=str, required=True,
                          help="Path to ablation_results.csv")
    p_charts.add_argument("--output-dir", type=str,
                          default=str(DEFAULT_OUTPUT_DIR / "charts"))

    # --- full subcommand ---
    p_full = subparsers.add_parser(
        "full",
        help="Run ablation then convergence analysis on generated diagnostics"
    )
    _add_training_args(p_full)
    p_full.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Helpers shared across handlers
    # -----------------------------------------------------------------------
    def _resolve_shared(args_ns):
        _, _, default_shared = MODEL_REGISTRY.get(args_ns.model, MODEL_REGISTRY["fedganblr"])
        shared = dict(default_shared)
        shared["num_rounds"]  = args_ns.num_rounds
        shared["num_clients"] = args_ns.num_clients
        shared["dir_alpha"]   = args_ns.dir_alpha
        return shared

    def _resolve_datasets(args_ns):
        selected = [s for s in DATASET_SPECS if s["name"] in args_ns.datasets]
        if not selected:
            print(f"No matching datasets for: {args_ns.datasets}")
            print(f"Available: {[s['name'] for s in DATASET_SPECS]}")
            sys.exit(1)
        return selected

    def _resolve_configs(args_ns):
        """Return the (possibly filtered) ablation-config dict for the chosen model."""
        _, default_configs, _ = MODEL_REGISTRY.get(args_ns.model, MODEL_REGISTRY["fedganblr"])
        requested = getattr(args_ns, "configs", None)
        if not requested:
            return default_configs
        unknown = [c for c in requested if c not in default_configs]
        if unknown:
            print(f"Unknown config name(s): {unknown}")
            print(f"Available: {list(default_configs.keys())}")
            sys.exit(1)
        return {c: default_configs[c] for c in requested}

    # -----------------------------------------------------------------------
    if args.command == "convergence":
        out_dir = Path(args.output_dir)
        run_convergence_analysis(Path(args.diagnostics_dir), out_dir)
        print("\n=== Convergence Analysis Complete ===")

    elif args.command == "reconstruct":
        out_dir = Path(args.output_dir)
        diag_root = Path(args.diagnostics_dir)
        if not diag_root.exists():
            print(f"Diagnostics directory not found: {diag_root}")
            sys.exit(1)
        df = reconstruct_from_diagnostics(diag_root, out_dir,
                                          eval_syn_frac=args.eval_syn_frac)
        if df.empty:
            print("No results to plot.")
            sys.exit(1)
        deltas = compute_ablation_deltas(df)
        plot_ablation_bar_chart(df, out_dir, metric_type="acc")
        plot_ablation_bar_chart(df, out_dir, metric_type="nll")
        plot_ablation_delta_heatmap(deltas, out_dir)
        emit_dataset_average(df, out_dir)
        print(f"\nCharts saved to: {out_dir}")
        print("=== Reconstruct Complete ===")

    elif args.command == "charts":
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"CSV not found: {csv_path}")
            sys.exit(1)
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
        print(f"Datasets: {df['dataset'].unique().tolist()}")
        print(f"Configs:  {df['config_name'].unique().tolist()}")
        deltas = compute_ablation_deltas(df)
        plot_ablation_bar_chart(df, out_dir, metric_type="acc")
        plot_ablation_bar_chart(df, out_dir, metric_type="nll")
        plot_ablation_delta_heatmap(deltas, out_dir)
        emit_dataset_average(df, out_dir)
        print(f"\nCharts saved to: {out_dir}")
        print("=== Charts Complete ===")

    elif args.command == "ablation":
        out_dir = Path(args.output_dir)
        selected = _resolve_datasets(args)
        shared   = _resolve_shared(args)
        cfgs     = _resolve_configs(args)
        df = run_ablation_study(selected, out_dir, configs=cfgs, shared_params=shared,
                                model=args.model, num_folds=args.num_folds)
        deltas = compute_ablation_deltas(df)
        plot_ablation_bar_chart(df, out_dir, metric_type="acc")
        plot_ablation_bar_chart(df, out_dir, metric_type="nll")
        plot_ablation_delta_heatmap(deltas, out_dir)
        emit_dataset_average(df, out_dir)
        print("\n=== Ablation Study Complete ===")

    elif args.command == "full":
        out_dir  = Path(args.output_dir)
        selected = _resolve_datasets(args)
        shared   = _resolve_shared(args)
        ablation_cfgs = _resolve_configs(args)

        # Step 1: ablation
        abl_dir = out_dir / "ablation"
        df = run_ablation_study(selected, abl_dir, configs=ablation_cfgs,
                                shared_params=shared, model=args.model,
                                num_folds=args.num_folds)
        deltas = compute_ablation_deltas(df)
        plot_ablation_bar_chart(df, abl_dir, metric_type="acc")
        plot_ablation_bar_chart(df, abl_dir, metric_type="nll")
        plot_ablation_delta_heatmap(deltas, abl_dir)
        emit_dataset_average(df, abl_dir)

        # Step 2: convergence on each config's diagnostics
        for spec in selected:
            for config_name in ablation_cfgs:
                diag_path = abl_dir / "diagnostics" / spec["name"] / config_name
                if diag_path.exists() and (diag_path / "global_round_stats.csv").exists():
                    conv_out = out_dir / "convergence" / spec["name"] / config_name
                    run_convergence_analysis(diag_path, conv_out)

        print("\n=== Full Analysis Complete ===")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
