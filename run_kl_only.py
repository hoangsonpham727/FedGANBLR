"""
FedGANBLR — KL-Only Variant
============================
Runs FedGANBLR with only server-side KL-weighted aggregation active.
All client-side regularisation components are disabled:

    gamma    = 0.25   ON  — server aggregation weights client_i by n_i * exp(-gamma * KL_i)
    beta_pow = 0.0    OFF — importance weighting (uniform W in KL loss)
    cpt_mix  = 0.0    OFF — post-training CPT interpolation toward global
    alpha_dir = 0.0   OFF — Dirichlet smoothing after aggregation

Motivation: ablation study showed the kl_only configuration consistently
outperforms the fully-regularised baseline across all classifiers.

Output: kl_only_results.csv  (per-fold)  +  kl_only_summary.csv  (per-dataset mean)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold

from utils import (
    fetch_openml_safely,
    discretize_train_test_no_leak,
    preprocess_covertype_binary_columns,
    _free_locals,
)
from evaluation import run_one_fold_fed_ganblr

# ---------------------------------------------------------------------------
# Datasets  (mirrors main.py)
# ---------------------------------------------------------------------------

DATASETS = [
    dict(name="nursery",           data_id=76,  target="class", ef_bins=None),
    dict(name="chess",             data_id=23,  target="class", ef_bins=None),
    dict(name="car",               data_id=19,  target="class", ef_bins=None),
    dict(name="adult",             data_id=2,   target="class", ef_bins=None),
    dict(name="magic",             data_id=159, target="class", ef_bins=None),
    dict(name="letter-recognition",data_id=59,  target="class", ef_bins=None),
    dict(name="Covertype",         data_id=31,  target="class", ef_bins=None),
    dict(name="Satellite",         data_id=146, target="class", ef_bins=None),
    dict(name="pokerhand",         data_id=158, target="class", ef_bins=None),
    dict(name="HTRU2",             data_id=372, target="class", ef_bins=None),
    dict(name="shuttle",           data_id=148, target="class", ef_bins=None),
    dict(name="connect-4",         data_id=26,  target="class", ef_bins=None),
    dict(name="bank-marketing",    data_id=222, target="class", ef_bins=None),
    dict(name="census-income-kdd", data_id=117, target="class", ef_bins=None),
    dict(name="spambase",          data_id=94,  target="class", ef_bins=None),
]

# ---------------------------------------------------------------------------
# KL-Only configuration  (all client-side components disabled)
# ---------------------------------------------------------------------------

KL_ONLY_CFG = dict(
    # --- architecture ---
    k_global     = 2,
    num_clients  = 5,
    num_rounds   = 30,
    dir_alpha    = 0.2,    # Dirichlet split heterogeneity (data split, not smoothing)
    local_epochs = 3,
    batch_size   = 512,
    disc_epochs  = 1,
    eval_syn_frac= 0.5,
    cap_train    = None,
    # --- components ---
    gamma        = 0.25,   # KL-weighted aggregation  ON
    beta_pow     = 0.0,    # importance weighting      OFF
    cpt_mix      = 0.0,    # CPT mixing toward global  OFF
    alpha_dir    = 0.0,    # Dirichlet smoothing        OFF
)

# ---------------------------------------------------------------------------
# CV settings
# ---------------------------------------------------------------------------

N_SPLITS     = 2
N_REPEATS    = 2
RANDOM_STATE = 2025
EF_BINS      = 12

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_kl_only():
    results_list = []

    for ds in DATASETS:
        name = ds["name"]
        print(f"\n{'='*55}")
        print(f"Dataset: {name}")
        print(f"{'='*55}")

        # --- fetch ---
        try:
            X, y = fetch_openml_safely(
                name=name, data_id=ds["data_id"], target=ds["target"]
            )
        except Exception as e:
            print(f"  Skipping — fetch failed: {e}")
            continue

        y = y.astype("category").cat.codes

        cv = RepeatedStratifiedKFold(
            n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
        )

        for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
            print(f"\n  Fold {fold_idx + 1}/{N_SPLITS * N_REPEATS}")

            Xtr_df, Xte_df = X.iloc[tr_idx], X.iloc[te_idx]
            ytr_sr, yte_sr = y.iloc[tr_idx], y.iloc[te_idx]

            if name.lower() == "covertype":
                Xtr_df = preprocess_covertype_binary_columns(Xtr_df)
                Xte_df = preprocess_covertype_binary_columns(Xte_df)

            ef_bins_use = ds.get("ef_bins") or EF_BINS

            try:
                Xtr_int, Xte_int, ytr_int, yte_int, card_feat, classes = (
                    discretize_train_test_no_leak(
                        Xtr_df, ytr_sr, Xte_df, yte_sr,
                        strategy="ef", ef_bins=ef_bins_use,
                    )
                )
                num_classes = len(classes)
            except Exception as e:
                print(f"    Discretization failed: {e}")
                continue
            finally:
                _free_locals(Xtr_df, Xte_df, ytr_sr, yte_sr)

            try:
                res = run_one_fold_fed_ganblr(
                    Xtr_int=Xtr_int, ytr_int=ytr_int,
                    Xte_int=Xte_int, yte_int=yte_int,
                    card_feat=card_feat, num_classes=num_classes,
                    ray_local_mode=True,
                    **KL_ONLY_CFG,
                )
            except Exception as e:
                print(f"    Training failed: {e}")
                continue

            row = {
                "dataset":  name,
                "fold":     fold_idx + 1,
                "n_train":  len(Xtr_int),
                "acc_lr":   res.get("acc_lr",  np.nan),
                "nll_lr":   res.get("nll_lr",  np.nan),
                "acc_mlp":  res.get("acc_mlp", np.nan),
                "nll_mlp":  res.get("nll_mlp", np.nan),
                "acc_rf":   res.get("acc_rf",  np.nan),
                "nll_rf":   res.get("nll_rf",  np.nan),
                "acc_xgb":  res.get("acc_xgb", np.nan),
                "nll_xgb":  res.get("nll_xgb", np.nan),
                "sim_time": res.get("train_time_sec", np.nan),
            }

            print(
                f"    LR  Acc={row['acc_lr']:.3f}  NLL={row['nll_lr']:.3f} | "
                f"MLP Acc={row['acc_mlp']:.3f}  NLL={row['nll_mlp']:.3f} | "
                f"RF  Acc={row['acc_rf']:.3f}  NLL={row['nll_rf']:.3f} | "
                f"XGB Acc={row['acc_xgb']:.3f}  NLL={row['nll_xgb']:.3f} | "
                f"t={row['sim_time']:.1f}s"
            )
            results_list.append(row)

    # --- save ---
    df = pd.DataFrame(results_list)

    fold_path = Path("kl_only_results.csv")
    df.to_csv(fold_path, index=False)
    print(f"\nPer-fold results saved → {fold_path}")

    metric_cols = [c for c in df.columns if c.startswith(("acc_", "nll_"))]
    summary = df.groupby("dataset")[metric_cols + ["sim_time"]].mean().round(4)
    summary_path = Path("kl_only_summary.csv")
    summary.to_csv(summary_path)
    print(f"Per-dataset summary saved → {summary_path}")
    print()
    print(summary.to_string())


if __name__ == "__main__":
    run_kl_only()
