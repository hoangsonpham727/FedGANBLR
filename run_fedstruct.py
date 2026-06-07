import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold

# Import original environment APIs
from evaluation import fetch_openml_safely, discretize_train_test_no_leak, _free_locals, preprocess_covertype_binary_columns

# Import New FedStruct Simulation Runner
from evaluation_fedstruct import run_one_fold_fed_ganblr_fedstruct

def run_experiment_5_datasets():
    # 1. Define 5 datasets of interest
    datasets = [
        {"name":"nursery",             "data_id":76,   "target":"class", "ef_bins":None},
        {"name": "chess",              "data_id": 23,  "target": "class", "ef_bins": None},
        {"name": "car",                "data_id": 19,  "target": "class", "ef_bins": None},
        {"name": "adult",              "data_id": 2,   "target": "class", "ef_bins": None},
        #{"name": "magic",              "data_id": 159, "target": "class", "ef_bins": None},
    ]
    
    # 2. Configure Evaluation Params (similar to main.py compare_cfg)
    n_splits = 2
    n_repeats = 1      # Set to 1 for faster testing
    random_state = 2025
    ef_bins_default = 12
    
    # Configure federated parameters
    fed_cfg = {
        "k_global": 2,
        "num_clients": 5,
        "num_rounds": 10,       # Will become 11 underneath since Round 1 is Structure
        "local_epochs": 3,
        "batch_size": 512,
        "disc_epochs": 1,
        "gamma": 0.25,
        "cpt_mix": 0.6,
        "dir_alpha": 0.2,       # Non-IID Dirichlet distribution alpha
        "eval_syn_frac": 0.5,
        "cap_train": None
    }
    
    results_list = []

    for ds in datasets:
        name = ds["name"]
        print(f"\n{'='*50}\nStarting Dataset: {name}\n{'='*50}")

        # Fetch Data
        try:
            X, y = fetch_openml_safely(name=name, data_id=ds["data_id"], target=ds["target"])
        except Exception as e:
            print(f"Skipping {name}: {e}")
            continue
            
        # Optional: drop NaNs here if needed
        # Convert y to codes if it's categorical
        y = y.astype("category").cat.codes
        
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        
        for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
            print(f"  --- Fold {fold_idx+1} ---")
            Xtr_df, Xte_df = X.iloc[tr_idx], X.iloc[te_idx]
            ytr_sr, yte_sr = y.iloc[tr_idx], y.iloc[te_idx]
            
            # Formatting specifics
            if name.lower() == "covertype":
                Xtr_df = preprocess_covertype_binary_columns(Xtr_df)
                Xte_df = preprocess_covertype_binary_columns(Xte_df)
            
            ef_bins_use = ds.get("ef_bins") or ef_bins_default
            
            # Non-leaky discretization (returns integers)
            try:
                Xtr_int, Xte_int, ytr_int, yte_int, card_feat, classes = discretize_train_test_no_leak(
                    Xtr_df, ytr_sr, Xte_df, yte_sr, strategy="ef", ef_bins=ef_bins_use
                )
                num_classes = len(classes)
            except Exception as e:
                print(f"Discretization failed for {name}: {e}")
                continue

            _free_locals(Xtr_df, Xte_df, ytr_sr, yte_sr)

            # Execution logic matching your requirements
            print(f"      Running FedStruct (Structure Phase -> Parameters Phase)...")
            res_fed = run_one_fold_fed_ganblr_fedstruct(
                Xtr_int=Xtr_int, ytr_int=ytr_int, 
                Xte_int=Xte_int, yte_int=yte_int, 
                card_feat=card_feat, num_classes=num_classes,
                ray_local_mode=True,
                **fed_cfg
            )
            
            # Store summary row
            res_row = {
                "dataset": name,
                "fold": fold_idx + 1,
                "n_train": len(Xtr_int),
                "acc_lr": res_fed.get("acc_lr", np.nan),
                "nll_lr": res_fed.get("nll_lr", np.nan),
                "acc_mlp": res_fed.get("acc_mlp", np.nan),
                "nll_mlp": res_fed.get("nll_mlp", np.nan),
                "acc_rf": res_fed.get("acc_rf", np.nan),
                "nll_rf": res_fed.get("nll_rf", np.nan),
                "acc_xgb": res_fed.get("acc_xgb", np.nan),
                "nll_xgb": res_fed.get("nll_xgb", np.nan),
                "sim_time": res_fed.get("train_time_sec", np.nan)
            }
            
            print(  f"Result: LR(Acc={res_row['acc_lr']:.3f}, NLL={res_row['nll_lr']:.3f}) | "
                    f"MLP(Acc={res_row['acc_mlp']:.3f}, NLL={res_row['nll_mlp']:.3f}) | "
                    f"RF(Acc={res_row['acc_rf']:.3f}, NLL={res_row['nll_rf']:.3f}) | "
                    f"XGB(Acc={res_row['acc_xgb']:.3f}, NLL={res_row['nll_xgb']:.3f}) | "
                    f"Time={res_row['sim_time']:.1f}s")
            
            results_list.append(res_row)

    # Summarize Results globally
    df_results = pd.DataFrame(results_list)
    out_file = Path("fedstruct_5_datasets_summary.csv")
    df_results.to_csv(out_file, index=False)
    print(f"\nAll Done! Saved summary mapping to -> {out_file}")
    
    # Calculate means dynamically for all acc/nll columns alongside sim_time
    metric_cols = [c for c in df_results.columns if c.startswith(("acc_", "nll_"))] + ["sim_time"]
    print(df_results.groupby("dataset")[metric_cols].mean())

if __name__ == "__main__":
    run_experiment_5_datasets()