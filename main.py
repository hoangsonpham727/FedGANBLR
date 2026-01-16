from evaluation import compare_real_central_fed_cv_all_datasets
from typing import Any

def main():
    DATASETS = [
    #     #OpenML datasets with proper names and data_id as fallback
    #     
    #    dict(name="letter", alt=["letter-recognition"], data_id=6, target="class", ef_bins=None),
    #    dict(name="chess", alt=[" chess"], data_id=23 , target="class", ef_bins=None ),
    #    dict(name="covertype", alt=["Covertype"], data_id=31, target="class", ef_bins=None),
    #    dict(name="Satellite", alt=["Statlog"], data_id=146, target="class", ef_bins=None),
    #    dict(name="pokerhand", alt=["Pokerhand"], data_id=158, target="class", ef_bins=None)
    ]

    # DATASETS = [
    #     dict(name="nursery",          alt=["nursery"],     data_id=76,           target="class", ef_bins=None),
    #     dict(name="car",              alt=["car"],       data_id=19,             target="class", ef_bins=None),
    #     dict(name="adult",            alt=["adult"],     data_id=2,             target="class", ef_bins=10),
    #     dict(name="magic",            alt=["magic"],     data_id=159,             target="class", ef_bins=12),
    #     dict(name="shuttle",          alt=["shuttle"],   data_id=148,            target="class", ef_bins=12), 
    #     dict(name="connect-4",        alt=["Connect-4", "connect"], data_id=26, target="class", ef_bins=None),
    #     dict(name="bank-marketing",              alt=["Bank"],       data_id=222,             target="class", ef_bins=None),
    #     dict(name="census-income-kdd",            alt=["census-income"],     data_id=117,             target="class", ef_bins=None),
    #     dict(name="spambase",          alt=["spambase"],   data_id=94,            target="class", ef_bins=None)
    # ]

    compare_cfg: dict[str, Any] = {
        "n_splits": 2,
        "n_repeats": 2,
        "random_state": 2025,
        "disc_strategy_use": None,
        "ef_bins_default": 12,
        "k_global": 2,
        "epochs_ganblr": 10,
        "batch_ganblr": 64,  
        "warmup_ganblr": 1,
        "num_clients": 5,
        "num_rounds": 30,
        "dir_alpha": 0.2,
        "gamma": 0.6,
        "local_epochs": 3,
        "batch_size": 64,  
        "disc_epochs": 1,
        "cpt_mix": 0.75,
        "alpha_dir": 1e-3,
        "cap_train": None
    }
    compare_real_central_fed_cv_all_datasets(DATASETS, **compare_cfg)
    
if __name__ == "__main__":
    main()
