from evaluation import compare_real_central_fed_cv_all_datasets
from typing import Any

def main():
    # OpenML datasets with proper names and data_id as fallback  
    DATASETS = [
    dict(name="letter-recognition",           alt=["letter-recognition"],       data_id=59,          target="class", ef_bins=None),
    dict(name="chess",            alt=["chess"],                   data_id=23 ,        target="class", ef_bins=None ),
    dict(name="Covertype",        alt=["covertype"],                data_id=31,         target="class", ef_bins=None),
    dict(name="Satellite",        alt=["Statlog"],                  data_id=146,        target="class", ef_bins=None),
    dict(name="pokerhand",        alt=["pokerhand"],                data_id=158,        target="class", ef_bins=None),
    dict(name="HTRU2",            alt=["htru2"],                    data_id=372,        target="class", ef_bins=None),
    dict(name="nursery",          alt=["nursery"],                  data_id=76,         target="class", ef_bins=None),
    dict(name="car",              alt=["car"],                      data_id=19,         target="class", ef_bins=None),
    dict(name="adult",            alt=["adult"],                    data_id=2,          target="class", ef_bins=None),
    dict(name="magic",            alt=["magic"],                    data_id=159,        target="class", ef_bins=None),
    dict(name="shuttle",          alt=["shuttle"],                  data_id=148,        target="class", ef_bins=None), 
    dict(name="connect-4",        alt=["Connect-4", "connect"],     data_id=26,         target="class", ef_bins=None),
    dict(name="bank-marketing",              alt=["Bank"],          data_id=222,        target="class", ef_bins=None),
    dict(name="census-income-kdd",            alt=["census-income"],     data_id=117,   target="class", ef_bins=None),
    dict(name="spambase",         alt=["spambase"],                 data_id=94,         target="class", ef_bins=None)
    ]


    compare_cfg: dict[str, Any] = {
        "n_splits": 2,
        "n_repeats": 2,
        "random_state": 2025,
        "disc_strategy_use": None,
        "ef_bins_default": 12,
        "k_global": 2,
        "epochs_ganblr": 10,
        "batch_ganblr": 512,  
        "warmup_ganblr": 1,
        "num_clients": 5,
        "num_rounds": 30,
        "dir_alpha": 0.2,
        "gamma": 0.25,
        "local_epochs": 3,
        "batch_size": 512,  
        "disc_epochs": 1,
        "cpt_mix": 0.6,
        "alpha_dir": 0.01,
        "cap_train": None,
    }
    df_cmp_folds, df_cmp_sum = compare_real_central_fed_cv_all_datasets(DATASETS, **compare_cfg)
    
if __name__ == "__main__":
    main()
