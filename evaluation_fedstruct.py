import numpy as np
import pandas as pd
import flwr as fl
import time
from pathlib import Path
import json

from evaluation import build_full_table, dirichlet_split, _evaluate_synthetic_classifiers, _gm_to_payload
from federated_models.FedGanblr import build_global_kdb_from_gm
from federated_models.FedGanblr_FedStruct import KDBGANStrategy_FedStruct, GANBLRFederatedClient_FedStruct

def run_one_fold_fed_ganblr_fedstruct(
    Xtr_int, ytr_int, Xte_int, yte_int, card_feat, num_classes,
    k_global=2, num_clients=5, num_rounds=5, dir_alpha=0.1,
    gamma=0.6, local_epochs=3, batch_size=1024, disc_epochs=1,
    cpt_mix=0.25, beta_pow=0.5, kl_lambda=0.5,
    use_theta_weights=True, alpha_mix=0.5, tau_floor=1e-6,
    cap_train=None, clf="lr", verbose=False,
    eval_syn_frac: float = 1.0,
    split_seed: int = 42,
    ray_local_mode: bool = True,
    diagnostics_dir: str | Path | None = None
):
    if cap_train is not None and len(Xtr_int) > cap_train:
        sel = np.random.default_rng(1).choice(len(Xtr_int), size=cap_train, replace=False)
        Xtr_int = Xtr_int[sel]; ytr_int = ytr_int[sel]

    card_all = [int(num_classes)] + [int(c) for c in card_feat]
    y_index = 0
    train_arr, test_arr = build_full_table(ytr_int, yte_int, Xtr_int, Xte_int)

    # Note: We do NOT use derive_global_meta now. 
    # Structure is learned during Round 1.
    # Deterministic client partition (paired across configs for a given split_seed).
    clients_data = dirichlet_split(Xtr_int, ytr_int, num_clients=num_clients, alpha=dir_alpha,
                                   rng=np.random.default_rng(int(split_seed)))

    diag_dir = Path(diagnostics_dir) if diagnostics_dir is not None else None
    if diag_dir:
        diag_dir.mkdir(parents=True, exist_ok=True)
        # We can dump initial stats similar to regular evaluation

    strategy = KDBGANStrategy_FedStruct(
        k=k_global,
        gamma=gamma,
        local_epochs=local_epochs,
        batch_size=batch_size,
        disc_epochs=disc_epochs,
        cpt_mix=cpt_mix,
        beta_pow=beta_pow,
        kl_lambda=kl_lambda,
        use_theta_weights=use_theta_weights,
        alpha_mix=alpha_mix,
        tau_floor=tau_floor,
        adversarial=True,
        nll_csv_path=None,
    )
    # Seed the strategy with bare minimum (card and y_index). `parents` starts empty.
    strategy.set_global_meta(card=card_all, parents={}, y_index=y_index)

    def client_fn(cid: str):
        i = int(cid)
        Xc, yc = clients_data[i]
        return GANBLRFederatedClient_FedStruct(cid, Xc, yc)

    import ray
    if not ray.is_initialized():
        try:
            ray.init(local_mode=ray_local_mode, ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")
        except Exception:
            pass

    t0 = time.time()
    # Add 1 to num_rounds for the initial structure learning round
    total_rounds = num_rounds + 1
    
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(clients_data),
        config=fl.server.ServerConfig(num_rounds=total_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    sim_time = time.time() - t0

    gm = strategy.global_model
    if gm is None:
        return dict(acc_lr=np.nan, nll_lr=np.nan, train_time_sec=sim_time, weights=None, s_y=None)

    gm["py"] = np.asarray(gm["py"], dtype=np.float32)
    gm["thetas"] = {v: np.asarray(t, dtype=np.float32) for v, t in gm["thetas"].items()}

    # Use the parents learned during Round 1, now preserved in strategy.meta["parents"]
    parents_glob_raw = strategy.meta["parents"]
    # Unpack parents map to exclude Y for building model
    parents_glob = {int(k): [p for p in v if p != y_index] for k, v in parents_glob_raw.items()}

    gen_global = build_global_kdb_from_gm(gm, card_all, parents_glob, y_index)

    n_full = len(Xtr_int)
    n_syn = max(1, int(n_full * float(eval_syn_frac)))
    syn_full = gen_global.sample(n=n_syn, rng=np.random.default_rng(0), order=None, return_y=True)
    X_syn, y_syn = syn_full[:, 1:], syn_full[:, 0]

    ev = _evaluate_synthetic_classifiers(X_syn, y_syn, Xte_int, yte_int)

    out = {"train_time_sec": sim_time}
    out.update(ev)

    try:
        out["fed_global_model"] = _gm_to_payload(gm, card_all, parents_glob, y_index)
    except Exception:
        out["fed_global_model"] = gm

    if ray_local_mode:
        try:
            ray.shutdown()
        except:
            pass

    return out
