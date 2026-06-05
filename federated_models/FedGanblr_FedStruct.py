import flwr as fl
import numpy as np
import json
import tensorflow as tf
from typing import Dict, Any, List

from federated_models.FedGanblr import GANBLRFederatedClient, KDBGANStrategy, build_global_kdb_from_gm
from base_models.KDependenceBayesian import _build_kdb_structure, MLE_KDB

class GANBLRFederatedClient_FedStruct(GANBLRFederatedClient):
    def fit(self, parameters, config):
        config_local = config or {}
        phase = config_local.get("phase", "parameters")
        
        if phase == "structure":
            # Structure Learning Phase
            meta_json = config_local.get("meta_json", "{}")
            meta = json.loads(str(meta_json))
            card = list(meta["card"])
            y_index = int(meta["y_index"])
            k = int(meta.get("k", 2))
            
            # Use local data for structure learning
            train_arr = np.column_stack([self.y, self.X]).astype(np.int32)
            parents, _ = _build_kdb_structure(train_arr, card, y_index, k=k)

            # Filter out the y_index entry (always empty, ignored by set_global_meta)
            parents_filtered = {v: ps for v, ps in parents.items() if v != y_index}

            # Fit MLE model on local data so server can generate synthetic data
            mle = MLE_KDB(card, y_index, parents_filtered, alpha=1.0)
            mle.fit(train_arr)

            metrics = {
                "cid": self.cid,
                "n": self.n,
                "local_parents_json": json.dumps(parents_filtered),
                "mle_py_json": json.dumps(mle.py.tolist()),
                "mle_thetas_json": json.dumps(
                    {str(v): t.tolist() for v, t in mle.theta.items()}
                ),
            }
            return [np.array([0.0], dtype=np.float32)], self.n, metrics
        else:
            # Traditional Parameters / Weights Training Phase
            return super().fit(parameters, config)

class KDBGANStrategy_FedStruct(KDBGANStrategy):
    def configure_fit(self, server_round, parameters, client_manager):
        # Store current round safely
        self.round = int(server_round)
        
        if self.round == 1:
            # Round 1 is strictly for Structure Learning
            cfg = {
                "phase": "structure",
                "meta_json": json.dumps({
                    "k": self.k,
                    "card": self.meta["card"],
                    "y_index": self.meta["y_index"]
                })
            }
            
            if client_manager is None:
                return []
                
            try:
                num_avail = client_manager.num_available()
                if callable(num_avail): num_avail = num_avail()
                # Sample all clients for structure consensus
                sampled = client_manager.sample(num_clients=int(num_avail))
            except Exception:
                sampled = client_manager.sample()
                
            # Normalize to list
            clients = []
            if isinstance(sampled, tuple):
                clients = list(sampled[0]) if len(sampled) >= 1 and sampled[0] is not None else []
            elif isinstance(sampled, list):
                clients = list(sampled)
            else:
                clients = [sampled]
                
            fit_instructions = []
            for c in clients:
                fit_instructions.append((c, fl.common.FitIns(parameters=parameters, config=cfg)))
            return fit_instructions
        else:
            # Rounds > 1 are for Parameter Training
            # Call parent configure_fit, then inject our phase flag
            fit_instructions = super().configure_fit(server_round, parameters, client_manager)
            for _, fit_ins in fit_instructions:
                fit_ins.config["phase"] = "parameters"
            return fit_instructions

    def aggregate_fit(self, server_round, results, failures):
        server_round = int(server_round)
        if server_round == 1:
            # FedStruct round 1 — structure aggregation.
            # Each client sent its locally-learned MLE model (structure + params).
            # The server generates synthetic data from every client's model, pools
            # it, and learns a single global structure on the combined synthetic data.
            k = int(self.k)
            y_index = int(self.meta["y_index"])
            card = self.meta["card"]
            V = len(card)

            syn_arrays = []
            n_reporting = 0
            for _, fit_res in results:
                metrics = fit_res.metrics or {}
                if "mle_py_json" not in metrics or "mle_thetas_json" not in metrics:
                    continue
                n_local = int(metrics.get("n", fit_res.num_examples))
                n_reporting += 1
                client_parents = {
                    int(v): list(ps)
                    for v, ps in json.loads(metrics["local_parents_json"]).items()
                }
                py = np.asarray(json.loads(metrics["mle_py_json"]), dtype=np.float64)
                thetas = {
                    int(v): np.asarray(t, dtype=np.float64)
                    for v, t in json.loads(metrics["mle_thetas_json"]).items()
                }
                gm = {"py": py, "thetas": thetas}
                mle_model = build_global_kdb_from_gm(gm, card, client_parents, y_index)
                syn_data = mle_model.sample(
                    n=n_local,
                    rng=np.random.default_rng(n_local),
                    order=None,
                    return_y=True,
                )
                syn_arrays.append(syn_data)

            if syn_arrays:
                combined_syn = np.vstack(syn_arrays)
                global_parents, _ = _build_kdb_structure(combined_syn, card, y_index, k)
                final_parents = {
                    v: ps for v, ps in global_parents.items() if v != y_index
                }
            else:
                # No client reported a model: fall back to Naive Bayes (Y-only parents).
                import warnings
                warnings.warn("FedStruct: No clients reported a model in round 1. "
                              "Falling back to Naive Bayes (Y-only parents).")
                final_parents = {v: [] for v in range(V) if v != y_index}

            self.set_global_meta(card, final_parents, y_index)

            return fl.common.ndarrays_to_parameters([]), {
                "phase": "structure_aggregated",
                "round": server_round,
                "n_reporting_clients": n_reporting,
            }
        else:
            # Regular parameter aggregation
            return super().aggregate_fit(server_round, results, failures)
