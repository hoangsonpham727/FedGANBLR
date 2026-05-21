import flwr as fl
import numpy as np
import json
import tensorflow as tf
from typing import Dict, Any, List

from federated_models.FedGanblr import GANBLRFederatedClient, KDBGANStrategy
from base_models.KDependenceBayesian import _build_kdb_structure

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
            metrics = {
                "cid": self.cid,
                "n": self.n,
                "local_parents_json": json.dumps(parents_filtered)
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
            # Collect per-edge vote counts weighted by client sample size
            edge_votes = {}    # (v, parent) -> total sample weight
            total_n = 0
            n_reporting = 0

            for _, fit_res in results:
                metrics = fit_res.metrics or {}
                if "local_parents_json" not in metrics:
                    continue
                n_local = int(metrics.get("n", fit_res.num_examples))
                n_reporting += 1
                total_n += n_local
                local_parents = json.loads(metrics["local_parents_json"])
                for v_str, parents_list in local_parents.items():
                    v = int(v_str)
                    for p in parents_list:
                        key = (v, int(p))
                        edge_votes[key] = edge_votes.get(key, 0) + n_local

            if n_reporting == 0:
                import warnings
                warnings.warn("FedStruct: No clients reported structure in round 1. "
                              "Falling back to Naive Bayes (Y-only parents).")

            # For each feature, keep only the top-k parents by vote weight
            k = int(self.k)
            y_index = int(self.meta["y_index"])
            card = self.meta["card"]
            V = len(card)
            final_parents = {}
            for v in range(V):
                if v == y_index:
                    continue
                # Gather all candidate parents for this feature with their votes
                candidates = []
                for (fv, p), weight in edge_votes.items():
                    if fv == v:
                        candidates.append((p, weight))
                # Sort by vote weight descending, take top-k
                candidates.sort(key=lambda t: -t[1])
                final_parents[v] = [p for p, _ in candidates[:k]]

            self.set_global_meta(card, final_parents, y_index)

            return fl.common.ndarrays_to_parameters([]), {
                "phase": "structure_aggregated",
                "round": server_round,
                "n_reporting_clients": n_reporting,
                "total_edges_before_cap": len(edge_votes),
            }
        else:
            # Regular parameter aggregation
            return super().aggregate_fit(server_round, results, failures)
