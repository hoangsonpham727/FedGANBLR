from __future__ import annotations
import flwr as fl
import numpy as np
import json
import tensorflow as tf
from typing import Dict, Any, List, Tuple, Optional
from base_models.KDependenceBayesian import MLE_KDB, DiscBN_KDB, _compute_strides, _build_kdb_structure
from base_models.Ganblr import GANBLR
from utils import _ohe_pipeline, LogisticRegression, _safe_log_loss
import warnings
import math
import csv
import os


def _local_cpt_counts(data: np.ndarray, card: list[int], y_index: int, parents_full: dict[int, list[int]]) -> dict[int, np.ndarray]:
    """
    Return empirical counts per feature row [num_cfg, r_v] using parents_full lists (Y first).
    Vectorized implementation: uses numpy operations and np.add.at instead of Python per-row loops.
    """
    V = len(card)
    out: dict[int, np.ndarray] = {}
    N = int(data.shape[0])
    for v in range(V):
        if v == y_index:
            continue
        pa = parents_full.get(v, [y_index])
        pa_cards = [int(card[p]) for p in pa]
        strides = _compute_strides(pa_cards)
        num_cfg = int(np.prod(pa_cards)) if pa_cards else 1
        rv = int(card[v])
        counts = np.zeros((num_cfg, rv), dtype=np.float64)
        if len(pa) == 0:
            vals_v = data[:, v].astype(np.int64)
            # all samples belong to single parent config (idx=0)
            np.add.at(counts, (np.zeros(N, dtype=np.int64), vals_v), 1.0)
        else:
            vals = data[:, pa].astype(np.int64)  # shape [N, len(pa)]
            if strides.size == 0:
                idx = np.zeros(N, dtype=np.int64)
            else:
                idx = np.sum(vals * strides[np.newaxis, :].astype(np.int64), axis=1).astype(np.int64)
            vals_v = data[:, v].astype(np.int64)
            np.add.at(counts, (idx, vals_v), 1.0)
        out[v] = counts
    return out

""" 
Compute client importance weights using marginal distributions mu_glob and mu_loc 

"""
def _mixture_kl_weights(counts_loc: dict[int, np.ndarray],
                        thetas_glob: dict[int, np.ndarray],
                        alpha_mix: float = 0.5,
                        beta: float = 0.5,
                        tau: float = 1e-6) -> dict[int, np.ndarray]:
    """Compute per-row normalized weights s[v] for KL weighting.
    - counts_loc: dict[v] -> counts array [num_cfg, r_v]
    - thetas_glob: dict[v] -> global theta array [num_cfg, r_v] (same shapes)
    Uses Laplace smoothing and vectorized normalization.
    We use a mixture of local and global marginal distributions to compute the importance weights
    """
    s: dict[int, np.ndarray] = {}
    a = float(alpha_mix)
    b = float(beta)
    t = float(tau)
    for v, cnt in counts_loc.items():
        cnt = np.asarray(cnt, dtype=np.float64)
        rv = cnt.shape[1]
        denom = cnt.sum(axis=1, keepdims=True) + rv
        p_loc = (cnt + 1.0) / np.maximum(denom, 1e-12)
        p_glb = thetas_glob.get(v, None)
        if p_glb is None or np.asarray(p_glb).shape != p_loc.shape:
            mix = p_loc
        else:
            mix = a * np.asarray(p_glb, dtype=np.float64) + (1.0 - a) * p_loc
        if b == 0.0:
            W = np.ones_like(mix, dtype=np.float64)
        else:
            W = np.power(np.maximum(mix, t), -b)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W = W / row_sums
        s[v] = W.astype(np.float32)
    return s


""" 
Compute client importance weights using models' parameters (probs)
"""
def _disc_local_thetas_from_logits(gen: 'DiscBN_KDB') -> dict[int, np.ndarray]:
    """
    Return local CPTs theta_loc[v] in probability space from a constrained DiscBN_KDB.
    """
    theta_loc: dict[int, np.ndarray] = {}
    V = len(gen.card)
    for v in range(V):
        if v == gen.y_index:
            continue
        if gen.logits_v is not None and v in gen.logits_v:
            theta_loc[v] = np.asarray(tf.nn.softmax(gen.logits_v[v], axis=-1), dtype=np.float64)
    return theta_loc

def _mixture_kl_weights_with_theta(counts_loc: dict[int, np.ndarray],
                        thetas_glob: dict[int, np.ndarray],
                        theta_loc: dict[int, np.ndarray],
                        alpha_mix: float = 0.5,
                        beta: float = 0.5,
                        tau: float = 1e-6) -> dict[int, np.ndarray]:
    """
    Compute per-row normalized weights s[v] for KL weighting using model parameters.
    
    If theta_loc is provided, uses mixture of local vs global θ parameters directly.
    Otherwise falls back to the original counts-based approach for backward compatibility.
    
    Args:
        counts_loc: dict[v] -> counts array [num_cfg, r_v] (used only for keys if theta_loc provided)
        thetas_glob: dict[v] -> global theta array [num_cfg, r_v] 
        theta_loc: dict[v] -> local theta array [num_cfg, r_v] (if None, uses counts_loc)
        alpha_mix: mixture weight (alpha_mix * global + (1-alpha_mix) * local)
        beta: importance weighting exponent
        tau: numerical floor
    
    Returns:
        s: dict[int, np.ndarray] with row-normalized importance weights
    """
    s: dict[int, np.ndarray] = {}
    a = float(alpha_mix)
    b = float(beta)
    t = float(tau)
    
    for v in counts_loc.keys():
        if theta_loc is not None:
            # New parameter-based approach
            th_loc = theta_loc.get(v, None)
            th_glb = thetas_glob.get(v, None)
            
            if th_loc is None and th_glb is None:
                continue
                
            # Ensure arrays and normalize row-wise
            def _norm_theta(th: np.ndarray) -> np.ndarray:
                th = np.asarray(th, dtype=np.float64)
                th = np.clip(th, 1e-12, None)
                th = th / np.maximum(th.sum(axis=1, keepdims=True), 1e-12)
                return th
            
            if th_loc is not None:
                th_loc = _norm_theta(th_loc)
            if th_glb is not None:
                th_glb = _norm_theta(th_glb)
                
            # Shape compatibility check
            if th_loc is None:
                mix = th_glb
            elif th_glb is None or th_glb.shape != th_loc.shape:
                mix = th_loc
            else:
                mix = a * th_glb + (1.0 - a) * th_loc
                mix = _norm_theta(mix)
        else:
            # Original counts-based approach (backward compatibility)
            cnt = np.asarray(counts_loc[v], dtype=np.float64)
            rv = cnt.shape[1]
            denom = cnt.sum(axis=1, keepdims=True) + rv
            p_loc = (cnt + 1.0) / np.maximum(denom, 1e-12)
            p_glb = thetas_glob.get(v, None)
            if p_glb is None or np.asarray(p_glb).shape != p_loc.shape:
                mix = p_loc
            else:
                mix = a * np.asarray(p_glb, dtype=np.float64) + (1.0 - a) * p_loc
        
        # Compute importance weights
        if b == 0.0:
            W = np.ones_like(mix, dtype=np.float64)
        else:
            W = np.power(np.maximum(mix, t), -b)
            
        # Row-normalize weights
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W = W / row_sums
        s[v] = W.astype(np.float32)
    
    return s



"""
Federated-GANBLR 
Key points:
- Server selects global K (kdb parents) using a bootstrap client (round 0 init).
- Metadata broadcast: {k, card, parents (list of parent lists per feature index), y_index}.
- Each client:
    * Receives metadata (if first round, builds its own if absent).
    * Adversarially trains local GANBLR on its integer-encoded (X_int, y_int).
    * Extracts generator parameters: class prior py and per-feature CPT theta[v].
    * Sends:
        - py (as probability vector)
        - For each feature v != y_index: theta[v] flattened
        - Per-feature shape info & parent list (in metadata JSON once)
        - Local sample count n
        - Local label marginal (for KL weighting)
        - Synthetic-vs-real marginal KL (diagnostic)
- Server aggregates:
    * Reconstructs per-client py/theta blocks
    * Computes reference (previous global) distribution
    * KL_i over (py + concatenated theta rows) vs reference
    * Weight w_i = n_i * exp(-gamma * KL_i) (gamma same semantics as FedGANBLR)
    * Weighted average in probability space -> new global py / theta.
    * Broadcasts updated global model next round.
"""



def serialize_kdb_generator(gen: 'MLE_KDB') -> Dict[str, Any]:
    """
    Extract generator parameters into JSON-safe dict.
    Returns:
        {
          'py': list,
          'thetas': { str(v): theta_ndarray.tolist() },
          'shapes': { str(v): [num_cfg, r_v] },
          'parents': { str(v): parent_list_with_Y_first },  # from gen.parent_lists
          'card': gen.card,
          'y_index': gen.y_index
        }
    """
    out = {
        "py": gen.py.tolist(),
        "thetas": {},
        "shapes": {},
        "parents": {},
        "card": list(gen.card),
        "y_index": int(gen.y_index),
    }
    for v, theta in gen.theta.items():
        out["thetas"][str(v)] = theta.tolist()
        out["shapes"][str(v)] = list(theta.shape)
        out["parents"][str(v)] = list(gen.parent_lists[v])  # includes Y first
    return out

def deserialize_kdb_generator(payload: Dict[str, Any]) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """Return (py, theta_dict) as numpy arrays."""
    py = np.asarray(payload["py"], dtype=np.float64)
    thetas: Dict[int, np.ndarray] = {}
    for k, arr in payload["thetas"].items():
        thetas[int(k)] = np.asarray(arr, dtype=np.float64)
    return py, thetas

def flatten_payload(py: np.ndarray, thetas: Dict[int, np.ndarray], y_index: int, V: int) -> List[np.ndarray]:
    """
    Convert py + theta dict into ordered list of ndarrays:
      [py, theta(v1), theta(v2), ...] in feature index order (skip Y).
    """
    seq = [py.astype(np.float64)]
    for v in range(V):
        if v == y_index:
            continue
        seq.append(thetas[v].astype(np.float64))
    return seq

def unflatten_payload(arrs: List[np.ndarray], y_index: int, card: List[int]) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    py = arrs[0]
    idx = 1
    thetas = {}
    V = len(card)
    for v in range(V):
        if v == y_index:
            continue
        thetas[v] = arrs[idx]
        idx += 1
    return py, thetas


def _concatenate_model_vector(py: np.ndarray, thetas: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Flatten py and all theta rows to a single 1D probability vector (for KL).
    (Concatenate row-wise probabilities.)
    """
    parts = [py.ravel()]
    for v in sorted(thetas.keys()):
        parts.append(thetas[v].ravel())
    vec = np.concatenate(parts, axis=0)
    # Guard: ensure sums of CPT rows individually sum to 1; for KL weighting we just normalize total vec.
    vec = np.clip(vec, 1e-12, None)
    return vec / vec.sum()

def _concatenate_model_vector_weighted(py: np.ndarray,
                                       thetas: Dict[int, np.ndarray],
                                       parents_full: Dict[int, List[int]],
                                       y_index: int,
                                       card: List[int],
                                       s_y: np.ndarray| None) -> np.ndarray:
    """
    Build a single 1D vector like _concatenate_model_vector, but scale segments
    by class weights s_y before normalization:
      - py segment weights: s_y[c]
      - For each theta_v: each row corresponds to a parent config with Y first;
        all probabilities in rows for class c are weighted by s_y[c].
    """
    V = len(card)
    parts = []
    wparts = []

    # P(Y)
    py_seg = np.clip(py.ravel(), 1e-12, None)
    parts.append(py_seg)
    wparts.append(np.asarray(s_y, dtype=np.float64).ravel())

    C = int(card[y_index])

    # CPTs
    for v in range(V):
        if v == y_index:
            continue
        th = np.asarray(thetas[v], dtype=np.float64)  # [num_cfg, r_v]
        th = np.clip(th, 1e-12, None)
        rv = th.shape[1]
        pa = list(parents_full.get(v, []))  # includes Y first
        if not pa or pa[0] != y_index:
            # Fallback: if schema malformed, treat equally weighted
            parts.append(th.ravel())
            wparts.append(np.ones_like(th.ravel(), dtype=np.float64))
            continue
        # stride for Y as first parent
        s0 = int(np.prod([card[p] for p in pa[1:]], dtype=np.int64)) if len(pa) > 1 else 1
        # Expected rows = C * s0
        num_rows = th.shape[0]
        # Build per-row weights by repeating s_y[c] for each configuration of non-Y parents
        row_weights = np.repeat(np.asarray(s_y, dtype=np.float64), repeats=s0)
        # If mismatch due to shape edge cases, pad/truncate safely
        if row_weights.shape[0] != num_rows:
            if row_weights.shape[0] < num_rows:
                row_weights = np.pad(row_weights, (0, num_rows - row_weights.shape[0]), mode="edge")
            else:
                row_weights = row_weights[:num_rows]
        # Expand weights to each value in row (rv columns)
        w_flat = np.repeat(row_weights, repeats=rv)
        parts.append(th.ravel())
        wparts.append(w_flat)

    vec = np.concatenate(parts, axis=0)
    wts = np.concatenate(wparts, axis=0)
    vecw = np.clip(vec * wts, 1e-12, None)
    return vecw / vecw.sum()

def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))

def aggregate_models(client_models: List[Dict[str, Any]],
                    gamma: float,
                    prev_global: Dict[str, Any] | None,
                    card: List[int],
                    parents_full: Dict[int, List[int]],
                    y_index: int,
                    s_y: np.ndarray | None = None) -> Dict[str, Any]:
    """
    Weighted aggregation in probability space using s_y-weighted KL-based weights.
    client_models[i] contains: 'py', 'thetas', 'n'
    s_y: class weights computed from client label marginals; if None -> uniform.
    """
    V = len(card)
    C = int(card[y_index])
    if s_y is None:
        s_y = np.ones(C, dtype=np.float64) / max(1, C)
    else:
        s_y = np.asarray(s_y, dtype=np.float64)
        s_y = np.clip(s_y, 1e-12, None); s_y = s_y / s_y.sum()

    # Reference vector
    if prev_global:
        ref_vec = _concatenate_model_vector_weighted(prev_global['py'], prev_global['thetas'],
                                                     parents_full, y_index, card, s_y)
    else:
        # Average of clients (unweighted) to seed reference
        first_v = _concatenate_model_vector_weighted(client_models[0]['py'], client_models[0]['thetas'],
                                                     parents_full, y_index, card, s_y)
        accum = np.zeros_like(first_v, dtype=np.float64)
        for m in client_models:
            v = _concatenate_model_vector_weighted(m['py'], m['thetas'],
                                                   parents_full, y_index, card, s_y)
            accum += v
        ref_vec = accum / float(len(client_models))

    weights_raw, kls = [], []
    for cm in client_models:
        kl_val = float(cm.get("kl_summary", np.nan))
        if not np.isfinite(kl_val):
            v = _concatenate_model_vector_weighted(cm['py'], cm['thetas'], parents_full, y_index, card, s_y)
            kl_val = _kl(v, ref_vec)
        kls.append(kl_val)
        weights_raw.append(cm['n'] * math.exp(-gamma * kl_val))
    weights_raw = np.asarray(weights_raw, dtype=np.float64)
    weights = (np.ones_like(weights_raw) / len(weights_raw)) if weights_raw.sum() == 0 else (weights_raw / weights_raw.sum())

    # Aggregate py
    py_stack = np.stack([m['py'] for m in client_models], axis=0)
    py_global = np.sum(weights[:, None] * py_stack, axis=0)
    py_global = np.clip(py_global, 1e-12, None); py_global /= py_global.sum()

    # Aggregate thetas per feature
    V = len(card)
    thetas_global = {}
    for v in range(V):
        if v == y_index:
            continue
        t_stack = np.stack([m['thetas'][v] for m in client_models], axis=0)  # [K, num_cfg, r_v]
        tg = np.sum(weights[:, None, None] * t_stack, axis=0)
        tg = np.clip(tg, 1e-12, None)
        tg /= tg.sum(axis=1, keepdims=True)
        thetas_global[v] = tg

    return {
        "py": py_global,
        "thetas": thetas_global,
        "weights": weights,
        "client_kls": kls,
        "s_y": s_y,
    }


def _validate_train_arr(train_arr: np.ndarray, card: List[int], client_id: str = "unknown", verbose: bool = False) -> np.ndarray:
    """
    Validate and sanitize train_arr to ensure all values are within valid ranges.
    
    Args:
        train_arr: Training array with shape [N, V] where V includes Y as first column
        card: Cardinalities for each variable (Y first, then features)
        client_id: Client ID for logging
        verbose: Whether to print warnings
    
    Returns:
        Validated and sanitized train_arr
    """
    train_arr_valid = train_arr.copy().astype(np.int32)
    for v_idx in range(train_arr_valid.shape[1]):
        if v_idx < len(card):
            max_val = int(card[v_idx]) - 1
            if max_val < 0:
                continue
            # Clip invalid values
            invalid_mask = (train_arr_valid[:, v_idx] < 0) | (train_arr_valid[:, v_idx] > max_val)
            if invalid_mask.any():
                if verbose:
                    warnings.warn(f"[Client {client_id}] train_arr column {v_idx} has {invalid_mask.sum()} invalid values (range: [0, {max_val}]); clipping")
                train_arr_valid[:, v_idx] = np.clip(train_arr_valid[:, v_idx], 0, max_val)
    return train_arr_valid


class GANBLRFederatedClient(fl.client.NumPyClient):
    """
    Federated client for Fed-GANBLR. Uses a constrained, trainable
    DiscBN_KDB generator and KL
    regularization toward server CPTs.
    """
    def __init__(self, cid: str, X_int: np.ndarray, y_int: np.ndarray):
        self.cid = cid
        self.X = X_int.astype(np.int32)
        self.y = y_int.astype(np.int32)
        self.meta = None
        self.local_model = None
        self.n = int(self.X.shape[0])
        self.last_payload = None

    def get_parameters(self, config):
        return []

    def set_parameters(self, parameters):
        return

    def fit(self, parameters, config):
        config_local: Dict[str, Any] = config or {}
        meta_json = config_local.get("meta_json", None)
        if meta_json is None:
            raise RuntimeError("Missing meta_json in server config.")
        
        self.meta = json.loads(str(meta_json))
        self.k = int(self.meta.get("k", 2))

        def _to_int(x, default):
            try:
                return int(x)
            except Exception:
                try:
                    return int(float(x))
                except Exception:
                    return int(default)

        local_epochs = _to_int(config.get("local_epochs", 3), 3)
        batch_size = _to_int(config.get("batch_size", 1024), 1024)
        disc_epochs = _to_int(config.get("disc_epochs", 1), 1)
        adversarial_flag = str(config.get("adversarial", "1")).lower() in ("1", "true", "yes")
        verbose = str(config.get("verbose", "0")).lower() in ("1", "true", "yes")
        try:
            cpt_mix = float(config.get("cpt_mix", 0.0))
        except Exception:
            cpt_mix = 0.0
        try:
            kl_lambda = float(config.get("kl_lambda", 0.0))
        except Exception:
            kl_lambda = 0.0
        try:
            alpha_mix = float(config.get("alpha_mix", 0.5))
        except Exception:
            alpha_mix = 0.5
        try:
            beta_pow = float(config.get("beta_pow", 0.5))
        except Exception:
            beta_pow = 0.0
        try:
            tau_floor = float(config.get("tau_floor", 1e-6))
        except Exception:
            tau_floor = 1e-6

        # Parse optional global model to warm-start / form KL targets
        py_glob, thetas_glob = None, None
        global_json = config.get("global_json", None)
        if global_json is not None:
            try:
                if isinstance(global_json, (str, bytes)):
                    g = json.loads(global_json)
                elif isinstance(global_json, dict):
                    g = global_json
                else:
                    g = json.loads(str(global_json))
                py_val = g.get("py", None) if isinstance(g, dict) else None
                py_glob = np.asarray(py_val, dtype=np.float64) if py_val is not None else None
                thetas_raw = (g.get("thetas", {}) if isinstance(g, dict) else {}) or {}
                thetas_glob = {int(v): np.asarray(t, dtype=np.float64) for v, t in thetas_raw.items()}
            except Exception:
                py_glob, thetas_glob = None, None

        s_y = None
        # s_y_json may be a JSON string or already a sequence/array — handle both safely
        s_y_json = config.get("s_y_json", None)
        if s_y_json is not None:
            try:
                if isinstance(s_y_json, (str, bytes)):
                    s_y_list = json.loads(s_y_json)
                elif isinstance(s_y_json, (list, tuple, np.ndarray)):
                    s_y_list = s_y_json
                elif isinstance(s_y_json, dict):
                    # permit dict -> take values (best-effort)
                    s_y_list = list(s_y_json.values())
                else:
                    s_y_list = json.loads(str(s_y_json))
                s_y = np.asarray(s_y_list, dtype=np.float64)
                s_y = np.clip(s_y, 1e-12, None)
                s_y = s_y / (s_y.sum() + 1e-12)
            except Exception:
                s_y = None

        # Server schema
        card = list(self.meta["card"])
        y_index = int(self.meta["y_index"])
        parents_full = {int(k_): list(v_) for k_, v_ in self.meta["parents"].items()}
        V = len(card)
        parents_excl_y = {v: [p for p in parents_full.get(v, []) if p != y_index] for v in range(V) if v != y_index}

        # Build training table (Y first)
        train_arr = np.column_stack([self.y, self.X]).astype(np.int32)
        N = self.X.shape[0]

        # Sanitize client-side integer arrays to match server card 
        for col in range(self.X.shape[1]):
            # card has Y first, then features at indices 1..V-1
            card_idx = col + 1
            if card_idx < len(card):
                max_allowed = int(card[card_idx])
                if max_allowed <= 0:
                    continue
                # any values >= max_allowed are invalid; clip to max_allowed-1
                mask = self.X[:, col] >= max_allowed
                if mask.any():
                    warnings.warn(f"Client {self.cid}: feature col {col} contains values >= card[{card_idx}]={max_allowed}; clipping to {max_allowed-1}")
                    self.X[mask, col] = max_allowed - 1

        # sanitize labels too
        max_y = int(card[y_index]) if y_index < len(card) else None
        if max_y is not None:
            masky = self.y >= max_y
            if masky.any():
                warnings.warn(f"Client {self.cid}: label values >= |Y|={max_y}; clipping to {max_y-1}")
                self.y[masky] = max_y - 1

        # Rebuild train_arr after sanitization
        train_arr = np.column_stack([self.y, self.X]).astype(np.int32)
        
        # Final validation: ensure all values in train_arr are within valid ranges
        train_arr = _validate_train_arr(train_arr, card, client_id=self.cid, verbose=verbose)

        # Parse global parameters if available (received from server)
        global_json = config.get("global_json", None)
        py_glob, thetas_glob = None, None
        if global_json is not None:
            try:
                if isinstance(global_json, (str, bytes)):
                    g = json.loads(global_json)
                elif isinstance(global_json, dict):
                    g = global_json
                else:
                    g = json.loads(str(global_json))
                py_val = g.get("py", None) if isinstance(g, dict) else None
                py_glob = np.asarray(py_val, dtype=np.float64) if py_val is not None else None
                thetas_raw = (g.get("thetas", {}) if isinstance(g, dict) else {}) or {}
                thetas_glob = {int(v): np.asarray(t, dtype=np.float64) for v, t in thetas_raw.items()}
            except Exception:
                py_glob, thetas_glob = None, None
        
        gen = DiscBN_KDB(card=card, y_index=y_index, parents=parents_excl_y, constrained=True,
                            seed=int(self.cid) if self.cid.isdigit() else 12345, federated=True)
        
        if py_glob is not None and thetas_glob:
            # Initialize from global model (federated learning: start from aggregated model)
            print(f"[Client {self.cid}] Initializing from global model parameters")
            gen.init_from_params(py_glob, thetas_glob, jitter_std=0.0)
        else:
            # Initialize from MLE for first round (no global model available)
            print(f"[Client {self.cid}] Initializing from MLE (first round)")
            mle = MLE_KDB(card, y_index, parents_excl_y, alpha=1.0)
            mle.fit(train_arr)
            gen.init_from_mle(mle, jitter_std=0.0)

        # If server provided global CPTs, prepare KL targets 
        kl_targets = None
        kl_s_weights = None

        # #COMPUTE WEIGHTS USING EMPIRICAL COUNTS
        # if thetas_glob:
        #     kl_targets = {int(v): np.asarray(t, dtype=np.float32) for v, t in thetas_glob.items()}
        #     # local counts per CPT row (parents with Y first, from server meta)
        #     counts_loc = _local_cpt_counts(train_arr, card, y_index, parents_full)
        #     kl_s_weights = _mixture_kl_weights(counts_loc, thetas_glob, alpha_mix=alpha_mix, beta=beta_pow, tau=tau_floor)

        #COMPUTE WEIGHTS USING MODELS PARAMETERS
        if thetas_glob:
            kl_targets = {int(v): np.asarray(t, dtype=np.float32) for v, t in thetas_glob.items()}
            
            # Extract local CPTs from current generator parameters (logits -> softmax)
            theta_loc = _disc_local_thetas_from_logits(gen)
            
            # Keep counts_loc only for defining which features to include (keys)
            counts_loc = _local_cpt_counts(train_arr, card, y_index, parents_full)
            
            # Compute importance weights based on mixture of global vs local learned parameters
            kl_s_weights = _mixture_kl_weights_with_theta(
                counts_loc=counts_loc,
                thetas_glob=thetas_glob,
                theta_loc=theta_loc,  # Pass the generator's learned parameters
                alpha_mix=alpha_mix,
                beta=beta_pow,
                tau=tau_floor,
            )

        # Use ganblr wrapper state where generator is the constrained DiscBN_KDB
        ganblr = GANBLR(alpha=1.0)
        ganblr.card = card
        ganblr.y_index = y_index
        ganblr.k = self.k
        ganblr.parents = parents_excl_y
        ganblr._X = self.X
        ganblr._y = self.y
        ganblr._train_table = train_arr
        ganblr.generator = gen

        # ---- Generator-only federated local training ----
        if (not adversarial_flag) or disc_epochs <= 0 or local_epochs <= 0 or N <= 0:
            if local_epochs > 0 and N > 0:
                if verbose:
                    print(f"[Client {self.cid}] Starting local training: {local_epochs} epochs")
                ganblr.generator.fit(
                    train_arr,
                    epochs=int(local_epochs),
                    batch_size=min(batch_size, train_arr.shape[0]),
                    l2=1e-5,
                    verbose=1 if verbose else 0,
                    kl_targets=({int(v): np.asarray(t, dtype=np.float32) for v, t in (thetas_glob or {}).items()} if thetas_glob else None),
                    kl_lambda=kl_lambda,
                    s_y=s_y,
                    kl_s_weights=kl_s_weights,
                )
                if verbose:
                    print(f"[Client {self.cid}] Local training completed")
            # serialize payload (softmax CPTs) and return
            py_soft = np.asarray(tf.nn.softmax(ganblr.generator.logits_y)).ravel().tolist()
            thetas_soft = {}
            V = len(card)
            for v in range(V):
                if v == y_index: continue
                if ganblr.generator.logits_v is not None: 
                    thetas_soft[str(v)] = np.asarray(tf.nn.softmax(ganblr.generator.logits_v[v], axis=-1)).tolist()
            payload = {"py": py_soft, "thetas": thetas_soft, "card": list(card), "y_index": y_index}
            py = np.asarray(payload["py"])
            thetas = {int(v): np.asarray(arr) for v, arr in payload["thetas"].items()}
            
            # Compute NLL on training data
            nll_value = None
            try:
                # Validate train_arr before computing NLL (in case it was modified during training)
                train_arr_valid = _validate_train_arr(train_arr, card, client_id=self.cid, verbose=verbose)
                proba = ganblr.generator.predict_proba(train_arr_valid)
                nll_value = _safe_log_loss(self.y, proba, card[y_index])
            except Exception as e:
                if verbose:
                    print(f"[Client {self.cid}] Warning: Could not compute NLL: {e}")
                    import traceback
                    traceback.print_exc()
                nll_value = float('nan')
            
            metrics = {
                "cid": self.cid,
                "n": self.n,
                "entropy": float(np.sum((_concatenate_model_vector(py, thetas)) * np.log(_concatenate_model_vector(py, thetas))) * -1.0),
                "py_json": json.dumps(payload["py"]),
                "thetas_json": json.dumps(payload["thetas"]),
                "label_marg_json": json.dumps((np.bincount(self.y, minlength=card[y_index]).astype(np.float64) / max(1.0, len(self.y))).tolist()),
            }
            if nll_value is not None:
                metrics["nll"] = float(nll_value)
            return [np.array([0.0], dtype=np.float32)], self.n, metrics

        # Note: Discriminator will be created fresh each epoch in _run_adversarial_training
        # No need to initialize here - the shared method handles it
        
        # Seed function for federated learning (client-specific)
        def _cid_seed(offset: int = 0) -> int:
            return (hash(self.cid) & 0xFFFFFFFF) + int(offset)
        
        rng_local = np.random.default_rng(12345)
        
        # Run adversarial training using shared method
        ganblr._run_adversarial_training(
            X=self.X,
            y=self.y,
            epochs=local_epochs,
            batch_size=batch_size,
            disc_epochs=disc_epochs,
            rng=np.random.default_rng(_cid_seed(1)),
            rng_local=rng_local,
            verbose=1 if verbose else 0,
            initial_disc_train=True,  # Federated learning does initial discriminator training
            kl_targets=kl_targets,
            kl_lambda=kl_lambda,
            s_y=s_y,
            kl_s_weights=kl_s_weights,
            client_id=self.cid,
            seed_fn=_cid_seed
        )
    
        # Store discriminator reference for potential future use
        self.disc = ganblr.disc

        # cpt_mix toward global (unchanged)
        if cpt_mix > 0.0 and py_glob is not None and thetas_glob:
            try:
                py_local = np.asarray(tf.nn.softmax(ganblr.generator.logits_y))
            except Exception:
                py_local = getattr(ganblr.generator, "py", None)
            if py_local is not None:
                py_new = (1.0 - cpt_mix) * py_local + cpt_mix * py_glob
                py_new = np.clip(py_new, 1e-12, None); py_new = py_new / py_new.sum()
                try:
                    ganblr.generator.logits_y.assign(tf.math.log(tf.add(tf.constant(py_new, dtype=tf.float32), 1e-12)))
                except Exception:
                    try:
                        if hasattr(ganblr.generator, "py"):
                            setattr(ganblr.generator, "py", py_new)
                        else:
                            setattr(ganblr.generator, "_py", py_new)
                    except Exception:
                        # give up silently if the generator doesn't accept direct assignment
                        pass
            for v in range(V):
                if v == y_index: continue
                tg = thetas_glob.get(v, None)
                if tg is None:
                    continue
                try:
                    if ganblr.generator.logits_v is not None and v in ganblr.generator.logits_v :
                        t_new = (1.0 - cpt_mix) * np.asarray(tf.nn.softmax(ganblr.generator.logits_v[v], axis=-1)) + cpt_mix * tg
                        t_new = np.clip(t_new, 1e-12, None)
                        t_new = t_new / t_new.sum(axis=1, keepdims=True)
                        ganblr.generator.logits_v[v].assign(tf.math.log(tf.add(tf.constant(t_new, dtype=tf.float32), 1e-12)))
                except Exception:
                    pass

        # Serialize constrained generator to payload (softmax CPTs)
        py_soft = np.asarray(tf.nn.softmax(ganblr.generator.logits_y)).ravel().tolist()
        thetas_soft = {}
        for v in range(V):
            if v == y_index: continue
            if ganblr.generator.logits_v is not None: 
                thetas_soft[str(v)] = np.asarray(tf.nn.softmax(ganblr.generator.logits_v[v], axis=-1)).tolist()

        payload = {"py": py_soft, "thetas": thetas_soft, "shapes": {}, "parents": {}, "card": list(card), "y_index": y_index}
        py = np.asarray(payload["py"])
        thetas = {int(v): np.asarray(arr) for v, arr in payload["thetas"].items()}

        # Optional client-side KL summary vs provided global
        kl_summary = None
        try:
            if "global_json" in config and py_glob is not None and thetas_glob:
                vec_loc = _concatenate_model_vector_weighted(py, thetas, parents_full, y_index, card, s_y if s_y is not None else np.ones(card[y_index])/card[y_index])
                vec_glb = _concatenate_model_vector_weighted(np.asarray(py_glob), thetas_glob, parents_full, y_index, card, s_y if s_y is not None else np.ones(card[y_index])/card[y_index])
                kl_summary = float(_kl(vec_loc, vec_glb))
        except Exception:
            kl_summary = None

        # Compute NLL on training data
        nll_value = None
        try:
            # Validate train_arr before computing NLL (in case it was modified during training)
            train_arr_valid = _validate_train_arr(train_arr, card, client_id=self.cid, verbose=verbose)
            proba = ganblr.generator.predict_proba(train_arr_valid)
            nll_value = _safe_log_loss(self.y, proba, card[y_index])
        except Exception as e:
            if verbose:
                print(f"[Client {self.cid}] Warning: Could not compute NLL: {e}")
                import traceback
                traceback.print_exc()
            nll_value = float('nan')

        metrics = {
            "cid": self.cid,
            "n": self.n,
            "entropy": float(np.sum((_concatenate_model_vector(py, thetas)) * np.log(_concatenate_model_vector(py, thetas))) * -1.0),
            "py_json": json.dumps(payload["py"]),
            "thetas_json": json.dumps(payload["thetas"]),
            "label_marg_json": json.dumps((np.bincount(self.y, minlength=card[y_index]).astype(np.float64) / max(1.0, len(self.y))).tolist()),
        }
        if kl_summary is not None:
            metrics["kl_summary"] = kl_summary
        if nll_value is not None:
            metrics["nll"] = float(nll_value)

        self.last_payload = payload
        return [np.array([0.0], dtype=np.float32)], self.n, metrics
    
    def evaluate(self, parameters, config):
        return float(0.0), self.n, {}
    
# ---------- Server Strategy ----------
class KDBGANStrategy(fl.server.strategy.FedAvg):
    """
    Server strategy performing KL-weighted aggregation (logic mirror of FedGANBLR):
      - Round 1: bootstrap global structure from first available client meta (card, parents, etc.)
      - Each round:
          * Parse client payload json (py, thetas)
          * KL-weighted average (py + each theta row) in probability space
          * Broadcast updated global model via meta_json (no direct parameter tensors)
    """
    def __init__(self, k: int, gamma: float = 0.6, local_epochs: int = 5, batch_size: int = 1024,
                 disc_epochs: int = 1, cpt_mix: float = 0.25, alpha_dir: float = 1e-3, adversarial: bool = False, 
                alpha_mix: float = 0.5, beta_pow: float = 0.0, tau_floor: float = 1e-6, nll_csv_path: str = "nll_convergence.csv"):
        super().__init__()
        self.k = int(k)
        self.gamma = float(gamma)
        self.local_epochs = int(local_epochs)
        self.batch_size = int(batch_size)
        self.disc_epochs = int(disc_epochs)
        self.cpt_mix = float(cpt_mix)
        self.alpha_dir = float(alpha_dir)
        self.adversarial = bool(adversarial)
        self.global_model = None
        self.meta = None
        self.round = 0
        self.alpha_mix = float(alpha_mix)
        self.beta_pow = float(beta_pow)
        self.tau_floor = float(tau_floor)
        self.nll_csv_path = str(nll_csv_path)
        # # Initialize CSV file with headers if it doesn't exist
        # if not os.path.exists(self.nll_csv_path):
        #     with open(self.nll_csv_path, 'w', newline='') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(['round', 'client_id', 'nll'])
        

    def initialize_parameters(self, client_manager):
        # No tensor parameters; we rely on config broadcast
        return fl.common.ndarrays_to_parameters([])

    def configure_fit(self, *args, **kwargs):
        """
        Backwards/forwards-compatible configure_fit for multiple Flower versions.

        Accepts positional signatures like (rnd, parameters, client_manager) or keyword-only
        ('server_round'/'rnd', 'parameters', 'client_manager'). Robust to different
        client_manager.sample return shapes. Critically, returns (ClientProxy, FitIns),
        not a raw (client, dict).
        """
        # Parse args
        if len(args) >= 3:
            rnd, parameters, client_manager = args[0], args[1], args[2]
        else:
            rnd = kwargs.get("server_round", kwargs.get("rnd", getattr(self, "round", 0)))
            parameters = kwargs.get("parameters", None)
            client_manager = kwargs.get("client_manager", None)

        # Track round safely
        try:
            self.round = int(rnd)
        except Exception:
            self.round = getattr(self, "round", 0)

        # Build config to send
        if self.meta is None:
            raise RuntimeError("Server meta not initialized. Call set_global_meta(...) before simulation.")
        cfg = {
            "meta_json": json.dumps(self.meta),
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "disc_epochs": self.disc_epochs,
            "cpt_mix": self.cpt_mix,
            "adversarial": 1 if self.adversarial else 0,
            "alpha_mix": self.alpha_mix,    
            "beta_pow": self.beta_pow,      
            "tau_floor": self.tau_floor,
            "verbose": 1,  # Enable verbose output for epoch printing
        }
        # Include current global model so clients warm-start instead of refitting from scratch
        if self.global_model is not None:
            py = self.global_model["py"].astype(float).tolist()
            thetas = {int(v): t.astype(float).tolist() for v, t in self.global_model["thetas"].items()}
            cfg["global_json"] = json.dumps({"py": py, "thetas": thetas})
            s_y = self.global_model.get("s_y", None)
            if s_y is not None:
                cfg["s_y_json"] = json.dumps(list(map(float, s_y)))
        # Normalize current Parameters
        try:
            cur_params = parameters if parameters is not None else fl.common.ndarrays_to_parameters([])
        except Exception:
            cur_params = fl.common.ndarrays_to_parameters([])

        if client_manager is None:
            return []

        # Sample clients robustly
        sampled = None
        try:
            # Prefer sampling all available clients for this simple strategy
            num_avail = None
            if hasattr(client_manager, "num_available"):
                try:
                    na = client_manager.num_available
                    num_avail = na() if callable(na) else int(na)
                except Exception:
                    num_avail = None
            if num_avail:
                sampled = client_manager.sample(num_clients=num_avail)
            else:
                sampled = client_manager.sample()
        except TypeError:
            try:
                sampled = client_manager.sample()
            except Exception:
                sampled = None
        except Exception:
            sampled = None

        # Normalize to list of client proxies
        clients = []
        if sampled is None:
            if hasattr(client_manager, "clients"):
                try:
                    clients = list(getattr(client_manager, "clients"))
                except Exception:
                    clients = []
        else:
            # Handle list, tuple, single
            if isinstance(sampled, tuple):
                clients = list(sampled[0]) if len(sampled) >= 1 and sampled[0] is not None else []
            elif isinstance(sampled, list):
                clients = list(sampled)
            else:
                clients = [sampled]

        # Build FitIns per client
        fit_instructions = []
        for c in clients:
            try:
                ins = fl.common.FitIns(parameters=cur_params, config=cfg)
                fit_instructions.append((c, ins))
            except Exception:
                # Best-effort: if c is packed inside a tuple
                try:
                    ins = fl.common.FitIns(parameters=cur_params, config=cfg)
                    fit_instructions.append((c[0], ins))
                except Exception:
                    continue

        return fit_instructions

    def aggregate_fit(self, rnd, results, failures):
        # Ensure meta has been initialized before attempting to aggregate
        if self.meta is None:
            # No meta available; cannot aggregate client models
            return fl.common.ndarrays_to_parameters([]), {}

        client_models = []
        # Derive expected shapes from global meta (enforced schema)
        try:
            card = list(self.meta["card"])
            y_index = int(self.meta["y_index"])
            parents_full = {int(k_): list(v_) for k_, v_ in self.meta["parents"].items()}  # includes Y first
            expected = {
                v: (int(np.prod([card[p] for p in parents_full[v]])), card[v])
                for v in range(len(card)) if v != y_index
            }
            exp_py_len = card[y_index]
        except Exception:
            expected, exp_py_len = None, None
            parents_full, card, y_index = {}, [], 0

        skipped = 0
        label_margs = []
        ns = []
        nll_data = []  # Store NLL values for CSV logging
        for _, fit_res in results:
            m = fit_res.metrics or {}
            try:
                py = np.asarray(json.loads(m["py_json"]), dtype=np.float64)
                thetas_dict_raw = json.loads(m["thetas_json"])
                thetas = {int(v): np.asarray(arr, dtype=np.float64) for v, arr in thetas_dict_raw.items()}
                lab = json.loads(m.get("label_marg_json", "null"))
                lab = np.asarray(lab, dtype=np.float64) if lab is not None else None
                n_local = int(m.get("n", fit_res.num_examples))
                client_id = m.get("cid", "unknown")

                # Collect NLL value if available
                nll_value = m.get("nll", None)
                if nll_value is not None:
                    try:
                        nll_float = float(nll_value)
                        nll_data.append((rnd, client_id, nll_float))
                    except (ValueError, TypeError):
                        pass

                # Validate payload against expected shapes
                if expected is not None:
                    if py.shape[0] != exp_py_len:
                        skipped += 1
                        continue
                    ok = True
                    for v, (rows, cols) in expected.items():
                        th = thetas.get(v, None)
                        if th is None or th.ndim != 2 or th.shape != (rows, cols):
                            ok = False
                            break
                    if not ok:
                        skipped += 1
                        continue

                client_models.append({
                    "py": py,
                    "thetas": thetas,
                    "n": n_local,
                    "kl_summary":float(m["kl_summary"]) if "kl_summary" in m else np.nan
                })
                if lab is not None and lab.size == exp_py_len:
                    label_margs.append(lab * max(1, n_local))
                    ns.append(max(1, n_local))
            except Exception:
                skipped += 1
                continue

        if not client_models:
            return fl.common.ndarrays_to_parameters([]), {}

        # Compute s_y from client label marginals (n-weighted); fallback to uniform
        if label_margs and sum(ns) > 0:
            s_y = np.sum(np.stack(label_margs, axis=0), axis=0)
            s_y = np.clip(s_y, 1e-12, None); s_y = s_y / s_y.sum()
        else:
            C = card[y_index]
            s_y = np.ones(C, dtype=np.float64) / max(1, C)

        # KL-weighted aggregation using s_y
        self.global_model = aggregate_models(client_models, self.gamma, self.global_model,
                                             card=card, parents_full=parents_full, y_index=y_index, s_y=s_y)
        # === Dirichlet smoothing per CPT row ===
        try:
            ad = max(0.0, float(self.alpha_dir))
        except Exception:
            ad = 1e-3
        for v, M in list(self.global_model["thetas"].items()):
            B = np.array(M, dtype=np.float64)
            if B.size == 0:
                continue
            B = np.clip(B + ad, 1e-12, None)
            B = B / B.sum(axis=1, keepdims=True)
            self.global_model["thetas"][v] = B
        py = np.clip(self.global_model["py"] + ad, 1e-12, None)
        self.global_model["py"] = py / py.sum()

        # Attach s_y for inspection
        self.global_model["s_y"] = s_y
        
        # Write NLL values to CSV file
        if nll_data:
            try:
                with open(self.nll_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for round_num, client_id, nll_val in nll_data:
                        writer.writerow([round_num, client_id, nll_val])
            except Exception as e:
                print(f"[Server] Warning: Could not write NLL to CSV: {e}")
    
        return fl.common.ndarrays_to_parameters([]), {
            "avg_kl": float(np.mean(self.global_model["client_kls"])),
            "round": rnd,
            "skipped_clients": skipped,
            "s_y": json.dumps(self.global_model["s_y"].tolist()),
        }

    def set_global_meta(self, card: List[int], parents: Dict[int, List[int]], y_index: int):
        """
        Must be called before start_simulation.
        parents: mapping v -> list of feature parents (EXCLUDING Y). We prepend Y when reconstructing CPT lists.
        """
        parents_full = {}
        V = len(card)
        for v in range(V):
            if v == y_index:
                continue
            parents_full[str(v)] = [y_index] + list(parents.get(v, []))
        self.meta = {
            "k": self.k,
            "card": list(card),
            "parents": parents_full,
            "y_index": int(y_index)
        }

# ---------- Helper to derive global structure (run once centrally) ----------
def derive_global_meta(train_arr: np.ndarray, card: List[int], y_index: int, k: int) -> Tuple[List[int], Dict[int, List[int]], int]:
    """
    Use build_kdb_structure (already defined in notebook) on a centralized bootstrap dataset
    to define parents. Returns (card, parents, y_index).
    """
    parents, _ = _build_kdb_structure(train_arr, card, y_index, k=k)
    return card, parents, y_index

def build_global_kdb_from_gm(gm, card, parents, y_index):
    """Rebuild an MLE_KDB from aggregated global parameters (no re-fitting)."""
    gen = MLE_KDB(card, y_index, parents, alpha=1.0)
    # Ensure py sums to exactly 1.0 (numerical precision issues from float32 conversion)
    py = np.asarray(gm["py"], dtype=np.float64)
    py = np.clip(py, 1e-12, None)
    gen.py = py / py.sum()
    # Ensure each theta row sums to exactly 1.0
    gen.theta = {}
    for v, t in gm["thetas"].items():
        theta_v = np.asarray(t, dtype=np.float64)
        theta_v = np.clip(theta_v, 1e-12, None)
        theta_v = theta_v / theta_v.sum(axis=1, keepdims=True)
        gen.theta[int(v)] = theta_v
    gen.parent_lists = {}
    gen.strides = {}
    V = len(card)
    for v in range(V):
        if v == y_index:
            continue
        pa = [y_index] + parents.get(v, [])
        gen.parent_lists[v] = pa
        gen.strides[v] = _compute_strides([card[p] for p in pa])
    return gen