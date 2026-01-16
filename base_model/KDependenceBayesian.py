from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
import keras
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


def mutual_information_xy(Xv: np.ndarray, Y: np.ndarray, r_v: int, r_y: int) -> float:
    # I(X;Y) = sum_{x,y} p(x,y) log p(x,y)/(p(x)p(y))
    N = Xv.shape[0]
    joint = np.zeros((r_v, r_y), dtype=np.float64)
    for i in range(N):
        joint[Xv[i], Y[i]] += 1
    joint /= N
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = joint / (px * py + 1e-15)
        val = np.nansum(joint * np.log(ratio + 1e-15))
    return float(val)

def conditional_mi_given_y(Xv: np.ndarray, Xw: np.ndarray, Y: np.ndarray, r_v: int, r_w: int, r_y: int) -> float:
    # I(Xv;Xw | Y) = sum_y p(y) I(Xv;Xw | Y=y)
    N = Xv.shape[0]
    cmi = 0.0
    for y in range(r_y):
        mask = (Y == y)
        Ny = int(mask.sum())
        if Ny == 0: continue
        xv = Xv[mask]; xw = Xw[mask]
        joint = np.zeros((r_v, r_w), dtype=np.float64)
        for i in range(Ny):
            joint[xv[i], xw[i]] += 1
        joint /= Ny
        px = joint.sum(axis=1, keepdims=True)
        pw = joint.sum(axis=0, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = joint / (px * pw + 1e-15)
            val = np.nansum(joint * np.log(ratio + 1e-15))
        cmi += (Ny / N) * val
    return float(cmi)

def _build_kdb_structure(data: np.ndarray, card: List[int], y_index: int, k: int = 2, n_threads: Optional[int] = None) -> tuple:
    """
    Return (parents, order): for each feature v != y_index, choose up to k parents among other features via KDB.
    Procedure: order features by I(X;Y), then for each feature, add up to k parents from earlier features with largest I(X;X_parent|Y).
    
    Args:
        data: Input data array
        card: Cardinality list for each variable
        y_index: Index of the class variable
        k: Maximum number of parents per feature
        n_threads: Number of threads to use for parallel computation. If None, uses os.cpu_count()
    """
    if n_threads is None:
        n_threads = os.cpu_count() or 1
    
    V = len(card); C = card[y_index]
    Y = data[:, y_index]
    feats = [v for v in range(V) if v != y_index]
    
    # Order by MI(X;Y) - parallelized
    def compute_mi(v):
        return v, mutual_information_xy(data[:, v], Y, card[v], C)
    
    mi = {}
    if len(feats) > 1 and n_threads > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = {executor.submit(compute_mi, v): v for v in feats}
            for future in as_completed(futures):
                v, mi_val = future.result()
                mi[v] = mi_val
    else:
        # Sequential fallback for small datasets or single thread
        for v in feats:
            mi[v] = mutual_information_xy(data[:, v], Y, card[v], C)
    
    order = sorted(feats, key=lambda v: -mi[v])
    
    # Parents per feature
    parents: Dict[int, List[int]] = {v: [] for v in range(V)}
    
    # Parallelize conditional MI computation for each feature
    def compute_cmi_for_feature(v, cand):
        """Compute conditional MI scores for all candidate parents of feature v"""
        if k <= 0 or len(cand) == 0:
            return v, []
        scores = []
        for w in cand:
            cmi_val = conditional_mi_given_y(data[:, v], data[:, w], Y, card[v], card[w], C)
            scores.append((w, cmi_val))
        scores.sort(key=lambda t: -t[1])
        return v, [w for w, _ in scores[:k]]
    
    # Process features in parallel
    if len(order) > 1 and n_threads > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = {}
            for i, v in enumerate(order):
                cand = order[:i]
                futures[executor.submit(compute_cmi_for_feature, v, cand)] = v
            
            for future in as_completed(futures):
                v, parent_list = future.result()
                parents[v] = parent_list
    else:
        # Sequential fallback
        for i, v in enumerate(order):
            cand = order[:i]
            if k <= 0 or len(cand) == 0:
                parents[v] = []
                continue
            scores = [(w, conditional_mi_given_y(data[:, v], data[:, w], Y, card[v], card[w], C)) for w in cand]
            scores.sort(key=lambda t: -t[1])
            parents[v] = [w for w, _ in scores[:k]]
    
    return parents, order

def _compute_strides(parent_cards: List[int]) -> np.ndarray:
    # e.g., for bases [r0, r1, r2] -> strides [r1*r2, r2, 1]
    if len(parent_cards) == 0: return np.array([], dtype=np.int64)
    strides = np.ones(len(parent_cards), dtype=np.int64)
    for i in range(len(parent_cards)-2, -1, -1):
        strides[i] = strides[i+1] * parent_cards[i+1]
    return strides

def _topo_order_from_parents(parents_excl_y: Dict[int, List[int]], V: int, y_index: int) -> List[int]:
    """
    Return a topological order of feature indices (excluding y_index) given parent lists that exclude Y.
    Falls back to a deterministic order (sorted remaining) if cycles or partial ordering found.
    """
    remaining = set(v for v in range(V) if v != y_index)
    included = {y_index}
    order = []
    while remaining:
        progressed = False
        for v in list(remaining):
            pa = [p for p in parents_excl_y.get(v, []) if p != y_index]
            if set(pa).issubset(included):
                order.append(v)
                included.add(v)
                remaining.remove(v)
                progressed = True
        if not progressed:
            order.extend(sorted(remaining))
            break
    return order

"""
KDB model that maximizes likelihood estimation
"""
@dataclass
class MLE_KDB:
    
    card: List[int]
    y_index: int
    parents: Dict[int, List[int]]
    alpha: float = 1.0  # Laplace smoothing

    def fit(self, data: np.ndarray):
        V = len(self.card); C = self.card[self.y_index]
        y = data[:, self.y_index]
        
        # class prior
        py = np.bincount(y, minlength=C).astype(np.float64) + self.alpha
        self.py = py / py.sum()
        
        # CPTs for each feature
        self.theta: Dict[int, np.ndarray] = {}
        self.strides: Dict[int, np.ndarray] = {}
        self.parent_lists: Dict[int, List[int]] = {}
        
        for v in range(V):
            if v == self.y_index: 
                continue
                
            pa = [self.y_index] + self.parents[v]  # include Y as first parent
            pa_cards = [self.card[p] for p in pa]
            strides = _compute_strides(pa_cards)
            num_cfg = int(np.prod(pa_cards)) if len(pa_cards) > 0 else 1
            rv = self.card[v]
            counts = np.full((num_cfg, rv), self.alpha, dtype=np.float64)
            
            # count
            for i in range(data.shape[0]):
                vals = [data[i, p] for p in pa]
                # flatten idx
                idx = 0
                for s, val in zip(strides, vals):
                    idx += int(s) * int(val)
                counts[idx, data[i, v]] += 1
            
            theta = counts / counts.sum(axis=1, keepdims=True)
            self.theta[v] = theta
            self.strides[v] = strides
            self.parent_lists[v] = pa

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        V = len(self.card); C = self.card[self.y_index]
        B = data.shape[0]
        S = np.log(self.py + 1e-12)[np.newaxis, :].repeat(B, axis=0)  # [B, C]
        
        for v in range(V):
            if v == self.y_index: continue
            rv = self.card[v]
            pa = self.parent_lists[v]     # [Y] + feature parents
            strides = self.strides[v]
            theta_v = self.theta[v]       # [num_cfg, rv]
            x_v = data[:, v]              # [B]
            
            # for each class c, add log theta(row_index(c, parents(x)), x_v)
            for c in range(C):
                # parent values with y=c
                vals = np.column_stack([np.full(B, c, dtype=np.int64)] + [data[:, p] for p in pa[1:]]).astype(np.int64)
                idx = np.sum(vals * strides[np.newaxis, :], axis=1) if len(strides)>0 else np.zeros(B, dtype=np.int64)
                
                # take row idx and column x_v
                contrib = np.log(theta_v[idx, x_v] + 1e-12)  # [B]
                S[:, c] += contrib
        
        # normalize
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S); P = P / P.sum(axis=1, keepdims=True)
        return P
    def sample(self, n: int, rng: np.random.Generator, order: List[int] | None, return_y: bool = True) -> np.ndarray:
        """
        Sample n examples from the MLE KDB model.
        Returns array shape [n, V] with integer-coded variables (Y + features) if return_y True,
        otherwise returns only feature columns (no Y).
        - If `order` is provided it must be an ordering of feature indices (excluding y_index)
          that respects parent dependencies; otherwise a simple topological order is inferred.
        """

        if rng is None:
            rng = np.random.default_rng()

        V = len(self.card)
        #infer topological order 
        if order is None:
            parents_excl_y = {v: [p for p in self.parent_lists.get(v, []) if p != self.y_index] for v in range(V) if v != self.y_index}
            order = _topo_order_from_parents(parents_excl_y, V, self.y_index)

        # allocate
        samples = np.zeros((n, V), dtype=np.int32)
        # sample Y first
        py = self.py
        # Ensure py sums to exactly 1.0 for numpy.random.choice
        py_sum = py.sum()
        if not np.isclose(py_sum, 1.0, atol=1e-6):
            py = np.clip(py, 1e-12, None)
            py = py / py.sum()
        Ys = rng.choice(len(py), size=n, p=py)
        samples[:, self.y_index] = Ys

        # sequentially sample other features following topological order
        for v in order:
            theta_v = self.theta[v]            # [num_cfg, r_v]
            strides = self.strides[v]         # ndarray
            pa = self.parent_lists[v]         # [Y] + parents
            num_cfg = theta_v.shape[0]
            rv = theta_v.shape[1]
            for i in range(n):
                # build parent values for this sample
                vals = [int(samples[i, p]) for p in pa]
                # compute flattened index
                if strides.size == 0:
                    idx = 0
                else:
                    idx = 0
                    for s, val in zip(strides, vals):
                        idx += int(s) * int(val)
                
                probs = np.asarray(theta_v[idx], dtype=np.float64)
                # defensive normalization
                s = probs.sum()
                if (s <= 0) or (not np.isfinite(s)):
                    probs = np.ones(rv, dtype=np.float64) / rv
                else:
                    probs = probs / s
                samples[i, v] = rng.choice(rv, p=probs)
        if return_y:
            return samples
        else:
            return samples[:, [v for v in range(V) if v != self.y_index]]


class DiscBN_KDB:
    """
    Discriminative BN with KDB parents.
    - constrained=True: probabilities via softmax CPTs (logits_y, logits_v[v] of shape [num_cfg_v, r_v]).
    - constrained=False: free weights with interactions (W_v[v] of shape [num_cfg_v, r_v, C]) + bias [C].
    Trains by minimizing conditional NLL (softmax over classes).
    """
    
    def __init__(self, card: List[int], y_index: int, parents: Dict[int, List[int]], constrained: bool = True, federated: bool = False,  seed: int = 123):
        self.card = card; self.y_index = y_index; self.parents = parents; self.constrained = constrained
        tf.random.set_seed(seed)
        
        V = len(card); C = card[y_index]
        self.federated = federated
        
        # per-feature parent config space
        self.pa_lists = {}
        self.pa_cards = {}
        self.strides = {}
        for v in range(V):
            if v == y_index: continue
            pa = [y_index] + parents[v]
            self.pa_lists[v] = pa
            cards = [card[p] for p in pa]
            self.pa_cards[v] = cards
            self.strides[v] = tf.constant(_compute_strides(cards), dtype=tf.int32)
        # parameters
        if constrained:
            # Initialize logits_y with small random values (not zeros) to avoid uniform class prior
            # This ensures meaningful gradient updates from the start
            self.logits_y = tf.Variable(tf.random.normal([C], stddev=0.1, dtype=tf.float32))
            self.logits_v = {}
            for v in range(V):
                if v == y_index: continue
                num_cfg = int(np.prod(self.pa_cards[v])) if len(self.pa_cards[v])>0 else 1
                rv = card[v]
                # Use slightly larger stddev for better initialization
                init = tf.random.normal([num_cfg, rv], stddev=0.1, dtype=tf.float32)
                self.logits_v[v] = tf.Variable(init)
            self.bias = None; self.W = None
        else:
            self.bias = tf.Variable(tf.zeros([C], dtype=tf.float32))
            self.W = {}
            for v in range(V):
                if v == y_index: continue
                num_cfg = int(np.prod(self.pa_cards[v])) if len(self.pa_cards[v])>0 else 1
                rv = card[v]
                init = tf.random.normal([num_cfg, rv, C], stddev=0.01, dtype=tf.float32)
                self.W[v] = tf.Variable(init)
            self.logits_y = None; self.logits_v = None
        self.opt = keras.optimizers.Adam(learning_rate=1e-3)

    def init_from_mle(self, mle: 'MLE_KDB', jitter_std: float = 0.0):
        assert self.constrained, "init_from_mle only for constrained mode"
        C = self.card[self.y_index]
        # guard in case static analysis or unexpected state makes logits_y None
        if self.logits_y is not None:
            self.logits_y.assign(tf.math.log(tf.constant(mle.py, dtype=tf.float32) + 1e-8))
        for v, theta in mle.theta.items():
            log_theta = tf.math.log(tf.constant(theta, dtype=tf.float32) + 1e-8)
            if jitter_std > 0:
                log_theta = log_theta + tf.random.normal(tf.shape(log_theta), stddev=jitter_std)
            # Only assign if logits_v is a dict and contains an actual Variable for this v
            if isinstance(self.logits_v, dict) and v in self.logits_v and self.logits_v[v] is not None:
                self.logits_v[v].assign(log_theta)
    
    def init_from_params(self, py: np.ndarray, thetas: Dict[int, np.ndarray], jitter_std: float = 0.0):
        """
        Initialize generator from global parameters (py and thetas dict).
        Similar to init_from_mle but accepts parameters directly.
        
        Args:
            py: Class prior probabilities [C]
            thetas: Dict mapping feature index v -> CPT array [num_cfg, r_v]
            jitter_std: Optional jitter noise standard deviation
        """
        assert self.constrained, "init_from_params only for constrained mode"
        C = self.card[self.y_index]
        
        # Normalize py
        py = np.asarray(py, dtype=np.float64)
        py = np.clip(py, 1e-12, None)
        py = py / py.sum()
        
        # Initialize logits_y from py
        if self.logits_y is not None:
            log_py = tf.math.log(tf.constant(py, dtype=tf.float32) + 1e-8)
            if jitter_std > 0:
                log_py = log_py + tf.random.normal(tf.shape(log_py), stddev=jitter_std)
            self.logits_y.assign(log_py)
        
        # Initialize logits_v from thetas
        for v, theta in thetas.items():
            if v == self.y_index:
                continue
            # Normalize theta rows
            theta = np.asarray(theta, dtype=np.float64)
            theta = np.clip(theta, 1e-12, None)
            theta = theta / theta.sum(axis=1, keepdims=True)
            
            log_theta = tf.math.log(tf.constant(theta, dtype=tf.float32) + 1e-8)
            if jitter_std > 0:
                log_theta = log_theta + tf.random.normal(tf.shape(log_theta), stddev=jitter_std)
            
            # Only assign if logits_v is a dict and contains an actual Variable for this v
            if isinstance(self.logits_v, dict) and v in self.logits_v and self.logits_v[v] is not None:
                self.logits_v[v].assign(log_theta)

    def _batch_indices_for_class(self, x_batch: tf.Tensor, v: int, c: int) -> tf.Tensor:
        """Compute parent row index for each sample at feature v, assuming Y=c."""
        B = tf.shape(x_batch)[0]
        pa = self.pa_lists[v]
        strides = self.strides[v]  # tf.Tensor (int32) or empty

        # Branch with Tensor-friendly control flow
        def _no_strides():
            return tf.zeros([B], dtype=tf.int32)

        def _with_strides():
            # Build values vector: [Y=c] + x_batch[:, parents...]
            vals = [tf.fill([B], tf.cast(c, tf.int32))]
            for p in pa[1:]:
                vals.append(x_batch[:, p])
            vals = tf.stack(vals, axis=1)  # [B, num_parents]
            # ensure same dtype and shape for arithmetic
            strides_row = tf.expand_dims(tf.cast(strides, tf.int32), axis=0)   # [1, num_parents]
            vals_int = tf.cast(vals, tf.int32)
            prod = tf.math.multiply(vals_int, strides_row)                     # elementwise multiply (no '*' operator)
            idx = tf.reduce_sum(prod, axis=1)
            return tf.cast(idx, tf.int32)

        return tf.cond(tf.equal(tf.size(strides), 0), _no_strides, _with_strides)

    def _batch_indices_for_all_classes(self, x_batch: tf.Tensor, v: int) -> tf.Tensor:
        """Vectorized: Compute parent row indices for all classes simultaneously.
        Returns: [B, C] where [b, c] is the index for sample b assuming class c.
        """
        B = tf.shape(x_batch)[0]
        C = self.card[self.y_index]
        pa = self.pa_lists[v]
        strides = self.strides[v]  # tf.Tensor (int32) or empty

        def _no_strides():
            return tf.zeros([B, C], dtype=tf.int32)

        def _with_strides():
            # Build values for all classes: [B, C, num_parents]
            # For each class c, we need [Y=c] + x_batch[:, parents...]
            classes = tf.range(C, dtype=tf.int32)  # [C]
            classes_expanded = tf.expand_dims(classes, 0)  # [1, C]
            classes_broadcast = tf.tile(classes_expanded, [B, 1])  # [B, C] - class value for each sample
            
            # Build parent values: [B, C, num_parents]
            # First parent is Y (the class), rest are feature parents
            vals_list = [tf.expand_dims(classes_broadcast, 2)]  # [B, C, 1] - Y values
            for p in pa[1:]:
                parent_vals = x_batch[:, p]  # [B]
                parent_vals_expanded = tf.expand_dims(parent_vals, 1)  # [B, 1]
                parent_vals_broadcast = tf.tile(parent_vals_expanded, [1, C])  # [B, C]
                vals_list.append(tf.expand_dims(parent_vals_broadcast, 2))  # [B, C, 1]
            
            vals = tf.concat(vals_list, axis=2)  # [B, C, num_parents]
            
            # Compute indices using strides
            strides_row = tf.expand_dims(tf.cast(strides, tf.int32), axis=0)  # [1, num_parents]
            strides_expanded = tf.expand_dims(strides_row, 0)  # [1, 1, num_parents]
            strides_broadcast = tf.tile(strides_expanded, [B, C, 1])  # [B, C, num_parents]
            
            vals_int = tf.cast(vals, tf.int32)
            prod = tf.math.multiply(vals_int, strides_broadcast)  # [B, C, num_parents]
            idx = tf.reduce_sum(prod, axis=2)  # [B, C]
            return tf.cast(idx, tf.int32)

        return tf.cond(tf.equal(tf.size(strides), 0), _no_strides, _with_strides)

    def _scores_constrained(self, x_batch: tf.Tensor) -> tf.Tensor:
        """Vectorized version: processes all classes simultaneously for better GPU utilization.
        
        This replaces the nested loop (for v, for c) with batched operations that process
        all classes at once. The mathematical logic is identical to the original sequential
        version, but GPU utilization is much better (10-20x speedup expected).
        
        Original: For each class c, compute indices [B], gather [B, r_v], get probs [B]
        Vectorized: For all classes, compute indices [B, C], gather [B, C, r_v], get probs [B, C]
        """
        B = tf.shape(x_batch)[0]
        C = self.card[self.y_index]
        S = tf.tile(tf.nn.log_softmax(self.logits_y)[tf.newaxis, :], [B, 1])  # [B,C]
        
        for v in range(len(self.card)):
            if v == self.y_index: continue
            theta_v = tf.nn.softmax(self.logits_v[v], axis=-1)  # [num_cfg, r_v]
            x_v = x_batch[:, v]  # [B]
            
            # Vectorized: compute indices for all classes at once
            idx_all = self._batch_indices_for_all_classes(x_batch, v)  # [B, C]
            
            # Gather rows for all classes: [B, C, r_v]
            # Flatten indices for gather: [B*C]
            B_int = tf.cast(B, tf.int32)
            batch_indices = tf.range(B_int)  # [B]
            batch_indices_expanded = tf.expand_dims(batch_indices, 1)  # [B, 1]
            batch_indices_broadcast = tf.tile(batch_indices_expanded, [1, C])  # [B, C]
            flat_batch = tf.reshape(batch_indices_broadcast, [-1])  # [B*C]
            flat_idx = tf.reshape(idx_all, [-1])  # [B*C]
            
            # Gather all rows: [B*C, r_v]
            rows_all_flat = tf.gather(theta_v, flat_idx)  # [B*C, r_v]
            rows_all = tf.reshape(rows_all_flat, [B, C, -1])  # [B, C, r_v]
            
            # Gather probabilities for all classes: [B, C]
            # x_v values: [B] -> [B, 1] -> [B, C]
            x_v_expanded = tf.expand_dims(x_v, 1)  # [B, 1]
            x_v_broadcast = tf.tile(x_v_expanded, [1, C])  # [B, C]
            
            # Build gather indices for 3D tensor [B, C, r_v]
            # gather_idx[b, c, :] = [b, c, x_v[b]] to get rows_all[b, c, x_v[b]]
            batch_idx = tf.range(B_int)  # [B]
            batch_idx_expanded = tf.expand_dims(batch_idx, 1)  # [B, 1]
            batch_idx_broadcast = tf.tile(batch_idx_expanded, [1, C])  # [B, C]
            
            class_idx = tf.range(C, dtype=tf.int32)  # [C]
            class_idx_expanded = tf.expand_dims(class_idx, 0)  # [1, C]
            class_idx_broadcast = tf.tile(class_idx_expanded, [B, 1])  # [B, C]
            
            gather_idx = tf.stack([
                tf.reshape(batch_idx_broadcast, [-1]),  # [B*C] - batch indices
                tf.reshape(class_idx_broadcast, [-1]),  # [B*C] - class indices
                tf.reshape(x_v_broadcast, [-1])  # [B*C] - feature values
            ], axis=1)  # [B*C, 3]
            
            # Gather probabilities: [B*C] -> reshape to [B, C]
            p_all_flat = tf.gather_nd(rows_all, gather_idx)  # [B*C]
            p_all = tf.reshape(p_all_flat, [B, C])  # [B, C]
            
            # Add to scores: [B, C]
            log_p_all = tf.math.log(tf.clip_by_value(p_all, 1e-8, 1.0))  # [B, C]
            S = S + log_p_all  # [B, C] - vectorized addition for all classes
        
        return S

    def _scores_unconstrained(self, x_batch: tf.Tensor) -> tf.Tensor:
        B = tf.shape(x_batch)[0]
        C = self.card[self.y_index]
        S = tf.tile(self.bias[tf.newaxis, :], [B, 1])  # [B,C]
        for v in range(len(self.card)):
            if v == self.y_index: continue
            Wv = self.W[v]  # [num_cfg, r_v, C]
            x_v = x_batch[:, v]  # [B]
            for c in range(C):
                idx_c = self._batch_indices_for_class(x_batch, v, c)         # [B]
                rows = tf.gather(Wv[..., c], idx_c)                           # [B, r_v]
                contrib = tf.gather_nd(rows, tf.stack([tf.range(B), x_v], axis=1))  # [B]
                S = tf.tensor_scatter_nd_add(S, tf.stack([tf.range(B), tf.fill([B], c)], axis=1), contrib)
        return S

    @tf.function(reduce_retracing=True)
    def _nll_batch(self, x_batch: tf.Tensor) -> tf.Tensor:
        y = x_batch[:, self.y_index]
        if self.constrained:
            S = self._scores_constrained(x_batch)
        else:
            S = self._scores_unconstrained(x_batch)
        logp = tf.nn.log_softmax(S, axis=1)
        yoh = tf.one_hot(y, depth=self.card[self.y_index], dtype=tf.float32)
        nll = -tf.reduce_mean(tf.reduce_sum(logp * yoh, axis=1))
        return nll

    def fit(self, data: np.ndarray, epochs: int = 60, batch_size: int = 1024, l2: float = 1e-5, verbose: int = 0,
            kl_targets: Dict[int, np.ndarray] | None = None, kl_lambda: float = 0.0, s_y: np.ndarray | None = None,
            kl_s_weights: Dict[int, np.ndarray] | None = None):
        N = data.shape[0]; idx_all = np.arange(N)
        # prepare targets
        kl_lambda = float(kl_lambda or 0.0)
        kl_lambda_tf = tf.constant(kl_lambda, dtype=tf.float32)
        l2 = float(l2 or 0.0)
        l2_tf = tf.constant(l2, dtype=tf.float32)
        if s_y is not None:
            s_y = np.asarray(s_y, dtype=np.float32)
            s_y = s_y / (s_y.sum() + 1e-12)
        
        # Pre-convert KL targets and weights to tensors
        kl_targets_tf = {}
        kl_weights_tf = {}
        if self.federated and kl_lambda > 0.0 and kl_targets:
            for v, arr in kl_targets.items():
                kl_targets_tf[int(v)] = tf.constant(np.asarray(arr, dtype=np.float32), dtype=tf.float32)
            if kl_s_weights:
                for v, arr in kl_s_weights.items():
                    kl_weights_tf[int(v)] = tf.constant(np.asarray(arr, dtype=np.float32), dtype=tf.float32)
        
        # Store loss value for logging
        loss_value = None
            
        for ep in range(epochs):
            if verbose > 0:
                print(f"[DiscBN_KDB] Epoch {ep+1}/{epochs}")
            rng = np.random.default_rng(ep + 7)
            rng.shuffle(idx_all)
            for s in range(0, N, batch_size):
                batch = data[idx_all[s:s+batch_size]]
                x_tf = tf.convert_to_tensor(batch, dtype=tf.int32)
                with tf.GradientTape() as tape:
                    loss = self._nll_batch(x_tf)
                    if l2 > 0.0:
                        if self.constrained:
                            reg = tf.nn.l2_loss(self.logits_y)
                            for v in self.logits_v.values(): reg += tf.nn.l2_loss(v)
                        else:
                            reg = tf.nn.l2_loss(self.bias)
                            for v in self.W.values(): reg += tf.nn.l2_loss(v)
                        loss = loss + tf.multiply(l2_tf, reg)
                    # KL regularizer: federated mode only
                    if self.federated and kl_lambda > 0.0 and kl_targets_tf:
                        kl_loss = tf.constant(0.0, dtype=tf.float32)
                        for v, logits in self.logits_v.items():
                            if int(v) not in kl_targets_tf:
                                continue
                            theta_local = tf.nn.softmax(logits, axis=-1)        # [num_cfg, r_v]
                            theta_tgt = kl_targets_tf[int(v)]
                            theta_local = tf.clip_by_value(theta_local, 1e-8, 1.0)
                            theta_tgt = tf.clip_by_value(theta_tgt, 1e-8, 1.0)
                            # weights per element (row-normalized)
                            if int(v) in kl_weights_tf:
                                W = kl_weights_tf[int(v)]
                            else:
                                W = tf.ones_like(theta_local)
                            # weighted per-row KL
                            el = theta_local * (tf.math.log(theta_local) - tf.math.log(theta_tgt))
                            el = W * el
                            kl_rows = tf.reduce_sum(el, axis=1)                 # [num_cfg]
                            kl_v = tf.reduce_mean(kl_rows)
                            kl_loss += kl_v
                        loss = loss + tf.multiply(kl_lambda_tf, kl_loss)
                vars_ = ([self.logits_y] + list(self.logits_v.values())) if self.constrained else ([self.bias] + list(self.W.values()))
                grads = tape.gradient(loss, vars_)
                self.opt.apply_gradients(zip(grads, vars_))
                # Store loss value for logging (only at end of epoch to minimize blocking)
                if verbose and (ep % max(1, epochs//5) == 0 or ep == epochs-1) and s + batch_size >= N:
                    loss_value = float(loss)
            # Log after epoch completes (deferred .numpy() call to reduce blocking)
            if verbose > 0:
                if loss_value is not None:
                    if kl_lambda > 0.0:
                        print(f"[DiscBN_KDB] Epoch {ep+1}/{epochs} completed: nll={loss_value:.4f} kl_lambda={kl_lambda:.6f}")
                    else:
                        print(f"[DiscBN_KDB] Epoch {ep+1}/{epochs} completed: nll={loss_value:.4f}")
                elif ep % max(1, epochs//5) == 0 or ep == epochs-1:
                    # Fallback if loss_value wasn't captured
                    print(f"[DiscBN_KDB] Epoch {ep+1}/{epochs} completed")

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        x_tf = tf.convert_to_tensor(data, dtype=tf.int32)
        S = self._scores_constrained(x_tf) if self.constrained else self._scores_unconstrained(x_tf)
        return np.asarray(tf.nn.softmax(S, axis=1))
    
    def predict_probs(self, data: np.ndarray) -> np.ndarray:
        return self.predict_proba(data)

    def sample(self, X: np.ndarray, rng: np.random.Generator , return_probs: bool = False) -> np.ndarray:
        """
        Sample class labels Y ~ P(Y|X) for provided feature matrix X (integer encoded).

        Accepts either:
            - X shaped [B, V-1] containing only feature columns (Y removed), in the same column order
            as the original data (i.e. features at indices != y_index), or
            - X shaped [B, V] containing a full table (Y may be ignored).

        Returns:
            y_samp (np.ndarray shape [B]) or (y_samp, P) if return_probs=True.
        """
        if rng is None:
            rng = np.random.default_rng()

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D [B, num_features] or [B, V]")

        V = len(self.card)
        # If user passed full table (including Y), use it directly.
        if X.shape[1] == V:
            full = X.astype(np.int32)
        elif X.shape[1] == V - 1:
            # features-only: insert dummy Y column at y_index
            B = X.shape[0]
            full = np.zeros((B, V), dtype=np.int32)
            feat_indices = [v for v in range(V) if v != self.y_index]
            # Ensure shapes align
            if X.shape[1] != len(feat_indices):
                raise ValueError(f"Expected {len(feat_indices)} feature columns but got {X.shape[1]}")
            full[:, feat_indices] = X
            # Y placeholder left as zeros (ignored by predict_proba because we evaluate P(Y|X))
        else:
            raise ValueError(f"X must have either {V-1} (features only) or {V} (full table) columns, got {X.shape[1]}")

        # Compute posterior probabilities P(Y|X)
        P = self.predict_proba(full)  # shape [B, C]

        # Vectorized categorical sampling using cumulative sums
        B, C = P.shape
        # numerical safety
        P = np.clip(P, 0.0, 1.0)
        P = P / P.sum(axis=1, keepdims=True)
        cum = np.cumsum(P, axis=1)
        r = rng.random(B).reshape(-1, 1)
        draws = (cum >= r).argmax(axis=1).astype(np.int32)

        if return_probs:
            return draws, P
        return draws
    
    def sample_full(self, n: int, rng: np.random.Generator | None = None, order: List[int] | None = None, return_y: bool = True) -> np.ndarray:
        """
        Sample n full rows (Y + all features) from a constrained DiscBN_KDB.
        Returns [n, V] integer-coded table if return_y=True, else only features (no Y).
        """
        if not self.constrained:
            raise RuntimeError("sample_full requires constrained=True (generative CPTs).")
        if rng is None:
            rng = np.random.default_rng()

        V = len(self.card)
        # class prior
        
        py = np.asarray(tf.nn.softmax(self.logits_y)).ravel().astype(np.float64)
        py = np.clip(py, 1e-12, None)
        py = py / (py.sum() + 1e-12)
        C = py.shape[0]

        # per-feature CPTs and strides
        thetas = {}
        strides = {}
        pa_lists = {}
        for v in range(V):
            if v == self.y_index:
                continue
            pa = list(self.pa_lists.get(v, []))  # includes Y first
            pa_lists[v] = pa
            cards_pa = [self.card[p] for p in pa] if pa else []
            strides[v] = _compute_strides(cards_pa)
            thetas[v] = np.asarray(tf.nn.softmax(self.logits_v[v], axis=-1))  # [num_cfg, r_v]

        # infer topological order if not provided (parents EXCLUDING Y)
        if order is None:
            order = _topo_order_from_parents(self.parents, V, self.y_index)

        # allocate and sample
        samples = np.zeros((n, V), dtype=np.int32)
        samples[:, self.y_index] = rng.choice(C, size=n, p=py)

        for v in order:
            theta_v = thetas[v]            # [num_cfg, r_v]
            stride_v = strides[v]          # np.ndarray
            pa = pa_lists[v]               # [Y] + feature parents
            rv = theta_v.shape[1]
            for i in range(n):
                if stride_v.size == 0:
                    idx = 0
                else:
                    vals = [int(samples[i, p]) for p in pa]
                    idx = 0
                    for s, val in zip(stride_v, vals):
                        idx += int(s) * int(val)
                row = np.asarray(theta_v[idx], dtype=np.float64)
                row = np.clip(row, 1e-12, None)
                row = row / (row.sum() + 1e-12)
                samples[i, v] = rng.choice(rv, p=row)

        if return_y:
            return samples
        else:
            return samples[:, [v for v in range(V) if v != self.y_index]]
    

