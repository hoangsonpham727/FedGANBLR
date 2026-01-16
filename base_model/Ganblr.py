from base_models.KDependenceBayesian import MLE_KDB, DiscBN_KDB, _build_kdb_structure, _topo_order_from_parents
from utils import _evaluate_synthetic_classifiers, _ohe_pipeline
from typing import Optional, Callable
import numpy as np
import keras
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils import _safe_log_loss

class SimpleGANBLR:
    """
    Simplified GANBLR: generator only (MLE_KDB).  
    """
    def __init__(self, alpha: float = 1.0, gpu_id=None):
        # clear initial values (no None)
        self.generator: Optional[MLE_KDB] = None    # will be an MLE_KDB after fit
        self.disc = False                 # no discriminator used
        self.alpha = float(alpha)
        self.k = int(2)
        self.gpu_id = gpu_id
        self.card = []                    # list[int] (will be [|Y|] + feature cards)
        self.y_index = 0
        self.parents = {}                 # mapping v -> parent list (excluding Y)
        self.order = []                   # topological order of features (excluding Y)
        self._X = np.empty((0, 0), dtype=np.int32)
        self._y = np.empty((0,), dtype=np.int32)
        self._train_table = np.empty((0, 0), dtype=np.int32)
        self._fitted = False

    # ---------- internal helpers ----------
    def _onehot_feats(self, X_int: np.ndarray) -> np.ndarray:
        parts = []
        feat_indices = [v for v in range(len(self.card)) if v != self.y_index]
        for j, v in enumerate(feat_indices):
            r = self.card[v]
            parts.append(np.eye(r, dtype=np.float32)[X_int[:, j]])
        return np.concatenate(parts, axis=1) if parts else np.zeros((X_int.shape[0], 0), dtype=np.float32)

    def _build_discriminator(self, input_dim: int):
        # kept for API compatibility but not used
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    # ---------- public API ----------
    def fit(self, X_int: np.ndarray, y_int: np.ndarray, k: int = 2, verbose: int = 1, warmup_epochs: int = 1):
        """
        Fit simplified GANBLR: learn KDB structure and fit an MLE_KDB generator.
        Accepts either:
         - X_int (N, V) full table with Y in first column (y_index==0), or
         - X_int (N, F) feature matrix and y_int (N,) labels.
        """
        # normalize inputs
        X_int = np.asarray(X_int, dtype=np.int32)
        if y_int is None:
            # assume full table with Y first column
            if X_int.ndim != 2 or X_int.shape[1] < 2:
                raise ValueError("Provide either (X_int, y_int) or a full table with Y in column 0")
            data = X_int
            y = data[:, 0].astype(np.int32)
            X = data[:, 1:].astype(np.int32)
        else:
            y = np.asarray(y_int, dtype=np.int32)
            X = X_int.astype(np.int32)
            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y must have the same number of rows")
            data = np.column_stack([y, X]).astype(np.int32)

        self.k = int(k)
        # build card: [|Y|] + per-feature cards
        y_card = int(y.max()) + 1
        feat_cards = [int(X[:, j].max()) + 1 for j in range(X.shape[1])]
        self.card = [y_card] + feat_cards
        self.y_index = 0

        # store training arrays
        self._X = X
        self._y = y
        self._train_table = data

        # build KDB parents/order
        self.parents, self.order = _build_kdb_structure(self._train_table, self.card, self.y_index, k=self.k)

        # create MLE generator and fit
        parents_excl_y = {v: [p for p in plist if p != self.y_index] for v, plist in self.parents.items()}
        self.generator = MLE_KDB(self.card, self.y_index, parents_excl_y, alpha=self.alpha)

        # warmup fits (keep API similar)
        for _ in range(max(1, warmup_epochs)):
            self.generator.fit(self._train_table)

        self._fitted = True
        if verbose:
            print(f"[SimpleGANBLR] fitted generator (k={self.k}). |Y|={self.card[self.y_index]}, features={len(feat_cards)}")
        return self

    def sample(self, size=None, return_decoded=False):
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        size = self._X.shape[0] if size is None else int(size)
        gen = self.generator
        if gen is None:
            raise RuntimeError("Generator not initialized. Call fit() first.")
        syn_full = gen.sample(n=size, rng=np.random.default_rng(), order=self.order, return_y=True)
        y_syn = syn_full[:, 0]
        X_syn = syn_full[:, 1:]
        if return_decoded:
            return X_syn, y_syn
        return X_syn, y_syn

    def evaluate(self, X_test_int: np.ndarray, y_test_int: np.ndarray, clf='lr'):
        X_syn, y_syn = self.sample(size=self._X.shape[0])
        # Reuse shared evaluator (returns acc_*/nll_* for lr/mlp/rf)
        return _evaluate_synthetic_classifiers(X_syn, y_syn, X_test_int, y_test_int)


class GANBLR:
    """
    GANBLR with adversarial training where both Generator and Discriminator are KDBs.
    - Generator: DiscBN_KDB — KDBe-style CPTs trained by gradient.
    - Discriminator: Simple Logistic Regression model — predicts is_real ∈ {0,1}.
    Logic: warm-start run with randomly initialized parameters, then adversarial rounds:
      1) sample synthetic rows from G,
      2) train D on real(1) vs fake(0) for a few epochs,
      3) compute discriminator confidence score and send back to the generator,
      4) continiue training the generator with the new loss and discriminator's information.  
    Notes:
      - KDB parent parameter k is the structure hyperparameter.
      - Use disc_epochs to control D training intensity each round.
    """
    def __init__(self, adversarial: bool = False, alpha: float = 1.0, seed: int = 123):
        self.alpha = float(alpha)
        self.seed = int(seed)

        # learned in fit()
        self.k = 0
        self.card: list[int] = []
        self.y_index = 0
        self.parents: dict[int, list[int]] = {}   # generator parents excluding Y
        self.order: list[int] = []                # topo order for features (excl Y)

        # models
        self.generator: Optional[DiscBN_KDB] = None  # constrained=True

        # caches
        self._X = np.empty((0, 0), dtype=np.int32)
        self._y = np.empty((0,), dtype=np.int32)
        self._train_table = np.empty((0, 0), dtype=np.int32)

        self._fitted = False
        self.adversarial = adversarial

    # ---------- internals ----------
    def _build_gen_from_mle(self, mle: 'MLE_KDB', parents_excl_y: dict[int, list[int]], jitter_std: float = 0.0) -> 'DiscBN_KDB':
        # Create generator with random initialization (MLE initialization removed per user request)
        gen = DiscBN_KDB(self.card, self.y_index, parents_excl_y, constrained=True, seed=self.seed)
        # Note: DiscBN_KDB.__init__ already initializes parameters randomly, so no need to call init_from_mle
        return gen

    def _topo_from_parents(self, V: int, y_index: int, parents_excl_y: dict[int, list[int]]) -> list[int]:
        return _topo_order_from_parents(parents_excl_y, V, y_index)

    def _run_adversarial_training(self,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  epochs: int,
                                  batch_size: int,
                                  disc_epochs: int,
                                  rng: np.random.Generator,
                                  rng_local: Optional[np.random.Generator] = None,
                                  verbose: int = 0,
                                  initial_disc_train: bool = False,
                                  kl_targets: Optional[dict[int, np.ndarray]] = None,
                                  kl_lambda: float = 0.0,
                                  s_y: Optional[np.ndarray] = None,
                                  kl_s_weights: Optional[dict[int, np.ndarray]] = None,
                                  client_id: Optional[str] = None,
                                  seed_fn: Optional[Callable] = None):
        """
        Shared adversarial training logic for both GANBLR and FedGANBLR.
        
        Args:
            X: Feature matrix [N, F]
            y: Labels [N]
            epochs: Number of adversarial epochs
            batch_size: Batch size for generator training
            disc_epochs: Epochs to train discriminator each round
            rng: Random number generator for sampling
            rng_local: Random number generator for resampling (if None, uses rng)
            verbose: Verbosity level
            initial_disc_train: If True, do an initial discriminator training before the loop
            kl_targets: Optional KL targets for federated learning
            kl_lambda: KL regularization weight
            s_y: Optional class weights for federated learning
            kl_s_weights: Optional importance weights for federated learning
            client_id: Optional client ID for logging
            seed_fn: Optional function to generate seeds (for federated learning)
        """
        if self.generator is None:
            raise RuntimeError("Generator not initialized")
        
        N = X.shape[0]
        if rng_local is None:
            rng_local = rng
        
        # Initialize discriminator if not already set
        if not hasattr(self, 'disc') or self.disc is None:
            self.disc = _ohe_pipeline(LogisticRegression(max_iter=500, multi_class="ovr", solver="lbfgs"), step_name="lr")
        
        # Optional initial discriminator training (for federated learning)
        # This is now mainly for consistency - the new implementation trains on real data directly
        # but we keep this for any potential future use or diagnostics
        if initial_disc_train:
            fake0 = self._sample_full_from_gen(n=max(1, N), rng=np.random.default_rng(12345), order=self.order)
            X_fake0 = fake0[:N, 1:].astype(np.int32)
            X_disc0_all = np.vstack([X, X_fake0]).astype(np.int32)
            y_disc0_all = np.concatenate([np.ones(N, dtype=np.int32), np.zeros(N, dtype=np.int32)])
            
            # Sample from combined dataset (consistent with main loop - uses 100% of data)
            n_disc0 = len(X_disc0_all)
            disc_indices0 = rng.choice(len(X_disc0_all), size=n_disc0, replace=False)
            X_disc0 = X_disc0_all[disc_indices0]
            y_disc0 = y_disc0_all[disc_indices0]
            
            # Create fresh discriminator (consistent with main loop)
            self.disc = _ohe_pipeline(LogisticRegression(max_iter=500, multi_class="ovr", solver="lbfgs"), step_name="lr")
            for _ in range(max(1, int(disc_epochs))):
                self.disc.fit(X_disc0, y_disc0)
            
            # Compute initial discriminator predictions for diagnostics
            # Note: In the new implementation, we don't use importance weights for resampling
            # but we keep this for potential future use or logging
            p_real = self.disc.predict_proba(X)[:, 1]
            p_real = np.clip(p_real, 1e-6, 1 - 1e-6)
            w = 1.0 - p_real
            probs = (w / w.sum()) if w.sum() > 0 else (np.ones_like(w) / max(1, len(w)))
        else:
            probs = None
        
        # Adversarial training loop
        log_prefix = f"[Client {client_id}]" if client_id else "[GANBLR]"
        if verbose > 0:
            print(f"{log_prefix} Starting adversarial training: {epochs} epochs")
        
        # Import tensorflow for manual gradient computation
        import tensorflow as tf
        
        for ep in range(1, epochs + 1):
            if verbose > 0:
                print(f"{log_prefix} Adversarial epoch {ep}/{epochs}")
            
            # 1) Sample synthetic data
            if seed_fn is not None:
                syn_full = self._sample_full_from_gen(n=N, rng=np.random.default_rng(seed_fn(ep)), order=self.order)
            else:
                syn_full = self._sample_full_from_gen(n=N, rng=rng, order=self.order)
            X_fake = syn_full[:, 1:]
            
            # 2) Train discriminator on combined data (fresh discriminator each epoch)
            X_disc_all = np.vstack([X, X_fake]).astype(np.int32)
            y_disc_all = np.concatenate([np.ones(N, dtype=np.int32), np.zeros(N, dtype=np.int32)])
            
            # Use all combined dataset (100% sampling)
            n_disc = len(X_disc_all)
            disc_indices = rng.choice(len(X_disc_all), size=n_disc, replace=False)
            X_disc = X_disc_all[disc_indices]
            y_disc = y_disc_all[disc_indices]
            
            # Create fresh discriminator each epoch
            self.disc = _ohe_pipeline(LogisticRegression(max_iter=500, multi_class="ovr", solver="lbfgs"), step_name="lr")
            for _ in range(max(1, int(disc_epochs))):
                self.disc.fit(X_disc, y_disc)
            
            # 3) Calculate generator loss using discriminator predictions on REAL data only
            # According to documentation: discriminator outputs probability that sample is "fake"
            # Our discriminator outputs probability of being "real" (class 1), so we interpret:
            # p_fake = 1 - p_real (probability discriminator thinks real data is fake)
            p_real = self.disc.predict_proba(X)[:, 1]
            p_real = np.clip(p_real, 1e-6, 1 - 1e-6)
            p_fake = 1.0 - p_real  # Probability discriminator thinks real data is fake
            # Generator loss: mean of negative log of (1 - p_fake)
            # = -mean(log(p_real))
            # If discriminator is uncertain about real data (p_real ≈ 0.5, p_fake ≈ 0.5),
            # this suggests generator is producing realistic data, so loss should be moderate
            # If discriminator is confident real is real (p_real ≈ 1), loss is low (good for generator)
            # If discriminator thinks real is fake (p_real ≈ 0), loss is high (bad for generator)
            # This encourages generator to produce data that makes discriminator confident real data is real
            disc_loss_np = -np.mean(np.log(p_real + 1e-8))
            disc_loss_tf = tf.constant(disc_loss_np, dtype=tf.float32)
            
            # 4) Update generator with combined loss: NLL + discriminator loss
            # Train generator on real data (not resampled) with combined loss
            data_real = np.column_stack([y, X]).astype(np.int32)
            
            # Manual training step to incorporate discriminator loss
            gen = self.generator
            N_batch = data_real.shape[0]
            idx_all = np.arange(N_batch)
            rng_batch = np.random.default_rng(ep + 1000)
            rng_batch.shuffle(idx_all)
            
            # Prepare KL regularization if needed
            kl_lambda_tf = tf.constant(float(kl_lambda or 0.0), dtype=tf.float32)
            l2_tf = tf.constant(1e-5, dtype=tf.float32)
            
            kl_targets_tf = {}
            kl_weights_tf = {}
            if gen.federated and kl_lambda > 0.0 and kl_targets:
                for v, arr in kl_targets.items():
                    kl_targets_tf[int(v)] = tf.constant(np.asarray(arr, dtype=np.float32), dtype=tf.float32)
                if kl_s_weights:
                    for v, arr in kl_s_weights.items():
                        kl_weights_tf[int(v)] = tf.constant(np.asarray(arr, dtype=np.float32), dtype=tf.float32)
            
            # Train for one epoch with combined loss
            for s in range(0, N_batch, batch_size):
                batch = data_real[idx_all[s:s+batch_size]]
                x_tf = tf.convert_to_tensor(batch, dtype=tf.int32)
                
                with tf.GradientTape() as tape:
                    # Standard NLL loss
                    loss = gen._nll_batch(x_tf)
                    
                    # Add L2 regularization
                    if l2_tf > 0.0:
                        if gen.constrained:
                            reg = tf.nn.l2_loss(gen.logits_y)
                            for v in gen.logits_v.values():
                                reg += tf.nn.l2_loss(v)
                        else:
                            reg = tf.nn.l2_loss(gen.bias)
                            for v in gen.W.values():
                                reg += tf.nn.l2_loss(v)
                        loss = loss + tf.multiply(l2_tf, reg)
                    
                    # Add KL regularization (for federated learning)
                    if gen.federated and kl_lambda > 0.0 and kl_targets_tf:
                        kl_loss = tf.constant(0.0, dtype=tf.float32)
                        for v, logits in gen.logits_v.items():
                            if int(v) not in kl_targets_tf:
                                continue
                            theta_local = tf.nn.softmax(logits, axis=-1)
                            theta_tgt = kl_targets_tf[int(v)]
                            theta_local = tf.clip_by_value(theta_local, 1e-8, 1.0)
                            theta_tgt = tf.clip_by_value(theta_tgt, 1e-8, 1.0)
                            if int(v) in kl_weights_tf:
                                W = kl_weights_tf[int(v)]
                            else:
                                W = tf.ones_like(theta_local)
                            el = theta_local * (tf.math.log(theta_local) - tf.math.log(theta_tgt))
                            el = W * el
                            kl_rows = tf.reduce_sum(el, axis=1)
                            kl_v = tf.reduce_mean(kl_rows)
                            kl_loss += kl_v
                        loss = loss + tf.multiply(kl_lambda_tf, kl_loss)
                    
                    # Add discriminator loss term (KL divergence-like term)
                    # This encourages generator to produce realistic data
                    loss = loss + disc_loss_tf
                
                # Compute gradients and update
                vars_ = ([gen.logits_y] + list(gen.logits_v.values())) if gen.constrained else ([gen.bias] + list(gen.W.values()))
                grads = tape.gradient(loss, vars_)
                gen.opt.apply_gradients(zip(grads, vars_))
            
            if verbose > 0:
                acc_disc_real = float(np.mean(self.disc.predict(X) == 1))
                disc_loss_val = float(disc_loss_np)
                print(f"{log_prefix} Epoch {ep}/{epochs} completed: D-acc(real)={acc_disc_real:.4f} disc_loss={disc_loss_val:.4f}")

    def _sample_full_from_gen(self, n: int, rng: np.random.Generator, order: Optional[list[int]] = None) -> np.ndarray:
        """
        Delegate full-table sampling to the constrained DiscBN_KDB generator.
        """
        gen = self.generator
        if gen is None:
            raise RuntimeError("Generator not initialized.")
        # reuse provided order if any; fallback handled inside sample_full
        return gen.sample_full(n=n, rng=rng, order=order)


    # ---------- public API ----------
    def fit(self,
            X_int: np.ndarray,
            y_int: np.ndarray,
            k: int = 2,
            epochs: int = 10,
            batch_size: int = 1024,
            disc_epochs: int = 1,
            warmup_epochs: int = 1,
            verbose: int = 0,
            adversarial: Optional[bool] = None):
        """
        Train GANBLR (Disc_KDB generator + Disc_KDB discriminator).
        - X_int: [N, F] integer features
        - y_int: [N] labels 0..C-1
        - k: KDB parent parameter (structure)
        - disc_epochs: epochs to train D each round (acts like D-steps)
        - adversarial: If provided, overrides the instance-level adversarial setting. 
                       If None, uses self.adversarial from initialization.
        """
        rng = np.random.default_rng(self.seed)

        X = np.asarray(X_int, dtype=np.int32)
        y = np.asarray(y_int, dtype=np.int32)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("X must be 2D and y 1D with matching rows.")

        self.k = int(k)
        self._X = X
        self._y = y
        data = np.column_stack([y, X]).astype(np.int32)
        self._train_table = data

        # cardinalities
        y_card = int(y.max()) + 1
        feat_cards = [int(X[:, j].max()) + 1 for j in range(X.shape[1])]
        self.card = [y_card] + feat_cards
        self.y_index = 0
        V = len(self.card)

        # Build structure for generator
        parents_full, order = _build_kdb_structure(data, self.card, self.y_index, k=self.k)
        parents_excl_y = {v: [p for p in parents_full.get(v, []) if p != self.y_index] for v in range(V) if v != self.y_index}
        self.parents = parents_excl_y
        self.order = order

        # Initialize generator from MLE
        # First, fit an MLE_KDB model
        mle = MLE_KDB(self.card, self.y_index, parents_excl_y, alpha=self.alpha)
        mle.fit(data)
        
        # Create generator and initialize from MLE
        gen = DiscBN_KDB(self.card, self.y_index, parents_excl_y, constrained=True, seed=self.seed)
        gen.init_from_mle(mle, jitter_std=0.0)
        
        if warmup_epochs > 0:
            if verbose > 0:
                print(f"[GANBLR] Starting warmup: {warmup_epochs} epochs")
            gen.fit(data, epochs=warmup_epochs, batch_size=min(batch_size, data.shape[0]), l2=1e-5, verbose=verbose)
        self.generator = gen
        
        # Determine if adversarial training should be used
        # Use parameter if provided, otherwise use instance-level setting
        use_adversarial = adversarial if adversarial is not None else self.adversarial
        
        # ---- Generator-only mode (no discriminator/adversarial rounds) ----
        if (not use_adversarial) or disc_epochs <= 0:
            if epochs > 0:
                if verbose > 0:
                    print(f"[GANBLR] Starting generator training: {epochs} epochs")
                self.generator.fit(
                    data,
                    epochs=int(epochs),
                    batch_size=min(batch_size, data.shape[0]),
                    l2=1e-5,
                    verbose=verbose
                )
            if verbose > 0:
                print(f"[GANBLR] Generator-only training completed (epochs={epochs}, warmup={warmup_epochs})")
            self._fitted = True
            return self
        # ---- Adversarial learning mode ----
        rng_local = np.random.default_rng(self.seed + 2025)
        self._run_adversarial_training(
            X=X,
            y=y,
            epochs=epochs,
            batch_size=batch_size,
            disc_epochs=disc_epochs,
            rng=rng,
            rng_local=rng_local,
            verbose=verbose,
            initial_disc_train=False
        )

        self._fitted = True
        return self

    def sample(self, size: Optional[int] = None, return_decoded: bool = False):
        """Return synthetic encoded ints: (X_syn, y_syn)."""
        if not self._fitted or self.generator is None:
            raise RuntimeError("Call fit() first.")
        n = self._X.shape[0] if size is None else int(size)
        rng = np.random.default_rng(self.seed)
        syn_full = self._sample_full_from_gen(n=n, rng=rng, order=self.order)
        y_syn = syn_full[:, 0]
        X_syn = syn_full[:, 1:]
        return (X_syn, y_syn) if return_decoded else (X_syn, y_syn)

    def evaluate(self, X_test_int: np.ndarray, y_test_int: np.ndarray, clf='lr'):
        """Train a classifier on synthetic data and evaluate on provided test set."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        X_syn, y_syn = self.sample(size=self._X.shape[0], return_decoded=False)
        models = dict(
            lr=LogisticRegression(max_iter=500, multi_class='multinomial'),
            rf=RandomForestClassifier(n_estimators=200, random_state=0),
            mlp=MLPClassifier(max_iter=300, random_state=0),
        )
        model = models[clf] if isinstance(clf, str) else clf
        pipe = _ohe_pipeline(model)
        pipe.fit(X_syn, y_syn)
        proba = pipe.predict_proba(X_test_int)
        preds = proba.argmax(axis=1)
        acc = accuracy_score(y_test_int, preds)
        nll = _safe_log_loss(y_test_int, proba, self.card[self.y_index])
        return dict(acc=acc, nll=nll)
