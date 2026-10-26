from base_models.KDependenceBayesian import MLE_KDB, DiscBN_KDB, _build_kdb_structure, _topo_order_from_parents
from utils import _evaluate_synthetic_classifiers, _ohe_pipeline
from typing import Optional, Callable, Union
import numpy as np
import keras
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from utils import _safe_log_loss

class SimpleGANBLR:
    """
    Simplified GANBLR: generator only (MLE_KDB).  
    """
    def __init__(self, alpha: float = 1.0):
        # clear initial values (no None)
        self.generator: Optional[MLE_KDB] = None    # will be an MLE_KDB after fit
        self.disc = False                 # no discriminator used
        self.alpha = float(alpha)
        self.k = int(2)
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
    def fit(self, X_int: np.ndarray, y_int: np.ndarray, k: int = 2, verbose: int = 0, warmup_epochs: int = 1):
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
    - Generator: Disc_KDB (here: DiscBN_KDB with constrained=True) — KDBe-style CPTs trained by gradient.
    - Discriminator: Simple Logisctic Regression model — predicts is_real ∈ {0,1}.
    Logic: generator initialized randomly, then adversarial rounds:
      1) sample synthetic rows from G,
      2) train D on real(1) vs fake(0) for a few epochs,
      3) compute weights w = 1 - P_D(is_real=1 | X_real),
      4) resample real rows by w and update G one discriminative epoch.
    Notes:
      - KDB parent parameter k is the structure hyperparameter (not the D-repeat count).
      - Use disc_epochs to control D training intensity each round.
      - Generator uses random initialization (not MLE) to ensure meaningful gradient updates.
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
        Adversarial training for GANBLR.

        Mapping to the paper:
          * Discriminator step  -> Eq. 7:  max_theta_d  log D(D_data) + log(1 - D(S_data))
          * Generator step      -> Eq. 2:  max_theta_g  log P(Y_g | X_g^k) - log(1 - D(S_data))
          * Combined game       -> Eq. 8.

        Key fix vs. the previous version
        --------------------------------
        The -log(1 - D(S_data)) term of Eq. 2 cannot be back-propagated: forward
        sampling (Eq. 6) is non-differentiable and the discriminator is sklearn.
        Adding D's loss as a tf.constant therefore left the generator gradient
        identical to the no-adversarial case (GANBLR-nAL).

        The faithful realisation is to fold the adversarial pressure into the CLL
        term as a per-row IMPORTANCE WEIGHT. With the optimal discriminator
            D(x) = p_data(x) / (p_data(x) + p_g(x)),
        the likelihood ratio is
            w(x) = D(x) / (1 - D(x)) = p_data(x) / p_g(x).
        Training log P(Y|X) weighted by w(x) pushes p_g toward p_data exactly in
        the regions D can still separate -- the differentiable stand-in for
        -log(1 - D(S_data)). Because the discriminator now changes WHICH rows
        dominate the gradient, theta_g actually moves, so GANBLR != GANBLR-nAL.
        """
        if self.generator is None:
            raise RuntimeError("Generator not initialized")

        import tensorflow as tf

        N = X.shape[0]
        if rng_local is None:
            rng_local = rng

        data_real = np.column_stack([y, X]).astype(np.int32)

        # discriminator: P(is_real = 1 | row)
        if not hasattr(self, 'disc') or self.disc is None or self.disc is False:
            self.disc = _ohe_pipeline(
                LogisticRegression(max_iter=500, multi_class="ovr", solver="lbfgs"),
                step_name="lr")

        # generator parameter list (constrained -> softmax logits; else raw W/bias)
        gen = self.generator
        vars_ = ([gen.logits_y] + list(gen.logits_v.values())) if gen.constrained \
            else ([gen.bias] + list(gen.W.values()))

        # static tensors for regularisation
        l2_tf = tf.constant(1e-5, dtype=tf.float32)
        kl_lambda_tf = tf.constant(float(kl_lambda or 0.0), dtype=tf.float32)
        kl_targets_tf, kl_weights_tf = {}, {}
        if getattr(gen, "federated", False) and kl_lambda > 0.0 and kl_targets:
            for v, arr in kl_targets.items():
                kl_targets_tf[int(v)] = tf.constant(np.asarray(arr, np.float32))
            if kl_s_weights:
                for v, arr in kl_s_weights.items():
                    kl_weights_tf[int(v)] = tf.constant(np.asarray(arr, np.float32))

        log_prefix = f"[Client {client_id}]" if client_id else "[GANBLR]"
        if verbose > 0:
            print(f"{log_prefix} Starting adversarial training: {epochs} epochs")

        for ep in range(1, epochs + 1):
            if verbose > 0:
                print(f"{log_prefix} Adversarial epoch {ep}/{epochs}")

            # -------------------------------------------------------------
            # 1) Sample synthetic data  (Eq. 6:  S_data = G_bar(.))
            # -------------------------------------------------------------
            samp_rng = np.random.default_rng(seed_fn(ep)) if seed_fn is not None else rng
            syn_full = self._sample_full_from_gen(n=N, rng=samp_rng, order=self.order)
            X_fake = syn_full[:, 1:].astype(np.int32)

            # -------------------------------------------------------------
            # 2) Train discriminator  (Eq. 7)
            #    real rows -> label 1 (Y_d = 1),  synthetic rows -> label 0 (Y_d = 0)
            # -------------------------------------------------------------
            X_disc = np.vstack([X, X_fake]).astype(np.int32)
            y_disc = np.concatenate([np.ones(N, np.int32), np.zeros(N, np.int32)])
            # fresh D each round; warm-starting one D across rounds also works
            self.disc = _ohe_pipeline(
                LogisticRegression(max_iter=500, multi_class="ovr", solver="lbfgs"),
                step_name="lr")
            for _ in range(max(1, int(disc_epochs))):
                self.disc.fit(X_disc, y_disc)

            # -------------------------------------------------------------
            # 3) Turn D's verdict on the REAL rows into importance weights.
            #    This is the differentiable surrogate for -log(1 - D(S_data))
            #    in Eq. 2:   w(x) = D(x) / (1 - D(x)) = p_data(x) / p_g(x).
            # -------------------------------------------------------------
            d_real = self.disc.predict_proba(X)[:, 1]          # D(x) = P(real | x)
            d_real = np.clip(d_real, 1e-6, 1.0 - 1e-6)
            w = d_real / (1.0 - d_real)                        # likelihood ratio
            w = np.clip(w, 1e-3, 1e3)                          # guard against blow-up
            w = w / w.mean()                                   # mean 1: keep step scale
            probs = w / w.sum()                                # sampling distribution

            # -------------------------------------------------------------
            # 4) Generator step (Eq. 2, first term, D-reweighted).
            #    Rows are drawn ∝ w(x), so D directly shapes theta_g's gradient.
            #    _nll_batch returns mean NLL = -mean log P(Y|X); minimising it
            #    maximises the CLL term of Eq. 2.
            # -------------------------------------------------------------
            bsz = min(batch_size, N)
            n_steps = max(1, N // bsz)
            step_rng = np.random.default_rng(ep + 1000)

            disc_acc = 0.0
            for _ in range(n_steps):
                idx = step_rng.choice(N, size=bsz, replace=True, p=probs)
                x_tf = tf.convert_to_tensor(data_real[idx], dtype=tf.int32)

                with tf.GradientTape() as tape:
                    loss = gen._nll_batch(x_tf)                # -log P(Y|X) (weighted via resampling)

                    # L2 regularisation
                    if gen.constrained:
                        reg = tf.nn.l2_loss(gen.logits_y)
                        for v in gen.logits_v.values():
                            reg += tf.nn.l2_loss(v)
                    else:
                        reg = tf.nn.l2_loss(gen.bias)
                        for v in gen.W.values():
                            reg += tf.nn.l2_loss(v)
                    loss = loss + tf.multiply(l2_tf, reg)

                    # KL term (federated only) -- unchanged from your version
                    if getattr(gen, "federated", False) and kl_lambda > 0.0 and kl_targets_tf:
                        kl_loss = tf.constant(0.0, dtype=tf.float32)
                        for v, logits in gen.logits_v.items():
                            if int(v) not in kl_targets_tf:
                                continue
                            theta_local = tf.clip_by_value(tf.nn.softmax(logits, axis=-1), 1e-8, 1.0)
                            theta_tgt = tf.clip_by_value(kl_targets_tf[int(v)], 1e-8, 1.0)
                            W = kl_weights_tf.get(int(v), tf.ones_like(theta_local))
                            el = W * (theta_local * (tf.math.log(theta_local) - tf.math.log(theta_tgt)))
                            kl_loss += tf.reduce_mean(tf.reduce_sum(el, axis=1))
                        loss = loss + tf.multiply(kl_lambda_tf, kl_loss)

                    # NOTE: no additive disc_loss constant here -- the adversarial
                    # signal now lives in `probs`, which controls the gradient.

                grads = tape.gradient(loss, vars_)
                gen.opt.apply_gradients(zip(grads, vars_))

            if verbose > 0:
                disc_acc = float(np.mean((d_real >= 0.5).astype(np.int32) == 1))
                print(f"{log_prefix} Epoch {ep}/{epochs}: "
                      f"D-acc(real)={disc_acc:.4f}  w[min/mean/max]="
                      f"{w.min():.3f}/{w.mean():.3f}/{w.max():.3f}")
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

        # Initialize generator from MLE model
        mle = MLE_KDB(self.card, self.y_index, parents_excl_y, alpha = self.alpha)
        mle.fit(data)
        # DiscBN_KDB.__init__ already initializes parameters randomly
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
