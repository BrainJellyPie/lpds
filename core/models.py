"""GBDT fitting and a uniform view of the fitted tree structure.

ModelView exposes, for scikit-learn, XGBoost and LightGBM:
  leaves(X)            leaf id per tree, shape (n, T)
  n_ids                number of leaf-id slots per tree
  stored_counts_flat() per-leaf counts stored in the model file
                       (scikit-learn n_node_samples, XGBoost cover,
                       LightGBM leaf_count); these are in-bag counts
  leaf_values_flat()   per-leaf output of each tree (shrinkage included)
  staged(X, iters)     predictions of the truncated ensembles
  active_feature_fraction(d)   fraction of features used in at least one split
"""
import numpy as np

from .config import GBDT_MAIN, LGBM_NUM_LEAVES


def fit_model(lib, X, y, seed, params=None, X_val=None, y_val=None,
              early_stopping=None):
    p = dict(GBDT_MAIN)
    if params:
        p.update(params)
    if lib == "sklearn":
        from sklearn.ensemble import GradientBoostingRegressor
        m = GradientBoostingRegressor(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"],
            learning_rate=p["learning_rate"], subsample=p["subsample"],
            random_state=seed)
        m.fit(X, y)
    elif lib == "xgb":
        import xgboost as xgb
        kw = dict(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                  min_child_weight=p["min_samples_leaf"],
                  learning_rate=p["learning_rate"], subsample=p["subsample"],
                  colsample_bytree=p.get("colsample", 1.0),
                  tree_method="hist", random_state=seed, n_jobs=1)
        if early_stopping:
            kw["early_stopping_rounds"] = early_stopping
        m = xgb.XGBRegressor(**kw)
        if early_stopping:
            m.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        else:
            m.fit(X, y)
    elif lib in ("lgbm", "lgbm_d4"):
        import lightgbm as lgb
        max_depth = p.get("lgbm_max_depth", -1)
        if lib == "lgbm_d4":
            max_depth = p["max_depth"]
        m = lgb.LGBMRegressor(
            n_estimators=p["n_estimators"],
            num_leaves=p.get("num_leaves", LGBM_NUM_LEAVES),
            max_depth=max_depth, min_child_samples=p["min_samples_leaf"],
            learning_rate=p["learning_rate"], subsample=p["subsample"],
            subsample_freq=1 if p["subsample"] < 1 else 0,
            colsample_bytree=p.get("colsample", 1.0),
            random_state=seed, n_jobs=1, verbose=-1)
        if early_stopping:
            m.fit(X, y, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
        else:
            m.fit(X, y)
    else:
        raise ValueError(lib)
    return ModelView(m, lib, p)


class ModelView:
    def __init__(self, model, lib, params):
        self.m = model
        self.lib = lib
        self.params = params
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        lib = self.lib
        if lib == "sklearn":
            ests = self.m.estimators_[:, 0]
            self.T = len(ests)
            self.n_ids = np.array([e.tree_.node_count for e in ests])
            lr = self.m.learning_rate
            # keep leaf entries only, so per-tree totals equal the in-bag size
            self._counts = [e.tree_.n_node_samples.astype(float) * (e.tree_.feature < 0)
                            for e in ests]
            self._values = [lr * e.tree_.value[:, 0, 0].astype(float)
                            for e in ests]
            self._is_leaf = [e.tree_.feature < 0 for e in ests]
            feats = set()
            for e in ests:
                f = e.tree_.feature
                feats.update(f[f >= 0].tolist())
            self._feats = feats
        elif lib == "xgb":
            booster = self.m.get_booster()
            best = getattr(self.m, "best_iteration", None)
            df = booster.trees_to_dataframe()
            if best is not None:
                df = df[df.Tree <= best]
            self.T = int(df.Tree.max()) + 1
            self.n_ids = np.zeros(self.T, dtype=int)
            self._counts, self._values, self._is_leaf = [], [], []
            feats = set()
            for t, g in df.groupby("Tree"):
                size = int(g.Node.max()) + 1
                c = np.zeros(size)
                v = np.zeros(size)
                isl = np.zeros(size, dtype=bool)
                lv = g[g.Feature == "Leaf"]
                c[lv.Node.to_numpy(int)] = lv.Cover.to_numpy(float)
                v[lv.Node.to_numpy(int)] = lv.Gain.to_numpy(float)
                isl[lv.Node.to_numpy(int)] = True
                self.n_ids[t] = size
                self._counts.append(c)
                self._values.append(v)
                self._is_leaf.append(isl)
                sp = g[g.Feature != "Leaf"].Feature.tolist()
                feats.update(int(s[1:]) if s.startswith("f") else s for s in sp)
            self._feats = feats
        else:  # lgbm
            booster = self.m.booster_
            best = getattr(self.m, "best_iteration_", None)
            df = booster.trees_to_dataframe()
            if best:
                df = df[df.tree_index < best]
            self.T = int(df.tree_index.max()) + 1
            self.n_ids = np.zeros(self.T, dtype=int)
            self._counts, self._values, self._is_leaf = [], [], []
            for t, g in df.groupby("tree_index"):
                lv = g[g.node_index.str.contains("-L")]
                ids = lv.node_index.str.split("-L").str[1].astype(int).to_numpy()
                size = int(ids.max()) + 1
                c = np.zeros(size)
                v = np.zeros(size)
                c[ids] = lv["count"].to_numpy(float)
                v[ids] = lv["value"].to_numpy(float)
                self.n_ids[t] = size
                self._counts.append(c)
                self._values.append(v)
                self._is_leaf.append(np.ones(size, dtype=bool))
            imp = booster.feature_importance("split")
            self._feats = set(np.nonzero(imp)[0].tolist())

    # ------------------------------------------------------------------
    def predict(self, X):
        if self.lib == "xgb" and getattr(self.m, "best_iteration", None) is not None:
            return self.m.predict(X, iteration_range=(0, self.T))
        if self.lib in ("lgbm", "lgbm_d4"):
            return self.m.predict(X, num_iteration=self.T)
        return self.m.predict(X)

    def leaves(self, X):
        if self.lib == "sklearn":
            L = self.m.apply(X)
        elif self.lib == "xgb":
            L = self.m.get_booster().predict(
                __import__("xgboost").DMatrix(X), pred_leaf=True,
                iteration_range=(0, self.T))
        else:
            L = self.m.predict(X, pred_leaf=True, num_iteration=self.T)
        return np.asarray(L).reshape(X.shape[0], -1)[:, :self.T].astype(np.int64)

    def stored_counts_flat(self):
        return np.concatenate(self._counts)

    def leaf_values_flat(self):
        return np.concatenate(self._values)

    def staged(self, X, iters):
        """Predictions of the ensembles truncated after each t in iters."""
        iters = list(iters)
        out = np.empty((len(iters), X.shape[0]))
        if self.lib == "sklearn":
            want = {t: k for k, t in enumerate(iters)}
            for t, pred in enumerate(self.m.staged_predict(X), start=1):
                if t in want:
                    out[want[t]] = pred
        elif self.lib == "xgb":
            for k, t in enumerate(iters):
                out[k] = self.m.predict(X, iteration_range=(0, t))
        else:
            for k, t in enumerate(iters):
                out[k] = self.m.predict(X, num_iteration=t)
        return out

    def active_feature_fraction(self, d):
        return len(self._feats) / float(d)
