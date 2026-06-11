from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import norm


class LogNormalMixture:
    def __init__(self, model: GaussianMixture | None, fallback_mean: float = 1000.0, cap: float | None = None):
        self.model = model
        self.fallback_mean = float(max(fallback_mean, 1e-6))
        self.cap = float(cap) if cap is not None and cap > 0 else None

    @classmethod
    def fit(cls, values, max_components: int = 2, seed: int = 123):
        x = np.asarray(values, dtype=float)
        x = x[np.isfinite(x) & (x > 0)]
        if len(x) < 20:
            return cls(None, float(np.nanmean(x)) if len(x) else 1000.0, float(np.nanquantile(x, 0.995)) if len(x) else None)
        z = np.log(x).reshape(-1, 1)
        best = None
        best_bic = np.inf
        for k in range(1, max_components + 1):
            gm = GaussianMixture(n_components=k, random_state=seed, covariance_type='full')
            gm.fit(z)
            bic = gm.bic(z)
            if bic < best_bic:
                best_bic, best = bic, gm
        obj = cls(best, float(np.mean(x)), float(np.nanquantile(x, 0.995)))
        obj.bic_ = best_bic
        obj.n_obs_ = len(x)
        return obj

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.model is None:
            vals = rng.lognormal(np.log(self.fallback_mean) - 0.5, 1.0, size=n)
            return np.minimum(vals, self.cap) if self.cap else vals
        comp = rng.choice(np.arange(self.model.n_components), size=n, p=self.model.weights_)
        vals = []
        for c in comp:
            mean = float(self.model.means_[c, 0])
            sd = float(np.sqrt(self.model.covariances_[c, 0, 0]))
            vals.append(rng.lognormal(mean, sd))
        arr = np.asarray(vals)
        return np.minimum(arr, self.cap) if self.cap else arr

    def cdf(self, values) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        out = np.zeros_like(x, dtype=float)
        mask = np.isfinite(x) & (x > 0)
        if not np.any(mask):
            return out
        z = np.log(x[mask])
        if self.model is None:
            # Approximation de repli : lognormale de moyenne fallback_mean et sigma=1
            mean = np.log(self.fallback_mean) - 0.5
            sd = 1.0
            out[mask] = norm.cdf((z - mean) / sd)
            return np.clip(out, 0, 1)
        cdf = np.zeros_like(z, dtype=float)
        for w, mean, cov in zip(self.model.weights_, self.model.means_.ravel(), self.model.covariances_.reshape(self.model.n_components, -1)):
            sd = float(np.sqrt(cov[0]))
            cdf += float(w) * norm.cdf((z - float(mean)) / max(sd, 1e-9))
        out[mask] = cdf
        return np.clip(out, 0, 1)

    def mean(self) -> float:
        if self.model is None:
            return self.fallback_mean
        m = 0.0
        for w, mean, cov in zip(self.model.weights_, self.model.means_.ravel(), self.model.covariances_.reshape(self.model.n_components, -1)):
            var = float(cov[0])
            m += float(w) * np.exp(float(mean) + 0.5 * var)
        return float(m)

    def summary(self) -> dict:
        if self.model is None:
            return {'components': 0, 'mean': self.fallback_mean, 'bic': np.nan, 'n_obs': 0}
        return {
            'components': int(self.model.n_components),
            'mean': self.mean(),
            'bic': getattr(self, 'bic_', np.nan),
            'n_obs': getattr(self, 'n_obs_', np.nan),
            'cap': self.cap,
            'weights': self.model.weights_.round(4).tolist(),
            'log_means': self.model.means_.ravel().round(4).tolist(),
            'log_sds': np.sqrt(self.model.covariances_.reshape(self.model.n_components, -1)[:, 0]).round(4).tolist(),
        }


class PaymentModel:
    def __init__(self, p1_model: LogNormalMixture, lr_models: dict, threshold: float):
        self.p1_model = p1_model
        self.lr_models = lr_models
        self.threshold = threshold

    @classmethod
    def fit(cls, df, panel, threshold: float = 50_000, max_components: int = 2, seed: int = 123):
        # P1: first strictly positive payment by claim
        p1_values = []
        p1_by_id = {}
        for cid, sub in panel.sort_values('DY').groupby('ID'):
            pos = sub.loc[sub['Pay'] > 0, 'Pay']
            if len(pos):
                p1 = float(pos.iloc[0])
                p1_values.append(p1)
                p1_by_id[cid] = p1
        p1_model = LogNormalMixture.fit(p1_values, max_components, seed)

        # LR : ratios de cumul positif aux dates d'augmentation du cumul
        lr_models = {}
        lr_rows = []
        for cid, sub in panel.sort_values('DY').groupby('ID'):
            pays = sub['Pay'].to_numpy(float)
            cum = np.cumsum(pays)
            event_idx = np.where(pays > 0)[0]
            if len(event_idx) < 2:
                continue
            p1 = p1_by_id.get(cid, np.nan)
            cls_hi = 'high' if p1 >= threshold else 'low'
            prev_cum = cum[event_idx[0]]
            for k, idx in enumerate(event_idx[1:], start=1):
                if prev_cum > 0 and cum[idx] > prev_cum:
                    lr = cum[idx] / prev_cum
                    lr_rows.append({'LambdaIndex': min(k, 5), 'Class': cls_hi, 'LR': lr})
                    prev_cum = cum[idx]
        lr_df = pd.DataFrame(lr_rows)
        for j in range(1, 6):
            for cls_name in ['low', 'high']:
                vals = lr_df.loc[(lr_df['LambdaIndex'] == j) & (lr_df['Class'] == cls_name), 'LR'] if len(lr_df) else []
                lr_models[(j, cls_name)] = LogNormalMixture.fit(vals, max_components, seed + j)
            # fallback all classes
            vals_all = lr_df.loc[lr_df['LambdaIndex'] == j, 'LR'] if len(lr_df) else []
            lr_models[(j, 'all')] = LogNormalMixture.fit(vals_all, max_components, seed + j)
        obj = cls(p1_model, lr_models, threshold)
        ult = panel.groupby('ID')['CumPay'].max().to_numpy(float)
        ult = ult[np.isfinite(ult) & (ult > 0)]
        obj.ultimate_cap = float(np.nanquantile(ult, 0.995) * 1.5) if len(ult) else float(p1_model.mean() * 20)
        return obj

    def sample_p1(self, rng: np.random.Generator) -> float:
        return float(self.p1_model.sample(1, rng)[0])

    def sample_lr(self, lambda_index: int, p1: float, rng: np.random.Generator) -> float:
        j = int(min(max(lambda_index, 1), 5))
        cls_name = 'high' if p1 >= self.threshold else 'low'
        model = self.lr_models.get((j, cls_name)) or self.lr_models.get((j, 'all'))
        val = float(model.sample(1, rng)[0])
        # Cap actuariel de robustesse : évite des ultimes irréalistes dus aux queues de mélange.
        return min(max(1.0, val), 3.0)

    def summary_table(self) -> pd.DataFrame:
        rows = [{'Variable': 'P1', 'Class': 'all', **self.p1_model.summary()}]
        for (j, cls), model in self.lr_models.items():
            if cls == 'all':
                continue
            rows.append({'Variable': f'Lambda{j}', 'Class': cls, **model.summary()})
        return pd.DataFrame(rows)
