from __future__ import annotations
import numpy as np
import pandas as pd
from .states import PHYSICAL_STATES, ABSORBING


def duration_cat(d: int, cap: int = 2) -> str:
    return str(d) if d < cap else f'{cap}+'


class TransitionModel:
    def __init__(self, probs: dict, smoothing: float = 0.5, dur_cap: int = 2):
        self.probs = probs
        self.smoothing = smoothing
        self.dur_cap = dur_cap

    @classmethod
    def fit(cls, aug: pd.DataFrame, smoothing: float = 0.5, dur_cap: int = 2,
            force_no_ibnr_to_ferme0: bool = True):
        data = aug.copy()
        data['Dcat'] = data['Previous_Duration'].apply(lambda x: duration_cat(int(x), dur_cap))
        probs = {}
        states = [s for s in PHYSICAL_STATES if s not in ABSORBING]
        destinations = PHYSICAL_STATES
        for st in states:
            for dcat in sorted(data.loc[data['Previous_State'] == st, 'Dcat'].unique().tolist() or ['1', f'{dur_cap}+']):
                sub = data[(data['Previous_State'] == st) & (data['Dcat'] == dcat)]
                counts = sub['Current_State'].value_counts().to_dict()
                weights = {}
                for dest in destinations:
                    if dest == 'Closed0' and st == 'IBNR' and force_no_ibnr_to_ferme0:
                        weights[dest] = 0.0
                    else:
                        weights[dest] = counts.get(dest, 0.0) + smoothing
                # transitions impossibles simples : ne pas revenir vers IBNR sauf rester IBNR
                if st != 'IBNR':
                    weights['IBNR'] = 0.0
                total = sum(weights.values())
                if total <= 0:
                    weights = {st: 1.0}
                else:
                    weights = {k: v / total for k, v in weights.items() if v > 0}
                probs[(st, dcat)] = weights
        # fallback par état
        for st in states:
            sub = data[data['Previous_State'] == st]
            counts = sub['Current_State'].value_counts().to_dict()
            weights = {dest: counts.get(dest, 0.0) + smoothing for dest in destinations}
            if st == 'IBNR' and force_no_ibnr_to_ferme0:
                weights['Closed0'] = 0.0
            if st != 'IBNR':
                weights['IBNR'] = 0.0
            total = sum(weights.values())
            probs[(st, 'fallback')] = {k: v / total for k, v in weights.items() if v > 0} if total else {st: 1.0}
        return cls(probs, smoothing, dur_cap)

    def next_probs(self, state: str, duration: int) -> dict:
        if state in ABSORBING:
            return {state: 1.0}
        key = (state, duration_cat(duration, self.dur_cap))
        return self.probs.get(key) or self.probs.get((state, 'fallback')) or {state: 1.0}

    def sample_next(self, state: str, duration: int, rng: np.random.Generator) -> str:
        p = self.next_probs(state, duration)
        states = list(p.keys())
        probs = np.array(list(p.values()), dtype=float)
        probs = probs / probs.sum()
        return str(rng.choice(states, p=probs))

    def table(self) -> pd.DataFrame:
        rows = []
        for (st, dcat), dist in self.probs.items():
            if dcat == 'fallback':
                continue
            for dest, p in dist.items():
                rows.append({'Previous_State': st, 'DurationCat': dcat, 'Current_State': dest, 'Probability': p})
        return pd.DataFrame(rows).sort_values(['Previous_State', 'DurationCat', 'Current_State'])
