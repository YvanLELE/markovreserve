from __future__ import annotations
import numpy as np
import pandas as pd


def pay_cols(df: pd.DataFrame):
    return sorted([c for c in df.columns if c.startswith('Pay')], key=lambda x: int(x[3:]))


def open_cols(df: pd.DataFrame):
    return sorted([c for c in df.columns if c.startswith('Open')], key=lambda x: int(x[4:]))


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'ClNr' in df.columns and 'ID' not in df.columns:
        df = df.rename(columns={'ClNr': 'ID'})
    if 'ID' not in df.columns:
        df.insert(0, 'ID', np.arange(1, len(df) + 1))
    if 'AY' not in df.columns:
        raise ValueError("Column AY is required.")
    if 'RepDel' not in df.columns:
        df['RepDel'] = 0
    for c in pay_cols(df):
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    for c in open_cols(df):
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
    df['AY'] = pd.to_numeric(df['AY'], errors='coerce').astype(int)
    df['RepDel'] = pd.to_numeric(df['RepDel'], errors='coerce').fillna(0).astype(int)
    return df


def validate_portfolio(df: pd.DataFrame) -> dict:
    df = standardize_columns(df)
    pcs = pay_cols(df)
    ocs = open_cols(df)
    issues = []
    if not pcs:
        issues.append("No Payj column detected.")
    if (df[pcs] < 0).any().any():
        issues.append("Negative payments detected: check recoveries/subrogation.")
    if ocs:
        open_mat = df[ocs].to_numpy()
        reopened = 0
        for row in open_mat:
            closed_seen = False
            for v in row:
                if v == 0:
                    closed_seen = True
                elif closed_seen and v == 1:
                    reopened += 1
                    break
        if reopened:
            issues.append(f"{reopened} claims with reopening detected.")
    return {
        'n_claims': len(df),
        'pay_columns': len(pcs),
        'open_columns': len(ocs),
        'total_paid': float(df[pcs].sum().sum()) if pcs else 0.0,
        'ay_min': int(df['AY'].min()),
        'ay_max': int(df['AY'].max()),
        'issues': issues,
    }


def generate_demo_portfolio(n_claims: int = 5000, seed: int = 100, ay_start: int = 1994,
                            n_years: int = 12, max_dev: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ay_values = np.arange(ay_start, ay_start + n_years)
    ay = rng.choice(ay_values, size=n_claims, replace=True)
    # Annual reporting delay: many short delays, some late reports.
    repdel = np.minimum(max_dev - 1, rng.gamma(shape=1.25, scale=0.9, size=n_claims).astype(int))
    repdel = np.minimum(repdel, np.maximum(0, max_dev - 1))

    age = rng.choice(np.arange(18, 81), size=n_claims, replace=True,
                     p=np.ones(63) / 63)
    cc = rng.integers(1, 54, size=n_claims)
    inj_part = rng.integers(1, 100, size=n_claims)

    # Lognormal ultimate calibrated for a right tail.
    sigma = 1.55
    meanlog = np.log(8000) - sigma**2 / 2
    ultimate = rng.lognormal(meanlog, sigma, size=n_claims)
    ultimate = np.clip(ultimate, 50, 1_500_000)

    pays = np.zeros((n_claims, max_dev))
    opens = np.ones((n_claims, max_dev), dtype=int)

    for i in range(n_claims):
        start = int(repdel[i])
        # Probability of a small claim with no terminal payment or no payment
        if rng.random() < 0.06:
            closure = min(max_dev - 1, start + rng.integers(0, 3))
            opens[i, closure:] = 0
            continue
        n_pay = int(rng.choice([1, 2, 3, 4, 5], p=[0.35, 0.28, 0.18, 0.12, 0.07]))
        possible = np.arange(start, max_dev)
        if len(possible) == 0:
            possible = np.array([max_dev - 1])
        n_pay = min(n_pay, len(possible))
        pay_times = np.sort(rng.choice(possible, size=n_pay, replace=False))
        weights = rng.dirichlet(np.ones(n_pay) * 1.2)
        amounts = ultimate[i] * weights
        amounts = np.round(amounts, 2)
        # correction exacte du total
        amounts[-1] += round(ultimate[i] - amounts.sum(), 2)
        for t, a in zip(pay_times, amounts):
            pays[i, int(t)] += max(0.0, float(a))
        closure_lag = int(rng.choice([0, 1, 2], p=[0.60, 0.30, 0.10]))
        closure = min(max_dev - 1, int(pay_times[-1]) + closure_lag)
        opens[i, closure:] = 0
        # Terminal payment sometimes occurs at closure
        if closure not in pay_times and rng.random() < 0.35:
            terminal = round(max(10.0, rng.lognormal(np.log(500), 0.9)), 2)
            pays[i, closure] += terminal

    data = {
        'ID': np.arange(1, n_claims + 1),
        'LoB': 1,
        'cc': cc,
        'AY': ay,
        'AQ': rng.integers(1, 5, size=n_claims),
        'age': age,
        'inj_part': inj_part,
        'RepDel': repdel,
        'Ultimate': np.round(pays.sum(axis=1), 2),
    }
    for j in range(max_dev):
        data[f'Pay{j}'] = pays[:, j]
    for j in range(max_dev):
        data[f'Open{j}'] = opens[:, j]
    return pd.DataFrame(data)
