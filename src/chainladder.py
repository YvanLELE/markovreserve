from __future__ import annotations
import numpy as np
import pandas as pd


def cumulative_triangle(panel: pd.DataFrame, valuation_year: int) -> pd.DataFrame:
    obs = panel[panel['CalendarYear'] <= valuation_year].copy()
    tri = obs.groupby(['AY', 'DY'])['Pay'].sum().unstack('DY').fillna(0).sort_index()
    tri = tri.cumsum(axis=1)
    # masquer les cellules futures selon diagonale
    for ay in tri.index:
        for dy in tri.columns:
            if ay + dy > valuation_year:
                tri.loc[ay, dy] = np.nan
    return tri


def chain_ladder_reserve(tri: pd.DataFrame):
    tri = tri.copy().astype(float)
    devs = list(tri.columns)
    factors = []
    for j in range(len(devs) - 1):
        c0 = tri[devs[j]]
        c1 = tri[devs[j + 1]]
        mask = c0.notna() & c1.notna() & (c0 > 0)
        f = c1[mask].sum() / c0[mask].sum() if mask.any() and c0[mask].sum() > 0 else 1.0
        factors.append(max(1.0, float(f)))
    ultimates = {}
    latest = {}
    latest_dev = {}
    pct_dev = {}
    for ay, row in tri.iterrows():
        obs = row.dropna()
        if len(obs) == 0:
            latest_val = 0.0
            latest_dev_idx = -1
        else:
            latest_val = float(obs.iloc[-1])
            latest_dev_idx = devs.index(obs.index[-1])
        prod = float(np.prod(factors[latest_dev_idx:])) if latest_dev_idx < len(factors) else 1.0
        pct = 1.0 / prod if prod > 0 else 1.0
        ultimates[ay] = latest_val * prod
        latest[ay] = latest_val
        latest_dev[ay] = latest_dev_idx
        pct_dev[ay] = pct
    out = pd.DataFrame({'AY': list(ultimates.keys()), 'LatestCumPaid': list(latest.values()),
                        'LatestDevIndex': [latest_dev[k] for k in ultimates.keys()],
                        'PctDeveloped': [pct_dev[k] for k in ultimates.keys()],
                        'UltimateCL': list(ultimates.values())})
    out['ReserveCL'] = out['UltimateCL'] - out['LatestCumPaid']
    return out, pd.Series(factors, name='DevelopmentFactor')


def benchmark_reserves(tri: pd.DataFrame, panel: pd.DataFrame, valuation_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes additional collective reserving benchmarks.

    Methods :
    - Deterministic Chain-Ladder.
    - Cape Cod: exposure = claim count by accident year, expected level calibrated by observed maturity.
    - Bornhuetter-Ferguson: ultimate prior = Cape Cod, reserve = prior * (1 - developed percentage).

    Note: where earned premiums or real exposures are not available, the exposure used is the number of claims
    by accident year. This is acceptable for demonstration purposes, but client work should use
    actual exposures or earned premiums.
    """
    cl, factors = chain_ladder_reserve(tri)
    counts = panel.groupby('AY')['ID'].nunique().reset_index(name='ExposureClaimCount')
    out = cl.merge(counts, on='AY', how='left')
    out['ExposureClaimCount'] = out['ExposureClaimCount'].fillna(1).clip(lower=1)
    # Cape Cod : Expected Ultimate per exposure = sum(latest) / sum(exposure * pct_dev)
    denom = (out['ExposureClaimCount'] * out['PctDeveloped']).sum()
    cc_rate = out['LatestCumPaid'].sum() / denom if denom > 0 else 0.0
    out['UltimateCapeCod'] = cc_rate * out['ExposureClaimCount']
    out['ReserveCapeCod'] = (out['UltimateCapeCod'] - out['LatestCumPaid']).clip(lower=0)
    # Bornhuetter-Ferguson with Cape Cod prior
    out['UltimateBF'] = out['LatestCumPaid'] + out['UltimateCapeCod'] * (1 - out['PctDeveloped'])
    out['ReserveBF'] = (out['UltimateCapeCod'] * (1 - out['PctDeveloped'])).clip(lower=0)
    summary = pd.DataFrame({
        'Method': ['Chain-Ladder', 'Cape Cod', 'Bornhuetter-Ferguson'],
        'Reserve': [out['ReserveCL'].sum(), out['ReserveCapeCod'].sum(), out['ReserveBF'].sum()],
        'Key assumption': [
            'Future development based on historical triangle factors',
            'Expected ultimate level calibrated by exposure and observed maturity',
            'Cape Cod prior weighted by the undeveloped proportion'
        ]
    })
    return summary, out
