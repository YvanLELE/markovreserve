from __future__ import annotations
import numpy as np
import pandas as pd
from .states import ABSORBING, portfolio_snapshot


def estimate_ibnr_counts(df: pd.DataFrame, valuation_year: int, max_dev: int) -> pd.DataFrame:
    tmp = df[['ID', 'AY', 'RepDel']].copy()
    tmp['ReportYear'] = tmp['AY'] + tmp['RepDel']
    tmp['Reported'] = tmp['ReportYear'] <= valuation_year
    rows = []
    # Empirical beta based on mature reported claims in the full available dataset
    beta = tmp['RepDel'].clip(0, max_dev - 1).value_counts(normalize=True).sort_index()
    cum_beta = beta.cumsum()
    for ay, sub in tmp.groupby('AY'):
        dev_obs = int(max(-1, min(max_dev - 1, valuation_year - ay)))
        if dev_obs < 0:
            continue
        obs = int(((sub['ReportYear'] <= valuation_year)).sum())
        prop = float(cum_beta.loc[cum_beta.index <= dev_obs].max()) if any(cum_beta.index <= dev_obs) else 0.0
        prop = max(prop, 1e-6)
        ultimate = obs / prop
        ibnr = max(0.0, ultimate - obs)
        rows.append({'AY': ay, 'ObservedReported': obs, 'DevObserved': dev_obs,
                     'ReportingProportion': prop, 'ExpectedUltimateCount': ultimate,
                     'ExpectedIBNR': ibnr})
    return pd.DataFrame(rows)


def simulate_one_path(state, duration, cum_paid, p1, transition_model, payment_model, rng,
                      max_steps: int = 30, one_year: bool = False):
    total_future = 0.0
    ultimate_cap = float(getattr(payment_model, 'ultimate_cap', np.inf))
    current_state = state
    current_duration = int(max(duration, 1))
    current_cum = float(cum_paid)
    first_payment = float(p1) if p1 and p1 > 0 else np.nan
    lambda_index = 1
    steps = 1 if one_year else max_steps
    for _ in range(steps):
        if current_state in ABSORBING:
            break
        nxt = transition_model.sample_next(current_state, current_duration, rng)
        pay = 0.0
        # Payment entry from IBNR/RBNP or paid closure before any prior payment
        if current_cum <= 0 and (nxt.startswith('RBNS') or nxt == 'Closed+'):
            pay = min(payment_model.sample_p1(rng), max(0.0, ultimate_cap - current_cum))
            first_payment = pay
            current_cum += pay
        elif current_cum > 0 and (nxt.startswith('RBNS') or nxt == 'Closed+'):
            lr = payment_model.sample_lr(lambda_index, first_payment if np.isfinite(first_payment) else current_cum, rng)
            new_cum = min(current_cum * lr, ultimate_cap)
            pay = max(0.0, new_cum - current_cum)
            current_cum = new_cum
            lambda_index += 1
        # Closed0: no terminal payment
        total_future += pay
        if current_cum >= ultimate_cap:
            current_state = 'Closed+'
            break
        if nxt == current_state:
            current_duration += 1
        else:
            current_state = nxt
            current_duration = 1
    return total_future, current_state, current_duration, current_cum, first_payment


def simulate_reserves(df, panel, valuation_year, transition_model, payment_model,
                      n_sims: int = 1000, seed: int = 123, include_ibnr: bool = True):
    rng = np.random.default_rng(seed)
    max_dev = int(panel['DY'].max() + 1)
    snap = portfolio_snapshot(panel, valuation_year)
    open_snap = snap[~snap['State'].isin(list(ABSORBING))].copy()
    # p1 observé par ID
    p1_map = panel[panel['Pay'] > 0].sort_values('DY').groupby('ID')['Pay'].first().to_dict()
    dur_map = panel.sort_values('DY').groupby('ID').tail(1).set_index('ID')['DY'].to_dict()

    ibnr_table = estimate_ibnr_counts(df, valuation_year, max_dev) if include_ibnr else pd.DataFrame()
    sim_totals = []
    sim_known = []
    sim_ibnr = []
    one_year_payments = []
    one_year_end_reserve_proxy = []

    for b in range(n_sims):
        known_total = 0.0
        oy_pay = 0.0
        oy_res_proxy = 0.0
        for _, r in open_snap.iterrows():
            cid = r['ID']
            p1 = p1_map.get(cid, np.nan)
            fut, _, _, _, _ = simulate_one_path(r['State'], 1, r['CumPay'], p1,
                                                transition_model, payment_model, rng, one_year=False)
            known_total += fut
            pay1, st1, dur1, cum1, p1_new = simulate_one_path(r['State'], 1, r['CumPay'], p1,
                                                              transition_model, payment_model, rng, one_year=True)
            oy_pay += pay1
            if st1 not in ABSORBING:
                # End-year reserve proxy: one conditional future simulation
                res1, *_ = simulate_one_path(st1, dur1, cum1, p1_new,
                                             transition_model, payment_model, rng, one_year=False)
                oy_res_proxy += res1
        ibnr_total = 0.0
        if include_ibnr and len(ibnr_table):
            for _, rr in ibnr_table.iterrows():
                n = rng.poisson(max(0.0, rr['ExpectedIBNR']))
                for _ in range(n):
                    fut, *_ = simulate_one_path('IBNR', max(1, int(rr['DevObserved'] + 1)), 0.0, np.nan,
                                                transition_model, payment_model, rng, one_year=False)
                    ibnr_total += fut
        sim_known.append(known_total)
        sim_ibnr.append(ibnr_total)
        sim_totals.append(known_total + ibnr_total)
        one_year_payments.append(oy_pay)
        one_year_end_reserve_proxy.append(oy_res_proxy)
    res = pd.DataFrame({'KnownReserve': sim_known, 'IBNRReserve': sim_ibnr, 'TotalReserve': sim_totals,
                        'OneYearPayments': one_year_payments, 'EndYearReserve': one_year_end_reserve_proxy})
    initial_reserve = res['TotalReserve'].mean()
    res['CDR'] = initial_reserve - res['OneYearPayments'] - res['EndYearReserve']
    return res, snap, ibnr_table


def risk_summary(x: pd.Series) -> dict:
    arr = np.asarray(x, dtype=float)
    q995 = np.quantile(arr, 0.995) if len(arr) else np.nan
    # For a reserve amount, TVaR uses the upper tail; for adverse CDR, the lower tail may be used depending on convention.
    tvar995 = arr[arr >= q995].mean() if len(arr) and np.any(arr >= q995) else np.nan
    return {
        'mean': float(np.mean(arr)), 'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'p50': float(np.quantile(arr, 0.50)), 'p75': float(np.quantile(arr, 0.75)),
        'p95': float(np.quantile(arr, 0.95)), 'p995': float(q995), 'tvar995': float(tvar995)
    }
