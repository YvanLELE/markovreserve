from __future__ import annotations
import numpy as np
import pandas as pd
from .data import pay_cols, open_cols, standardize_columns

PHYSICAL_STATES = ['IBNR', 'RBNP', 'RBNS1', 'RBNS2', 'RBNS3', 'RBNS4', 'RBNS5+', 'Closed0', 'Closed+']
ABSORBING = {'Closed0', 'Closed+'}


def _closure_year(pay_row: np.ndarray, open_row: np.ndarray | None) -> int:
    k = len(pay_row)
    if open_row is not None and len(open_row) == k:
        zeros = np.where(open_row == 0)[0]
        if len(zeros):
            return int(zeros[0])
    pos = np.where(pay_row > 0)[0]
    if len(pos):
        return min(k - 1, int(pos[-1]) + 1)
    return k - 1


def build_state_panel(df: pd.DataFrame, rbns_cap: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds the annual state panel and the augmented database.

    Convention: annual discrete time. The claim state is observed at the end of each development year.
    Closed0 = closure without terminal payment. Closed+ = closure with terminal payment.
    """
    df = standardize_columns(df)
    pcs = pay_cols(df)
    ocs = open_cols(df)
    if not pcs:
        raise ValueError('No Payj column detected.')
    max_dev = len(pcs)
    pay_mat = df[pcs].to_numpy(float)
    open_mat = df[ocs].to_numpy(int) if len(ocs) == max_dev else None

    state_rows = []
    aug_rows = []

    for idx, row in df.reset_index(drop=True).iterrows():
        claim_id = row['ID']
        ay = int(row['AY'])
        rep = int(max(0, min(max_dev - 1, row.get('RepDel', 0))))
        pays = pay_mat[idx]
        opens = open_mat[idx] if open_mat is not None else None
        closure = _closure_year(pays, opens)
        cum = np.cumsum(pays)
        states = []
        pos_pay_count = 0
        for dy in range(max_dev):
            if dy < rep:
                st = 'IBNR'
            elif dy >= closure:
                st = 'Closed+' if pays[closure] > 0 else 'Closed0'
            else:
                pos_pay_count = int(np.sum(pays[:dy + 1] > 0))
                if cum[dy] <= 0:
                    st = 'RBNP'
                else:
                    j = min(pos_pay_count, rbns_cap)
                    st = f'RBNS{j}' if j < rbns_cap else f'RBNS{rbns_cap}+'
            states.append(st)
            state_rows.append({
                'ID': claim_id, 'AY': ay, 'DY': dy, 'CalendarYear': ay + dy,
                'State': st, 'Pay': float(pays[dy]), 'CumPay': float(cum[dy]),
                'RepDel': rep, 'ClosureDY': closure,
            })

        # durations in current state at each dy
        durations = []
        d = 1
        for dy, st in enumerate(states):
            if dy > 0 and states[dy - 1] == st:
                d += 1
            else:
                d = 1
            durations.append(d)

        for dy in range(1, max_dev):
            prev, curr = states[dy - 1], states[dy]
            if prev in ABSORBING:
                # états absorbants : pas utile pour l'estimation des transitions futures
                continue
            any_transition = int(curr != prev)
            exit_closed = int(curr in ABSORBING and curr != prev)
            closed_plus = int(curr == 'Closed+' and curr != prev)
            # LR défini uniquement quand le cumul positif augmente après un cumul positif
            lr = np.nan
            if cum[dy - 1] > 0 and cum[dy] > cum[dy - 1]:
                lr = cum[dy] / cum[dy - 1]
            aug_rows.append({
                'ID': claim_id, 'AY': ay, 'DY': dy,
                'Previous_State': prev, 'Current_State': curr,
                'Previous_Duration': durations[dy - 1],
                'Duration': durations[dy],
                'AnyTransition': any_transition,
                'ExitToClosed': exit_closed,
                'ClosedWithPayment': closed_plus,
                'Pay': float(pays[dy]),
                'CumPay': float(cum[dy]),
                'CumPayLag': float(cum[dy - 1]),
                'LR': float(lr) if np.isfinite(lr) else np.nan,
                'RepDel': rep,
            })

    panel = pd.DataFrame(state_rows)
    aug = pd.DataFrame(aug_rows)
    return panel, aug


def portfolio_snapshot(panel: pd.DataFrame, valuation_year: int) -> pd.DataFrame:
    snap = panel[panel['CalendarYear'] <= valuation_year].sort_values(['ID', 'DY']).groupby('ID').tail(1)
    return snap.reset_index(drop=True)
