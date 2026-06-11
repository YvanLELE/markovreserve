# MarkovReserve

**MarkovReserve** is a client-ready Streamlit application for individual non-life claims reserving using a semi-Markov multi-state framework.

It models claim trajectories through IBNR, RBNP, RBNS and closed states, estimates reserves through Monte Carlo simulation, computes one-year CDR, and compares results with collective reserving benchmarks.

## Main features

- CSV and Excel import.
- Demo claim portfolio generation.
- Automatic construction of states: `IBNR`, `RBNP`, `RBNS1..RBNS5+`, `Closed0`, `Closed+`.
- Duration-dependent transition modelling.
- Payment modelling using lognormal mixtures for first payment `P1` and link ratios `Lambda_j`.
- Monte Carlo reserve simulation.
- One-year CDR, VaR and TVaR.
- Benchmarks: Individual Semi-Markov, Chain-Ladder, Cape Cod, Bornhuetter-Ferguson.
- Sensitivity analysis.
- Stress testing and quick backtesting.
- Theory and methodology explanations for non-actuarial users.
- Stress testing and quick backtesting.
- Excel and PDF exports.
- Client metadata, currency selection, inflation and discounting approximation.

## Expected input format

Minimum columns:

- `ID` or `ClNr`: unique claim identifier.
- `AY`: accident / occurrence year.
- `Pay0`, ..., `PayK`: payments by development year.

Recommended columns:

- `RepDel`: reporting delay in development years (`0` = reported in accident year).
- `Open0`, ..., `OpenK`: open/closed indicator at end of development (`1` open, `0` closed).

Optional columns:

- `LoB`, `cc`, `age`, `inj_part`, `AQ`.

A template file is provided in `data/template_import_sinistres.xlsx`.

## Business conventions

- `Closed0` = closure without terminal payment.
- `Closed+` = closure with terminal payment.
- Closed states are absorbing.
- CDR = opening reserve - one-year payments - closing reserve.

## Local launch

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.fileWatcherType=none
```

## Deployment on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload the contents of this folder at the root of the repository.
3. Go to https://streamlit.io/cloud.
4. Create a new app.
5. Select the GitHub repository and set the main file path to `app.py`.
6. Deploy.

## Confidentiality note

The public demo should only use simulated or anonymised data. Do not upload confidential client data to a public Streamlit deployment. For confidential use, deploy on a secured private environment with authentication and HTTPS.

## Actuarial note

MarkovReserve is a decision-support prototype. Results must be reviewed and validated by a qualified actuary before any regulatory, accounting or business decision use.
