# =============================================================================
# DGS Forecast Engine — Test Algoritmi MA
# =============================================================================

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.algorithms_ma import sma, ewma, batch, excel_like, linreg, flat_forecast
from core.kpi import compute_train_kpi, ci_poisson

HISTORY = np.array([
    120, 135, 128, 142, 118, 155, 162, 148, 170, 158, 145, 180,
    190, 175, 200, 188, 165, 210, 205, 195, 215, 220, 198, 235,
    240, 225, 250, 238, 215, 260, 255, 245, 265, 270, 248, 280
], dtype=float)

LAG         = 6
N_FORECAST  = 12
MASE_PERIOD = 12


def separator(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


def print_kpi(kpi):
    print(f"  MASE : {kpi['MASE']:.4f}  (< 1.0 = meglio del naïve)")
    print(f"  lnQ  : {kpi['lnQ']:.4f}   (vicino a 0 = ottimo)")
    print(f"  MAE  : {kpi['MAE']:.2f}")
    print(f"  RMSE : {kpi['RMSE']:.2f}")
    print(f"  R2   : {kpi['Rsq']:.4f}")


separator("1. SMA — Simple Moving Average (mode=0)")
result = flat_forecast(HISTORY, lag=LAG, mode=0, n_forecast=N_FORECAST, mase_period=MASE_PERIOD)
print(f"  Fitted ultimi 6 : {result['fitted'][-6:]}")
print(f"  Forecast 12m    : {result['forecast']}")
lo3 = result['lower'][-3:].astype(int)
hi3 = result['upper'][-3:].astype(int)
print(f"  CI lower ultimi 3: {lo3}")
print(f"  CI upper ultimi 3: {hi3}")
print_kpi(result['kpi'])

separator("2. EWMA — Exp. Weighted MA (mode=1)")
result = flat_forecast(HISTORY, lag=LAG, mode=1, n_forecast=N_FORECAST, mase_period=MASE_PERIOD)
print(f"  Fitted ultimi 6 : {result['fitted'][-6:]}")
print(f"  Forecast 12m    : {result['forecast']}")
print_kpi(result['kpi'])

separator("3. BATCH — MA Batch (mode=-1)")
result = flat_forecast(HISTORY, lag=LAG, mode=-1, n_forecast=N_FORECAST, mase_period=MASE_PERIOD)
print(f"  Fitted ultimi 6 : {result['fitted'][-6:]}")
print(f"  Forecast 12m    : {result['forecast']}")
print_kpi(result['kpi'])

separator("4. EXCEL-LIKE — MA Rolling (mode=2)")
result = flat_forecast(HISTORY, lag=LAG, mode=2, n_forecast=N_FORECAST, mase_period=MASE_PERIOD)
print(f"  Fitted ultimi 6 : {result['fitted'][-6:]}")
print(f"  Forecast 12m    : {result['forecast']}")
print_kpi(result['kpi'])

separator("5. LINREG — Regressione Lineare (mode=3)")
result = flat_forecast(HISTORY, lag=LAG, mode=3, n_forecast=N_FORECAST, mase_period=MASE_PERIOD)
print(f"  Fitted ultimi 6 : {result['fitted'][-6:]}")
print(f"  Forecast 12m    : {result['forecast']}")
print_kpi(result['kpi'])

separator("6. Confronto MASE tra algoritmi")
modes = {'SMA': 0, 'EWMA': 1, 'BATCH': -1, 'EXCEL-LIKE': 2, 'LINREG': 3}
results = {}
for name, mode in modes.items():
    results[name] = flat_forecast(HISTORY, lag=LAG, mode=mode,
                                  n_forecast=N_FORECAST, mase_period=MASE_PERIOD)

mase_values = {n: r['kpi']['MASE'] for n, r in results.items() if not np.isnan(r['kpi']['MASE'])}
best = min(mase_values, key=mase_values.get) if mase_values else None

print(f"\n  {'Algoritmo':<14} {'MASE':>8}  {'lnQ':>8}")
print(f"  {'-'*34}")
for name, r in results.items():
    kpi = r['kpi']
    flag = " <- BEST" if name == best else ""
    mase_str = f"{kpi['MASE']:.4f}" if not np.isnan(kpi['MASE']) else "     NaN"
    lnq_str  = f"{kpi['lnQ']:.4f}"  if not np.isnan(kpi['lnQ'])  else "     NaN"
    print(f"  {name:<14} {mase_str:>8}  {lnq_str:>8}{flag}")

separator("7. Test serie corta (6 valori)")
short = np.array([50, 48, 55, 52, 58, 54], dtype=float)
r_short = flat_forecast(short, lag=3, mode=0, n_forecast=12, mase_period=1)
print(f"  Serie    : {short}")
print(f"  Forecast : {r_short['forecast']}")
mase_s = r_short['kpi']['MASE']
print(f"  MASE: {mase_s:.4f}" if not np.isnan(mase_s) else "  MASE: NaN")

separator("8. Test CI Poisson")
vals = np.array([100., 110., 105., 115., 108.])
lo, hi = ci_poisson(vals, ci_type=False, alpha=0.05, zeta=0.0)
print(f"  Valori  : {vals}")
print(f"  CI lower: {lo.astype(int)}")
print(f"  CI upper: {hi.astype(int)}")

print(f"\n{'='*55}")
print("  Tutti i test completati.")
print('='*55)