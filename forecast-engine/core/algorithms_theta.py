# =============================================================================
# DGS Forecast Engine — AutoTheta
# -----------------------------------------------------------------------------
# AutoTheta è il metodo che ha vinto la competizione M4 Forecasting (2018)
# su 100.000 serie reali eterogenee. Supera ARIMA e HW nella maggior parte
# delle serie con trend + stagionalità moderata.
#
# QUANDO USARLO (vs altri algoritmi):
#   ✓ Serie con trend chiaro ma stagionalità non fortissima
#   ✓ Serie di medio-breve storia (18-48 mesi)
#   ✓ Come alternativa veloce a SARIMA quando si vuole evitare
#     la ricerca degli ordini p,d,q
#   ✗ Non ideale per domanda intermittente (usa Croston)
#   ✗ Non ideale per serie con stagionalità molto marcata (usa HW/SARIMA)
#
# PARAMETRI IN INGRESSO (dal backend C#):
#   history        : array float, serie storica ordine vecchio→recente
#   n_forecast     : int, periodi futuri (default 12)
#   season_length  : int, lunghezza stagione (12=mensile, 4=trimestrale)
#   decomposition  : str, 'multiplicative' o 'additive'
#                    - 'multiplicative': default, funziona meglio con trend
#                      e stagionalità proporzionale ai valori
#                    - 'additive': usa quando la stagionalità è costante
#                      indipendentemente dal livello della serie
#   ts_number      : int, periodi hold-out per lnQ (default 12)
#   mase_period    : int, periodo MASE (default 12)
#   ci_alpha       : float, livello CI Poisson (default 0.05 = 95%)
#   ci_zeta        : float, parametro aggiuntivo CI (default 0.0)
#
# OUTPUT (identico agli altri moduli):
#   fitted, forecast, fore_all, lower, upper, kpi, kpi_array, method
# =============================================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from statsforecast import StatsForecast
from statsforecast.models import AutoTheta
from core.kpi import compute_train_kpi, compute_test_kpi, merge_kpi, ci_poisson, kpi_to_array


def theta_forecast(
    history:        np.ndarray,
    n_forecast:     int   = 12,
    season_length:  int   = 12,
    decomposition:  str   = 'multiplicative',
    ts_number:      int   = 12,
    mase_period:    int   = 12,
    ci_alpha:       float = 0.05,
    ci_zeta:        float = 0.0,
) -> dict:
    """
    AutoTheta — selezione automatica del modello Theta ottimale.

    Internamente prova le varianti:
      - SES  (Simple Exponential Smoothing)
      - DOT  (Damped Trend)
      - OTM  (Optimized Theta Model)  ← quasi sempre il migliore
      - DSES (Damped SES)
    e sceglie quella con AIC minimo.

    Parametri
    ---------
    history       : serie storica (ordine vecchio→recente)
    n_forecast    : periodi futuri da prevedere
    season_length : lunghezza stagione (12=mensile, 4=trimestrale, 1=nessuna)
    decomposition : 'multiplicative' (default) o 'additive'
    ts_number     : periodi hold-out per lnQ
    mase_period   : periodo per calcolo MASE
    ci_alpha      : livello significatività CI Poisson
    ci_zeta       : parametro aggiuntivo banda CI

    Ritorna
    -------
    dict con:
        fitted      : array in-sample (lunghezza = len(history))
        forecast    : array forecast futuro (lunghezza = n_forecast)
        fore_all    : fitted + forecast concatenati
        lower       : CI inferiore (Poisson)
        upper       : CI superiore (Poisson)
        kpi         : dict {MASE, lnQ, MAE, RMSE, R, Rsq, MSE, ErrorMean, ErrorStd, Num}
        kpi_array   : array numpy [MSE,RMSE,MAE,lnQ,ErrorMean,ErrorStd,R,Rsq,Num,MASE]
        method      : stringa descrittiva (es. 'Theta-OTM')
        theta_model : variante Theta selezionata (SES/DOT/OTM/DSES)
    """
    data = np.asarray(history, dtype=float).ravel()
    n    = len(data)

    if n < 4:
        return _fallback(data, n_forecast, ts_number, mase_period,
                         ci_alpha, ci_zeta, reason='serie troppo corta')

    data_safe = np.where(data <= 0, 0.1, data)
    dates = pd.date_range(start='2020-01-01', periods=n, freq='MS')
    df = pd.DataFrame({'unique_id': ['s1'] * n, 'ds': dates, 'y': data_safe})

    model = AutoTheta(
        season_length=season_length,
        decomposition_type=decomposition,
        alias='Theta',
    )
    sf = StatsForecast(models=[model], freq='MS', n_jobs=1)

    try:
        sf.fit(df)
        forecast_df  = sf.predict(h=n_forecast)
        forecast_raw = forecast_df['Theta'].values
        forecast     = np.maximum(0, np.round(forecast_raw))

        inner      = sf.fitted_[0, 0].model_
        fitted_raw = np.asarray(inner['fitted'], dtype=float)
        fitted     = np.maximum(0, np.round(fitted_raw))
        theta_model = inner.get('modeltype', 'OTM')

    except Exception as e:
        return _fallback(data, n_forecast, ts_number, mase_period,
                         ci_alpha, ci_zeta, reason=str(e))

    train_kpi = compute_train_kpi(data, fitted_raw, mase_period)
    test_kpi  = compute_test_kpi(data[-ts_number:], fitted_raw[-ts_number:]) \
                if n > ts_number else {'lnQ': float('nan')}
    kpi = merge_kpi(train_kpi, test_kpi)

    fore_all = np.concatenate([fitted, forecast])
    lower, upper = ci_poisson(fore_all, ci_type=False, alpha=ci_alpha, zeta=ci_zeta)

    return {
        'fitted':      fitted,
        'forecast':    forecast,
        'fore_all':    fore_all,
        'lower':       lower,
        'upper':       upper,
        'kpi':         kpi,
        'kpi_array':   kpi_to_array(kpi),
        'method':      f'Theta-{theta_model}',
        'theta_model': theta_model,
    }


def _fallback(data, n_forecast, ts_number, mase_period, ci_alpha, ci_zeta, reason=''):
    from core.algorithms_ma import flat_forecast
    r = flat_forecast(data, lag=min(6, max(1, len(data)-1)), mode=0,
                      n_forecast=n_forecast, ts_number=ts_number,
                      mase_period=mase_period, ci_alpha=ci_alpha, ci_zeta=ci_zeta)
    r['method']      = 'fallback_SMA'
    r['theta_model'] = 'N/A'
    r['fallback']    = reason
    return r
