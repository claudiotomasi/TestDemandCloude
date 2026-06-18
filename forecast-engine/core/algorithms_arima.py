# =============================================================================
# DGS Forecast Engine — ARMA / ARIMA / SARIMA / SARIMAX
# Equivalente a: SarimaModel() in coreAlgorithmMA.m
# =============================================================================

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from core.kpi import compute_train_kpi, compute_test_kpi, merge_kpi, ci_poisson, kpi_to_array

DEFAULT_SEASON_LENGTH = 12


def sarima(
    history:       np.ndarray,
    mode:          str   = 'SARIMA',
    n_forecast:    int   = 12,
    season_length: int   = DEFAULT_SEASON_LENGTH,
    X_hist:        np.ndarray = None,
    X_fut:         np.ndarray = None,
    ts_number:     int   = 12,
    mase_period:   int   = 12,
    ci_alpha:      float = 0.05,
    ci_zeta:       float = 0.0,
    max_p:         int   = 5,
    max_q:         int   = 5,
    max_P:         int   = 2,
    max_Q:         int   = 2,
) -> dict:
    """
    ARMA / ARIMA / SARIMA / SARIMAX con selezione automatica degli ordini.
    Equivalente a SarimaModel() in coreAlgorithmMA.m.

    Modalità:
      'ARMA'    → d=0, D=0, no stagionalità
      'ARIMA'   → d=auto, D=0, no stagionalità
      'SARIMA'  → d=auto, D=auto, stagionalità automatica
      'SARIMAX' → come SARIMA + variabili esogene (X_hist, X_fut)
    """
    data = np.asarray(history, dtype=float).ravel()
    n    = len(data)

    if n < 4:
        return _fallback_arima(data, n_forecast, ts_number, mase_period,
                               ci_alpha, ci_zeta, reason='serie troppo corta')

    data_safe = np.where(data <= 0, 0.1, data)
    mode_upper = mode.upper()

    # --- Configurazione modello per modalità ---
    if mode_upper == 'ARMA':
        arima_model = AutoARIMA(
            d=0, D=0, max_p=max_p, max_q=max_q,
            max_P=0, max_Q=0, seasonal=False, season_length=1,
            ic='aicc', stepwise=True, allowdrift=False, allowmean=True,
            alias='ARIMA'
        )
    elif mode_upper == 'ARIMA':
        arima_model = AutoARIMA(
            D=0, max_p=max_p, max_q=max_q,
            max_P=0, max_Q=0, seasonal=False, season_length=1,
            ic='aicc', stepwise=True, allowdrift=True, allowmean=True,
            alias='ARIMA'
        )
    else:
        # SARIMA e SARIMAX
        arima_model = AutoARIMA(
            max_p=max_p, max_q=max_q, max_P=max_P, max_Q=max_Q,
            seasonal=True, season_length=season_length,
            ic='aicc', stepwise=True, allowdrift=True, allowmean=True,
            alias='ARIMA'
        )

    dates = pd.date_range(start='2020-01-01', periods=n, freq='MS')
    df = pd.DataFrame({'unique_id': ['s1'] * n, 'ds': dates, 'y': data_safe})

    use_exog = (mode_upper == 'SARIMAX'
                and X_hist is not None and X_fut is not None
                and len(X_hist) == n)

    sf = StatsForecast(models=[arima_model], freq='MS', n_jobs=1)

    try:
        if use_exog:
            X_h = np.atleast_2d(np.asarray(X_hist, dtype=float))
            X_f = np.atleast_2d(np.asarray(X_fut,  dtype=float))
            if X_h.shape[0] != n:
                X_h = X_h.T
            sf.fit(df, X=X_h)
            forecast_df = sf.predict(h=n_forecast, X=X_f)
        else:
            sf.fit(df)
            forecast_df = sf.predict(h=n_forecast)

        forecast_raw = forecast_df['ARIMA'].values
        forecast     = np.maximum(0, np.round(forecast_raw))

        # Estrai fitted values: x - residuals (struttura ARIMA)
        inner  = sf.fitted_[0, 0].model_
        x      = np.asarray(inner['x'],         dtype=float)
        resid  = np.asarray(inner['residuals'],  dtype=float)
        fitted_raw = x - resid
        fitted     = np.maximum(0, np.round(fitted_raw))

        # Ordini ARIMA(p,d,q)(P,D,Q)[m]
        arma = inner['arma']
        p, q, P, Q, m, d, D = (int(v) for v in arma[:7])
        method = f'ARIMA({p},{d},{q})({P},{D},{Q})[{m}]'

    except Exception as e:
        return _fallback_arima(data, n_forecast, ts_number, mase_period,
                               ci_alpha, ci_zeta, reason=str(e))

    train_kpi = compute_train_kpi(data, fitted_raw, mase_period)

    if n > ts_number:
        test_kpi = compute_test_kpi(data[-ts_number:], fitted_raw[-ts_number:])
    else:
        test_kpi = {'lnQ': float('nan')}

    kpi = merge_kpi(train_kpi, test_kpi)

    fore_all = np.concatenate([fitted, forecast])
    lower, upper = ci_poisson(fore_all, ci_type=False, alpha=ci_alpha, zeta=ci_zeta)

    return {
        'fitted':         fitted,
        'forecast':       forecast,
        'fore_all':       fore_all,
        'lower':          lower,
        'upper':          upper,
        'kpi':            kpi,
        'kpi_array':      kpi_to_array(kpi),
        'order':          (p, d, q),
        'seasonal_order': (P, D, Q, m),
        'method':         method,
    }


def _fallback_arima(data, n_forecast, ts_number, mase_period,
                    ci_alpha, ci_zeta, reason=''):
    from core.algorithms_ma import flat_forecast
    result = flat_forecast(
        data, lag=min(6, max(1, len(data) - 1)), mode=0,
        n_forecast=n_forecast, ts_number=ts_number,
        mase_period=mase_period, ci_alpha=ci_alpha, ci_zeta=ci_zeta
    )
    result['order']          = (0, 0, 0)
    result['seasonal_order'] = (0, 0, 0, 0)
    result['method']         = f'fallback_SMA ({reason})'
    return result