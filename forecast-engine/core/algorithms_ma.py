# =============================================================================
# DGS Forecast Engine — Algoritmi Moving Average
# Equivalente a: FlatFore() e le sue sotto-funzioni in coreAlgorithmMA.m
#
# Algoritmi implementati:
#   - SMA          : Simple Moving Average         (foreEWMA = 0)
#   - EWMA         : Exp. Weighted Moving Average  (foreEWMA = 1)
#   - BATCH        : Moving Average Batch          (foreEWMA = -1)
#   - EXCEL_LIKE   : MA Excel-Like rolling         (foreEWMA = 2)
#   - LINREG       : Regressione Lineare (Trend)   (foreEWMA = 3)
#
# Ogni funzione accetta la serie storica (array 1D, ordine cronologico
# dal più vecchio al più recente) e il parametro lag, e restituisce
# il vettore fitted (in-sample) + forecast (futuri).
# =============================================================================

import numpy as np
from core.kpi import compute_train_kpi, compute_test_kpi, merge_kpi, ci_poisson, kpi_to_array


# =============================================================================
# PREPROCESSING — equivalente a dataPreparation() in Matlab
# =============================================================================

def prepare_series(history: np.ndarray, tr_number: int = None) -> np.ndarray:
    """
    Prepara la serie storica:
    - Tronca a tr_number se specificato
    - Rimuove NaN finali (trimming)
    - Sostituisce NaN interni con 0.2 (identico a Matlab)
    - Restituisce array float 1D in ordine cronologico (vecchio → recente)
    """
    data = np.asarray(history, dtype=float).ravel()

    # Tronca a tr_number
    if tr_number is not None and len(data) > tr_number:
        data = data[:tr_number]

    # Trimming NaN finali (identico al loop Matlab)
    k = len(data)
    for j in range(len(data) - 1, -1, -1):
        if np.isnan(data[j]):
            k -= 1
        else:
            break
    data = data[:k]

    # Sostituisce NaN interni con 0.2 (scelta Matlab: né 0 né media)
    data = np.where(np.isnan(data), 0.2, data)

    return data


def adapt_lag(lag: int, series_length: int) -> int:
    """Clamp lag nell'intervallo [1, series_length-1] come in Matlab."""
    lag = max(1, round(lag))
    if lag >= series_length:
        lag = max(1, series_length - 1)
    return lag


def adapt_mase_period(period: int, series_length: int) -> int:
    """Clamp masePeriod come in Matlab."""
    period = max(1, round(period))
    if period >= series_length:
        period = max(1, series_length - 1)
    return period


# =============================================================================
# SMA — Simple Moving Average
# Equivalente a sma() in Matlab
# =============================================================================

def sma(series: np.ndarray, lag: int) -> np.ndarray:
    """
    Calcola la Simple Moving Average in-sample.
    Output: array della stessa lunghezza della serie,
            NaN per i primi lag-1 valori (identico a Matlab).
    """
    n   = len(series)
    lag = adapt_lag(lag, n)
    out = np.full(n, np.nan)

    kernel = np.ones(lag) / lag
    ma     = np.convolve(series, kernel, mode='full')[:n]
    out[lag - 1:] = ma[lag - 1:]
    return out


# =============================================================================
# EWMA — Exponentially Weighted Moving Average
# Equivalente a ewma() in Matlab
# =============================================================================

def ewma(series: np.ndarray, lag: int) -> np.ndarray:
    """
    Calcola l'EWMA in-sample con k = 2/(lag+1).
    Identico alla funzione ewma() in Matlab.
    """
    n   = len(series)
    lag = adapt_lag(lag, n)
    out = np.full(n, np.nan)

    k = 2.0 / (lag + 1)

    # Inizializzazione: media semplice dei primi lag valori
    out[lag - 1] = np.mean(series[:lag])
    # Prima applicazione della formula EWMA
    out[lag - 1] = series[lag - 1] * k + out[lag - 1] * (1 - k)

    # Iterazione ricorsiva
    for i in range(lag, n):
        out[i] = series[i] * k + out[i - 1] * (1 - k)

    return out


# =============================================================================
# BATCH — Moving Average Batch (media fissa sugli ultimi lag periodi)
# Equivalente a batcha() in Matlab
# =============================================================================

def batch(series: np.ndarray, lag: int) -> np.ndarray:
    """
    Media degli ultimi lag valori della serie, costante per tutti i periodi
    nell'intervallo [n-lag, n-1].
    Identico a batcha() in Matlab.
    """
    n   = len(series)
    lag = adapt_lag(lag, n)
    out = np.full(n, np.nan)

    start = n - lag
    mean_val = np.mean(series[start:])
    out[start:] = mean_val
    return out


# =============================================================================
# EXCEL-LIKE MA — Rolling forecast auto-alimentato
# Equivalente alla logica foreEWMA==2 in FlatFore() di Matlab
# =============================================================================

def excel_like(series: np.ndarray, lag: int, n_forecast: int) -> tuple:
    """
    MA Excel-Like: il forecast di ogni mese futuro è la media degli ultimi
    lag valori, dove i valori futuri già calcolati entrano nel calcolo
    dei successivi (rolling auto-alimentato).

    Identico alla logica foreEWMA==2 in Matlab.

    Ritorna
    -------
    fitted    : array in-sample (SMA classico sulla storia)
    forecast  : array n_forecast valori futuri
    """
    fitted   = sma(series, lag)
    extended = list(series)

    forecast = []
    for _ in range(n_forecast):
        window = extended[-lag:]
        val    = np.nanmean(window)
        forecast.append(val)
        extended.append(val)

    return fitted, np.array(forecast)


# =============================================================================
# LINREG — Regressione Lineare (Trend)
# Equivalente a linreg() in Matlab
# =============================================================================

def linreg(series: np.ndarray, lag: int) -> tuple:
    """
    Regressione lineare sugli ultimi lag valori della serie.
    Ritorna i fitted values e i coefficienti W = [intercetta, slope].
    Identico a linreg() in Matlab (usa pseudo-inversa).
    """
    n   = len(series)
    lag = adapt_lag(lag, n)

    start = n - lag
    X_idx = np.arange(start + 1, n + 1, dtype=float)   # indici 1-based come Matlab
    Y     = series[start:]

    # Matrice di design [1, x] — identica a [ones(lag,1) X] in Matlab
    A = np.column_stack([np.ones(lag), X_idx])
    # Minimi quadrati (pinv equivalente a Matlab)
    W, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)

    out = np.full(n, np.nan)
    out[start:] = A @ W

    return out, W


# =============================================================================
# FLAT FORECAST — equivalente a FlatFore() in Matlab
# Calcola fitted + forecast + KPI per tutti gli algoritmi MA/EWMA/Batch/LinReg
# =============================================================================

def flat_forecast(
    history:      np.ndarray,
    lag:          int,
    mode:         int,        # 0=SMA, 1=EWMA, -1=Batch, 2=ExcelLike, 3=LinReg
    n_forecast:   int   = 12,
    ts_number:    int   = 12, # periodi hold-out per lnQ (come opt.tsNumber in Matlab)
    mase_period:  int   = 1,
    ci_type:      bool  = False,
    ci_alpha:     float = 0.05,
    ci_zeta:      float = 0.0,
) -> dict:
    """
    Calcola forecast completo per un algoritmo MA.
    Equivalente a FlatFore() in coreAlgorithmMA.m.

    Parametri
    ---------
    history     : serie storica (ordine vecchio→recente)
    lag         : finestra MA
    mode        : 0=SMA, 1=EWMA, -1=Batch, 2=ExcelLike, 3=LinReg
    n_forecast  : numero di periodi futuri da prevedere
    ts_number   : periodi finali usati come hold-out per lnQ
    mase_period : periodo MASE (1=non stagionale, 12=stagionale)
    ci_type     : tipo CI Poisson (False=Pearson, True=exact)
    ci_alpha    : livello significatività CI
    ci_zeta     : parametro aggiuntivo CI

    Ritorna
    -------
    dict con:
        'fitted'   : array in-sample (lunghezza = len(history))
        'forecast' : array forecast futuro (lunghezza = n_forecast)
        'fore_all' : fitted + forecast concatenati
        'lower'    : CI inferiore su fore_all
        'upper'    : CI superiore su fore_all
        'kpi'      : dict KPI (MASE, lnQ, MSE, ...)
        'kpi_array': array [MSE,RMSE,MAE,lnQ,ErrorMean,ErrorStd,R,Rsq,Num,MASE]
    """
    data = np.asarray(history, dtype=float).ravel()
    n    = len(data)

    lag         = adapt_lag(lag, n)
    mase_period = adapt_mase_period(mase_period, n)

    # --- Calcolo fitted in-sample ---
    if mode == 1:
        fitted_raw = ewma(data, lag)
    elif mode == -1:
        fitted_raw = batch(data, lag)
    elif mode == 2:
        fitted_raw, _ = excel_like(data, lag, n_forecast)
    elif mode == 3:
        fitted_raw, W = linreg(data, lag)
    else:  # mode == 0, default SMA
        fitted_raw = sma(data, lag)

    fitted = np.maximum(0, np.round(fitted_raw))

    # --- Calcolo forecast futuro ---
    if mode in (0, 1, -1):
        # Forecast piatto: media degli ultimi ts_number fitted valori
        if n > ts_number:
            fc_val = np.round(np.nanmean(fitted[-ts_number:]))
        else:
            fc_val = np.round(np.nanmean(fitted))
        forecast = np.full(n_forecast, max(0, fc_val))

    elif mode == 2:
        # Excel-Like: rolling auto-alimentato
        _, forecast_raw = excel_like(data, lag, n_forecast)
        forecast = np.maximum(0, np.round(forecast_raw))

    else:  # mode == 3, LinReg
        X_fut = np.arange(n + 1, n + n_forecast + 1, dtype=float)
        A_fut = np.column_stack([np.ones(n_forecast), X_fut])
        forecast = np.maximum(0, np.round(A_fut @ W))

    # --- KPI ---
    train_kpi = compute_train_kpi(data, fitted_raw, mase_period)

    if n > ts_number:
        hold_targets = data[-ts_number:]
        hold_fitted  = np.full(ts_number, np.nanmean(fitted[-ts_number:]))
        test_kpi     = compute_test_kpi(hold_targets, hold_fitted)
    else:
        test_kpi = {'lnQ': float('nan')}

    kpi = merge_kpi(train_kpi, test_kpi)

    # --- Concatena fitted + forecast per CI ---
    fore_all = np.concatenate([fitted, forecast])
    lower, upper = ci_poisson(fore_all, ci_type, ci_alpha, ci_zeta)

    return {
        'fitted':    fitted,
        'forecast':  forecast,
        'fore_all':  fore_all,
        'lower':     lower,
        'upper':     upper,
        'kpi':       kpi,
        'kpi_array': kpi_to_array(kpi),
    }