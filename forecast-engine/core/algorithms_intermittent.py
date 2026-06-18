# =============================================================================
# DGS Forecast Engine — Algoritmi per Domanda Intermittente
# -----------------------------------------------------------------------------
# Questi algoritmi sono progettati specificamente per serie con molti zeri
# (slow-mover, prodotti stagionali, pezzi di ricambio, ecc.).
# Su questo tipo di serie ARIMA, HW e LSTM producono risultati pessimi.
#
# ALGORITMI IMPLEMENTATI:
#
#   CROSTON     — Croston Ottimizzato
#   --------
#   Separa la serie in due componenti: dimensione della domanda (quando c'è)
#   e intervallo tra domande. Applica SES a ciascuna componente separatamente.
#   È lo standard industriale per domanda intermittente dal 1972.
#   QUANDO USARLO: serie con domanda rara ma costante (es. pezzi di ricambio
#   costosi, prodotti speciali). Ottimo quando gli zeri sono > 30% dei valori.
#
#   ADIDA       — Aggregate-Disaggregate Intermittent Demand Approach
#   -----
#   Aggrega la serie temporalmente (somma in bucket più grandi), applica SES,
#   poi disaggrega il forecast. Riduce la variabilità degli zeri.
#   QUANDO USARLO: serie con domanda molto sporadica e irregolare,
#   dove Croston tende a sovrastimare. Meglio di Croston quando gli
#   intervalli tra domande sono molto variabili.
#
#   IMAPA       — Intermittent Multiple Aggregation Prediction Algorithm
#   -----
#   Combina forecast a più livelli di aggregazione temporale (1,2,3,...,n mesi)
#   e ne fa la media. È la versione più moderna e robusta dei tre.
#   QUANDO USARLO: quando non si sa se la serie è "pura" intermittente o
#   "lumpy" (intermittente con domanda variabile). IMAPA è conservativo
#   e generalmente il più sicuro tra i tre.
#
# COME SCEGLIERE TRA I TRE (regola pratica per il backend):
#   - Calcola ADI (Average Demand Interval) = n_periodi / n_periodi_non_zero
#   - Calcola CV2 = (std_domanda_non_zero / mean_domanda_non_zero)^2
#   - Se ADI > 1.32 e CV2 < 0.49  → CROSTON (intermittente regolare)
#   - Se ADI > 1.32 e CV2 >= 0.49 → ADIDA   (intermittente irregolare/lumpy)
#   - Se ADI <= 1.32               → IMAPA o algoritmi standard (non intermittente)
#   Questa classificazione è quella di Syntetos-Boylan (2005), standard SIA.
#
# PARAMETRI IN INGRESSO (dal backend C#):
#   history        : array float, serie storica ordine vecchio→recente
#                    Gli zeri devono essere VERI zeri (non NaN, non -1)
#   n_forecast     : int, periodi futuri (default 12)
#   algorithm      : str, 'CROSTON' | 'ADIDA' | 'IMAPA' | 'AUTO'
#                    AUTO applica la regola ADI/CV2 sopra descritta
#   ts_number      : int, periodi hold-out per lnQ (default 12)
#   mase_period    : int, periodo MASE (default 1 per serie intermittenti,
#                    non 12 — la stagionalità non ha senso su serie con molti zeri)
#   ci_alpha       : float, livello CI Poisson (default 0.05 = 95%)
#   ci_zeta        : float, parametro aggiuntivo CI (default 0.0)
#
# OUTPUT (identico agli altri moduli):
#   fitted, forecast, fore_all, lower, upper, kpi, kpi_array, method,
#   + adi (Average Demand Interval), cv2 (Coefficient of Variation squared),
#   + intermittency_class ('smooth'|'intermittent'|'lumpy'|'erratic')
# =============================================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from statsforecast import StatsForecast
from statsforecast.models import CrostonOptimized, ADIDA, IMAPA
from core.kpi import compute_train_kpi, compute_test_kpi, merge_kpi, ci_poisson, kpi_to_array


# =============================================================================
# Classificazione domanda intermittente (Syntetos-Boylan 2005)
# =============================================================================

def classify_intermittency(series: np.ndarray) -> dict:
    """
    Classifica la serie secondo la matrice ADI/CV2 di Syntetos-Boylan.

    Ritorna
    -------
    dict con:
        adi   : Average Demand Interval (n_periodi / n_periodi_con_domanda)
        cv2   : CV² della domanda non-zero
        class : 'smooth' | 'intermittent' | 'lumpy' | 'erratic'
        recommended : algoritmo raccomandato
    """
    data     = np.asarray(series, dtype=float).ravel()
    nonzero  = data[data > 0]
    n        = len(data)
    n_nonzero = len(nonzero)

    if n_nonzero == 0:
        return {'adi': float('inf'), 'cv2': 0.0,
                'class': 'zero_demand', 'recommended': 'CROSTON'}

    adi = n / n_nonzero
    if n_nonzero > 1:
        cv2 = (np.std(nonzero) / np.mean(nonzero)) ** 2
    else:
        cv2 = 0.0

    # Matrice Syntetos-Boylan
    if adi <= 1.32 and cv2 < 0.49:
        cls = 'smooth'
        rec = 'SMA_or_HW'          # non è veramente intermittente
    elif adi > 1.32 and cv2 < 0.49:
        cls = 'intermittent'
        rec = 'CROSTON'
    elif adi <= 1.32 and cv2 >= 0.49:
        cls = 'erratic'
        rec = 'IMAPA'
    else:
        cls = 'lumpy'
        rec = 'ADIDA'

    return {
        'adi':         round(adi, 3),
        'cv2':         round(cv2, 3),
        'class':       cls,
        'recommended': rec,
    }


# =============================================================================
# Funzione principale
# =============================================================================

def intermittent_forecast(
    history:      np.ndarray,
    algorithm:    str   = 'AUTO',
    n_forecast:   int   = 12,
    ts_number:    int   = 12,
    mase_period:  int   = 1,        # 1 = non stagionale, standard per intermittente
    ci_alpha:     float = 0.05,
    ci_zeta:      float = 0.0,
) -> dict:
    """
    Forecast per domanda intermittente.

    Parametri
    ---------
    history      : serie storica (ordine vecchio→recente, zeri inclusi)
    algorithm    : 'CROSTON' | 'ADIDA' | 'IMAPA' | 'AUTO'
                   AUTO usa la classificazione ADI/CV2 per scegliere
    n_forecast   : periodi futuri da prevedere
    ts_number    : periodi hold-out per lnQ
    mase_period  : periodo MASE (1 per intermittente, non 12)
    ci_alpha     : livello significatività CI Poisson
    ci_zeta      : parametro aggiuntivo CI

    Ritorna
    -------
    dict con: fitted, forecast, fore_all, lower, upper, kpi, kpi_array,
              method, adi, cv2, intermittency_class
    """
    data = np.asarray(history, dtype=float).ravel()
    n    = len(data)

    # Classificazione intermittenza (sempre calcolata, utile per il backend)
    intermittency = classify_intermittency(data)

    # Selezione algoritmo
    algo = algorithm.upper().strip()
    if algo == 'AUTO':
        algo = intermittency['recommended']
        if algo == 'SMA_or_HW':
            algo = 'IMAPA'  # fallback sicuro se AUTO su serie non intermittente

    # Mappa nome → classe statsforecast
    model_map = {
        'CROSTON': CrostonOptimized(alias='Intermittent'),
        'ADIDA':   ADIDA(alias='Intermittent'),
        'IMAPA':   IMAPA(alias='Intermittent'),
    }
    if algo not in model_map:
        algo = 'IMAPA'
    sf_model = model_map[algo]

    if n < 4:
        return _fallback_intermittent(
            data, n_forecast, ts_number, mase_period, ci_alpha, ci_zeta,
            intermittency, reason='serie troppo corta')

    # Piccolo offset per evitare zeri puri (statsforecast gestisce male zeri esatti)
    data_safe = np.where(data <= 0, 0.0001, data)
    dates = pd.date_range(start='2020-01-01', periods=n, freq='MS')
    df = pd.DataFrame({'unique_id': ['s1'] * n, 'ds': dates, 'y': data_safe})

    sf = StatsForecast(models=[sf_model], freq='MS', n_jobs=1)

    try:
        sf.fit(df)
        forecast_df  = sf.predict(h=n_forecast)
        forecast_raw = forecast_df['Intermittent'].values
        forecast     = np.maximum(0, np.round(forecast_raw))

        # Croston/ADIDA/IMAPA non hanno fitted values individuali —
        # restituiscono solo la media prevista (model_['mean']).
        # Usiamo la media come fitted costante sugli ultimi ts_number periodi
        # (identico al comportamento Batch MA, appropriato per intermittente)
        inner    = sf.fitted_[0, 0].model_
        mean_val = float(inner.get('mean', np.mean(data_safe[data_safe > 0.001])))

        fitted_raw = np.full(n, float('nan'))
        # Popola gli ultimi ts_number periodi (quelli confrontabili)
        start_fit = max(0, n - ts_number)
        fitted_raw[start_fit:] = mean_val
        fitted = np.maximum(0, np.round(fitted_raw))

    except Exception as e:
        return _fallback_intermittent(
            data, n_forecast, ts_number, mase_period, ci_alpha, ci_zeta,
            intermittency, reason=str(e))

    # KPI — solo sugli ultimi ts_number periodi dove abbiamo fitted
    # (i NaN nei periodi precedenti vengono ignorati automaticamente da compute_train_kpi)
    train_kpi = compute_train_kpi(data, fitted_raw, mase_period)
    test_kpi  = compute_test_kpi(data[-ts_number:], np.full(ts_number, mean_val)) \
                if n > ts_number else {'lnQ': float('nan')}
    kpi = merge_kpi(train_kpi, test_kpi)

    fore_all = np.concatenate([
        np.where(np.isnan(fitted), 0, fitted),
        forecast
    ])
    lower, upper = ci_poisson(fore_all, ci_type=False, alpha=ci_alpha, zeta=ci_zeta)

    return {
        'fitted':              fitted,
        'forecast':            forecast,
        'fore_all':            fore_all,
        'lower':               lower,
        'upper':               upper,
        'kpi':                 kpi,
        'kpi_array':           kpi_to_array(kpi),
        'method':              algo,
        'adi':                 intermittency['adi'],
        'cv2':                 intermittency['cv2'],
        'intermittency_class': intermittency['class'],
        'recommended':         intermittency['recommended'],
    }


def _fallback_intermittent(data, n_forecast, ts_number, mase_period,
                           ci_alpha, ci_zeta, intermittency, reason=''):
    from core.algorithms_ma import flat_forecast
    r = flat_forecast(data, lag=min(6, max(1, len(data)-1)), mode=-1,
                      n_forecast=n_forecast, ts_number=ts_number,
                      mase_period=mase_period, ci_alpha=ci_alpha, ci_zeta=ci_zeta)
    r['method']              = 'fallback_BATCH'
    r['adi']                 = intermittency.get('adi', float('nan'))
    r['cv2']                 = intermittency.get('cv2', float('nan'))
    r['intermittency_class'] = intermittency.get('class', 'unknown')
    r['recommended']         = intermittency.get('recommended', 'N/A')
    r['fallback']            = reason
    return r
