# =============================================================================
# DGS Forecast Engine — Dispatcher Centrale v2
# -----------------------------------------------------------------------------
# Algoritmi disponibili:
#   MA family  : SMA, EWMA, BATCH, EXCEL_LIKE, LINREG
#   Statistical: HW, ARMA, ARIMA, SARIMA, SARIMAX
#   Advanced   : THETA, CROSTON, ADIDA, IMAPA
#   Neural     : LSTM
#   Meta       : ENSEMBLE, AUTO
#
# Parametri comuni a tutti gli algoritmi:
#   history        array float  serie storica (vecchio→recente)
#   n_forecast     int          periodi futuri          default 12
#   season_length  int          lunghezza stagione      default 12
#   mase_period    int          periodo MASE            default 12
#   ts_number      int          hold-out per lnQ        default 12
#   ci_alpha       float        livello CI Poisson      default 0.05
#   ci_zeta        float        parametro aggiuntivo CI default 0.0
#   tr_number      int|None     tronca la storia        default None
#
# Parametri specifici per algoritmo:
#   lag            int     (SMA/EWMA/BATCH/EXCEL_LIKE/LINREG)  default 6
#   decomposition  str     (THETA) 'multiplicative'|'additive' default 'multiplicative'
#   X_hist/X_fut   array   (SARIMAX) variabili esogene
#   lstm_steps     int     (LSTM) epoche training               default 300
#   lstm_hidden    int     (LSTM) neuroni hidden                default 64
#   lstm_layers    int     (LSTM) strati LSTM                   default 2
#   ensemble_algos list    (ENSEMBLE) algoritmi da combinare
#   ensemble_top_n int     (ENSEMBLE) top N per MASE            default None
#   intermittent   str     (AUTO) forza modalità intermittente   default None
# =============================================================================

import numpy as np
import time
from core.algorithms_ma           import flat_forecast, prepare_series
from core.algorithms_hw           import holt_winters
from core.algorithms_arima        import sarima
from core.algorithms_lstm         import lstm_forecast
from core.algorithms_theta        import theta_forecast
from core.algorithms_intermittent import intermittent_forecast, classify_intermittency
from core.algorithms_ensemble     import ensemble_forecast
from core.kpi                     import kpi_to_array

_MA_MODES = {
    'SMA':        0,
    'EWMA':       1,
    'BATCH':     -1,
    'EXCEL_LIKE': 2,
    'LINREG':     3,
}

# Algoritmi veloci inclusi in AUTO (esclude LSTM per latenza)
_AUTO_ALGORITHMS = ['SMA', 'EWMA', 'HW', 'THETA', 'ARIMA', 'SARIMA']

# Soglia ADI per rilevamento automatico domanda intermittente in AUTO
_INTERMITTENT_ADI_THRESHOLD = 1.32


def run_forecast(
    history:        np.ndarray,
    algorithm:      str   = 'AUTO',
    n_forecast:     int   = 12,
    lag:            int   = 6,
    season_length:  int   = 12,
    mase_period:    int   = 12,
    ts_number:      int   = 12,
    ci_alpha:       float = 0.05,
    ci_zeta:        float = 0.0,
    decomposition:  str   = 'multiplicative',
    X_hist:         np.ndarray = None,
    X_fut:          np.ndarray = None,
    lstm_steps:     int   = 300,
    lstm_hidden:    int   = 64,
    lstm_layers:    int   = 2,
    ensemble_algos: list  = None,
    ensemble_top_n: int   = None,
    tr_number:      int   = None,
) -> dict:
    """
    Dispatcher principale del DGS Forecast Engine.
    Tutti i parametri sono documentati nell'header del file.
    """
    t0   = time.time()
    data = prepare_series(np.asarray(history, dtype=float), tr_number=tr_number)
    algo = algorithm.upper().strip()

    common = dict(
        n_forecast=n_forecast, ts_number=ts_number,
        mase_period=mase_period, ci_alpha=ci_alpha, ci_zeta=ci_zeta,
    )

    # ── Routing ──────────────────────────────────────────────────────────────

    if algo in _MA_MODES:
        result = flat_forecast(data, lag=lag, mode=_MA_MODES[algo], **common)
        result.setdefault('method', algo)

    elif algo == 'HW':
        result = holt_winters(data, season_length=season_length, **common)

    elif algo in ('ARMA', 'ARIMA', 'SARIMA'):
        result = sarima(data, mode=algo, season_length=season_length, **common)

    elif algo == 'SARIMAX':
        result = sarima(data, mode='SARIMAX', season_length=season_length,
                        X_hist=X_hist, X_fut=X_fut, **common)

    elif algo == 'THETA':
        result = theta_forecast(data, season_length=season_length,
                                decomposition=decomposition, **common)

    elif algo in ('CROSTON', 'ADIDA', 'IMAPA'):
        result = intermittent_forecast(data, algorithm=algo, **common)

    elif algo == 'INTERMITTENT_AUTO':
        # Rileva automaticamente il miglior algoritmo intermittente
        result = intermittent_forecast(data, algorithm='AUTO', **common)

    elif algo == 'LSTM':
        result = lstm_forecast(
            data, input_size=season_length,
            encoder_hidden_size=lstm_hidden,
            encoder_n_layers=lstm_layers,
            max_steps=lstm_steps, **common,
        )
        result.setdefault('method', 'LSTM')

    elif algo == 'ENSEMBLE':
        result = ensemble_forecast(
            data, lag=lag, season_length=season_length,
            algorithms=ensemble_algos, top_n=ensemble_top_n, **common,
        )

    elif algo == 'AUTO':
        result = _run_auto(data, common, lag, season_length, decomposition)

    else:
        raise ValueError(
            f"Algoritmo '{algorithm}' non riconosciuto. "
            f"Valori validi: SMA, EWMA, BATCH, EXCEL_LIKE, LINREG, HW, "
            f"ARMA, ARIMA, SARIMA, SARIMAX, THETA, CROSTON, ADIDA, IMAPA, "
            f"INTERMITTENT_AUTO, LSTM, ENSEMBLE, AUTO"
        )

    result['algorithm']   = algo
    result['elapsed_sec'] = round(time.time() - t0, 3)
    return result


def _run_auto(data, common, lag, season_length, decomposition):
    """
    AUTO: seleziona il best algoritmo per MASE.
    Se la serie è intermittente (ADI > soglia), usa IMAPA direttamente.
    """
    # Check intermittenza preventivo
    interm = classify_intermittency(data)
    if interm['adi'] > _INTERMITTENT_ADI_THRESHOLD:
        result = intermittent_forecast(data, algorithm='AUTO', **common)
        result['auto_best']    = result['method']
        result['auto_scores']  = {result['method']: {
            'MASE': result['kpi']['MASE'], 'lnQ': result['kpi']['lnQ']
        }}
        result['auto_mode']    = 'intermittent'
        return result

    # Serie normale: prova tutti gli algoritmi veloci
    candidates = {}
    for algo in _AUTO_ALGORITHMS:
        try:
            if algo in _MA_MODES:
                r = flat_forecast(data, lag=lag, mode=_MA_MODES[algo], **common)
                r.setdefault('method', algo)
            elif algo == 'HW':
                r = holt_winters(data, season_length=season_length, **common)
            elif algo == 'THETA':
                r = theta_forecast(data, season_length=season_length,
                                   decomposition=decomposition, **common)
            else:
                r = sarima(data, mode=algo, season_length=season_length, **common)
            candidates[algo] = r
        except Exception:
            pass

    if not candidates:
        r = flat_forecast(data, lag=min(lag, len(data)-1), mode=0, **common)
        r['method'] = 'SMA_emergency'
        return r

    valid = {
        n: r['kpi']['MASE']
        for n, r in candidates.items()
        if r['kpi']['MASE'] == r['kpi']['MASE']
    }
    if valid:
        best_name = min(valid, key=valid.get)
    else:
        valid_lnq = {
            n: r['kpi']['lnQ']
            for n, r in candidates.items()
            if r['kpi']['lnQ'] == r['kpi']['lnQ']
        }
        best_name = min(valid_lnq, key=valid_lnq.get) if valid_lnq \
                    else list(candidates.keys())[0]

    result = candidates[best_name]
    result['auto_best']   = best_name
    result['auto_mode']   = 'standard'
    result['auto_scores'] = {
        n: {'MASE': r['kpi']['MASE'], 'lnQ': r['kpi']['lnQ']}
        for n, r in candidates.items()
    }
    return result
