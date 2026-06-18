# =============================================================================
# DGS Forecast Engine — Ensemble (media pesata per MASE inverso)
# -----------------------------------------------------------------------------
# L'ensemble combina i forecast di più algoritmi in un unico forecast finale,
# pesando ciascuno in base alla sua accuratezza storica (MASE inverso).
# In letteratura migliora il MASE del 5-15% rispetto al singolo best.
#
# PERCHÉ FUNZIONA:
# Ogni algoritmo ha punti di forza diversi. SMA è stabile ma lento a reagire.
# SARIMA cattura bene la stagionalità ma può overfittare. HW/ETS è robusto
# su serie regolari. L'ensemble mitiga i punti deboli di ciascuno.
#
# QUANDO USARLO:
#   ✓ Come alternativa o complemento alla modalità AUTO
#   ✓ Quando si vuole ridurre il rischio di scelta sbagliata dell'algoritmo
#   ✓ Su cataloghi grandi dove non si ha tempo di analizzare ogni SKU
#   ✗ Non su serie intermittenti (usare Croston/ADIDA/IMAPA)
#   ✗ Non quando un singolo algoritmo ha MASE chiaramente dominante
#
# PARAMETRI IN INGRESSO (dal backend C#):
#   history        : array float, serie storica ordine vecchio→recente
#   n_forecast     : int, periodi futuri (default 12)
#   algorithms     : list[str], algoritmi da includere nell'ensemble
#                    default: ['SMA','EWMA','HW','ARIMA','SARIMA']
#                    possibili: qualsiasi subset degli algoritmi del motore
#   weights        : list[float] | None
#                    None   → pesi automatici (1/MASE normalizzato) — raccomandato
#                    lista  → pesi manuali fissi (devono sommare a 1.0)
#                    Se un algoritmo ha MASE=NaN viene escluso automaticamente
#   lag            : int, finestra MA per SMA/EWMA/Batch (default 6)
#   season_length  : int, lunghezza stagione (default 12)
#   mase_period    : int, periodo MASE (default 12)
#   ts_number      : int, periodi hold-out per lnQ (default 12)
#   ci_alpha       : float, livello CI Poisson (default 0.05 = 95%)
#   ci_zeta        : float, parametro aggiuntivo CI (default 0.0)
#   top_n          : int | None, se specificato usa solo i top_n algoritmi
#                    per MASE (es. top_n=3 → ensemble dei migliori 3)
#
# OUTPUT:
#   fitted, forecast, fore_all, lower, upper, kpi, kpi_array
#   + weights_used   : dict {algo: peso_finale}
#   + scores         : dict {algo: {MASE, lnQ}}
#   + components     : dict {algo: forecast_array} — i singoli forecast
# =============================================================================

import numpy as np
from core.kpi import compute_train_kpi, compute_test_kpi, merge_kpi, ci_poisson, kpi_to_array


# Algoritmi disponibili per l'ensemble
_ENSEMBLE_ALGORITHMS = ['SMA', 'EWMA', 'HW', 'ARIMA', 'SARIMA', 'THETA']

# Default ensemble leggero (esclude LSTM per velocità in produzione)
_DEFAULT_ENSEMBLE = ['SMA', 'EWMA', 'HW', 'ARIMA', 'SARIMA']


def ensemble_forecast(
    history:       np.ndarray,
    n_forecast:    int        = 12,
    algorithms:    list       = None,
    weights:       list       = None,
    lag:           int        = 6,
    season_length: int        = 12,
    mase_period:   int        = 12,
    ts_number:     int        = 12,
    ci_alpha:      float      = 0.05,
    ci_zeta:       float      = 0.0,
    top_n:         int        = None,
) -> dict:
    """
    Ensemble di algoritmi con pesi automatici basati su MASE inverso.

    Parametri
    ---------
    history       : serie storica (ordine vecchio→recente)
    n_forecast    : periodi futuri da prevedere
    algorithms    : lista algoritmi da combinare
                    default: ['SMA','EWMA','HW','ARIMA','SARIMA']
    weights       : pesi manuali (None = automatici da MASE)
    lag           : finestra MA per SMA/EWMA
    season_length : lunghezza stagione
    mase_period   : periodo MASE
    ts_number     : periodi hold-out per lnQ
    ci_alpha      : livello CI Poisson
    ci_zeta       : parametro aggiuntivo CI
    top_n         : se specificato, usa solo i migliori top_n per MASE

    Ritorna
    -------
    dict con: fitted, forecast, fore_all, lower, upper, kpi, kpi_array,
              weights_used, scores, components
    """
    from core.algorithms_ma    import flat_forecast
    from core.algorithms_hw    import holt_winters
    from core.algorithms_arima import sarima
    from core.algorithms_theta import theta_forecast

    data = np.asarray(history, dtype=float).ravel()
    n    = len(data)

    algo_list = algorithms if algorithms else _DEFAULT_ENSEMBLE
    algo_list = [a.upper() for a in algo_list]

    common = dict(
        n_forecast=n_forecast, ts_number=ts_number,
        mase_period=mase_period, ci_alpha=ci_alpha, ci_zeta=ci_zeta,
    )

    # --- Esegui tutti gli algoritmi ---
    MA_MODES = {'SMA': 0, 'EWMA': 1, 'BATCH': -1, 'EXCEL_LIKE': 2, 'LINREG': 3}
    raw_results = {}

    for algo in algo_list:
        try:
            if algo in MA_MODES:
                r = flat_forecast(data, lag=lag, mode=MA_MODES[algo], **common)
            elif algo == 'HW':
                r = holt_winters(data, season_length=season_length, **common)
            elif algo in ('ARMA', 'ARIMA', 'SARIMA'):
                r = sarima(data, mode=algo, season_length=season_length, **common)
            elif algo == 'THETA':
                r = theta_forecast(data, season_length=season_length, **common)
            else:
                continue
            raw_results[algo] = r
        except Exception:
            pass

    if not raw_results:
        return _fallback_ensemble(data, n_forecast, ts_number, mase_period,
                                  ci_alpha, ci_zeta)

    # --- Calcola pesi ---
    scores = {
        algo: {'MASE': r['kpi']['MASE'], 'lnQ': r['kpi']['lnQ']}
        for algo, r in raw_results.items()
    }

    if weights is not None and len(weights) == len(raw_results):
        # Pesi manuali normalizzati
        w_arr  = np.array(weights[:len(raw_results)], dtype=float)
        w_norm = w_arr / w_arr.sum()
        weights_used = dict(zip(raw_results.keys(), w_norm))
    else:
        # Pesi automatici: 1/MASE normalizzato
        mase_vals = {
            algo: r['kpi']['MASE']
            for algo, r in raw_results.items()
            if r['kpi']['MASE'] == r['kpi']['MASE'] and r['kpi']['MASE'] > 0
        }

        if not mase_vals:
            # Tutti NaN → pesi uguali
            w = 1.0 / len(raw_results)
            weights_used = {algo: w for algo in raw_results}
        else:
            inv_mase = {algo: 1.0 / mase for algo, mase in mase_vals.items()}
            total    = sum(inv_mase.values())
            weights_used = {algo: v / total for algo, v in inv_mase.items()}

    # Applica top_n: tieni solo i migliori per MASE e rinormalizza
    if top_n is not None and top_n < len(weights_used):
        valid_mase = {
            a: raw_results[a]['kpi']['MASE']
            for a in weights_used
            if raw_results[a]['kpi']['MASE'] == raw_results[a]['kpi']['MASE']
        }
        top_algos = sorted(valid_mase, key=valid_mase.get)[:top_n]
        weights_used = {a: weights_used[a] for a in top_algos}
        total = sum(weights_used.values())
        weights_used = {a: v / total for a, v in weights_used.items()}

    # --- Combina i forecast (media pesata) ---
    forecast_matrix = np.array([
        raw_results[algo]['forecast']
        for algo in weights_used
    ])
    w_vector = np.array([weights_used[algo] for algo in weights_used])

    # Broadcast pesi e media pesata
    ensemble_forecast_raw = (forecast_matrix * w_vector[:, np.newaxis]).sum(axis=0)
    ensemble_forecast     = np.maximum(0, np.round(ensemble_forecast_raw))

    # --- Combina fitted (media pesata, allineando le lunghezze) ---
    fitted_arrays = []
    for algo in weights_used:
        f = raw_results[algo]['fitted'].astype(float)
        fitted_arrays.append(f)

    fitted_matrix = np.array(fitted_arrays)
    ensemble_fitted_raw = (fitted_matrix * w_vector[:, np.newaxis]).sum(axis=0)
    ensemble_fitted     = np.maximum(0, np.round(ensemble_fitted_raw))

    # --- KPI sull'ensemble ---
    train_kpi = compute_train_kpi(data, ensemble_fitted_raw, mase_period)
    test_kpi  = compute_test_kpi(data[-ts_number:], ensemble_fitted_raw[-ts_number:]) \
                if n > ts_number else {'lnQ': float('nan')}
    kpi = merge_kpi(train_kpi, test_kpi)

    fore_all = np.concatenate([ensemble_fitted, ensemble_forecast])
    lower, upper = ci_poisson(fore_all, ci_type=False, alpha=ci_alpha, zeta=ci_zeta)

    # Componenti individuali (utile per debug e UI)
    components = {
        algo: raw_results[algo]['forecast'].tolist()
        for algo in weights_used
    }

    return {
        'fitted':       ensemble_fitted,
        'forecast':     ensemble_forecast,
        'fore_all':     fore_all,
        'lower':        lower,
        'upper':        upper,
        'kpi':          kpi,
        'kpi_array':    kpi_to_array(kpi),
        'method':       f'Ensemble({",".join(weights_used.keys())})',
        'weights_used': {a: round(v, 4) for a, v in weights_used.items()},
        'scores':       scores,
        'components':   components,
    }


def _fallback_ensemble(data, n_forecast, ts_number, mase_period, ci_alpha, ci_zeta):
    from core.algorithms_ma import flat_forecast
    r = flat_forecast(data, lag=min(6, max(1, len(data)-1)), mode=0,
                      n_forecast=n_forecast, ts_number=ts_number,
                      mase_period=mase_period, ci_alpha=ci_alpha, ci_zeta=ci_zeta)
    r['method']       = 'fallback_SMA'
    r['weights_used'] = {'SMA': 1.0}
    r['scores']       = {}
    r['components']   = {}
    return r
