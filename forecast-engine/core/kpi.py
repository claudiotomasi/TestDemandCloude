# =============================================================================
# DGS Forecast Engine — KPI Module
# Equivalente a: PlotResults() in coreAlgorithmMA.m e coreAlgorithmLSTM.m
#
# Calcola tutti gli indicatori di accuratezza:
#   MASE, lnQ, MSE, RMSE, MAE, ErrorMean, ErrorStd, R, Rsq, Num
#
# Output identico a fitMeasures in Matlab:
#   [MSE, RMSE, MAE, lnQ, ErrorMean, ErrorStd, R, Rsq, Num, MASE]
# =============================================================================

import numpy as np


def compute_train_kpi(targets: np.ndarray, outputs: np.ndarray, mase_period: int = 1) -> dict:
    """
    Calcola i KPI sul training set.
    Equivalente al branch 'Train Data' di PlotResults() in Matlab.

    Parametri
    ---------
    targets     : valori reali (storia)
    outputs     : valori previsti (in-sample fitted)
    mase_period : periodo per il calcolo MASE (1 = non stagionale, 12 = stagionale mensile)

    Ritorna
    -------
    dict con chiavi: MSE, RMSE, MAE, lnQ, ErrorMean, ErrorStd, R, Rsq, Num, MASE
    """
    targets = np.asarray(targets, dtype=float).ravel()
    outputs = np.asarray(outputs, dtype=float).ravel()

    # Filtro identico a Matlab: mantieni solo valori >= 1 oppure == 0
    # (esclude valori negativi e NaN)
    ind = (targets >= 1) | (targets == 0)
    true_targets = targets[ind]
    true_outputs = outputs[ind]

    # Rimuovi dove outputs è NaN
    valid = ~np.isnan(true_outputs)
    true_targets = true_targets[valid]
    true_outputs = true_outputs[valid]

    if len(true_targets) == 0:
        return _empty_train_kpi()

    errors = true_targets - true_outputs

    mse       = float(np.nanmean(errors ** 2))
    rmse      = float(np.sqrt(mse))
    mae       = float(np.nanmean(np.abs(errors)))
    err_mean  = float(np.nanmean(errors))
    err_std   = float(np.nanstd(errors))
    num       = int(len(errors))

    # Correlazione R e R²
    valid_corr = ~np.isnan(true_outputs)
    if np.sum(valid_corr) > 1 and np.std(true_targets[valid_corr]) > 0 and np.std(true_outputs[valid_corr]) > 0:
        r_matrix = np.corrcoef(true_targets[valid_corr], true_outputs[valid_corr])
        r   = float(r_matrix[0, 1])
        rsq = float(r ** 2)
    else:
        r   = float('nan')
        rsq = float('nan')

    # MASE — Mean Absolute Scaled Error
    # Formula: Hyndman & Koehler (2006)
    m = mase_period
    if len(true_targets) > m:
        naive_errors = np.abs(true_targets[m:] - true_targets[:-m])
        mean_naive   = float(np.nanmean(naive_errors))
        if mean_naive > 0:
            scaled_errors = errors / mean_naive
            mase = float(np.nanmean(np.abs(scaled_errors)))
        else:
            mase = float('nan')
    else:
        mase = float('nan')

    return {
        'MSE':      mse,
        'RMSE':     rmse,
        'MAE':      mae,
        'lnQ':      float('nan'),   # lnQ si calcola solo sul test set
        'ErrorMean': err_mean,
        'ErrorStd': err_std,
        'R':        r,
        'Rsq':      rsq,
        'Num':      num,
        'MASE':     mase,
    }


def compute_test_kpi(targets: np.ndarray, outputs: np.ndarray) -> dict:
    """
    Calcola lnQ sul test set (ultimi N periodi della storia usati come hold-out).
    Equivalente al branch 'Test Data' di PlotResults() in Matlab.

    lnQ = sum( log(output/target)^2 ) per i valori > 0
    Valori ideali: lnQ vicino a 0. Valori alti indicano overfitting.

    Parametri
    ---------
    targets : valori reali (hold-out)
    outputs : valori previsti

    Ritorna
    -------
    dict con chiave lnQ (tutti gli altri campi sono NaN come in Matlab)
    """
    targets = np.asarray(targets, dtype=float).ravel()
    outputs = np.asarray(outputs, dtype=float).ravel()

    # Filtro identico a Matlab
    ind = (targets >= 1) | (targets == 0)
    true_targets = targets[ind]
    true_outputs = outputs[ind]

    valid = ~np.isnan(true_outputs)
    true_targets = true_targets[valid]
    true_outputs = true_outputs[valid]

    # lnQ: solo dove entrambi > 0
    positive = (true_targets > 0) & (true_outputs > 0)
    if np.any(positive):
        lnq = float(np.sum((np.log(true_outputs[positive] / true_targets[positive])) ** 2))
    else:
        lnq = float('nan')

    return {
        'MSE':      float('nan'),
        'RMSE':     float('nan'),
        'MAE':      float('nan'),
        'lnQ':      lnq,
        'ErrorMean': float('nan'),
        'ErrorStd': float('nan'),
        'R':        float('nan'),
        'Rsq':      float('nan'),
        'Num':      float('nan'),
        'MASE':     float('nan'),
    }


def merge_kpi(train_kpi: dict, test_kpi: dict) -> dict:
    """
    Unisce i KPI di train e test in un unico dizionario,
    inserendo lnQ dal test nel risultato finale.
    Equivalente all'output fitMeasures in Matlab.
    """
    result = train_kpi.copy()
    result['lnQ'] = test_kpi['lnQ']
    return result


def kpi_to_array(kpi: dict) -> np.ndarray:
    """
    Converte il dizionario KPI in un array numpy nell'ordine identico a Matlab:
    [MSE, RMSE, MAE, lnQ, ErrorMean, ErrorStd, R, Rsq, Num, MASE]

    Utile per l'interfaccia con il backend C#.
    """
    return np.array([
        kpi['MSE'],
        kpi['RMSE'],
        kpi['MAE'],
        kpi['lnQ'],
        kpi['ErrorMean'],
        kpi['ErrorStd'],
        kpi['R'],
        kpi['Rsq'],
        kpi['Num'],
        kpi['MASE'],
    ])


def ci_poisson(values: np.ndarray, ci_type: bool = False,
               alpha: float = 0.05, zeta: float = 0.0) -> tuple:
    """
    Intervalli di confidenza Poisson.
    Equivalente a ciMeanPoisson() in Matlab.

    Parametri
    ---------
    values   : vettore di valori (forecast o fitted)
    ci_type  : False = Pearson approximation, True = exact chi-squared
    alpha    : livello di significatività (default 0.05 → CI 95%)
    zeta     : se > 0, aggiunge zeta*sqrt(u) alla banda superiore

    Ritorna
    -------
    (lower, upper) : array numpy dei limiti inferiore e superiore
    """
    from scipy import stats

    values = np.asarray(values, dtype=float)

    if ci_type:
        # Exact: chi-squared
        lower = stats.chi2.ppf(alpha / 2, 2 * values) / 2
        upper = stats.chi2.ppf(1 - alpha / 2, 2 * (values + 1)) / 2
    else:
        # Pearson approximation (default Matlab)
        a = stats.chi2.ppf(1 - alpha, 1)
        b = np.sqrt(a) * np.sqrt(values + a / 4)
        c = values + a / 2
        lower = np.maximum(np.zeros_like(c), c - b)
        upper = c + b

    if zeta <= 0:
        # Quantile Poisson (identico a Matlab poissinv)
        # Usa approssimazione normale per stabilità numerica
        lower = np.maximum(0, stats.poisson.ppf(alpha / 2, np.maximum(0, lower)))
        upper = stats.poisson.ppf(1 - alpha / 2, np.maximum(0, upper))
    else:
        upper = upper + zeta * np.sqrt(np.maximum(0, upper))
        lower = np.maximum(0, lower - zeta * np.sqrt(np.maximum(0, lower)))

    return lower, upper


# --- helper privato ---
def _empty_train_kpi() -> dict:
    nan = float('nan')
    return {
        'MSE': nan, 'RMSE': nan, 'MAE': nan, 'lnQ': nan,
        'ErrorMean': nan, 'ErrorStd': nan, 'R': nan,
        'Rsq': nan, 'Num': nan, 'MASE': nan,
    }
