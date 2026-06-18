# Test di verifica installazione ambiente DGS Forecast Engine
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SimpleExponentialSmoothing, AutoARIMA, HoltWinters
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from scipy import stats
from sklearn import metrics

print("=" * 50)
print("DGS Forecast Engine — Verifica ambiente")
print("=" * 50)

import numpy
import pandas
import statsforecast
import neuralforecast
import torch

print(f"numpy         : {numpy.__version__}")
print(f"pandas        : {pandas.__version__}")
print(f"statsforecast : {statsforecast.__version__}")
print(f"neuralforecast: {neuralforecast.__version__}")
print(f"torch         : {torch.__version__}")
print(f"GPU disponibile: {torch.cuda.is_available()}")

print()
print("Test calcolo SMA su serie di esempio...")
data = np.array([10, 12, 11, 14, 13, 15, 16, 14, 17, 18, 16, 19,
                 20, 18, 21, 22, 20, 23, 24, 22, 25, 26, 24, 27], dtype=float)
lag = 3
sma = np.convolve(data, np.ones(lag)/lag, mode='valid')
print(f"Serie input    : {data[:6]} ...")
print(f"SMA (lag={lag})   : {sma[:6].round(2)} ...")

print()
print("Tutto OK — ambiente pronto!")