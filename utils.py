import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# Load your time series data
df = pd.read_csv('your_time_series_data.csv', index_col='date', parse_dates=True)
ts = df['your_time_series_column']

# Initialize the TimeSeriesSplit object
tscv = TimeSeriesSplit(n_splits=5)

# Store the results
results = []

# Perform time series cross-validation
for train_index, test_index in tscv.split(ts):
    train, test = ts.iloc[train_index], ts.iloc[test_index]

    # Fit the model
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()

    # Make predictions
    predictions = model_fit.forecast(steps=len(test))

    # Evaluate the model
    mse = mean_squared_error(test, predictions)
    results.append(mse)

print(f'Mean MSE: {np.mean(results):.3f}')
