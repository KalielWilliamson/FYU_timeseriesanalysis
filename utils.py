import pandas as pd
import numpy as np
from collections import defaultdict
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
import seaborn as sns
from scipy.interpolate import LSQUnivariateSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class DataUtils:
    def __init__(self, length_scale_bounds=(1e-2, 1e2), alpha=1e-2):
        self.length_scale_bounds = length_scale_bounds
        self.alpha = alpha
        self.scalers = {}

    @staticmethod
    def read_data(filepath):
        df = pd.read_csv(filepath)
        df['Date Time'] = pd.to_datetime(df['Date Time'])
        df = df.set_index('Date Time')
        df = df.applymap(lambda x: float(x.replace(',', '')) if isinstance(x, str) else x)
        df = df[1:]
        df = df.interpolate()
        df = df.loc[df.index < '2024-04-22']
        return df

    @staticmethod
    def log_transform_open_interest(df):
        df['Open Interest'] = np.log(df['Open Interest'])
        return df

    def scale_data(self, df):
        for column in df.columns:
            scaler = StandardScaler()
            df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
            self.scalers[column] = scaler
        return df

    @staticmethod
    def stationarize_data(df, columns=None):
        if columns is None:
            columns = ['Open', 'High', 'Low', 'Close', 'VWAP', 'Open Interest', 'Implied Volatility']
        df = df.copy()

        for column in columns:
            df[column] = df[column] - df[column].shift(1)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.interpolate()
        df['x'] = df.index.map(pd.Timestamp.toordinal).values - df.index.map(pd.Timestamp.toordinal).values.min()
        df.set_index('x', inplace=True)

        return df

    def backfill_implied_volatility(self, df):
        num_missing_values = int(df['Implied Volatility'].isna().sum())
        series = df['Implied Volatility'].values.reshape(-1, 1)[:-1]

        X = np.arange(len(series) - num_missing_values).reshape(-1, 1)
        y = series[num_missing_values:]

        kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, length_scale_bounds=self.length_scale_bounds)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=self.alpha)
        gp.fit(X, y)

        X_pred = np.arange(num_missing_values).reshape(-1, 1)
        forecast = gp.predict(X_pred)

        df['Implied Volatility'][:num_missing_values] = forecast
        return df

    @staticmethod
    def transform_bspline_univariate(df, degree=3, num_knots=8):
        splines = []

        for col in df.columns:
            assert df[col].notna().all(), f"Column {col} contains missing values."
            y = df[col].values
            x = df.index.values
            t = np.linspace(x.min(), x.max(), num_knots + 2)[1:-1]
            spl = LSQUnivariateSpline(x, y, t=t, k=degree)
            y_spline = spl(x)
            mse = mean_squared_error(y, y_spline)
            splines.append({
                'column': col,
                'bspline': spl,
                't': t,
                'c': spl.get_coeffs(),
                'k': degree,
                'mse': mse
            })
        splines = pd.DataFrame(splines)
        splines.set_index('column', inplace=True)
        return splines

    @staticmethod
    def drop_na(df):
        return df.dropna()

    @staticmethod
    def remove_outliers(df, threshold=3.0):
        for column in df.columns:
            df = df[(np.abs(df[column] - df[column].mean()) / df[column].std()) <= threshold]
        return df


class VisualizationUtils:
    @staticmethod
    def visualize_data(df):
        num_cols = 3
        num_rows = -(-len(df.columns) // num_cols)

        plt.figure(figsize=(20 * num_cols, 10 * num_rows))
        for i, column in enumerate(df.columns, start=1):
            plt.subplot(num_rows, num_cols, i)
            plt.plot(df.index, df[column])
            plt.xlabel('Date Time')
            plt.ylabel(str(column))
            plt.title(column, fontsize=80)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_lagged_correlation(df, max_lag=30):
        lags, correlations = DataAnalysisUtils.lagged_correlation(df, max_lag)
        correlations = pd.DataFrame(correlations, index=lags)
        plt.figure(figsize=(10, 6))
        plt.plot(correlations, marker='o')
        plt.title('Lagged Time Series Correlation')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.legend(correlations.columns)
        plt.grid(True)
        plt.show()

    @staticmethod
    def visualize_correlation_matrix(df):
        corr = df.corr(method='spearman')

        sns.set_context("poster")
        plt.figure(figsize=(20, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)

        plt.show()


class DataAnalysisUtils:

    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def evaluate(self, df, fit_callback, predict_callback, column, ):
        ts = df[column]
        results = []
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        for train_index, test_index in tscv.split(ts):
            train, test = ts.iloc[train_index], ts.iloc[test_index]
            model = fit_callback(train)
            predictions = predict_callback(model, test)
            mse = mean_squared_error(test, predictions)
            results.append(mse)
        return np.mean(results)

    @staticmethod
    def lagged_correlation(time_series, max_lag):
        correlations = defaultdict(list)
        lags = range(1, max_lag + 1)
        for col in time_series.columns:
            for lag in lags:
                correlation = time_series[col].autocorr(lag=lag)
                correlations[col].append(correlation)
        return lags, correlations

    @staticmethod
    def adf_test(time_series):
        results = {}
        for col in time_series.columns:
            result = adfuller(time_series[col])
            results[col] = result
        return pd.DataFrame(results).T

    @staticmethod
    def autoarima_fit_callback(series):
        model = pm.auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True)
        return model

    @staticmethod
    def autoarima_forecast_callback(model, series):
        forecast = model.predict(n_periods=len(series))
        return forecast

