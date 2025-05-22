import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

np.random.seed(42)
date_range = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
seasonal = 10 + 5 * np.sin(2 * np.pi * date_range.dayofyear / 365)
noise = np.random.normal(0, 1, len(date_range))
energy = seasonal + noise
data = pd.DataFrame({'ds': date_range, 'y': energy}).set_index('ds')

train = data.iloc[:-30]
test = data.iloc[-30:]
forecast_horizon = len(test)


def exp_smoothing_forecast(train, test):
    model = SimpleExpSmoothing(train['y']).fit()
    forecast = model.forecast(len(test))
    rmse = np.sqrt(mean_squared_error(test['y'], forecast))
    return forecast, rmse

exp_forecast, exp_rmse = exp_smoothing_forecast(train, test)


def arima_forecast(train, test):
    model = ARIMA(train['y'], order=(2, 0, 2))
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test['y'], forecast))
    return forecast, rmse

arima_forecast_vals, arima_rmse = arima_forecast(train, test)


def prophet_forecast(train, test):
    df_train = train.reset_index().rename(columns={'ds': 'ds', 'y': 'y'})
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(df_train)

    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)

    y_pred = forecast[['ds', 'yhat']].set_index('ds').loc[test.index]
    rmse = np.sqrt(mean_squared_error(test['y'], y_pred['yhat']))

    return y_pred['yhat'], rmse


prophet_forecast_vals, prophet_rmse = prophet_forecast(train, test)


plt.figure(figsize=(12, 6))
plt.plot(data.index, data['y'], label='Actual', alpha=0.6)
plt.plot(test.index, exp_forecast, label=f'Exp Smoothing (RMSE: {exp_rmse:.2f})', linestyle='--')
plt.plot(test.index, arima_forecast_vals, label=f'ARIMA (RMSE: {arima_rmse:.2f})', linestyle='--')
plt.plot(test.index, prophet_forecast_vals, label=f'Prophet (RMSE: {prophet_rmse:.2f})', linestyle='--')
plt.title("Прогноз енергоспоживання (30 днів)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"RMSE (Exp Smoothing): {exp_rmse:.2f}")
print(f"RMSE (ARIMA): {arima_rmse:.2f}")
print(f"RMSE (Prophet): {prophet_rmse:.2f}")
