import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)
date_range = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
seasonal = 10 + 5 * np.sin(2 * np.pi * date_range.dayofyear / 365)
noise = np.random.normal(0, 1, len(date_range)) 
energy_daily = seasonal + noise
data_daily = pd.DataFrame({'date': date_range, 'energy': energy_daily}).set_index('date')

data_weekly = data_daily.resample('W').mean()

def build_and_evaluate_model_with_seasonality(data, label):
    data = data.copy()
    data['day'] = np.arange(len(data))
    data['sin'] = np.sin(2 * np.pi * data['day'] / 365)
    data['cos'] = np.cos(2 * np.pi * data['day'] / 365)

    X = data[['day', 'sin', 'cos']].values
    y = data['energy'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    plt.figure(figsize=(10, 4))
    plt.plot(data.index, y, label='Actual')
    plt.plot(data.index[len(X_train):], y_pred, label='Predicted', linestyle='--')
    plt.title(f'{label} Data Forecast (з сезонністю) | RMSE: {rmse:.2f}')
    plt.legend()
    plt.show()

    return rmse

rmse_daily = build_and_evaluate_model_with_seasonality(data_daily, 'Daily')
rmse_weekly = build_and_evaluate_model_with_seasonality(data_daily, 'Weekly')

print(f"RMSE (щоденні дані): {rmse_daily:.2f}")
print(f"RMSE (тижневі середні): {rmse_weekly:.2f}")
