import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

np.random.seed(123)
date_rng = pd.date_range(start='2020-01-01', end='2023-01-01', freq='M')
data = np.random.randn(len(date_rng)) + np.arange(len(date_rng)) * 0.1

df = pd.DataFrame(data, columns = ['value'])

df['date'] = date_rng

df['date'] = df['date'].dt.strftime('%d.%m.%Y')

df.set_index('date', inplace=True)

df.to_excel('timeseries_data.xlsx')
print("DONE..")

df = pd.read_excel('timeseries_data.xlsx', index_col='date', parse_dates=True, date_parser=lambda x: pd.to_datetime(x, format='%d.%m.%Y'))

plt.figure(figsize=(10,6))
plt.plot(df.index, df['value'])
plt.title('Time Series Data')
plt.show()

arima_model = ARIMA(df['value'], order=(1,1,1))
arima_result = arima_model.fit()

print(arima_result.summary())

arima_forecast = arima_result.forecast(steps=12)

plt.figure(figsize=(10,6))
plt.plot(df.index, df['value'], label='Actuel Data')
plt.plot(pd.date_range(df.index[-1], periods=12, freq='M'), arima_forecast, label='ARIMA FORECAST', color='red')
plt.legend()
plt.show()

#SARIMA MODEL
sarima_model = SARIMAX(df['value'], order=(1,1,1), seasonal_order=(1,1,1,12))
sarima_result = sarima_model.fit()

print(sarima_result.summary())

sarima_forecast = sarima_result.forecast(steps=12)

plt.figure(figsize=(10,6))
plt.plot(df.index, df['value'], label='Actual Data')
plt.plot(pd.date_range(df.index[-1], periods=12, freq='M'), sarima_forecast, label='SARIMA FORECAST', color='red')
plt.title('SARIMA MODEL FORECAST')
plt.legend()
plt.show()