#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import data
df = pd.read_csv("extracted.csv")

# Extract features (including coordinates) and target (energy)
X = df[["Mean Temperature", "Max Temperature", "Min Temperature", "Longitude", "Latitude"]].values
y = df["Energy"].values

# Add date features (You may need to customize this based on your dataset)
df["Date"] = pd.to_datetime(df["Date"])
X = np.column_stack((X, df['Date'].dt.day, df['Date'].dt.dayofweek, df['Date'].dt.month, df['Date'].dt.year))

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

plot = df[["Date", "Energy"]].plot(x = "Date")

#%%
from statsmodels.tsa.stattools import adfuller
result = adfuller(df["Energy"].diff()[1:])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
  print('\t%s: %.3f' % (key, value))

sm.graphics.tsa.plot_acf(df["Energy"], lags=20)
sm.graphics.tsa.plot_pacf(df["Energy"], lags=20)
plt.show()

"""
# Create an ARIMAX model
(p, d, q) = (1, 1, 0)
model = sm.tsa.ARIMA(y, exog=X, order=(p, d, q))
model_fit = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")

# Now you have a unified model that takes geographical coordinates into account for energy prediction.
# You can use this model for making predictions.
forecast = model_fit.get_forecast(steps=10)  # Replace steps with the number of future steps to forecast
forecast_values = forecast.predicted_mean
confidence_intervals = forecast.conf_int()"""
# %%
