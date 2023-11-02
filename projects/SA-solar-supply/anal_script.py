#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm

# Import data
df = pd.read_csv("extracted.csv")

# Extract features (including coordinates) and target (energy)
X = df[["Mean Temperature", "Max Temperature", "Min Temperature", "Longitude", "Latitude"]].values
y = df["Energy"].values

# Add date features (You may need to customize this based on your dataset)
df["Date"] = pd.to_datetime(df["Date"])
X = np.column_stack((X, df['Date'].dt.day, df['Date'].dt.dayofweek, df['Date'].dt.month, df['Date'].dt.year))

#%%
# Pick one area first
sub_df = df[df["Name"] == "BNGSF1"]
window = 13
# exponential smooth?
sub_df["Smoothed Max Energy"] = sub_df["Energy"].rolling(window, center=True).max()
is_nulls = sub_df["Smoothed Max Energy"].isnull()
sub_df.loc[is_nulls, "Smoothed Max Energy"] = sub_df.loc[is_nulls, "Energy"]

lag = 365
train, test = sub_df[:-lag], sub_df[-lag:]

plt.draw()
plt.plot(train["Energy"], label="Energy")
plt.show()

# Seasonal plot?

# %%
# Fit model on max energy
def sinusoidal_func(t, A, f, w, b):
    return A*np.sin(2*np.pi*f*(t + w)) + b
train_t = range(len(train))
test_t = range(len(test))
initial_guess = [350, 1/365, 300, 960]
popt, pcov = curve_fit(sinusoidal_func, train_t, train["Smoothed Max Energy"], p0=initial_guess)
print(popt, pcov)

plt.draw()
plt.plot(train["Energy"], label='Original Data')
plt.plot(train_t, sinusoidal_func(train_t, *popt), label='Fitted Max Data')
plt.legend()
plt.show()

#%%

# Regularised by max energy, seasonal diff (year, week) 
scaled_seasonal_diff_train = (train["Energy"]/train["Smoothed Max Energy"]).diff(lag).dropna()
scaled_seasonal_diff2_train = scaled_seasonal_diff_train.diff(7).dropna()

mu, std = norm.fit(scaled_seasonal_diff2_train) # std = 0.5
x = np.linspace(-2, 2, 100)
print(mu, std)

plt.draw()
#plt.plot(scaled_seasonal_diff2_train)
plt.plot(x, p, 'k', linewidth=2)
plt.hist(scaled_seasonal_diff2_train, density=True, bins=200) # somewhat a sum of uniform
plt.legend()
plt.show()
#%%
# Model: (A_t/max - A_(t-365)/max)

plt.plot(train_t, train["Energy"], label='Residual')
plt.plot(train_t, model.predict(n_periods=len(train_t)), label='Predicted Residual')
plt.show()
