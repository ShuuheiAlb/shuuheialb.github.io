#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pmdarima as pm

# Import data
df = pd.read_csv("extracted.csv")

# Extract features (including coordinates) and target (energy)
X = df[["Mean Temperature", "Max Temperature", "Min Temperature", "Longitude", "Latitude"]].values
y = df["Energy"].values

# Add date features (You may need to customize this based on your dataset)
df["Date"] = pd.to_datetime(df["Date"])
X = np.column_stack((X, df['Date'].dt.day, df['Date'].dt.dayofweek, df['Date'].dt.month, df['Date'].dt.year))

# By exploraion, the max is sinusoid
sub_df = df[df["Name"] == "BNGSF1"]
sub_df.index = sub_df["Date"]
window = 14
sub_df["Smoothed Max Energy"] = sub_df["Energy"].rolling(window, center=True).max()
is_nulls = sub_df["Smoothed Max Energy"].isnull()
sub_df.loc[is_nulls, "Smoothed Max Energy"] = sub_df.loc[is_nulls, "Energy"]

gap = 365
train, test = sub_df[:-gap], sub_df[-gap:]

# %%
#==== Check for seasonal trend
import statsmodels.api as sm

dh = train["Energy"].asfreq("M")
decomposition = sm.tsa.seasonal_decompose(dh, model='additive', 
extrapolate_trend='freq') #additive or multiplicative is data specific
fig = decomposition.plot()
plt.plot(train["Energy"])
plt.show()

# %%
# Fit model
def sinusoidal_func(t, A1, f1, w1, A2, f2, w2, b):
    return A1*np.cos(2*np.pi*f1*t + w1) + A2*np.sin(2*np.pi*f2*t + w2) + b
train_t = range(len(train))
test_t = range(len(test))
initial_guess = [350, 1/365, 180, 350, 1/365, 180, 950]
popt, pcov = curve_fit(sinusoidal_func, train_t, train["Smoothed Max Energy"], p0=initial_guess)

print(popt, pcov)


plt.plot(train["Energy"][:160], label='Original Data')
#plt.plot(train["Date"], sinusoidal_func(train_t, *popt), label='Fitted Model') # assuming normal dist error
plt.legend()
plt.show()


#%%

# And the ratio is weather dependent etc, i.e. time series
residual_train = sinusoidal_func(train_t, *popt) - train["Energy"]
model = pm.auto_arima(train["Energy"], m = 4) # 4 years of data
print(model.summary()) # ARIMA(1, 1, 1)

plt.close()
plt.plot(train_t, train["Energy"], label='Residual')
plt.plot(train_t, model.predict(n_periods=len(train_t)), label='Predicted Residual')
plt.show()


#%%

def energy_predict(t):
    return sinusoidal_func(t, *popt) - model.predict(n_periods=len(t))

plt.close()
plt.plot(test_t, test["Energy"], label='Test Data')
plt.plot(test_t, energy_predict(test_t), label='Test Model') 
plt.legend()
plt.show()
