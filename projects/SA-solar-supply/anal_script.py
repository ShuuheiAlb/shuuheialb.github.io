#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
window = 10
sub_df["Smoothed Max Energy"] = sub_df["Energy"].rolling(window).max()
sub_df.loc[:window, "Smoothed Max Energy"] = sub_df.loc[:window, "Energy"]

# Fit model
gap = 365
train, test = sub_df[:-gap], sub_df[-gap:]
def sinusoidal_func(t, amplitude, frequency, phase, bitude, reduction_factor):
    return amplitude*np.sin(2*np.pi*frequency*t + phase) + bitude
t = range(len(train))
initial_guess = [350, 1/365, 180, 950, 0.2]
popt, pcov = curve_fit(sinusoidal_func, t, train["Smoothed Max Energy"], p0=initial_guess)
print(popt, pcov)

plt.plot(train["Date"], train["Energy"], label='Original Data')
plt.plot(train["Date"], sinusoidal_func(t, *popt), label='Fitted Model')
plt.legend()
plt.show()

# And the residual is weather dependent etc, i.e. time series

