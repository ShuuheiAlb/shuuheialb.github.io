#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm

# Import data
DATA = pd.read_csv("extracted.csv")

# I'll set aside 10% verification dataa
DATA_LOC1 = DATA[DATA["Name"] == "BNGSF1"]
VERI = DATA_LOC1[-round(0.1*len(DATA_LOC1)):]
df = DATA_LOC1[:-round(0.1*len(DATA_LOC1))]

# Extract features (including coordinates, date, temp) and target (energy)
X = df[["Mean Temperature", "Max Temperature", "Min Temperature", "Longitude", "Latitude"]].values
y = df["Energy"].values
df["Date"] = pd.to_datetime(df["Date"])
X = np.column_stack((X, df['Date'].dt.day, df['Date'].dt.dayofweek, df['Date'].dt.month, df['Date'].dt.year))

# Train-test
lag = 365
window = 13
train, test = df[:-lag], df[-lag:]
train["Smoothed Max Energy"] = train["Energy"].rolling(window, center=True).max()
is_nulls = train["Smoothed Max Energy"].isnull()
train.loc[is_nulls, "Smoothed Max Energy"] = train.loc[is_nulls, "Energy"]

# Fit max energy
def sinusoidal_func(t, A, f, w, b):
    return A*np.sin(2*np.pi*f*(t + w)) + b
train_t = range(len(train))
test_t = range(len(test))
initial_guess = [350, 1/365, 300, 960]
popt, pcov = curve_fit(sinusoidal_func, train_t, train["Smoothed Max Energy"], p0=initial_guess)
print(popt, pcov)

plt.close()
plt.plot(train["Energy"], label='Original Data')
plt.plot(train_t, sinusoidal_func(train_t, *popt), label='Fitted Max Data')
plt.legend()
plt.show()

#%%
# Regularised by max energy, seasonal diff (year, week) 
scaled_seasonal_diff_train = (train["Energy"]/train["Smoothed Max Energy"]).diff(lag).dropna()
plt.close()
plt.plot(train["Energy"].diff(lag))
plt.show()

#%%
# Model: (A_t/max - A_(t-365)/max)

# Benchmark
def mean_model(df):
    return df["Energy"].mean()
def naive_model(df):
    return df.iloc[len(df)-1, ]["Energy"]
def seasonal_model(df):
    gap = 365
    return df.iloc[len(df)-gap, ]["Energy"]
def seasonal_naive_model(df):
    gap = 365
    return (df.iloc[len(df)-gap, ]["Energy"] + df.iloc[len(df)-1, ]["Energy"])/2
def drift_model(df):
    start = df.iloc[0, ]
    end = df.iloc[len(df)-1, ]
    return end["Energy"] + (end["Energy"]-start["Energy"])/(end["Date"]-start["Date"]).days * 1
def sine_model(df):
    popt, pcov = curve_fit(sinusoidal_func, train_t, df["Energy"])
    return sinusoidal_func(train_t[-1]+1, *popt)

benchmarks = {
    "mean": mean_model,
    "naive": naive_model,
    "seasonal": seasonal_model,
    "seasonal-naive": seasonal_naive_model,
    "drift": drift_model,
    "sine": sine_model
}

# Test
start = 800
mid = 400
end = 1600
plt.close()
plt.draw()
method = "sine"
plt.plot(range(start, end), df.iloc[start:end]["Energy"] -
                            np.array([benchmarks[method](df[i-mid:i]) for i in range(start, end)]), label=method)
plt.legend()
plt.show()

# %%
