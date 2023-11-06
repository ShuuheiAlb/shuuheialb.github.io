#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA

# Import data
DATA = pd.read_csv("extracted.csv")

# I'll set aside 10% verification data
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
window_max = 13
df["Smoothed Max Energy"] = df["Energy"].rolling(window_max, center=True).max()
is_nulls = df["Smoothed Max Energy"].isnull()
df.loc[is_nulls, "Smoothed Max Energy"] = df.loc[is_nulls, "Energy"]
train, test = df[:-lag], df[-lag:]

plt.close()
plt.plot(train["Energy"], label='Original Data')
plt.legend()
plt.show()

#%%

# Find best ARIMA
train_scaled = train["Energy"]/train["Smoothed Max Energy"] # ARIMA(1,1,1)
train_double_diff = train["Energy"].diff(365).dropna() # ARIMA(2,0,2)
model = auto_arima(
        train_scaled,
        trace=True,
        error_action="ignore",
	    suppress_warnings=True)

#%%

# Candidates
def simple_exp_smooth_model(df):
    model = SimpleExpSmoothing(df["Energy"])
    return model.fit().forecast(1).iloc[0]
def seasonal_recent_model(df):
    # Based on "stabilised" seasonal pattern + recency
    gap = 365
    return (df.iloc[(len(df)-gap-3):(len(df)-gap+3), ]["Energy"].mean() + df.iloc[len(df)-1, ]["Energy"])/2
def arima_model(df):
    # Based on pmdarima
    # put the last 100
    df_scaled = df["Energy"]/df["Smoothed Max Energy"]
    model_fit = ARIMA(df_scaled, order=(1,1,1)).fit()
    return model_fit.predict(start=len(df), end=len(df)) * \
                np.array(df["Smoothed Max Energy"])[len(df)-1-(window_max+1)//2]
def experimental_model(df):
    return 0

# Benchmark
def mean_model(df):
    return df["Energy"].mean()
def k_mean_model(df, k=7):
    return df.iloc[(len(df)-k-1):(len(df)-1), ]["Energy"].mean()
def naive_model(df):
    return df.iloc[len(df)-1, ]["Energy"]
def seasonal_model(df):
    gap = 365
    return df.iloc[len(df)-gap, ]["Energy"]
def drift_model(df):
    start = df.iloc[0, ]
    end = df.iloc[len(df)-1, ]
    return end["Energy"] + (end["Energy"]-start["Energy"])/(end["Date"]-start["Date"]).days * 1
def sinusoidal_func(t, A, f, w, b):
    return A*np.sin(2*np.pi*f*(t + w)) + b
def sine_model(df): #  min 400 samples
    initial_guess = [350, 1/365, 300, 960]
    popt, _ = curve_fit(sinusoidal_func, range(len(df)), df["Energy"], p0=initial_guess)
    return sinusoidal_func(len(df), *popt)  


models = {
    "simple-exp-smooth": simple_exp_smooth_model, # 260, less variance
    "seasonal-recent": seasonal_recent_model, # min 365 samples, 255
    "arima": arima_model, #
    "k-mean": k_mean_model, # MSE 320 for (800-1600, given 400 data)
    "naive": naive_model, # MSE 290
    "drift": drift_model, # MSE 290
    "sine": sine_model, # min 400? samples, MSE 260
    "experimental": experimental_model
}

# Residuals, RMSE
start = 800
mid = 100
end = 1600
plt.close()
plt.draw()
method = "arima"
#[models[method](df[i-mid:i])  for i in range(start, end)]
residual = df.iloc[start:end]["Energy"] - np.array([models[method](df[i-mid:i]) for i in range(start, end)])
plt.plot(range(start, end), residual, label=method)
print(np.sqrt(np.square(residual).mean()))
plt.legend()
plt.show()

# %%

# Prediction
plt.close()
plt.plot(train["Energy"], label='Original Data')
plt.plot(range(400, len(train)), np.array([seasonal_average_naive_model(train[i:(i+400)]) for i in range(len(train)-400)]), label='Fitted Data')
plt.legend()
plt.show()