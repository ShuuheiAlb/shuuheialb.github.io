#%%

import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

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
window_min = 7
df["Smoothed Max Energy"] = df["Energy"].rolling(window_max, center=True).max()
df["Smoothed Min Energy"] = df["Energy"].rolling(window_min, center=True).min()
df["Smoothed Max Min Energy"] = df["Smoothed Min Energy"].rolling(window_min, center=True).max()
for col in ["Smoothed Max Energy", "Smoothed Min Energy", "Smoothed Max Min Energy"]:
    is_nulls = df[col].isnull()
    df.loc[is_nulls, col] = df.loc[is_nulls, "Energy"]
train, test = df[:-lag], df[-lag:]

# Fit max energy
def sinusoidal_func(t, A, f, w, b):
    return A*np.sin(2*np.pi*f*(t + w)) + b
train_t = range(len(train))
test_t = range(len(test))
initial_guess = [600, 1/365, 300, 600]
popt, pcov = curve_fit(sinusoidal_func, train_t, train["Smoothed Max Min Energy"], p0=initial_guess)
print(popt, pcov)

plt.close()
plt.plot(train["Energy"], label='Original Data')
plt.plot(train_t, sinusoidal_func(train_t, *popt), label='Fitted Max Data')
plt.legend()
plt.show()

#%%

# Prediction
plt.close()
plt.plot(train["Energy"], label='Original Data')
plt.plot(range(400, len(train)), np.array([seasonal_average_naive_model(train[i:(i+400)]) for i in range(len(train)-400)]), label='Fitted Data')
plt.legend()
plt.show()

#%%

# Candidates
def simple_exp_smooth_model(df):
    model = SimpleExpSmoothing(df["Energy"])
    return model.fit().forecast(1).iloc[0]
def seasonal_recent_model(df):
    # Based on "stabilised" seasonal pattern + recency
    gap = 365
    return (df.iloc[(len(df)-gap-3):(len(df)-gap+3), ]["Energy"].mean() + df.iloc[len(df)-1, ]["Energy"])/2
def arima_model(df): # from auto arima: 1, 1, 1
    # ARIMA (1, 1, 1)
    exogenous_features = ["Mean Temperature", "Max Temperature", "Min Temperature"]
    model = auto_arima(
        train["Energy"],
        exogenous=train[exogenous_features],
        trace=True,
        error_action="ignore",
	    suppress_warnings=True)
    return model.predict(n_periods=1,  exogenous=df_valid[exogenous_features])
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
def sine_model(df):
    initial_guess = [350, 1/365, 300, 960]
    popt, _ = curve_fit(sinusoidal_func, range(len(df)), df["Energy"], p0=initial_guess)
    return sinusoidal_func(len(df), *popt)  


models = {
    "simple-exp-smooth": simple_exp_smooth_model, # 260, less variance
    "seasonal-recent": seasonal_recent_model, # 255
    "arima": arima_model,
    "k-mean": k_mean_model, # MSE 320 for (800-1600, given 400 data)
    "ma": moving_average_model,
    "naive": naive_model, # MSE 290
    "drift": drift_model, # MSE 290
    "sine": sine_model, # MSE 260
    "experimental": experimental_model
}

# Residuals, RMSE
start = 800
mid = 400
end = 1600
plt.close()
plt.draw()
arima_sample_model = ARIMA(df.iloc[start-mid:start]["Energy"], order=(30,1,0))
method = "seasonal-recent"
#[models[method](df[i-mid:i])  for i in range(start, end)]
residual = df.iloc[start:end]["Energy"] - np.array(arima_sample_model.fit().forecast())
plt.plot(range(start, end), residual, label=method)
print(np.sqrt(np.square(residual).mean()))
plt.legend()
plt.show()

# %%
