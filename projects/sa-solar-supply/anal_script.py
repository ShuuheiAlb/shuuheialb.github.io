#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoTheta, WindowAverage

# Import data, set aside 10% verification data
df = pd.read_csv("extracted.csv")
#DATA = pd.read_csv("extracted.csv")
#VERI = DATA[-round(0.1*len(DATA)):]
#df = DATA[:-round(0.1*len(DATA))]

for name in df["Name"].unique():
    plt.close()
    plt.plot(df[df["Name"] == name]["Energy"], label=name)
    plt.legend()
    plt.show()

# Select the data from establishment dates
df_est = pd.DataFrame(df)
for name in df_est["Name"].unique():
    start_date_idx = df[df["Name"] == name].index[0]
    est_date_idx = df[(df["Name"] == name) & (df["Energy"] > 0)].index[0]
    df_est.drop(index = list(range(start_date_idx, est_date_idx)), inplace=True)

#%%

# Trial nixtla
# Plan 3: MLForecast - LGBM/XGB/RF
df_nixtla = df[["Name", "Date", "Energy", "Temperature", "Solar Irradiance"]] \
                .replace("", np.nan).dropna() \
                .rename(columns = {
                    "Name": "unique_id",
                    "Date": "ds",
                    "Energy": "y"
                })

StatsForecast.plot(df_nixtla, plot_random = False)

models = [AutoARIMA(), AutoTheta(), WindowAverage(7)]
sf = StatsForecast(models, freq="D", df=df_nixtla)
cv_df_nixtla = sf.cross_validation(df=df_nixtla, h=7, n_windows=5, step_size = 100)

from utilsforecast.losses import mse
from utilsforecast.evaluation import evaluate
def evaluate_cross_validation(df, metric):
    models = df.drop(columns=['unique_id', 'ds', 'cutoff', 'y']).columns.tolist()
    evals = []
    # Calculate loss for every unique_id and cutoff.    
    for cutoff in df['cutoff'].unique():
        eval_ = evaluate(df[df['cutoff'] == cutoff], metrics=[metric], models=models)
        evals.append(eval_)
    evals = pd.concat(evals)
    evals = evals.groupby('unique_id').mean(numeric_only=True) # Averages the error metrics for all cutoffs for every combination of model and unique_id
    evals['best_model'] = evals.idxmin(axis=1)
    return evals
eval_df_nixtla = evaluate_cross_validation(cv_df_nixtla.reset_index(), mse)
eval_df_nixtla

#%%

# Train-test
lag = 365
window_max = 13
df["Smoothed Max Energy"] = df["Energy"].rolling(window_max, center=True).max()
is_nulls = df["Smoothed Max Energy"].isnull()
df.loc[is_nulls, "Smoothed Max Energy"] = df.loc[is_nulls, "Energy"]

# Other candidates
def seasonal_recent_model(df):
    # Based on "stabilised" seasonal pattern + recency
    gap = 365
    return (df.iloc[(len(df)-gap-3):(len(df)-gap+3), ]["Energy"].mean() + df.iloc[len(df)-1, ]["Energy"])/2

from scipy.optimize import curve_fit
def sinusoidal_func(t, A, f, w, b):
    return A*np.sin(2*np.pi*f*(t + w)) + b
def sine_model(df): #  min 400 samples
    initial_guess = [350, 1/365, 300, 960]
    popt, _ = curve_fit(sinusoidal_func, range(len(df)), df["Energy"], p0=initial_guess)
    return sinusoidal_func(len(df), *popt)  

models = {
    # MSE 1 or 800-1600 with 400 data
    # MSE 2 for 800-1600 with 30 data
    # MSE 3 for 1000-1100 with 400 data
    "seasonal-recent": seasonal_recent_model, # MSE 255 (1), -, 250 (3)
                                              # => min 365 samples
    "sine": sine_model, # MSE 260 (1), -, 230 (3)
                        # => min 400? samples
}

# Residuals, RMSE
start = 800
mid = 400
end = 1600
plt.close()
plt.draw()
method = "seasonal-recent"
#[models[method](df[i-mid:i])  for i in range(start, end)]
residual = df.iloc[start:end]["Energy"] - np.array([models[method](df[i-mid:i]) for i in range(start, end)])
plt.plot(range(start, end), residual, label=method)
plt.legend()
plt.show()
print(np.sqrt(np.square(residual).mean()))