#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoTheta, WindowAverage

from os.path import isfile
import pickle
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mse

# Import data, set aside 10% verification data
#DATA = pd.read_csv("extracted.csv")
#VERI = DATA[-round(0.1*len(DATA)):]
#df = DATA[:-round(0.1*len(DATA))]

df = pd.read_csv("extracted.csv")

# Preview plot
for name in df["Name"].unique():
    plt.close()
    plt.plot(df[df["Name"] == name]["Energy"], label=name)
    plt.title("Solar PV Generated Energy")
    plt.legend()
    plt.show()

# Select the data from establishment dates (actually should be in ETL)
df_est = pd.DataFrame(df)
for name in df_est["Name"].unique():
    start_date_idx = df[df["Name"] == name].index[0]
    est_date_idx = df[(df["Name"] == name) & (df["Energy"] > 0)].index[0]
    df_est.drop(index = list(range(start_date_idx, est_date_idx)), inplace=True)

#%%

# Modelling using StatsForecast
df_sf = df_est[["Name", "Date", "Energy", "Temperature", "Solar Irradiance"]] \
                .replace("", np.nan).dropna() \
                .rename(columns = {
                    "Name": "unique_id",
                    "Date": "ds",
                    "Energy": "y"
                })
models = [AutoARIMA(), AutoTheta(), WindowAverage(7)]
sf = StatsForecast(models, freq="D", df=df_sf)

# Model selection: cross-val separated based on short vs long time series
ts_lens = df_sf.groupby("unique_id").size()
ts_len_limit = 500
df_sf_filter = df_sf["unique_id"].isin(ts_lens[ts_lens > ts_len_limit].index)
df_sf_long = df_sf[df_sf_filter]
df_sf_short = df_sf[-df_sf_filter]
# Soooo long like 10 mins, so save it in pickle
if not isfile("crossval_statsforecast.pkl"):
    cv_df_sf_long = sf.cross_validation(df=df_sf_long, h=7, n_windows=5, step_size = 100)
    cv_df_sf_short = sf.cross_validation(df=df_sf_short, h=7, n_windows=10, step_size = 10)
    with open('crossval_statsforecast.pkl', 'wb') as outp:
        pickle.dump(cv_df_sf_long, outp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(cv_df_sf_short, outp, pickle.HIGHEST_PROTOCOL)
else:
    with open('crossval_statsforecast.pkl', 'rb') as inp:
        cv_df_sf_long = pickle.load(inp)
        cv_df_sf_short = pickle.load(inp)

# Calculate the error
def evaluate_cross_validation(df, metric):
    models = df.drop(columns=['unique_id', 'ds', 'cutoff', 'y']).columns.tolist()
    evals = []    
    for cutoff in df['cutoff'].unique():
        eval_ = evaluate(df[df['cutoff'] == cutoff], metrics=[metric], models=models)
        evals.append(eval_)
    evals = pd.concat(evals)
    evals = evals.groupby('unique_id').mean(numeric_only=True) # Averages the error metrics for all cutoffs for every combination of model and unique_id
    evals['best_model'] = evals.idxmin(axis=1)
    return evals
error_df_sf_long = evaluate_cross_validation(cv_df_sf_long.reset_index(), mse)
error_df_sf_short = evaluate_cross_validation(cv_df_sf_short.reset_index(), mse)
error_df_sf_long
error_df_sf_short
# StatsForecast.plot(df_sf_long, cv_df_sf_long[["ds", "AutoARIMA", "AutoTheta", "WindowAverage"]])
#%%

# ====
# Note:
# 1. BNGSF1, BNGSF2, TBSF are very regular
# 2. MWPS, PAREPW, HVWW; MBPS2 and MAPS2 are somewhat regular with a bit volatility
# 3. BOLIVAR, ADP; TB2SF and CBWWBA are like, wtf. Need 

# Basic visualisation



#%%

# Potentially an alternative basic model: "stabilised" seasonal avg
def seasonal_recent_model(df):
    gap = 365
    return (df.iloc[(len(df)-gap-3):(len(df)-gap+3), ]["Energy"].mean() + df.iloc[len(df)-1, ]["Energy"])/2

from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from numba import njit
from window_ops.expanding import expanding_mean
from window_ops.rolling import rolling_mean

@njit
def rolling_mean_28(x):
    return rolling_mean(x, window_size=28)
fcst = MLForecast(
    models=models,
    freq='D',
    lags=[7, 14],
    lag_transforms={
        1: [expanding_mean],
        7: [rolling_mean_28]
    },
    date_features=['dayofweek'],
    target_transforms=[Differences([1])],
)