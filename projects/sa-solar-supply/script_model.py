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

# Import data, set aside 10% based on forward-chain
#DATA = pd.read_csv("sol_data.csv")
#VERI = DATA[-round(0.1*len(DATA)):]
#sol = DATA[:-round(0.1*len(DATA))]
sol = pd.read_csv("sol_data.csv")

# Preview plot
for name in sol["Name"].unique():
    plt.close()
    plt.plot(sol[sol["Name"] == name]["Energy"], label=name)
    plt.title("Solar PV Generated Energy")
    plt.legend()
    plt.show()

# Select the data from establishment dates (actually should be in ETL)
sol_from_estb_dt = pd.DataFrame(sol)
for name in sol_from_estb_dt["Name"].unique():
    start_date_idx = sol[sol["Name"] == name].index[0]
    est_date_idx = sol[(sol["Name"] == name) & (sol["Energy"] > 0)].index[0]
    sol_from_estb_dt.drop(index = list(range(start_date_idx, est_date_idx)), inplace=True)
# Change header for StatsForecast
sol_sf = sol_from_estb_dt[["Name", "Date", "Energy", "Temperature", "Solar Irradiance"]] \
                .replace("", np.nan).dropna() \
                .rename(columns = {
                    "Name": "unique_id",
                    "Date": "ds",
                    "Energy": "y"
                })

#%%

# Modelling
models = [AutoARIMA(), AutoTheta(), WindowAverage(7)]
sf = StatsForecast(models, freq="D", df=sol_sf)

# Model selection: cross-val separated based on short vs long time series
ds_counts = sol_sf.groupby("unique_id").size()
ds_count_limit = 500
sol_sf_filter = sol_sf["unique_id"].isin(ds_counts[ds_counts > ds_count_limit].index)
sol_sf_long = sol_sf[sol_sf_filter]
sol_sf_short = sol_sf[-sol_sf_filter]
# Soooo long like 10 mins, so save it in pickle
if not isfile("save_model.pkl"):
    cv_sol_sf_long = sf.cross_validation(df=sol_sf_long, h=7, n_windows=5, step_size = 100)
    cv_sol_sf_short = sf.cross_validation(df=sol_sf_short, h=7, n_windows=10, step_size = 10)
    with open('save_model.pkl', 'wb') as outp:
        pickle.dump(cv_sol_sf_long, outp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(cv_sol_sf_short, outp, pickle.HIGHEST_PROTOCOL)
else:
    with open('save_model.pkl', 'rb') as inp:
        cv_sol_sf_long = pickle.load(inp)
        cv_sol_sf_short = pickle.load(inp)

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
model_error_sol_sf_long = evaluate_cross_validation(cv_sol_sf_long.reset_index(), mse)
model_error_sol_sf_short = evaluate_cross_validation(cv_sol_sf_short.reset_index(), mse)
model_error_sol_sf_long
model_error_sol_sf_short
# StatsForecast.plot(sol_sf_long, cv_sol_sf_long[["ds", "AutoARIMA", "AutoTheta", "WindowAverage"]])
# AIC, BIC?
#%%

# ====
# Note:
# 1. BNGSF1, BNGSF2, TBSF are very regular. Model: ARIMA
# 2. MWPS, PAREPW, HVWW; MBPS2 and MAPS2 are somewhat regular with a bit volatility
# 3. BOLIVAR, ADP; TB2SF and CBWWBA are like, wtf. Model: CrostonClassic

# Collect final predictions
pred_list = []
for col, row in model_error_sol_sf_long.iterrows():
    loc_str = col
    best_model = globals()[row["best_model"]] # Maybe: globals()[best_model_str]
    best_sf = StatsForecast([best_model], freq="D", sol=sol_sf[sol_sf["unique_id"] == loc_str])
    pred = best_sf.forecast(7, X_sol = []) # !! Exogenous future vars
    pred_list.append(pred)
preds = pd.concat(pred_list)
hists = sol_from_estb_dt

# Basic visualisation

import plotly.graph_objects as go
from plotly.subplots import make_subplots

selected_unique_id = ""
fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=hists["ds"],
                         y=hists["y"], mode='lines'))
# not this actually.


plt.figure(figsize=(10, 6))
plt.plot(timestamps, predicted_values, label='Predicted sol Power')
plt.xlabel('Time') 
plt.ylabel('sol Power')
plt.title('sol Power Prediction')
plt.legend()
plt.show()


#%%

# Potentially an alternative basic model: "stabilised" seasonal avg
def seasonal_recent_model(df):
    gap = 365
    return (df.iloc[(len(df)-gap-3):(len(df)-gap+3), ]["Energy"].mean() + df.iloc[len(df)-1, ]["Energy"])/2

