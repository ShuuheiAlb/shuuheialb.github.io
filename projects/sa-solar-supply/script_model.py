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
#TEST = DATA[-round(0.1*len(DATA)):] # watch out for the missing exog data
#sol = DATA[:-round(0.1*len(DATA))]
sol = pd.read_csv("sol_data.csv")

# Select the data from establishment dates (actually should be in ETL)
for name in sol["Name"].unique():
    start_date_idx = sol[sol["Name"] == name].index[0]
    est_date_idx = sol[(sol["Name"] == name) & (sol["Energy"] > 0)].index[0]
    sol.drop(index = list(range(start_date_idx, est_date_idx)), inplace=True)
# Preview plot
for name in sol["Name"].unique():
    plt.close()
    plt.plot(sol[sol["Name"] == name]["Energy"], label=name)
    plt.title("Solar PV Generated Energy")
    plt.legend()
    plt.show()

#%%

# Model
sol_sf = sol[["Name", "Date", "Energy", "Temperature", "Solar Irradiance"]] \
                .replace("", np.nan).dropna() \
                .rename(columns = {
                    "Name": "unique_id",
                    "Date": "ds",
                    "Energy": "y"
                })
models = [AutoARIMA(), AutoTheta(), WindowAverage(7)]
sf = StatsForecast(models, freq="D", df=sol_sf)

# Potentially an alternative basic model: "stabilised" seasonal avg
def seasonal_recent_model(df):
    gap = 365
    return (df.iloc[(len(df)-gap-3):(len(df)-gap+3), ]["Energy"].mean() + df.iloc[len(df)-1, ]["Energy"])/2

# %%

# Cross-val separated based on short vs long time series
# Note:
# 1. BNGSF1, BNGSF2, TBSF are very regular. Model: ARIMA
# 2. MWPS, PAREPW, HVWW; MBPS2 and MAPS2 are somewhat regular with a bit volatility
# 3. BOLIVAR, ADP; TB2SF and CBWWBA are like, wtf. Model: CrostonClassic
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

#%%

# Calculate the error
# AIC, BIC?
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
print(model_error_sol_sf_long)
print(model_error_sol_sf_short)

#%%

# Collect final predictions
pred_list = []
for col, row in model_error_sol_sf_long.iterrows():
    loc_str = col
    best_model_str = row["best_model"]
    best_model_class = globals()[row["best_model"]]
    best_model = best_model_class() if best_model_str != "WindowAverage"  else best_model_class(7)
    sol_sf_loc = sol_sf[sol_sf["unique_id"] == loc_str]
    best_sf = StatsForecast([best_model], freq="D", df=sol_sf_loc[:-7])
    pred = best_sf.forecast(7, df=sol_sf_loc[:-7], X_df = sol_sf_loc[-7:].drop("y", axis=1))
    pred.columns.values[1] = "y"
    pred_list.append(pred)
preds = pd.concat(pred_list)
hists = sol_sf

# %%

# Basic visualisation
import plotly.graph_objects as go

selected_uid = "PAREPW"
selected_hists = hists[hists["unique_id"] == selected_uid]
selected_preds = preds[preds.index == selected_uid]
fig = go.Figure()
fig.add_trace(go.Scatter(x = selected_hists["ds"][-30:-6], y = selected_hists["y"][-30:-6], mode='lines', name = "Historic"))
fig.add_trace(go.Scatter(x = selected_hists["ds"][-7:], y = selected_hists["y"][-7:], mode='lines', name = "Actual"))
fig.add_trace(go.Scatter(x = selected_preds["ds"], y = selected_preds["y"], mode='lines', name = "Forecast"))
fig.update_layout(barmode = 'overlay', template = "plotly_white")
fig.update_layout(
    updatemenus = [dict(type = "buttons",
                        direction = "left",
                        buttons=list([
                            dict(
                                args=["type", "surface"],
                                label="3D Surface",
                                method="restyle"
                            ),
                            dict(
                                args=["type", "heatmap"],
                                label="Heatmap",
                                method="restyle"
                            )
                        ]),
                    pad={"r": 10, "t": 10}, showactive=True,
                    x=0.11, xanchor="left", y=1.1, yanchor="top")]
)
fig
