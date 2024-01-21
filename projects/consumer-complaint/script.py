#%%

import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

# Importing data, save some for verification
np.random.seed(100)
DATA = pd.read_csv("complaints_data.csv")
SIZE = round(0.1*len(DATA))
RAND = np.random.permutation(len(DATA))
VERI = DATA.loc[RAND[:SIZE], ]
complaints = DATA.loc[RAND[SIZE:], ]

profile = ProfileReport(complaints)
profile.to_file("your_report.html")

print(complaints)


# %%
