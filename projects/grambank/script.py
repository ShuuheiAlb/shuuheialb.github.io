
#%%

import numpy as np
import pandas as pd

DATA = pd.read_csv("values.csv")
VERI = DATA[-round(0.05*len(DATA)):]
values = DATA[:-round(0.05*len(DATA))]

# Most of missing value (75623) are ""
print(values.head())
print(values.info()) # Data formats and nulls
print(sum(values.duplicated()))
for col in values:
    print(values[col].value_counts())

df = values[["Parameter_ID", "Language_ID", "Value"]]
df_group_count = df.groupby(["Parameter_ID", "Language_ID"])['Value'].count()
df_group_count = 


