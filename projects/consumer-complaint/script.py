#%%

import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns

# Importing data, save some for verification
np.random.seed(100)
DATA = pd.read_csv("complaints_data.csv")
SIZE = round(0.1*len(DATA))
RAND = np.random.permutation(len(DATA))
VERI = DATA.loc[RAND[:SIZE], ]
complaints = DATA.loc[RAND[SIZE:], ]

profile = ProfileReport(complaints)
profile.to_file("complaint_result.html")

# Preview, data types
print(complaints.head())
print(complaints.info())
# Data ranges, missing values, correlation
category_counts = complaints['Product'].value_counts().sort_values(ascending=False)
sns.barplot(x=category_counts.index, y=category_counts.values)
#sns.heatmap(complaints.isna(), cmap="Greens")

# ..
dummies = pd.get_dummies(df['Category']).rename(columns=lambda x: 'Category_' + str(x))
df = pd.concat([df, dummies], axis=1)
df = df.drop(['Category'], inplace=True, axis=1)
#sns.heatmap(complaints.drop([""], axis=1).corr(), cmap="Greens", annot=False)
plt.xticks(rotation=45)
plt.show()

# Date should be date format, remove 
# retain tags + consumer_disputed, 

# %%

# Jump to prediction, yooooooooo


