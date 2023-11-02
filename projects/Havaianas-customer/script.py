
# Why this dataset: Many retail/restaurants have POS receipt data.
#
# I wanna toy around with the most robust clusters, via k-medoid/PAM.
# Metrics: 1) sales, 2) num of items (mimicking frequency), 3) 
# 

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

receipts = pd.read_excel("receipts.xls")
# Set verification data aside

print(receipts.head())
print(receipts.info()) # Data formats and nulls
print(sum(receipts.duplicated()))
print(receipts.describe())
for column in receipts.columns:
    uniques = receipts[column].unique()
    print(f"{column}: {uniques[:np.minimum(30, len(uniques))]}")

# Some null columns on color_code and group => linked to style_x
# Format data (time, size), filter cols, group/degroup categories
# Then outliers

receipts_origin = receipts
receipts = receipts[receipts["amount_total"] < 300] # From histogram

plt.draw()
plt.hist(receipts["amount_total"], bins=30)
plt.show()

#%%

# Cluster


# %%
