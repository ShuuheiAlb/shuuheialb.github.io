
#%%

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import jaccard_similarity_score
import matplotlib.pyplot as plt
import seaborn as sns

# For sensitivity test
np.random.seed(209)
DATA = pd.read_csv("values.csv")
values, VERI = train_test_split(DATA, test_size=0.05)

# Missing value (75K/420K)
# Param-language combos not appearing, but each param has > 75%
print(values.head())
print(values.info()) # Data formats and nulls
print(sum(values.duplicated()))
for col in values:
    print(values[col].value_counts())

# Imputing values
def jaccard_distance(x, y):
  return 1 - jaccard_similarity_score(x, y)
values_matrix = values[["Parameter_ID", "Language_ID", "Value"]] \
        .pivot(index="Parameter_ID", columns="Language_ID", values="Value")
values_matrix[values_matrix == "?"] = np.nan
values_matrix = values_matrix.astype(float)
nan_proportion = (values_matrix.isna().sum() / len(values_matrix))
values_matrix_dropped = values_matrix[values_matrix.columns[nan_proportion < 0.5]]
values_imputed = KNNImputer(n_neighbors=10,
                            metrics = jaccard_distance) \
                            .fit_transform(values_matrix_dropped)

from scipy.linalg import svd
u, s, vt = svd(values_imputed)
pca = PCA(n_components=0.95).fit(values_imputed)  
principalComponents = pca.transform(values_imputed)

# Run DBSCAN, hierarchial clustering, k-means
for eps in range(0.1, 0.6, 0.1):
    for min_samples in range(5, 21):
        db = DBSCAN(eps=eps, min_samples=10).fit(principalComponents)
        silhouette_avg = silhouette_score(principalComponents, db.labels_)
        print("Silhouette coefficient: ", silhouette_avg)







# %%
