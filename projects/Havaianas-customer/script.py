import kmedoids

c = kmedoids.fasterpam(distmatrix, 5)
print("Loss is:", c.loss)

# Generate sample data (replace with your actual data)
X = make_blobs(n_samples=100, n_features=8, centers=5, random_state=42)[0]

# Compute pairwise distances
distances = pairwise_distances(X)

# Perform k-medoids clustering using FastPAM algorithm
k = 3  # Number of clusters
kmedoids = KMedoids(n_clusters=k, metric='precomputed', method='pam', random_state=42)
kmedoids.fit(distances)

# Get cluster labels and medoid indices
cluster_labels = kmedoids.labels_
medoid_indices = kmedoids.medoid_indices_

# Print the cluster labels and medoids
for cluster in range(k):
    print(f"Cluster {cluster + 1} - Medoid Index: {medoid_indices[cluster]}")
    cluster_samples = X[cluster_labels == cluster]
    print(f"Samples in Cluster {cluster + 1}:")
    print(cluster_samples)
    print()