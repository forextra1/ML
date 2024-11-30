'''
DBSCAN is a clustering algorithm that is based on the density of data points.
It groups together points that are closely packed and marks points that are far away from any group as noise (outliers).
Unlike K-Means, DBSCAN does not require specifying the number of clusters beforehand. It has two key parameters:

eps (epsilon): The maximum distance between two points to be considered as neighbors.
min_samples: The minimum number of points to form a dense region (i.e., a cluster).
DBSCAN is particularly effective for:

Datasets with noise (outliers).
Datasets with clusters of different shapes and sizes

Core Points: A point is a core point if it has at least min_samples points (including itself) within a distance of eps.
Border Points: A point is a border point if it is not a core point but is within the eps distance of a core point.
Noise: A point is considered noise if it is neither a core point nor a border point.
'''


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Generate a sample dataset (You can replace this with your own dataset)
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Standardize the features (important for distance-based algorithms like DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Print the unique labels (the -1 label represents noise)
print(f"Unique labels (including noise): {np.unique(dbscan_labels)}")

# Plot the clustering results
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster Label')
plt.show()

# Evaluate the results
# (If you have ground truth labels, you can use clustering metrics such as adjusted Rand Index, F1-score, etc.)
# Example: Adjusted Rand Index (ARI)
from sklearn.metrics import adjusted_rand_score

# Assuming you have ground truth labels (here, we use 'y' from make_blobs for demonstration)
ari = adjusted_rand_score(y, dbscan_labels)
print(f"Adjusted Rand Index: {ari:.4f}")



''''
Data Preparation:

The code generates a synthetic dataset using make_blobs for demonstration. You can replace it with your own dataset.
StandardScaler is used to scale the features, which is important for distance-based algorithms like DBSCAN.
DBSCAN Clustering:

The DBSCAN algorithm is applied with two main parameters: eps and min_samples.
eps: The maximum distance between two points to consider them as neighbors. You should experiment with different values based on your data.
min_samples: The minimum number of points to form a dense cluster.
fit_predict() returns the cluster labels for each data point. A label of -1 indicates a noise point.
Visualization:

The clustered data is plotted using a scatter plot. Points that belong to the same cluster are assigned the same color. Noise points (label -1) are marked separately.
Evaluation (Optional):

The Adjusted Rand Index (ARI) is used to compare the clustering results with ground truth labels. If you have ground truth labels, you can calculate clustering metrics like ARI, F1-score, or silhouette score.
Parameters of DBSCAN:
eps: The radius that defines the neighborhood of a point.
min_samples: The minimum number of points required to form a dense region (a cluster).
metric: The distance metric used to calculate the distance between points (Euclidean by default).
algorithm: The algorithm used to compute nearest neighbors (auto, ball_tree, kd_tree, or brute).
leaf_size: The leaf size used in the ball_tree or kd_tree algorithms (useful for large datasets).
DBSCAN vs K-Means:
DBSCAN is density-based, meaning it groups together points that are close and have enough points in the neighborhood, whereas K-Means tries to form spherical clusters of fixed size, requiring the number of clusters to be specified.
DBSCAN can find clusters of arbitrary shape, while K-Means works best for spherical clusters.
DBSCAN can identify and label outliers, while K-Means doesn’t have an explicit mechanism for detecting outliers.

'''


