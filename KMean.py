'''
a popular unsupervised machine learning algorithm used for clustering.
Clustering is the process of grouping similar data points together into "clusters" based on certain features
it doesn't require labeled data. Instead, it learns patterns and structure from the input features alone.

-Initialization:
Choose K random points as the initial centroids. These can be selected randomly from the data points or using more sophisticated methods (e.g., KMeans++).
-Assignment Step:
Assign each data point to the nearest centroid based on Euclidean distance. This creates K clusters.
-Update Step:
After the points are assigned, calculate the new centroids by taking the average of all points assigned to each cluster.
-Repeat:
Repeat the Assignment and Update steps until the centroids no longer change (or change very little), indicating that the algorithm has converged.

K: The number of clusters. You must specify K before running the algorithm.
Max Iterations: The maximum number of iterations to run the algorithm.
Tolerance: A threshold to stop the algorithm if the centroids' movement is smaller than this value.
'''


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Generate some random data for demonstration
X = np.random.rand(100, 2)

# Define the number of clusters
k = 3

# Create the KMeans model
kmeans = KMeans(n_clusters=k, random_state=42)

# Fit the model
kmeans.fit(X)

# Get the cluster centers (centroids)
centroids = kmeans.cluster_centers_

# Predict the labels (which cluster each point belongs to)
labels = kmeans.labels_

# Visualize the clusters and centroids
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='red', s=200)  # Plot centroids
plt.title(f'KMeans Clustering (k={k})')
plt.show()



and------> 



import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Step 1: Generate or Load Data
# Example: Create a synthetic dataset for clustering (You can replace this with your dataset)
X, y = make_blobs(n_samples=500, centers=4, random_state=42)

# Step 2: Preprocessing (Scaling the Data)
# Standardize the features (important for KMeans)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply KMeans Clustering
# Define the number of clusters (k)
k = 4

# Initialize and fit the KMeans model
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

# Step 4: Get Cluster Centers and Labels
centroids = kmeans.cluster_centers_  # Cluster centroids
labels = kmeans.labels_  # Cluster labels for each data point

# Step 5: Visualize the Clusters
plt.figure(figsize=(8, 6))

# Plot the points with their cluster labels
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', marker='o', s=50, edgecolors='k', alpha=0.6)

# Plot the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

# Add labels and title
plt.title(f'KMeans Clustering (k={k})', fontsize=14)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Step 6: Evaluation (Optional)
# You can also use the inertia to evaluate the model (lower inertia = better clustering)
print(f'Inertia: {kmeans.inertia_}')



#and--------> 

'''
Choosing the Optimal Number of Clusters (k):

Instead of arbitrarily choosing k=4, you can use the Elbow Method to find the optimal number of clusters.
This involves running KMeans for different values of k and plotting the inertia to look for the "elbow" point,
where the inertia stops decreasing sharply.
'''
inertia = []
for k in range(1, 11):  # Test for k from 1 to 10
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()


#and------> 


'''
> Silhouette Score: 

The Silhouette Score is a metric used to evaluate how well-separated and well-formed the clusters are in a clustering algorithm like KMeans.
It measures the quality of the clustering by comparing two things:

Cohesion: How close the points in the same cluster are to each other.
Separation: How far the points in one cluster are from points in other clusters.

(Silhouette Score Evaluation):
- Clustering: The code runs KMeans clustering with different values of k (number of clusters) from 2 to 9 for a subset of liver pixels (sampled randomly).

- Silhouette Score Calculation: For each k, the code calculates the Silhouette Score using the KMeans model's predicted labels (the cluster assignments for the sample data).
A higher Silhouette score indicates better clustering.

- Best k Selection: The code selects the value of k that yields the highest Silhouette score as the "best" number of clusters for that slice of the CT scan.

- Visualization: The Silhouette scores for different values of k are plotted, showing how the score changes with the number of clusters.
This allows you to visually inspect which k leads to the most optimal clustering.
'''

    silhouette_scores = []
    best_score = -1
    best_k = None
    best_kmeans = None

    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(pixels_sample)
        sample_labels = kmeans.predict(pixels_sample)
        score = silhouette_score(pixels_sample, sample_labels)
        silhouette_scores.append(score)
        print(f"k: {k:<2} , Silhouette Score: {score:.3f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_kmeans = kmeans


#and------> 


import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Sample data (replace with your dataset)
# Assuming 'data' is a 2D array where rows are samples and columns are features
# Example: X = np.random.randn(100, 5)  # 100 samples, 5 features

# Data preprocessing
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)  # Standardize the data

# Sample size for silhouette score calculation (optional, to speed up)
sample_size = 1000  # You can change this based on your dataset size
n_samples = data_scaled.shape[0]
indices = shuffle(np.arange(n_samples), random_state=42)[:sample_size]
sample_data = data_scaled[indices]

# List of k values to evaluate
k_values = range(2, 11)  # From 2 clusters to 10 clusters

# Initialize variables to store results
silhouette_scores = []
best_score = -1
best_k = None
best_kmeans = None

# Evaluate silhouette scores for different k values
print("Evaluating silhouette scores:")
for k in k_values:
    # Fit the KMeans model
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(sample_data)
    
    # Calculate silhouette score
    labels = kmeans.predict(sample_data)
    score = silhouette_score(sample_data, labels)
    silhouette_scores.append(score)
    
    print(f"k: {k}, Silhouette Score: {score:.3f}")
    
    # Update best score and best k if needed
    if score > best_score:
        best_score = score
        best_k = k
        best_kmeans = kmeans

# Plot silhouette scores
plt.figure(figsize=(8, 6))
plt.plot(k_values, silhouette_scores, marker='o', color='b')
plt.title('Silhouette Score vs Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# Print the best k value
print(f"Best number of clusters (k) based on silhouette score: {best_k}")
print(f"Best silhouette score: {best_score:.3f}")

# Final clustering with best k
kmeans = best_kmeans
labels = kmeans.predict(data_scaled)

# Optionally, visualize the clusters (for 2D data)
if data_scaled.shape[1] == 2:
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, cmap='viridis')
    plt.title(f'Clustering with k={best_k}')
    plt.show()



#and-------> 

#Use KMeans for Prediction

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Example: Assume 'data' is your training dataset
# Replace this with your actual dataset
data = np.random.randn(100, 5)  # 100 samples, 5 features

# Step 1: Preprocess the data (standardize it)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)  # Standardize the data

# Step 2: Train the KMeans model
k = 3  # Set the number of clusters
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
kmeans.fit(data_scaled)  # Fit KMeans to the data

# Step 3: Predict cluster labels for new/unseen data points
new_data = np.random.randn(5, 5)  # 5 new samples, same number of features (5)
new_data_scaled = scaler.transform(new_data)  # Don't fit, only transform the new data

predicted_labels = kmeans.predict(new_data_scaled)  # Predict the cluster labels for new data

print("Predicted cluster labels for new data:", predicted_labels)


#and--------> 

'''
Approximating Data Using KMeans Centroids:
Here's how you might approximate the data after clustering with KMeans by using the cluster centroids:

Train a KMeans model on your data.
Assign each point to a cluster.
Replace each point with the centroid of its assigned cluster to get a reconstruction-like effect.
'''


import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Example: Assume 'data' is your original dataset
data = np.random.randn(100, 5)  # 100 samples, 5 features

# Step 1: Preprocess the data (standardize it)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)  # Standardize the data

# Step 2: Train the KMeans model
k = 3  # Set the number of clusters
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
kmeans.fit(data_scaled)  # Fit KMeans to the data

# Step 3: Predict the cluster labels for the data
cluster_labels = kmeans.predict(data_scaled)

# Step 4: Approximate the data by replacing each point with its cluster centroid
centroids = kmeans.cluster_centers_  # Get the centroids of the clusters
reconstructed_data = centroids[cluster_labels]  # Replace each point with the corresponding centroid

# Step 5: Inverse scale the data to get it back to the original scale
reconstructed_data_original_scale = scaler.inverse_transform(reconstructed_data)

# Step 6: Plot the original and reconstructed data (for comparison)
plt.figure(figsize=(10, 6))

# Plot the original data (first feature vs second feature)
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], color='blue', label='Original Data')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot the reconstructed data (first feature vs second feature)
plt.subplot(1, 2, 2)
plt.scatter(reconstructed_data_original_scale[:, 0], reconstructed_data_original_scale[:, 1], color='red', label='Reconstructed Data')
plt.title('Reconstructed Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

'''
Step 1: Standardize the original data using StandardScaler to make it more suitable for clustering.
Step 2: Fit a KMeans model with k = 3 (3 clusters).
Step 3: Use the predict() method to assign each data point to a cluster.
Step 4: Approximate the data by replacing each point with the corresponding centroid of the cluster it was assigned to.
Step 5: Apply inverse scaling using scaler.inverse_transform() to bring the data back to the original scale.
Step 6: Plot the original data and the reconstructed data (approximated by the centroids) for visual comparison.
''''


