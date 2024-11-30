'''
Gaussian Mixture Models (GMM) is a probabilistic model used for clustering and density estimation.
Unlike K-Means, which assigns each data point to a single cluster, GMM assumes that the data is generated from a mixture of several Gaussian distributions, 
each corresponding to a cluster. GMM provides a more flexible approach than K-Means because it allows for clusters to have different shapes and densities.
'''


# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Create a sample dataset (you can replace this with your own dataset)
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Standardize the features (important for GMM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit the Gaussian Mixture Model
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
gmm.fit(X_scaled)

# Predict the labels for each point (cluster assignment)
labels = gmm.predict(X_scaled)

# Get the probabilities (responsibilities) of each data point belonging to each cluster
probabilities = gmm.predict_proba(X_scaled)

# Plot the data points with cluster assignments
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('Gaussian Mixture Model Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster Label')
plt.show()

# Visualize the Gaussian Mixtures (Gaussian ellipses)
plt.figure(figsize=(8, 6))

# Get the means and covariances of the fitted GMM
means = gmm.means_
covariances = gmm.covariances_

# Plot each Gaussian component
for i in range(4):
    mean = means[i]
    cov = covariances[i]
    
    # Create an ellipse for each Gaussian component
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    
    # Plot the ellipse
    ax = plt.gca()
    angle = 180.0 * angle / np.pi
    ell = plt.matplotlib.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color='r', alpha=0.5)
    ax.add_patch(ell)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('Gaussian Mixture Model Clusters with Ellipses')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Evaluate the GMM model using metrics like AIC or BIC
print(f"AIC: {gmm.aic(X_scaled):.2f}")
print(f"BIC: {gmm.bic(X_scaled):.2f}")


'''
Data Preparation:

We generate a synthetic dataset using make_blobs, which you can replace with your own dataset.
We use StandardScaler to standardize the data before applying GMM.
GMM Clustering:

We create a GaussianMixture object specifying the number of components (clusters) and fit it to the scaled data.
The fit() method is used to train the GMM, while predict() assigns cluster labels to each data point.
Visualization:

The clusters are plotted using a scatter plot with the labels assigned by GMM. The clusters are color-coded.
Additionally, ellipses are drawn to visualize the Gaussian components, showing the shape and spread of each cluster.
Model Evaluation:

We print the AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion), which are used for model selection. Lower values of AIC and BIC suggest a better fit of the model.
Choosing the Number of Clusters:
To select the optimal number of components for GMM, you can use metrics like AIC or BIC. By iterating over different values of n_components, you can compare the AIC or BIC values and choose the best model.


'''


