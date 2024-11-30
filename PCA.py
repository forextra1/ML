from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Example data with 3 features
data = np.array([[2, 3, 1],
                 [3, 4, 2],
                 [4, 5, 3],
                 [5, 6, 4]])

# Step 1: Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Step 2: Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 components
data_pca = pca.fit_transform(data_scaled)

print(f'Original data shape: {data.shape}')
print(f'Transformed data shape: {data_pca.shape}')
print(f'Explained variance ratio: {pca.explained_variance_ratio_}')






#and ------------>>  

# Import necessary libraries
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris  # For example purposes, using the Iris dataset

# Step 1: Load and prepare the dataset
# (You can replace this with your own dataset)
data = load_iris()
X = data.data  # Features
y = data.target  # Labels

# Step 2: Data Preprocessing (Scaling)
# PCA is affected by the scale of the data, so we standardize it
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA to reduce dimensionality
# Set the number of components you want to keep (for example, 2 components for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 4: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Step 5: Train a machine learning model
# Here we use RandomForestClassifier as an example
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Step 7: Output results
print(f'Accuracy of the model: {accuracy * 100:.2f}%')
# Optional: Explore the explained variance of each principal component
print(f'Explained variance ratio: {pca.explained_variance_ratio_}')










#and -------------------> 
pca = PCA().fit(x_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
'''
The explained_variance_ratio_ gives the percentage of variance each principal component explains. This tells you how much of the total variance is captured by each
'''

cumulative_variance = explained_variance_ratio.cumsum()
'''
cumulative_variance = explained_variance_ratio.cumsum()
'''


'''
PCA is a technique used to reduce the dimensionality of a dataset while preserving as much variance as possible. Here's how it works:

PCA takes your original features and combines them in a new way.
The new features (called principal components) are linear combinations of the original features.
The first principal component captures the most variance (the most "spread" in the data).
The second principal component captures the second most variance, but is uncorrelated with the first.

explained variance ratio: This is important because it tells you how much of the original data's variance is captured by each principal component.

After PCA has been applied, we often want to know how much total variance is explained by a certain number of principal components.
This is where cumulative variance comes into play:
The cumulative explained variance is the running total of the variance explained by the first few principal components.


'''

from sklearn.decomposition import PCA


scaler = StandardScaler()
x_scaled = scaler.fit_transform(df.iloc[:, :-1])  

pca = PCA().fit(x_scaled)

explained_variance_ratio = pca.explained_variance_ratio_

cumulative_variance = explained_variance_ratio.cumsum()

#>

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel('number of components')
plt.ylabel('cumulative explained var')
plt.title('cumulative explained var vs number of components')
plt.axhline(y=0.90, color='r')  # 90%  var
plt.axhline(y=0.95, color='g')  #  95%  var
plt.show()

n_components_90 = (cumulative_variance >= 0.90).argmax() + 1
n_components_95 = (cumulative_variance >= 0.95).argmax() + 1

print(f"optimal number of components for 90% var: {n_components_90}")
print(f"optimal number of components for 95% var: {n_components_95}")









