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











# Optional: Explore the explained variance of each principal component
print(f'Explained variance ratio: {pca.explained_variance_ratio_}')
