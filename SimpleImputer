import numpy as np
from sklearn.impute import SimpleImputer

# Example data with missing values (NaN)
X = np.array([[1, 2, np.nan], [3, np.nan, 6], [7, 8, 9]])

# Create an imputer for filling missing values with the mean
imputer = SimpleImputer(strategy='mean')

# Learn the mean of each column and replace NaNs
imputer.fit(X)
X_imputed = imputer.transform(X)

print("Original Data:\n", X)
print("Imputed Data:\n", X_imputed)
