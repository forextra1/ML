
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
# Replace 'your_dataset.csv' with the path to your dataset
data = pd.read_csv('your_dataset.csv')

# Inspect the data
print(data.head())
print(data.info())

# Separate features (X) and target variable (y)
# Replace 'target_column' with the actual column name of your target variable
X = data.drop(columns=['target_column'])
y = data['target_column']

# Perform train-test split
# Adjust test_size (e.g., 0.2 for 20% test set) and random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the resulting datasets
print("Training set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# The data is now ready for model training
# Example: Save or pass it to a machine learning model
# model.fit(X_train, y_train)

# Optionally save the splits to disk for later use
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
