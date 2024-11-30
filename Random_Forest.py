''''
Random Forest is an ensemble learning method that can be used for both classification and regression tasks. 
It works by creating a collection (or "forest") of decision trees and merging their results to improve prediction accuracy and control overfitting.
In a Random Forest classifier, each tree is built using a random subset of the data and features,
and the final prediction is made by aggregating the individual tree predictions (e.g., using majority voting for classification).
'''
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load dataset (using the Iris dataset as an example)
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for Random Forest)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest model
rf_model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{cm}')

# Plot Confusion Matrix
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(iris.target_names))
plt.xticks(tick_marks, iris.target_names)
plt.yticks(tick_marks, iris.target_names)

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# Hyperparameter tuning using GridSearchCV (optional)
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Display the best parameters and the corresponding accuracy
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

# Train a new model with the best parameters
best_rf_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred_best = best_rf_model.predict(X_test_scaled)

# Evaluate the best model
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f'Accuracy (Best Model): {accuracy_best:.4f}')
print(f'Classification Report (Best Model):\n{classification_report(y_test, y_pred_best)}')




'''
Data Loading and Preprocessing:

The Iris dataset is used for this classification task.
The data is split into training and test sets using train_test_split().
We standardize the features using StandardScaler to ensure the model performs optimally.
Random Forest Model:

A RandomForestClassifier is initialized and trained with the scaled training data using fit().
The number of trees in the forest is set using the n_estimators parameter (default is 100).
Model Evaluation:

The model makes predictions using the predict() method.
We evaluate the model's performance using:
Accuracy: The proportion of correct predictions.
Classification Report: Shows precision, recall, F1-score for each class.
Confusion Matrix: A matrix showing the true vs predicted labels.
Hyperparameter Tuning (Optional):

GridSearchCV is used to tune hyperparameters such as:
n_estimators: Number of trees in the forest.
max_depth: The maximum depth of the trees.
min_samples_split: The minimum number of samples required to split an internal node.
The cv=5 argument performs 5-fold cross-validation to choose the best hyperparameters.
Model with Best Hyperparameters:

After finding the best parameters, we retrain the model using the best configuration and evaluate the performance again.
Hyperparameters in Random Forest:
n_estimators: The number of trees in the forest. More trees usually improve performance, but also increase computation time.
max_depth: The maximum depth of each tree. Limiting depth helps prevent overfitting.
min_samples_split: The minimum number of samples required to split an internal node.
min_samples_leaf: The minimum number of samples required to be at a leaf node. Higher values prevent overfitting.
max_features: The number of features to consider when looking for the best split. You can use a subset of features to make the model more efficient.
bootstrap: Whether or not to use bootstrap sampling when building trees. Default is True.

'''
