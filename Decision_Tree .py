'''
A Decision Tree Classifier is a machine learning algorithm used for classification tasks. 
It splits the data into branches to make decisions based on feature values, resulting in a tree-like structure.
Each internal node represents a feature, each branch represents a decision rule, and each leaf node represents a class label.
'''

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load dataset (using the Iris dataset as an example)
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for better model performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree Classifier
dt_model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = dt_model.predict(X_test_scaled)

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
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Display the best parameters and the corresponding accuracy
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

# Train a new model with the best parameters
best_dt_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred_best = best_dt_model.predict(X_test_scaled)

# Evaluate the best model
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f'Accuracy (Best Model): {accuracy_best:.4f}')
print(f'Classification Report (Best Model):\n{classification_report(y_test, y_pred_best)}')

# Visualize the Decision Tree (Optional)
plt.figure(figsize=(12, 8))
plot_tree(dt_model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True)
plt.title('Decision Tree Visualization')
plt.show()




'''

Data Loading and Preprocessing:

The Iris dataset is loaded for this example.
The data is split into training and testing sets using train_test_split().
We standardize the features using StandardScaler to improve model performance.
Decision Tree Classifier Model:

We initialize a DecisionTreeClassifier and fit it to the training data.
fit() trains the model on the data, and predict() is used to make predictions on the test data.
Model Evaluation:

We use accuracy to evaluate the model's performance.
Classification Report provides precision, recall, F1-score, and support for each class.
Confusion Matrix shows the number of correct and incorrect predictions in a matrix format.
We visualize the confusion matrix to understand how well the model performed on each class.
Hyperparameter Tuning (Optional):

GridSearchCV is used to search over a grid of hyperparameters, such as max_depth, min_samples_split, and min_samples_leaf.
The best parameters are found through cross-validation, and the model is retrained using those best parameters.
Decision Tree Visualization:

You can visualize the trained decision tree using plot_tree(). This helps to understand how decisions are being made at each node of the tree.
Hyperparameters in Decision Tree:
max_depth: The maximum depth of the tree. Limiting depth can help prevent overfitting.
min_samples_split: The minimum number of samples required to split an internal node. Higher values prevent creating nodes with few samples, which may cause overfitting.
min_samples_leaf: The minimum number of samples required to be at a leaf node. Setting this parameter helps prevent overfitting by ensuring each leaf node contains enough data points.
max_features: The number of features to consider when splitting a node. By default, it uses all features, but restricting the number of features can improve generalization.
criterion: The function to measure the quality of a split. Common values are 'gini' (Gini impurity) and 'entropy' (information gain).


''

